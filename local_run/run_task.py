#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
import logging
import random
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional


def resolve_base_dir() -> str:
    """
    期望结构：
      /.../evalscope/local_run/run_task.py
      /.../evalscope/local_run/configs/default.json
    """
    cur = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(cur, "configs")):
        return cur
    return os.path.dirname(cur)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize(s: str) -> str:
    # 用于 tag / 目录名
    return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])


def normalize_gpus_csv(gpus_csv: str) -> List[str]:
    gpus_csv = (gpus_csv or "").strip()
    if not gpus_csv:
        return ["0"]
    parts = [p.strip() for p in gpus_csv.split(",") if p.strip()]
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"--gpus 必须是逗号分隔整数，例如 0 / 0,1 / 4,7。你给的是: {gpus_csv}")
    return parts


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("EvalBench")


def set_global_seed(seed: int):
    """
    只做“常规可复现”，不要强制 deterministic_algorithms，
    否则你之前遇到的 CuBLAS deterministic 报错会回来。
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ✅ 不强制 deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # torch.use_deterministic_algorithms(True)  # ❌ 不要开
    except Exception:
        pass

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass


def build_generation_config(cfg: Dict[str, Any], timeout_s: float, seed: int) -> Dict[str, Any]:
    """
    generation 参数都在 python 里（你要求“conf 就一个，别再拆生成 conf 文件”）。
    为防止 AIME 某题无限长推理卡住，max_tokens/max_new_tokens 限到 4096（强烈建议）。
    """
    # 允许你未来在 default.json 里加一个可选字段 generation_override 来覆盖默认值（不强制）
    override = cfg.get("generation_override", {}) if isinstance(cfg.get("generation_override", {}), dict) else {}

    gen = {
        "timeout": timeout_s,
        "batch_size": 1,

        "max_tokens": 16384,
        "max_new_tokens": 16384,

        "top_p": 0.95,
        "temperature": 0.6,
        "do_sample": True,
        "top_k": 20,

        "n": 1,
        "num_return_sequences": 1,
        "seed": seed,

        
        "chat_template_kwargs": {
            "add_generation_prompt": True
            
            },
    }

    # 覆盖（如果你以后真想只改 json，不改 python）
    gen.update(override)
    # 保证 add_generation_prompt 不被不小心关掉
    if "chat_template_kwargs" not in gen or not isinstance(gen["chat_template_kwargs"], dict):
        gen["chat_template_kwargs"] = {"add_generation_prompt": True}
    gen["chat_template_kwargs"]["add_generation_prompt"] = True

    return gen


def build_tasks_from_config(cfg: Dict[str, Any], batch_tag: str, only: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """
    自动识别 subset：
      - 如果 subset_list 存在且长度>1：拆成多个子任务（dataset + subset）
      - 否则：一个任务（dataset，不指定 subset）
    返回 (dataset_name, tag, subset_or_empty)
    """
    datasets_cfg = cfg.get("datasets", {})
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        raise ValueError("default.json 里 datasets 配置为空或格式不对。")

    only_set = set(only) if only else None
    tasks: List[Tuple[str, str, str]] = []

    for ds_name, ds_cfg in datasets_cfg.items():
        if only_set is not None and ds_name not in only_set:
            continue
        if not isinstance(ds_cfg, dict):
            raise ValueError(f"datasets.{ds_name} 不是对象 dict")

        subset_list = ds_cfg.get("subset_list", [])
        if isinstance(subset_list, list) and len(subset_list) > 1:
            for subset in subset_list:
                subset = str(subset)
                tag = f"{batch_tag}__{sanitize(ds_name)}__{sanitize(subset)}"
                tasks.append((ds_name, tag, subset))
        else:
            tag = f"{batch_tag}__{sanitize(ds_name)}"
            tasks.append((ds_name, tag, ""))

    if not tasks:
        raise ValueError("根据 --only 过滤后没有任何任务可跑。")
    return tasks


def dispatch(tasks: List[Tuple[str, str, str]], gpus: List[str], runner_path: str) -> int:
    """
    GPU 工人池：
      - 每个 GPU 同时最多跑 1 个任务（避免互抢显存/速度变慢）
      - 任务自动排队，哪个 GPU 空了就跑下一个
    """
    queue = tasks[:]
    running: Dict[str, subprocess.Popen] = {}
    running_meta: Dict[str, Tuple[str, str, str]] = {}
    exit_code = 0

    def start_one(gpu: str, task: Tuple[str, str, str]):
        ds, tag, subset = task
        cmd = [
            sys.executable, runner_path,
            "--mode", "worker",
            "--dataset", ds,
            "--tag", tag,
            "--gpu", gpu,
        ]
        if subset:
            cmd += ["--subset", subset]
        print(f"[DISPATCH] GPU {gpu} START: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd)
        running[gpu] = proc
        running_meta[gpu] = task

    # 先填满所有 GPU
    for gpu in gpus:
        if queue:
            start_one(gpu, queue.pop(0))

    # 轮询等待
    while running:
        time.sleep(2)
        finished = []
        for gpu, proc in running.items():
            ret = proc.poll()
            if ret is not None:
                ds, tag, subset = running_meta[gpu]
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[DISPATCH] GPU {gpu} DONE: dataset={ds} subset={subset or '(auto)'} tag={tag} => {status}", flush=True)
                if ret != 0 and exit_code == 0:
                    exit_code = ret
                finished.append(gpu)

        for gpu in finished:
            running.pop(gpu, None)
            running_meta.pop(gpu, None)
            if queue:
                start_one(gpu, queue.pop(0))

    return exit_code


def run_one_worker(cfg: Dict[str, Any], dataset: str, tag: str, subset: str, gpu: str) -> int:
    """
    单个任务执行器（绑定单 GPU）
    """
    # ✅ 每个 worker 只看见一张 GPU：你给的是物理 id（如 4/7）
    if gpu:
        if not gpu.isdigit():
            raise ValueError(f"--gpu 必须是单个整数 id，例如 0 或 7。你给的是: {gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        # 让日志里看得更清楚
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    outputs_root = cfg.get("outputs_root") or cfg.get("outputs_dir")
    logs_root = cfg.get("logs_root") or cfg.get("logs_dir")
    if not outputs_root or not logs_root:
        raise ValueError("default.json 必须包含 outputs_root 和 logs_root（或 outputs_dir/logs_dir）。")

    common = cfg.get("common_args", {})
    seed = int(common.get("seed", 42))
    timeout_s = float(common.get("timeout_ms", 60000)) / 1000.0
    eval_batch_size = int(common.get("eval_batch_size", 1))
    use_cache = bool(common.get("use_cache", False))

    set_global_seed(seed)

    datasets_cfg = cfg.get("datasets", {})
    if dataset not in datasets_cfg:
        raise KeyError(f"Dataset '{dataset}' not found in configs/default.json. Available: {list(datasets_cfg.keys())}")

    ds_cfg = datasets_cfg[dataset]
    dataset_id = ds_cfg["dataset_id"]

    # subset：worker 若收到 subset 就只跑该 subset，否则用 config（通常 dispatch 会传 subset）
    if subset.strip():
        subset_list = [subset.strip()]
    else:
        subset_list = ds_cfg.get("subset_list", ["default"])

    # 输出/日志目录
    work_dir = os.path.join(outputs_root, dataset, tag)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    logger = setup_logger(os.path.join(logs_root, f"{dataset}_{tag}.log"))

    logger.info(f"Running dataset={dataset}, subset_list={subset_list}")
    logger.info(f"Model: {cfg['model_path']}")
    logger.info(f"dataset_id: {dataset_id}")
    logger.info(f"Seed={seed}, timeout_s={timeout_s}, eval_batch_size={eval_batch_size}, use_cache={use_cache}")
    if gpu:
        logger.info(f"Bound single GPU (physical id) = {gpu} via CUDA_VISIBLE_DEVICES")

    # ✅ import evalscope 必须在设置 CUDA_VISIBLE_DEVICES 之后
    from evalscope import TaskConfig, run_task

    generation_config = build_generation_config(cfg, timeout_s=timeout_s, seed=seed)

    dataset_args = {
        dataset: {
            "dataset_id": dataset_id,
            "subset_list": subset_list,
        }
    }

    task_cfg = TaskConfig(
        model=cfg["model_path"],
        datasets=[dataset],
        dataset_args=dataset_args,
        generation_config=generation_config,
        work_dir=work_dir,
        eval_batch_size=eval_batch_size,
        seed=seed,
        use_cache=use_cache,
        no_timestamp=True,
    )

    # meta 方便复现
    with open(os.path.join(work_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(task_cfg.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    run_task(task_cfg)
    return 0


def main():
    base_dir = resolve_base_dir()
    cfg_path = os.path.join(base_dir, "configs", "default.json")
    cfg = load_json(cfg_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dispatch", choices=["dispatch", "worker"],
                        help="dispatch: one command runs everything with GPU scheduling; worker: internal single task")
    # dispatch 参数
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated physical GPU ids. e.g. 0 or 0,1,2,3 or 4,7")
    parser.add_argument("--tag", type=str, default="",
                        help="Batch tag. If empty, auto timestamp.")
    parser.add_argument("--only", type=str, default="",
                        help="Optional: only run these datasets (comma-separated keys in default.json), e.g. aime25,math_500")

    # worker 参数（内部用）
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--gpu", type=str, default="")

    args = parser.parse_args()

    if args.mode == "worker":
        if not args.dataset or not args.tag:
            raise ValueError("worker 模式必须提供 --dataset 和 --tag")
        return_code = run_one_worker(cfg, dataset=args.dataset, tag=args.tag, subset=args.subset, gpu=args.gpu)
        raise SystemExit(return_code)

    # dispatch 模式（你日常只用这个）
    gpus = normalize_gpus_csv(args.gpus)

    if args.tag.strip():
        batch_tag = sanitize(args.tag.strip())
    else:
        batch_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    only_list = None
    if args.only.strip():
        only_list = [x.strip() for x in args.only.split(",") if x.strip()]

    tasks = build_tasks_from_config(cfg, batch_tag=batch_tag, only=only_list)

    print(f"[DISPATCH] Config: {cfg_path}")
    print(f"[DISPATCH] GPUs: {gpus}")
    print(f"[DISPATCH] Batch tag: {batch_tag}")
    print(f"[DISPATCH] Total tasks: {len(tasks)}")
    for ds, tag, subset in tasks:
        print(f"  - dataset={ds} subset={subset or '(auto)'} tag={tag}")
    print("", flush=True)

    runner_path = os.path.abspath(__file__)
    code = dispatch(tasks=tasks, gpus=gpus, runner_path=runner_path)
    raise SystemExit(code)


if __name__ == "__main__":
    main()


# ========================= run_task.py CLI 参数说明 =========================
#
# 该脚本支持两种模式：dispatch（调度） / worker（执行单任务）
#
# ----------------------------- 1) 通用参数 -----------------------------
#
# --mode {dispatch,worker}
#   - 默认: dispatch
#   - dispatch: 读取 default.json -> 自动拆分 subset -> 按 GPU 列表排队分发 -> 启动多个 worker 子进程
#   - worker  : 只执行一个最小任务（dataset + 可选 subset），通常由 dispatch 自动启动
#
# --tag <string>
#   - 作用：
#       dispatch: 作为本次批次 batch_tag，用于生成每个任务的输出 tag（目录名的一部分）
#       worker  : 必填（用于输出目录名与日志名），通常是 dispatch 生成的 `${batch_tag}__${dataset}__${subset}`
#   - 规则：会被 sanitize() 处理，非 [a-zA-Z0-9_-] 的字符替换成 "_"
#   - 默认：
#       dispatch: 若不传，会自动生成 "run_%Y%m%d_%H%M%S"
#       worker  : 必须传，否则直接报错
#
# ----------------------------- 2) dispatch 模式参数 -----------------------------
#
# --gpus "0,1,2,3"
#   - 仅在 dispatch 有意义（worker 不用）
#   - 含义：可用的“物理 GPU id 列表”，用逗号分隔
#   - 示例：
#       --gpus "0"           -> 只用物理卡 0
#       --gpus "0,1,3,6"     -> 只在物理卡 0/1/3/6 上跑任务
#       --gpus "4,7"         -> 只在物理卡 4/7 上跑任务
#   - 行为：每个 GPU 同时最多跑 1 个任务；任务排队，空闲 GPU 取下一个任务
#
# --only "aime25,math_500"
#   - 仅在 dispatch 有意义
#   - 含义：只跑 default.json 的 datasets 中列出的这些 key（逗号分隔）
#   - 示例：
#       --only "aime25"              -> 只跑 aime25（会自动拆 AIME2025-I/II）
#       --only "math_500,aime25"     -> 只跑这两个
#   - 注意：必须写 default.json 里 datasets 的 key 名，不是 pretty_name，不是 dataset_id。
#
# ----------------------------- 3) worker 模式参数（内部用，dispatch 会自动填） -----------------------------
#
# --dataset <string>
#   - worker 必填
#   - 取值：default.json 的 datasets key，比如 "aime25" / "aime24" / "math_500"
#
# --subset <string>
#   - worker 可选
#   - 含义：若传，则只跑该 subset；否则使用 default.json 中该 dataset 的 subset_list（或默认 ["default"]）
#   - 示例：
#       --subset "AIME2025-I"
#       --subset "Level 3"
#
# --gpu <int>
#   - worker 可选（但 dispatch 调用 worker 时一定会传）
#   - 含义：绑定到某个“物理 GPU id”
#   - 实现：会设置环境变量 CUDA_VISIBLE_DEVICES=<gpu>
#   - 重要细节：
#       CUDA_VISIBLE_DEVICES=4 表示“本进程只看见物理卡 4，并且它会变成逻辑的 cuda:0”
#       所以 transformers/evalscope 里看到的通常是 cuda:0，但实际对应物理卡 4
#
# ----------------------------- 4) 常见用法 -----------------------------
#
# A) 一条命令跑全部（自动拆 subset + 自动分配 GPU）
#   python run_task.py --mode dispatch --gpus "0,1,2,3"
#
# B) 只跑某些数据集
#   python run_task.py --mode dispatch --gpus "0,1" --only "aime25,math_500"
#
# C) 指定批次 tag（输出目录可控）
#   python run_task.py --mode dispatch --gpus "0,1" --tag "0120_3"
#
# D) 单独调试 worker（只跑一个 subset）
#   python run_task.py --mode worker --dataset aime25 --subset "AIME2025-I" --gpu 0 --tag "debug_aime25_I"
#
# ======================================================================================
