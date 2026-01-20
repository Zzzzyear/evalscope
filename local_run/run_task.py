import os
import json
import argparse
import logging
import random
from typing import Any, Dict

from evalscope import TaskConfig, run_task


def setup_logger(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("EvalBench")


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_base_dir() -> str:
    """
    默认结构：
      /.../evalscope/local_run/run_task.py
      /.../evalscope/local_run/configs/default.json
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_dir, "configs")):
        return current_dir
    return os.path.dirname(current_dir)


def set_global_seed(seed: int):
    """
    目标：尽量可复现 + 绝不因为 deterministic_algorithms 触发 CuBLAS 报错。

    ✅ 做：
      - random / numpy / torch / transformers set_seed
    ❌ 不做：
      - torch.use_deterministic_algorithms(True)  （会报错）
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

        # 不强制 deterministic（避免 CuBLAS deterministic 报错）
        # 这两行更偏“性能”，也更贴近大多数人默认行为
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # ❌ 千万别开：会让你在 Qwen3 RoPE + CuBLAS 上直接 RuntimeError
        # torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass


def build_generation_config(timeout_s: float, seed: int) -> Dict[str, Any]:
    """
    说明：
      - 同时写 max_tokens / max_new_tokens，避免不同后端字段差异
      - Qwen3-4B-Base：不写 enable_thinking
    """
    return {
        "timeout": timeout_s,
        "batch_size": 1,

        "max_tokens": 32768,
        "max_new_tokens": 32768,

        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "do_sample": True,

        "n": 1,
        "num_return_sequences": 1,

        "seed": seed,

        "chat_template_kwargs": {
            "add_generation_prompt": True
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["math_500", "aime24", "aime25"])
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    base_dir = resolve_base_dir()
    cfg_path = os.path.join(base_dir, "configs", "default.json")
    print(f"[INFO] Loading config from: {cfg_path}")
    cfg = load_json(cfg_path)

    outputs_root = cfg.get("outputs_root") or cfg.get("outputs_dir")
    logs_root = cfg.get("logs_root") or cfg.get("logs_dir")
    if not outputs_root:
        raise ValueError("Config missing 'outputs_root' (or outputs_dir)")
    if not logs_root:
        raise ValueError("Config missing 'logs_root' (or logs_dir)")

    ds_name = args.dataset
    if ds_name not in cfg["datasets"]:
        raise ValueError(f"Dataset '{ds_name}' not found. Available: {list(cfg['datasets'].keys())}")

    ds_cfg = cfg["datasets"][ds_name]
    common = cfg["common_args"]

    seed = int(common["seed"])
    timeout_s = float(common["timeout_ms"]) / 1000.0

    set_global_seed(seed)

    work_dir = os.path.join(outputs_root, ds_name, args.tag)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    logger = setup_logger(os.path.join(logs_root, f"{ds_name}_{args.tag}.log"))
    logger.info(f"Running {ds_name} on model: {cfg['model_path']}")
    logger.info(f"Seed={seed}, timeout_s={timeout_s}")
    logger.info(f"Dataset remote id: {ds_cfg.get('dataset_id')}, subsets: {ds_cfg.get('subset_list')}")

    generation_config = build_generation_config(timeout_s=timeout_s, seed=seed)

    # ✅ 用“远端 dataset_id 全称”最省心，不再传 data_files/extra_params
    runtime_dataset_args = {
        ds_name: {
            "dataset_id": ds_cfg["dataset_id"],
            "subset_list": ds_cfg.get("subset_list", ["default"])
        }
    }

    task_cfg = TaskConfig(
        model=cfg["model_path"],
        datasets=[ds_name],
        dataset_args=runtime_dataset_args,
        generation_config=generation_config,
        work_dir=work_dir,
        eval_batch_size=int(common["eval_batch_size"]),
        seed=seed,
        use_cache=bool(common["use_cache"]),
        no_timestamp=True
    )

    with open(os.path.join(work_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(task_cfg.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    run_task(task_cfg)


if __name__ == "__main__":
    main()
