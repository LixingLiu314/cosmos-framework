"""Cosmos3 policy server compatible with StarVLA RoboCasa-GR1 evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

STARVLA_REPO = Path(os.environ.get("STARVLA_REPO", "/root/workspace/lixing/starVLA"))
COSMOS_REPO = Path(os.environ.get("COSMOS_REPO", "/root/workspace/lixing/cosmos-framework"))
EXAMPLES_DIR = Path(__file__).resolve().parent

# Remove script directory from sys.path to avoid shadowing real packages (e.g. hydra)
_script_dir = str(EXAMPLES_DIR)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
# Add repos to path (append to avoid priority conflicts)
for path in (str(STARVLA_REPO), str(COSMOS_REPO), _script_dir):
    if path not in sys.path:
        sys.path.append(path)

from cosmos3_forward_probe import (  # noqa: E402
    build_cosmos3_policy_sample,
    build_data_batch_from_sample,
    resolve_checkpoint_path,
    transform_cosmos3_policy_sample,
)

DEFAULT_CHECKPOINT_REPO = "nvidia/Cosmos3-Nano-Policy-DROID"
DEFAULT_OUTPUT_DIR = Path("/tmp/cosmos3_robocasa_policy_server")
DEFAULT_ACTION_HORIZON = 16
ROBOCASA29_ACTION_DIM = 29
GR1_44_ACTION_DIM = 44
DEFAULT_RAW_ACTION_DIM = ROBOCASA29_ACTION_DIM
DEFAULT_MAX_ACTION_DIM = 64
ROBOCASA29_DOMAIN_ID = 21
GR1_44_DOMAIN_ID = 31
DEFAULT_DOMAIN_ID = ROBOCASA29_DOMAIN_ID
DEFAULT_ACTION_SCHEMA = "gr1_44"
DEFAULT_ACTION_DENORM = "auto"
DEFAULT_ACTION_HISTORY = "auto"
DEFAULT_GR1_ACTION_STATS_PATH = (
    COSMOS_REPO / "cosmos_framework/data/vfm/action/datasets/gr1_lerobot_normalization.json"
)
DEFAULT_GR1_STATE_STATS_ROOT = Path(
    os.environ.get("GR1_DATA_ROOT", "/root/workspace/mengya/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot")
)
DEFAULT_FPS = 20
WAN22_VAE_CACHE_PATH = Path(
    "/root/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/"
    "snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
)
_ORIGINAL_HF_DOWNLOAD = None
ROBOCASA29_PARTS = (
    ("left_arm", 0, 7),
    ("right_arm", 7, 14),
    ("left_hand", 14, 20),
    ("right_hand", 20, 26),
    ("waist", 26, 29),
)
GR1_29_ACTION_DIM = 29
GR1_29_DOMAIN_ID = 31  # same domain_id as gr1_44

# 29D active parts in training order: left_arm, right_arm, left_hand, right_hand, waist
GR1_29_PARTS = (
    ("left_arm", 0, 7),
    ("right_arm", 7, 14),
    ("left_hand", 14, 20),
    ("right_hand", 20, 26),
    ("waist", 26, 29),
)

# Mapping from 44D GR1 original indices to 29D active-parts order
_GR1_44_TO_29_SLICES = [
    (0, 7),    # left_arm   -> 44D[0:7]
    (22, 29),  # right_arm  -> 44D[22:29]
    (7, 13),   # left_hand  -> 44D[7:13]
    (29, 35),  # right_hand -> 44D[29:35]
    (41, 44),  # waist      -> 44D[41:44]
]



def _cached_wan22_hf_download(cmd_args: list[str]) -> str:
    if (
        len(cmd_args) >= 2
        and cmd_args[0] == "Wan-AI/Wan2.2-TI2V-5B"
        and cmd_args[-1] == "Wan2.2_VAE.pth"
        and WAN22_VAE_CACHE_PATH.exists()
    ):
        logging.info("Using cached Wan2.2 VAE checkpoint: %s", WAN22_VAE_CACHE_PATH)
        return str(WAN22_VAE_CACHE_PATH)
    if _ORIGINAL_HF_DOWNLOAD is None:
        raise RuntimeError("Original checkpoint downloader was not registered")
    return _ORIGINAL_HF_DOWNLOAD(cmd_args)


def patch_cached_wan_vae_download() -> None:
    """Avoid re-downloading the Wan2.2 VAE when the HF cache already has it."""
    if not WAN22_VAE_CACHE_PATH.exists():
        logging.info("Cached Wan2.2 VAE not found; Cosmos downloader will handle it")
        return
    import cosmos_framework.utils.checkpoint_db as checkpoint_db

    global _ORIGINAL_HF_DOWNLOAD
    if checkpoint_db._hf_download is _cached_wan22_hf_download:
        return
    _ORIGINAL_HF_DOWNLOAD = checkpoint_db._hf_download
    checkpoint_db._hf_download = _cached_wan22_hf_download


def _checkpoint_run_dir(checkpoint_path: str | Path) -> Path | None:
    """Find the training run directory that owns a checkpoint."""
    path = Path(checkpoint_path).expanduser()
    for candidate in (path, *path.parents):
        if (candidate / "config.pkl").is_file() or (candidate / "config.yaml").is_file():
            return candidate
    return None


def _defrost_config_tree(value: Any, seen: set[int] | None = None) -> None:
    """Clear Cosmos config freeze flags so inference overrides can be applied."""
    import attrs

    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if "_is_frozen" in getattr(value, "__dict__", {}):
        object.__setattr__(value, "_is_frozen", False)

    if attrs.has(value.__class__):
        children = (getattr(value, field.name) for field in attrs.fields(value.__class__))
    elif isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple, set)):
        children = value
    else:
        return

    for child in children:
        if isinstance(child, (str, bytes, int, float, bool, type(None))):
            continue
        _defrost_config_tree(child, seen)


def _load_training_run_config(checkpoint_path: str | Path) -> tuple[Any, Path]:
    run_dir = _checkpoint_run_dir(checkpoint_path)
    if run_dir is None:
        raise FileNotFoundError(
            f"Could not find config.pkl/config.yaml in parents of checkpoint path {checkpoint_path}"
        )
    config_pkl = run_dir / "config.pkl"
    if not config_pkl.is_file():
        raise FileNotFoundError(
            f"Checkpoint run dir {run_dir} does not contain config.pkl; YAML fallback is not supported here"
        )
    with config_pkl.open("rb") as f:
        config = pickle.load(f)
    _defrost_config_tree(config)
    return config, run_dir


def _load_model_from_training_run_config(
    *,
    checkpoint_path: str,
    sampler: str,
    use_ema_weights: bool,
    setup_args: Any,
) -> tuple[Any, Any]:
    """Load a Cosmos DCP checkpoint using the config saved by its training run."""
    from cosmos_framework.inference.model import Cosmos3OmniModel
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils.vfm import model_loader

    config, run_dir = _load_training_run_config(checkpoint_path)
    logging.info("Using checkpoint-local Cosmos config from %s", run_dir / "config.pkl")

    checkpoint_load_path = str(checkpoint_path)
    is_safetensors = model_loader._is_safetensors_checkpoint(checkpoint_load_path, None)
    if is_safetensors:
        raise ValueError("Checkpoint-local config fallback currently supports DCP checkpoints only")
    if not checkpoint_load_path.strip("/").endswith("model"):
        checkpoint_load_path = os.path.join(checkpoint_load_path, "model")

    parallelism = config.model.config.parallelism
    parallelism.enable_inference_mode = True
    parallelism.data_parallel_shard_degree = setup_args.dp_shard_size
    parallelism.context_parallel_shard_degree = setup_args.cp_size
    parallelism.cfg_parallel_shard_degree = setup_args.cfgp_size

    compile_cfg = config.model.config.compile
    compile_cfg.enabled = setup_args.use_torch_compile
    compile_cfg.use_cuda_graphs = (
        setup_args.use_cuda_graphs
        and setup_args.dp_shard_size * setup_args.cp_size * setup_args.cfgp_size == 1
    )
    compile_cfg.compiled_region = setup_args.compiled_region
    compile_cfg.compile_dynamic = setup_args.compile_dynamic

    config.model.config.activation_checkpointing.mode = "none"
    config.model.config.ema.enabled = False
    config.model.config.rectified_flow_inference_config.scheduler_type = sampler

    keys_to_skip_loading = []
    checkpoint_config = getattr(config, "checkpoint", None)
    configured_skip_keys = (
        getattr(checkpoint_config, "keys_to_skip_loading", None) if checkpoint_config is not None else None
    )
    if configured_skip_keys is not None:
        # The checkpoint-local training config still contains the warm-start
        # skip list used to initialize from the base Cosmos3 checkpoint.  When
        # evaluating a saved posttrain checkpoint, action pathway weights must
        # be loaded from that checkpoint; otherwise the action heads stay at
        # their fresh initialization and training-set probes become invalid.
        keys_to_skip_loading = [key for key in configured_skip_keys if key == "net_ema."]
        dropped_skip_keys = sorted(set(configured_skip_keys) - set(keys_to_skip_loading))
        if dropped_skip_keys:
            logging.info("Loading posttrain action weights; ignored warm-start skip keys: %s", dropped_skip_keys)

    config.validate()
    config.freeze()
    misc.set_random_seed(seed=0, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark

    Cosmos3OmniModel.before_load_model()
    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()
        model.on_train_start()
    model_loader._load_model(
        model,
        checkpoint_path=checkpoint_load_path,
        credential_path=None,
        enable_gcs_patch_in_boto3=False,
        load_ema_to_reg=use_ema_weights,
        keys_to_skip_loading=keys_to_skip_loading,
    )
    Cosmos3OmniModel.after_load_model(model)
    return model, config


def _is_missing_hydra_experiment_error(exc: BaseException) -> bool:
    text = str(exc)
    return "Could not find 'experiment/" in text or "MissingConfigException" in exc.__class__.__name__


@dataclass(frozen=True)
class PreparedEvalExample:
    image_uint8: torch.Tensor
    prompt: str
    action_template: torch.Tensor
    raw_state: Any | None = None
    sincos_state: Any | None = None
    env_name: str | None = None


def _coerce_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        return _coerce_prompt(value[0])
    return str(value)


def _extract_first_image_uint8(images: Any) -> torch.Tensor:
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("example[image] must contain at least one image")
        image = images[0]
    else:
        image = images

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = array.copy()
    return torch.from_numpy(array).contiguous()


def prepare_eval_example(example: dict[str, Any], *, action_horizon: int, action_dim: int) -> PreparedEvalExample:
    if "image" not in example:
        raise ValueError("RoboCasa eval example is missing key image")
    image_uint8 = _extract_first_image_uint8(example["image"])
    prompt = _coerce_prompt(example.get("lang", ""))
    action_template = torch.zeros((action_horizon, action_dim), dtype=torch.float32)
    env_name = _coerce_prompt(example["env_name"]) if "env_name" in example else None
    return PreparedEvalExample(
        image_uint8=image_uint8,
        prompt=prompt,
        action_template=action_template,
        raw_state=example.get("raw_state"),
        sincos_state=example.get("state"),
        env_name=env_name,
    )


def normalize_action_schema(action_schema: str) -> str:
    schema = action_schema.lower().replace("-", "_").strip()
    aliases = {
        "29": "robocasa29",
        "robocasa_29": "robocasa29",
        "robocasa29": "robocasa29",
        "44": "gr1_44",
        "gr1_44": "gr1_44",
        "gr1_44_to_robocasa29": "gr1_44",
        "gr1_29": "gr1_29",
        "gr1_29d": "gr1_29",
    }
    if schema not in aliases:
        raise ValueError(f"Unknown action_schema={action_schema!r}; expected robocasa29, gr1_44, or gr1_29")
    return aliases[schema]


def schema_defaults(action_schema: str) -> tuple[int, int]:
    schema = normalize_action_schema(action_schema)
    if schema == "gr1_44":
        return GR1_44_ACTION_DIM, GR1_44_DOMAIN_ID
    if schema == "gr1_29":
        return GR1_29_ACTION_DIM, GR1_29_DOMAIN_ID
    return ROBOCASA29_ACTION_DIM, ROBOCASA29_DOMAIN_ID


def normalize_action_denorm(action_denorm: str, action_schema: str) -> str:
    mode = action_denorm.lower().strip()
    if mode == "auto":
        return "minmax" if normalize_action_schema(action_schema) in ("gr1_44", "gr1_29") else "none"
    if mode not in {"none", "minmax"}:
        raise ValueError(f"Unknown action_denorm={action_denorm!r}; expected auto, none, or minmax")
    return mode


def normalize_action_history(action_history: str, action_schema: str) -> str:
    mode = action_history.lower().strip()
    if mode == "auto":
        return "state" if normalize_action_schema(action_schema) in ("gr1_44", "gr1_29") else "none"
    if mode not in {"none", "zero", "state"}:
        raise ValueError(f"Unknown action_history={action_history!r}; expected auto, none, zero, or state")
    if mode == "state" and normalize_action_schema(action_schema) not in ("gr1_44", "gr1_29"):
        raise ValueError("action_history=state is only supported for action_schema=gr1_44 or gr1_29")
    return mode


def next_request_seed(*, seed: int, request_count: int, deterministic_seed: bool) -> int:
    if deterministic_seed:
        return int(seed)
    return int(seed) + int(request_count)


def clamp_normalized_action(action: torch.Tensor, *, action_dim: int, enabled: bool) -> torch.Tensor:
    if not enabled:
        return action
    if action.shape[-1] < action_dim:
        raise ValueError(f"Action dim {action.shape[-1]} is smaller than action_dim={action_dim}")
    output = action.clone()
    output[..., :action_dim] = output[..., :action_dim].clamp(-1.0, 1.0)
    return output



def _round_float(value: Any) -> float:
    return round(float(value), 6)


def _tensor_debug_stats(tensor: torch.Tensor) -> dict[str, float]:
    if tensor.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "max_abs": 0.0}
    values = tensor.detach().float().cpu()
    return {
        "min": _round_float(values.min().item()),
        "max": _round_float(values.max().item()),
        "mean": _round_float(values.mean().item()),
        "max_abs": _round_float(values.abs().max().item()),
    }


def summarize_normalized_action_debug(action: torch.Tensor, *, action_dim: int) -> dict[str, Any]:
    tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2:
        raise ValueError(f"Expected normalized action [T,D], got {tuple(tensor.shape)}")
    if tensor.shape[1] < action_dim:
        raise ValueError(f"Action dim {tensor.shape[1]} is smaller than action_dim={action_dim}")
    active = tensor[:, :action_dim]
    out_of_bounds = active.abs() > 1.0
    return {
        "shape": list(active.shape),
        **_tensor_debug_stats(active),
        "out_of_bounds_frac": _round_float(out_of_bounds.float().mean().item()),
        "out_of_bounds_count": int(out_of_bounds.sum().item()),
    }


def summarize_robocasa29_action_debug(
    action: torch.Tensor,
    *,
    current_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2 or tensor.shape[1] != ROBOCASA29_ACTION_DIM:
        raise ValueError(f"Expected RoboCasa action [T,29], got {tuple(tensor.shape)}")
    state = None
    if current_state is not None:
        state = torch.as_tensor(current_state, dtype=torch.float32).detach().cpu().reshape(-1)
        if state.shape[0] != ROBOCASA29_ACTION_DIM:
            raise ValueError(f"current_state must have dim 29, got {tuple(state.shape)}")

    parts: dict[str, Any] = {}
    for name, start, end in ROBOCASA29_PARTS:
        part = tensor[:, start:end]
        step_delta = part[1:] - part[:-1] if part.shape[0] > 1 else torch.zeros((0, end - start))
        first_delta_max_abs = None
        if state is not None and part.shape[0] > 0:
            first_delta_max_abs = _round_float((part[0] - state[start:end]).abs().max().item())
        parts[name] = {
            **_tensor_debug_stats(part),
            "step_delta_max_abs": _round_float(step_delta.abs().max().item()) if step_delta.numel() else 0.0,
            "first_delta_max_abs": first_delta_max_abs,
        }

    return {
        "shape": list(tensor.shape),
        "overall": _tensor_debug_stats(tensor),
        "parts": parts,
    }

def _as_single_state_row(value: Any, *, key: str, dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    if tensor.ndim != 2 or tensor.shape[-1] != dim:
        raise ValueError(f"raw_state[{key!r}] must have trailing dim {dim}, got {tuple(tensor.shape)}")
    return tensor[-1:].contiguous()


def pack_robocasa_raw_state_to_gr1_44(raw_state: Any) -> torch.Tensor:
    if isinstance(raw_state, dict):
        def part(name: str, dim: int) -> torch.Tensor:
            if name in raw_state:
                return _as_single_state_row(raw_state[name], key=name, dim=dim)
            dotted = f"state.{name}"
            if dotted in raw_state:
                return _as_single_state_row(raw_state[dotted], key=dotted, dim=dim)
            raise ValueError(f"raw_state is missing {name!r}")

        left_arm = part("left_arm", 7)
        left_hand = part("left_hand", 6)
        right_arm = part("right_arm", 7)
        right_hand = part("right_hand", 6)
        waist = part("waist", 3)
        output = torch.zeros((1, GR1_44_ACTION_DIM), dtype=torch.float32)
        output[:, 0:7] = left_arm
        output[:, 7:13] = left_hand
        output[:, 22:29] = right_arm
        output[:, 29:35] = right_hand
        output[:, 41:44] = waist
        return output

    tensor = torch.as_tensor(raw_state, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    if tensor.ndim != 2:
        raise ValueError(f"raw_state must be [T,D], got {tuple(tensor.shape)}")
    tensor = tensor[-1:].contiguous()
    if tensor.shape[-1] == GR1_44_ACTION_DIM:
        return tensor
    if tensor.shape[-1] != ROBOCASA29_ACTION_DIM:
        raise ValueError(f"raw_state dim must be 29 or 44, got {tensor.shape[-1]}")
    output = torch.zeros((1, GR1_44_ACTION_DIM), dtype=torch.float32)
    output[:, 0:7] = tensor[:, 0:7]
    output[:, 22:29] = tensor[:, 7:14]
    output[:, 7:13] = tensor[:, 14:20]
    output[:, 29:35] = tensor[:, 20:26]
    output[:, 41:44] = tensor[:, 26:29]
    return output


def sincos_state_to_gr1_44(state_sincos: Any) -> torch.Tensor:
    """Decode sin/cos encoded 58D state back to raw 44D GR1 joint angles.

    The starVLA_dev client encodes 29D state as sin/cos pairs per part, yielding 58D:
        [sin(left_arm)(7), cos(left_arm)(7),
         sin(right_arm)(7), cos(right_arm)(7),
         sin(left_hand)(6), cos(left_hand)(6),
         sin(right_hand)(6), cos(right_hand)(6),
         sin(waist)(3), cos(waist)(3)]

    We recover the original angles via arctan2(sin, cos) and map to 44D GR1 layout.
    """
    state_sincos = np.asarray(state_sincos, dtype=np.float64).reshape(-1)
    # If multi-row (e.g. history), take last row
    if state_sincos.shape[0] > 58 and state_sincos.shape[0] % 58 == 0:
        state_sincos = state_sincos[-58:]
    if state_sincos.shape[0] != 58:
        raise ValueError(f"Expected 58D sin/cos state, got {state_sincos.shape[0]}D")

    # Part dimensions in the 29D RoboCasa layout
    parts = [
        ("left_arm", 7),
        ("right_arm", 7),
        ("left_hand", 6),
        ("right_hand", 6),
        ("waist", 3),
    ]

    # Decode each part: layout is [sin(n), cos(n)] per part
    raw_parts = {}
    offset = 0
    for name, dim in parts:
        sin_vals = state_sincos[offset : offset + dim]
        cos_vals = state_sincos[offset + dim : offset + 2 * dim]
        raw_parts[name] = np.arctan2(sin_vals, cos_vals).astype(np.float32)
        offset += 2 * dim

    # Map to GR1 44D layout
    output = torch.zeros((1, GR1_44_ACTION_DIM), dtype=torch.float32)
    output[0, 0:7] = torch.from_numpy(raw_parts["left_arm"])
    output[0, 7:13] = torch.from_numpy(raw_parts["left_hand"])
    # 13:19 = left_leg (zeros)
    # 19:22 = neck (zeros)
    output[0, 22:29] = torch.from_numpy(raw_parts["right_arm"])
    output[0, 29:35] = torch.from_numpy(raw_parts["right_hand"])
    # 35:41 = right_leg (zeros)
    output[0, 41:44] = torch.from_numpy(raw_parts["waist"])
    return output


def pack_robocasa_raw_state_to_gr1_29(raw_state: Any) -> torch.Tensor:
    """Pack raw RoboCasa state into 29D GR1 active-parts layout.

    Output layout: [left_arm(7), right_arm(7), left_hand(6), right_hand(6), waist(3)]
    """
    if isinstance(raw_state, dict):
        def part(name: str, dim: int) -> torch.Tensor:
            if name in raw_state:
                return _as_single_state_row(raw_state[name], key=name, dim=dim)
            dotted = f"state.{name}"
            if dotted in raw_state:
                return _as_single_state_row(raw_state[dotted], key=dotted, dim=dim)
            raise ValueError(f"raw_state is missing {name!r}")

        return torch.cat([
            part("left_arm", 7),
            part("right_arm", 7),
            part("left_hand", 6),
            part("right_hand", 6),
            part("waist", 3),
        ], dim=1)

    tensor = torch.as_tensor(raw_state, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    if tensor.ndim != 2:
        raise ValueError(f"raw_state must be [T,D], got {tuple(tensor.shape)}")
    tensor = tensor[-1:].contiguous()
    if tensor.shape[-1] == GR1_29_ACTION_DIM:
        return tensor
    if tensor.shape[-1] == GR1_44_ACTION_DIM:
        return torch.cat([tensor[:, s:e] for s, e in _GR1_44_TO_29_SLICES], dim=1)
    if tensor.shape[-1] == ROBOCASA29_ACTION_DIM:
        return tensor  # RoboCasa 29D is already the same layout
    raise ValueError(f"raw_state dim must be 29 or 44, got {tensor.shape[-1]}")


def sincos_state_to_gr1_29(state_sincos: Any) -> torch.Tensor:
    """Decode sin/cos 58D state to 29D GR1 active-parts layout."""
    state_sincos = np.asarray(state_sincos, dtype=np.float64).reshape(-1)
    if state_sincos.shape[0] > 58 and state_sincos.shape[0] % 58 == 0:
        state_sincos = state_sincos[-58:]
    if state_sincos.shape[0] != 58:
        raise ValueError(f"Expected 58D sin/cos state, got {state_sincos.shape[0]}D")

    parts_spec = [("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3)]
    decoded = []
    offset = 0
    for _name, dim in parts_spec:
        sin_vals = state_sincos[offset:offset + dim]
        cos_vals = state_sincos[offset + dim:offset + 2 * dim]
        decoded.append(torch.from_numpy(np.arctan2(sin_vals, cos_vals).astype(np.float32)))
        offset += 2 * dim

    return torch.cat(decoded, dim=0).unsqueeze(0)



def _stats_to_min_range(stats: dict[str, Any], *, expected_dim: int, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if "min" not in stats or "max" not in stats:
        raise ValueError(f"Min/max stats require min and max arrays: {path}")
    stat_min = torch.tensor(stats["min"], dtype=torch.float32)
    stat_max = torch.tensor(stats["max"], dtype=torch.float32)
    if stat_min.numel() < expected_dim or stat_max.numel() < expected_dim:
        raise ValueError(f"Stats dim {stat_min.numel()} is smaller than expected_dim={expected_dim}: {path}")
    stat_min = stat_min[:expected_dim].contiguous()
    stat_range = (stat_max[:expected_dim] - stat_min).clamp(min=1e-8).contiguous()
    return stat_min, stat_range


def normalize_minmax_tensor(tensor: torch.Tensor, *, stat_min: torch.Tensor, stat_range: torch.Tensor) -> torch.Tensor:
    dim = int(stat_min.shape[0])
    if tensor.shape[-1] < dim:
        raise ValueError(f"Tensor dim {tensor.shape[-1]} is smaller than stats dim {dim}")
    stat_min = stat_min.to(device=tensor.device, dtype=tensor.dtype)
    stat_range = stat_range.to(device=tensor.device, dtype=tensor.dtype)
    output = tensor.clone()
    output[..., :dim] = (2.0 * (output[..., :dim] - stat_min) / stat_range - 1.0).clamp(-1.0, 1.0)
    return output


def summarize_minmax_normalization_debug(
    tensor: torch.Tensor,
    *,
    stat_min: torch.Tensor,
    stat_range: torch.Tensor,
    action_dim: int,
) -> dict[str, Any]:
    values = torch.as_tensor(tensor, dtype=torch.float32).detach().cpu()
    if values.ndim != 2:
        raise ValueError(f"Expected tensor [T,D], got {tuple(values.shape)}")
    if values.shape[-1] < action_dim:
        raise ValueError(f"Tensor dim {values.shape[-1]} is smaller than action_dim={action_dim}")
    if stat_min.shape[0] < action_dim or stat_range.shape[0] < action_dim:
        raise ValueError("Stats are smaller than action_dim")

    stat_min = torch.as_tensor(stat_min[:action_dim], dtype=torch.float32).detach().cpu()
    stat_range = torch.as_tensor(stat_range[:action_dim], dtype=torch.float32).detach().cpu().clamp(min=1e-8)
    active = values[:, :action_dim]
    normalized_before_clamp = 2.0 * (active - stat_min) / stat_range - 1.0
    normalized_after_clamp = normalized_before_clamp.clamp(-1.0, 1.0)
    out_of_bounds = normalized_before_clamp.abs() > 1.0
    return {
        "shape": list(active.shape),
        "before_clamp": {
            **_tensor_debug_stats(normalized_before_clamp),
            "out_of_bounds_frac": _round_float(out_of_bounds.float().mean().item()),
            "out_of_bounds_count": int(out_of_bounds.sum().item()),
        },
        "after_clamp": {
            **_tensor_debug_stats(normalized_after_clamp),
        },
    }


def load_minmax_state_stats(stats_path: str | Path, *, expected_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"State stats file does not exist: {path}")
    with path.open("r") as f:
        raw_stats = json.load(f)
    if "observation.state" in raw_stats:
        stats = raw_stats["observation.state"]
    elif "state" in raw_stats:
        stats = raw_stats["state"]
    else:
        stats = raw_stats
    return _stats_to_min_range(stats, expected_dim=expected_dim, path=path)


def _dataset_name_from_robocasa_env(env_name: str | None) -> str | None:
    if not env_name or "/" not in env_name:
        return None
    prefix, rest = env_name.split("/", 1)
    task = rest.split("_GR1", 1)[0]
    if not prefix or not task:
        return None
    return f"{prefix}.{task}"


def resolve_gr1_state_stats_path(
    *,
    env_name: str | None,
    explicit_path: str | None,
    stats_root: str | Path,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    root = Path(stats_root)
    dataset_name = _dataset_name_from_robocasa_env(env_name)
    if dataset_name is not None:
        candidate = root / dataset_name / "meta" / "stats.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No GR1 state stats found under {root}")


def load_minmax_action_stats(stats_path: str | Path, *, expected_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action stats file does not exist: {path}")
    with path.open("r") as f:
        raw_stats = json.load(f)
    stats = raw_stats.get("global", raw_stats) if isinstance(raw_stats, dict) else raw_stats
    if not isinstance(stats, dict) or "min" not in stats or "max" not in stats:
        raise ValueError(f"Min/max action stats require min and max arrays: {path}")
    action_min = torch.tensor(stats["min"], dtype=torch.float32)
    action_max = torch.tensor(stats["max"], dtype=torch.float32)
    if action_min.numel() < expected_dim or action_max.numel() < expected_dim:
        raise ValueError(
            f"Action stats dim {action_min.numel()} is smaller than expected_dim={expected_dim}: {path}"
        )
    action_min = action_min[:expected_dim].contiguous()
    action_range = (action_max[:expected_dim] - action_min).clamp(min=1e-6).contiguous()
    return action_min, action_range


def load_minmax_action_stats_29d(
    stats_path: str | Path,
    *,
    expected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load action normalization stats from per-dataset stats.json, reordered to 29D.

    Reads the 'action' field, extracts active-part slices in 29D training order.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action stats file does not exist: {path}")
    with path.open("r") as f:
        raw_stats = json.load(f)
    if "action" in raw_stats:
        stats = raw_stats["action"]
    else:
        stats = raw_stats.get("global", raw_stats) if isinstance(raw_stats, dict) else raw_stats
    if not isinstance(stats, dict) or "min" not in stats or "max" not in stats:
        raise ValueError(f"Min/max action stats require min and max arrays: {path}")

    full_min = torch.tensor(stats["min"], dtype=torch.float32)
    full_max = torch.tensor(stats["max"], dtype=torch.float32)

    reordered_min = torch.cat([full_min[s:e] for s, e in _GR1_44_TO_29_SLICES])
    reordered_max = torch.cat([full_max[s:e] for s, e in _GR1_44_TO_29_SLICES])

    if reordered_min.numel() < expected_dim:
        raise ValueError(f"Reordered stats dim {reordered_min.numel()} < expected {expected_dim}")
    reordered_min = reordered_min[:expected_dim].contiguous()
    reordered_range = (reordered_max[:expected_dim] - reordered_min).clamp(min=1e-8).contiguous()
    return reordered_min, reordered_range


def load_minmax_state_stats_29d(
    stats_path: str | Path,
    *,
    expected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load state normalization stats reordered to 29D active-parts layout."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"State stats file does not exist: {path}")
    with path.open("r") as f:
        raw_stats = json.load(f)
    if "observation.state" in raw_stats:
        stats = raw_stats["observation.state"]
    elif "state" in raw_stats:
        stats = raw_stats["state"]
    else:
        stats = raw_stats

    full_min = torch.tensor(stats["min"], dtype=torch.float32)
    full_max = torch.tensor(stats["max"], dtype=torch.float32)

    reordered_min = torch.cat([full_min[s:e] for s, e in _GR1_44_TO_29_SLICES])
    reordered_max = torch.cat([full_max[s:e] for s, e in _GR1_44_TO_29_SLICES])

    if reordered_min.numel() < expected_dim:
        raise ValueError(f"Reordered state stats dim {reordered_min.numel()} < expected {expected_dim}")
    reordered_min = reordered_min[:expected_dim].contiguous()
    reordered_range = (reordered_max[:expected_dim] - reordered_min).clamp(min=1e-8).contiguous()
    return reordered_min, reordered_range



def denormalize_minmax_action(
    action: torch.Tensor,
    *,
    action_min: torch.Tensor,
    action_range: torch.Tensor,
) -> torch.Tensor:
    dim = int(action_min.shape[0])
    if action.shape[-1] < dim:
        raise ValueError(f"Action dim {action.shape[-1]} is smaller than denorm stats dim {dim}")
    output = action.clone()
    action_min = action_min.to(device=output.device, dtype=output.dtype)
    action_range = action_range.to(device=output.device, dtype=output.dtype)
    output[..., :dim] = (output[..., :dim] + 1.0) / 2.0 * action_range + action_min
    return output


def remap_gr1_44_to_robocasa29(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] < GR1_44_ACTION_DIM:
        raise ValueError(f"Action dim {tensor.shape[1]} is smaller than GR1 44D schema")
    return torch.cat(
        [
            tensor[:, 0:7],  # left_arm
            tensor[:, 22:29],  # right_arm
            tensor[:, 7:13],  # left_hand
            tensor[:, 29:35],  # right_hand
            tensor[:, 41:44],  # waist
        ],
        dim=1,
    )


def format_policy_response(
    actions: Sequence[torch.Tensor],
    *,
    raw_action_dim: int,
    action_schema: str = DEFAULT_ACTION_SCHEMA,
) -> dict[str, np.ndarray]:
    schema = normalize_action_schema(action_schema)
    formatted = []
    for action in actions:
        tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
        if tensor.ndim != 2:
            raise ValueError(f"Expected action [T,D], got {tuple(tensor.shape)}")
        if tensor.shape[1] < raw_action_dim:
            raise ValueError(f"Action dim {tensor.shape[1]} is smaller than raw_action_dim={raw_action_dim}")
        if schema == "gr1_44":
            tensor = remap_gr1_44_to_robocasa29(tensor)
        elif schema == "gr1_29":
            # 29D is already in RoboCasa-compatible layout
            tensor = tensor[:, :GR1_29_ACTION_DIM]
        else:
            tensor = tensor[:, :ROBOCASA29_ACTION_DIM]
        formatted.append(tensor.numpy().astype(np.float32, copy=False))
    return {"actions": np.stack(formatted, axis=0)}


def decode_vision_latent_to_uint8_video(model: Any, vision_latent: torch.Tensor) -> np.ndarray:
    """Decode a Cosmos vision latent to time-major RGB uint8 video frames."""
    decoded = model.decode(vision_latent)
    video = ((decoded[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    return video.permute(1, 2, 3, 0).detach().cpu().numpy()


@dataclass(frozen=True)
class PolicyPrediction:
    action: torch.Tensor
    video: np.ndarray | None = None


class Cosmos3RobocasaPolicy:
    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        checkpoint_repo: str = DEFAULT_CHECKPOINT_REPO,
        checkpoint_revision: str | None = None,
        allow_download: bool = True,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        resolution: str = "256",
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        raw_action_dim: int | None = None,
        max_action_dim: int = DEFAULT_MAX_ACTION_DIM,
        domain_id: int | None = None,
        action_schema: str = DEFAULT_ACTION_SCHEMA,
        action_denorm: str = DEFAULT_ACTION_DENORM,
        action_stats_path: str | None = None,
        action_history: str = DEFAULT_ACTION_HISTORY,
        action_state_stats_path: str | None = None,
        action_state_stats_root: str | Path = DEFAULT_GR1_STATE_STATS_ROOT,
        conditioning_fps: int = DEFAULT_FPS,
        sampler: str = "unipc",
        guidance: float = 1.0,
        num_steps: int = 30,
        shift: float = 5.0,
        use_ema_weights: bool = False,
        seed: int = 0,
        deterministic_seed: bool = False,
        action_clamp: bool = False,
        decode_video: bool = False,
        model_input_debug_dir: str | None = None,
        model_input_debug_max: int = 10,
    ) -> None:
        from cosmos_framework.inference.common.init import init_output_dir, init_script

        init_script()

        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.scripts.action_policy_server_utils import (
            disable_runtime_ema_for_frozen_config,
            maybe_init_distributed,
        )

        self.resolution = resolution
        self.action_horizon = action_horizon
        self.action_schema = normalize_action_schema(action_schema)
        default_raw_action_dim, default_domain_id = schema_defaults(self.action_schema)
        self.raw_action_dim = raw_action_dim if raw_action_dim is not None else default_raw_action_dim
        self.max_action_dim = max_action_dim
        self.domain_id = domain_id if domain_id is not None else default_domain_id
        self.action_denorm = normalize_action_denorm(action_denorm, self.action_schema)
        if action_stats_path is None and self.action_denorm == "minmax" and self.action_schema == "gr1_44":
            action_stats_path = str(DEFAULT_GR1_ACTION_STATS_PATH)
        # gr1_29 uses per-dataset action stats (resolved per-example), no global default
        self.action_stats_path = action_stats_path
        self.action_min: torch.Tensor | None = None
        self.action_range: torch.Tensor | None = None
        self._action_stats_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.action_denorm == "minmax":
            if self.action_schema == "gr1_29":
                # gr1_29 uses per-dataset action stats, resolved per-example
                self.action_min = None
                self.action_range = None
            else:
                if self.action_stats_path is None:
                    raise ValueError("action_denorm=minmax requires action_stats_path")
                self.action_min, self.action_range = load_minmax_action_stats(
                    self.action_stats_path, expected_dim=self.raw_action_dim
                )
        self.action_history = normalize_action_history(action_history, self.action_schema)
        self.action_state_stats_path = action_state_stats_path
        self.action_state_stats_root = Path(action_state_stats_root)
        self._state_stats_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
        self.conditioning_fps = conditioning_fps
        self.guidance = guidance
        self.num_steps = num_steps
        self.shift = shift
        self.seed = seed
        self.deterministic_seed = deterministic_seed
        self.action_clamp = action_clamp
        self.decode_video = decode_video
        self._request_count = 0
        self._model_input_debug_dir = Path(model_input_debug_dir) if model_input_debug_dir else None
        self._model_input_debug_max = model_input_debug_max

        checkpoint = resolve_checkpoint_path(
            checkpoint_path,
            checkpoint_repo,
            allow_download=allow_download,
            revision=checkpoint_revision,
        )

        patch_cached_wan_vae_download()
        maybe_init_distributed()
        init_output_dir(output_dir)
        setup_args = OmniSetupOverrides.model_validate(
            {
                "checkpoint_path": checkpoint,
                "output_dir": str(output_dir),
                "sampler": sampler,
                "guardrails": False,
                "use_ema_weights": use_ema_weights,
            }
        ).build_setup()
        setup_args = disable_runtime_ema_for_frozen_config(setup_args)

        logging.info("Loading Cosmos3 policy checkpoint: %s", checkpoint)
        try:
            pipe = OmniInference.create(setup_args)
            self.model = pipe.model.eval()
        except Exception as exc:
            if not _is_missing_hydra_experiment_error(exc):
                raise
            logging.warning(
                "Cosmos Hydra experiment config is missing for checkpoint %s; "
                "falling back to checkpoint-local training config. Original error: %s",
                checkpoint,
                exc,
            )
            model, _config = _load_model_from_training_run_config(
                checkpoint_path=checkpoint,
                sampler=sampler,
                use_ema_weights=use_ema_weights,
                setup_args=setup_args,
            )
            self.model = model.eval()
        self.checkpoint_path = checkpoint
        logging.info(
            "Cosmos3 policy ready: max_action_dim=%s action_schema=%s action_denorm=%s action_history=%s action_clamp=%s",
            getattr(self.model.config, "max_action_dim", None),
            self.action_schema,
            self.action_denorm,
            self.action_history,
            self.action_clamp,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "env": "cosmos3_robocasa_policy_server",
            "checkpoint_path": self.checkpoint_path,
            "action_chunk_size": self.action_horizon,
            "action_schema": self.action_schema,
            "action_denorm": self.action_denorm,
            "action_stats_path": self.action_stats_path,
            "action_history": self.action_history,
            "action_state_stats_path": self.action_state_stats_path,
            "action_state_stats_root": str(self.action_state_stats_root),
            "raw_action_dim": self.raw_action_dim,
            "response_action_dim": ROBOCASA29_ACTION_DIM,
            "max_action_dim": self.max_action_dim,
            "domain_id": self.domain_id,
            "conditioning_fps": self.conditioning_fps,
            "num_steps": self.num_steps,
            "guidance": self.guidance,
            "shift": self.shift,
            "seed": self.seed,
            "deterministic_seed": self.deterministic_seed,
            "action_clamp": self.action_clamp,
            "decode_video": self.decode_video,
        }

    def _denormalize_action(
        self,
        action: torch.Tensor,
        *,
        action_min: torch.Tensor | None = None,
        action_range: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.action_denorm == "none":
            return action
        if self.action_denorm == "minmax":
            a_min = action_min if action_min is not None else self.action_min
            a_range = action_range if action_range is not None else self.action_range
            if a_min is None or a_range is None:
                raise RuntimeError("minmax denorm requested but no action stats available")
            return denormalize_minmax_action(action, action_min=a_min, action_range=a_range)
        raise ValueError(f"Unsupported action_denorm={self.action_denorm!r}")

    def _state_stats_for_example(self, prepared: PreparedEvalExample) -> tuple[torch.Tensor, torch.Tensor]:
        stats_path = resolve_gr1_state_stats_path(
            env_name=prepared.env_name,
            explicit_path=self.action_state_stats_path,
            stats_root=self.action_state_stats_root,
        )
        cache_key = (stats_path, self.action_schema)
        cached = self._state_stats_cache.get(cache_key)
        if cached is None:
            if self.action_schema == "gr1_29":
                cached = load_minmax_state_stats_29d(stats_path, expected_dim=GR1_29_ACTION_DIM)
            else:
                cached = load_minmax_state_stats(stats_path, expected_dim=GR1_44_ACTION_DIM)
            self._state_stats_cache[cache_key] = cached
        return cached


    def _action_stats_for_example(self, prepared: PreparedEvalExample) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve per-dataset action normalization stats (used by gr1_29 schema)."""
        stats_path = resolve_gr1_state_stats_path(
            env_name=prepared.env_name,
            explicit_path=self.action_stats_path,
            stats_root=self.action_state_stats_root,
        )
        cache_key = (stats_path, "action_29d")
        cached = self._action_stats_cache.get(cache_key)
        if cached is None:
            cached = load_minmax_action_stats_29d(stats_path, expected_dim=GR1_29_ACTION_DIM)
            self._action_stats_cache[cache_key] = cached
        return cached

    def _build_history_action(self, prepared: PreparedEvalExample) -> torch.Tensor | None:
        if self.action_history == "none":
            return None
        if self.action_history == "zero":
            return torch.zeros((1, self.raw_action_dim), dtype=torch.float32)
        if self.action_history == "state":
            state = None
            if self.action_schema == "gr1_29":
                if prepared.raw_state is not None:
                    state = pack_robocasa_raw_state_to_gr1_29(prepared.raw_state)
                elif prepared.sincos_state is not None:
                    state = sincos_state_to_gr1_29(prepared.sincos_state)
            else:
                if prepared.raw_state is not None:
                    state = pack_robocasa_raw_state_to_gr1_44(prepared.raw_state)
                elif prepared.sincos_state is not None:
                    state = sincos_state_to_gr1_44(prepared.sincos_state)
            if state is None:
                raise ValueError("action_history=state requires raw_state or sin/cos state")
            stat_min, stat_range = self._state_stats_for_example(prepared)
            return normalize_minmax_tensor(state, stat_min=stat_min, stat_range=stat_range)
        raise ValueError(f"Unsupported action_history={self.action_history!r}")

    def _history_action_debug(self, prepared: PreparedEvalExample) -> dict[str, Any] | None:
        if self.action_history != "state":
            return None
        if prepared.raw_state is None and prepared.sincos_state is None:
            return None
        try:
            if self.action_schema == "gr1_29":
                if prepared.raw_state is not None:
                    state = pack_robocasa_raw_state_to_gr1_29(prepared.raw_state)
                else:
                    state = sincos_state_to_gr1_29(prepared.sincos_state)
            else:
                if prepared.raw_state is not None:
                    state = pack_robocasa_raw_state_to_gr1_44(prepared.raw_state)
                else:
                    state = sincos_state_to_gr1_44(prepared.sincos_state)
            stat_min, stat_range = self._state_stats_for_example(prepared)
            return summarize_minmax_normalization_debug(
                state,
                stat_min=stat_min,
                stat_range=stat_range,
                action_dim=self.raw_action_dim,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path should not break inference.
            return {"error": str(exc)}



    def _current_robocasa29_state(self, prepared: PreparedEvalExample) -> torch.Tensor | None:
        if prepared.raw_state is None and prepared.sincos_state is None:
            return None
        try:
            if self.action_schema == "gr1_29":
                if prepared.raw_state is not None:
                    return pack_robocasa_raw_state_to_gr1_29(prepared.raw_state).reshape(-1)
                else:
                    return sincos_state_to_gr1_29(prepared.sincos_state).reshape(-1)
            if prepared.raw_state is not None:
                state_44 = pack_robocasa_raw_state_to_gr1_44(prepared.raw_state)
            else:
                state_44 = sincos_state_to_gr1_44(prepared.sincos_state)
            return remap_gr1_44_to_robocasa29(state_44).reshape(-1)
        except Exception as exc:  # pragma: no cover - diagnostic path should not break inference.
            logging.info("ACTION_DEBUG failed to pack current state: %s", exc)
            return None

    def _log_action_debug(
        self,
        *,
        request_index: int,
        seed: int,
        history_len: int,
        normalized_action: torch.Tensor,
        denormalized_action: torch.Tensor,
        normalized_action_before_clamp: torch.Tensor | None = None,
        prepared: PreparedEvalExample,
        history_action_debug: dict[str, Any] | None = None,
    ) -> None:
        if os.environ.get("ACTION_DEBUG", "0") != "1":
            return
        limit = int(os.environ.get("ACTION_DEBUG_LIMIT", "5"))
        if request_index >= limit:
            return
        if self.action_schema == "gr1_44":
            robocasa_action = remap_gr1_44_to_robocasa29(denormalized_action)
        else:
            robocasa_action = denormalized_action[:, :ROBOCASA29_ACTION_DIM]
        payload = {
            "request_index": request_index,
            "seed": seed,
            "history_len": history_len,
            "env_name": prepared.env_name,
            "prompt": prepared.prompt,
            "action_schema": self.action_schema,
            "action_denorm": self.action_denorm,
            "action_clamp": self.action_clamp,
            "normalized": summarize_normalized_action_debug(normalized_action, action_dim=self.raw_action_dim),
            "robocasa29": summarize_robocasa29_action_debug(
                robocasa_action,
                current_state=self._current_robocasa29_state(prepared),
            ),
        }
        if history_action_debug is not None:
            payload["history_action"] = history_action_debug
        if normalized_action_before_clamp is not None:
            payload["normalized_before_clamp"] = summarize_normalized_action_debug(
                normalized_action_before_clamp, action_dim=self.raw_action_dim
            )
        logging.info("ACTION_DEBUG %s", json.dumps(payload, sort_keys=True))

    def _save_model_input_debug(self, request_index: int, data_batch: dict, prepared: Any, seed: int) -> None:
        if self._model_input_debug_dir is None or request_index >= self._model_input_debug_max:
            return
        import json as _json
        debug_dir = self._model_input_debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"request_{request_index:04d}"
        # Save prompt
        info = {
            "request_index": request_index,
            "seed": seed,
            "prompt": prepared.prompt,
            "env_name": prepared.env_name,
            "action_schema": self.action_schema,
            "action_history": self.action_history,
            "raw_action_dim": self.raw_action_dim,
            "resolution": self.resolution,
        }
        # Save caption actually fed to model
        if "ai_caption" in data_batch:
            cap = data_batch["ai_caption"]
            info["ai_caption"] = cap[0] if isinstance(cap, list) else cap
        # Save action shape and values
        if "action" in data_batch:
            action = data_batch["action"]
            if isinstance(action, list) and len(action) > 0:
                a = action[0]
                if isinstance(a, list) and len(a) > 0:
                    a = a[0]
                if isinstance(a, torch.Tensor):
                    info["action_shape"] = list(a.shape)
                    info["action_nonzero_count"] = int((a != 0).sum().item())
                    info["action_first_row"] = a[0, :self.raw_action_dim].tolist()
        # Save sequence plan
        if "sequence_plan" in data_batch:
            sp = data_batch["sequence_plan"]
            if isinstance(sp, list):
                sp = sp[0]
            if hasattr(sp, "as_dict"):
                info["sequence_plan"] = sp.as_dict()
        # Save domain_id
        if "domain_id" in data_batch:
            d = data_batch["domain_id"]
            if isinstance(d, list):
                d = d[0]
            if isinstance(d, torch.Tensor):
                info["domain_id"] = int(d.item())
        # Save image (first frame)
        if "video" in data_batch:
            video = data_batch["video"]
            if isinstance(video, list) and len(video) > 0:
                v = video[0]
                if isinstance(v, list) and len(v) > 0:
                    v = v[0]
                if isinstance(v, torch.Tensor):
                    info["video_shape"] = list(v.shape)
                    # Save first frame as png
                    first_frame = v[:, 0].permute(1, 2, 0).cpu().numpy()
                    if first_frame.dtype == np.uint8 or first_frame.max() > 1:
                        img = first_frame.astype(np.uint8)
                    else:
                        img = (first_frame * 255).astype(np.uint8)
                    import cv2
                    cv2.imwrite(str(debug_dir / f"{prefix}_image.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        meta_path = debug_dir / f"{prefix}_model_input.json"
        meta_path.write_text(_json.dumps(info, indent=2, default=str))


    def _save_pipeline_debug(
        self,
        request_index: int,
        prepared: "PreparedEvalExample",
        history_action: "torch.Tensor | None",
        normalized_action: torch.Tensor,
        denormalized_action: torch.Tensor,
        final_action_29d: torch.Tensor,
        history_len: int,
    ) -> None:
        """Save intermediate pipeline variables for train-eval mismatch verification.

        Activated by --model-input-debug-dir. Saves one JSON per request for
        the first ``_model_input_debug_max`` requests.
        """
        if self._model_input_debug_dir is None or request_index >= self._model_input_debug_max:
            return

        debug_dir = self._model_input_debug_dir / "pipeline_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        def _t2l(t: torch.Tensor | None) -> list | None:
            if t is None:
                return None
            return t.detach().cpu().float().tolist()

        dump: dict = {
            "request_index": request_index,
            "env_name": prepared.env_name,
            "prompt": prepared.prompt,
            "action_schema": self.action_schema,
            "raw_action_dim": self.raw_action_dim,
        }

        # --- Input side ---
        # 1. Raw state from simulation
        if prepared.raw_state is not None:
            if isinstance(prepared.raw_state, dict):
                dump["raw_state"] = {
                    k: (v.tolist() if hasattr(v, 'tolist') else list(v))
                    for k, v in prepared.raw_state.items()
                }
            else:
                raw_t = torch.as_tensor(prepared.raw_state, dtype=torch.float32)
                dump["raw_state"] = _t2l(raw_t)

        # 2. Packed state (after reordering to 29D)
        packed_state = None
        if self.action_schema == "gr1_29" and prepared.raw_state is not None:
            packed_state = pack_robocasa_raw_state_to_gr1_29(prepared.raw_state)
            dump["packed_state_29d"] = _t2l(packed_state)
            dump["packed_state_29d_shape"] = list(packed_state.shape)

        # 3. State normalization stats
        try:
            s_min, s_range = self._state_stats_for_example(prepared)
            dump["state_stats_min"] = _t2l(s_min)
            dump["state_stats_range"] = _t2l(s_range)
        except Exception as e:
            dump["state_stats_error"] = str(e)

        # 4. Normalized state (= history_action fed to model)
        if history_action is not None:
            dump["history_action_normalized"] = _t2l(history_action)
            dump["history_action_shape"] = list(history_action.shape)

        # --- Output side ---
        # 5. Model output (normalized action, after removing history)
        dump["model_output_normalized"] = _t2l(normalized_action)
        dump["model_output_normalized_shape"] = list(normalized_action.shape)

        # 6. Action normalization stats
        if self.action_schema == "gr1_29" and self.action_denorm == "minmax":
            try:
                a_min, a_range = self._action_stats_for_example(prepared)
                dump["action_stats_min"] = _t2l(a_min)
                dump["action_stats_range"] = _t2l(a_range)
            except Exception as e:
                dump["action_stats_error"] = str(e)

        # 7. Denormalized action (after inverse normalization)
        dump["denormalized_action"] = _t2l(denormalized_action)
        dump["denormalized_action_shape"] = list(denormalized_action.shape)

        # 8. Final 29D action sent to simulation
        dump["final_action_29d"] = _t2l(final_action_29d)
        dump["final_action_29d_shape"] = list(final_action_29d.shape)

        # --- Per-part breakdown (for easy reading) ---
        parts = [("left_arm", 0, 7), ("right_arm", 7, 14), ("left_hand", 14, 20),
                 ("right_hand", 20, 26), ("waist", 26, 29)]

        if packed_state is not None:
            state_row = packed_state[0]
            dump["packed_state_per_part"] = {
                name: _t2l(state_row[s:e]) for name, s, e in parts
            }

        if history_action is not None and history_action.shape[-1] >= 29:
            ha_row = history_action[0]
            dump["history_action_per_part"] = {
                name: _t2l(ha_row[s:e]) for name, s, e in parts
            }

        if final_action_29d.shape[0] > 0 and final_action_29d.shape[-1] >= 29:
            fa_row = final_action_29d[0]
            dump["final_action_first_step_per_part"] = {
                name: _t2l(fa_row[s:e]) for name, s, e in parts
            }

        out_path = debug_dir / f"request_{request_index:04d}_pipeline.json"
        out_path.write_text(json.dumps(dump, indent=2, default=str))
        logging.info("PIPELINE_DEBUG saved: %s", out_path)

    def _predict_one(self, example: dict[str, Any]) -> PolicyPrediction:
        prepared = prepare_eval_example(
            example,
            action_horizon=self.action_horizon,
            action_dim=self.raw_action_dim,
        )
        history_action_debug = self._history_action_debug(prepared)
        history_action = self._build_history_action(prepared)
        history_len = 0 if history_action is None else int(history_action.shape[0])
        use_gr1_train_metadata = self.action_schema in ("gr1_44", "gr1_29")
        sample = build_cosmos3_policy_sample(
            image_uint8=prepared.image_uint8,
            action=prepared.action_template,
            prompt=prepared.prompt,
            domain_id=self.domain_id,
            conditioning_fps=self.conditioning_fps,
            history_action=history_action,
            viewpoint="ego_view" if use_gr1_train_metadata else None,
            idle_frames=0 if use_gr1_train_metadata else None,
        )
        transformed = transform_cosmos3_policy_sample(
            sample,
            resolution=self.resolution,
            max_action_dim=self.max_action_dim,
            append_viewpoint_info=use_gr1_train_metadata,
            append_idle_frames=use_gr1_train_metadata,
        )
        data_batch = build_data_batch_from_sample(transformed)
        request_index = self._request_count
        seed = next_request_seed(
            seed=self.seed,
            request_count=request_index,
            deterministic_seed=self.deterministic_seed,
        )
        self._request_count += 1
        self._save_model_input_debug(request_index, data_batch, prepared, seed)

        with torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                data_batch,
                guidance=self.guidance,
                seed=[seed],
                num_steps=self.num_steps,
                shift=self.shift,
            )
        video = None
        if self.decode_video:
            video = decode_vision_latent_to_uint8_video(self.model, samples["vision"][0])
        normalized_action_before_clamp = torch.as_tensor(samples["action"][0], dtype=torch.float32)
        normalized_action = clamp_normalized_action(
            normalized_action_before_clamp,
            action_dim=self.raw_action_dim,
            enabled=self.action_clamp,
        )
        # For gr1_29, resolve per-example action stats
        denorm_kwargs = {}
        if self.action_schema == "gr1_29" and self.action_denorm == "minmax":
            a_min, a_range = self._action_stats_for_example(prepared)
            denorm_kwargs = {"action_min": a_min, "action_range": a_range}
        action = self._denormalize_action(normalized_action, **denorm_kwargs)
        if history_len:
            if action.shape[0] <= history_len:
                raise ValueError(f"Generated action length {action.shape[0]} is not longer than history_len={history_len}")
            action = action[history_len:]
            normalized_action = normalized_action[history_len:]
            normalized_action_before_clamp = normalized_action_before_clamp[history_len:]
        self._log_action_debug(
            request_index=request_index,
            seed=seed,
            history_len=history_len,
            normalized_action=normalized_action,
            denormalized_action=action,
            normalized_action_before_clamp=normalized_action_before_clamp if self.action_clamp else None,
            prepared=prepared,
            history_action_debug=history_action_debug,
        )
        # --- Pipeline debug dump ---
        if self.action_schema == "gr1_29":
            _final_29d = action[:, :GR1_29_ACTION_DIM]
        elif self.action_schema == "gr1_44":
            _final_29d = remap_gr1_44_to_robocasa29(action)
        else:
            _final_29d = action[:, :ROBOCASA29_ACTION_DIM]
        self._save_pipeline_debug(
            request_index=request_index,
            prepared=prepared,
            history_action=history_action,
            normalized_action=normalized_action,
            denormalized_action=action,
            final_action_29d=_final_29d,
            history_len=history_len,
        )

        return PolicyPrediction(action=action, video=video)

    def predict_action(self, examples: list[dict[str, Any]], **_: Any) -> dict[str, np.ndarray]:
        if not isinstance(examples, list) or len(examples) == 0:
            raise ValueError("predict_action expects a non-empty list under examples")
        predictions = [self._predict_one(example) for example in examples]
        actions = [prediction.action for prediction in predictions]
        response = format_policy_response(actions, raw_action_dim=self.raw_action_dim, action_schema=self.action_schema)
        if self.decode_video:
            videos = [prediction.video for prediction in predictions]
            if any(video is None for video in videos):
                raise RuntimeError("decode_video=True but at least one prediction has no decoded video")
            response["videos"] = np.stack([np.asarray(video) for video in videos], axis=0)
        logging.info("Predicted actions shape=%s", response["actions"].shape)
        return response

    def close(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="/root/workspace/mengya/cosmos-framework/outputs/train/cosmos3/gr1_robot_policy/gr1_robot_policy_posttrain/checkpoints/iter_000020000/")
    parser.add_argument("--checkpoint-repo", default=DEFAULT_CHECKPOINT_REPO)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5682)
    parser.add_argument("--idle-timeout", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", default="256")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--raw-action-dim", type=int, default=None)
    parser.add_argument("--max-action-dim", type=int, default=DEFAULT_MAX_ACTION_DIM)
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--action-schema", default=DEFAULT_ACTION_SCHEMA, choices=("robocasa29", "gr1_44", "gr1_29"))
    parser.add_argument("--action-denorm", default=DEFAULT_ACTION_DENORM, choices=("auto", "none", "minmax"))
    parser.add_argument("--action-stats-path", default=None)
    parser.add_argument("--action-history", default=DEFAULT_ACTION_HISTORY, choices=("auto", "none", "zero", "state"))
    parser.add_argument("--action-state-stats-path", default=None)
    parser.add_argument("--action-state-stats-root", default=str(DEFAULT_GR1_STATE_STATS_ROOT))
    parser.add_argument("--conditioning-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--sampler", default="unipc")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--use-ema-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--model-input-debug-dir", type=str, default="/root/workspace/lixing/eval_results/server_debug",
                        help="Directory to save model inputs for train/test mismatch debugging")
    parser.add_argument("--model-input-debug-max", type=int, default=10,
                        help="Max number of requests to save debug info for")
    parser.add_argument("--action-clamp", action="store_true")
    parser.add_argument("--decode-video", action="store_true", default=True)
    parser.add_argument("--no-decode-video", dest="decode_video", action="store_false")
    return parser


def main() -> None:
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    args = build_argparser().parse_args()
    policy = Cosmos3RobocasaPolicy(
        checkpoint_path=args.checkpoint_path,
        checkpoint_repo=args.checkpoint_repo,
        checkpoint_revision=args.checkpoint_revision,
        allow_download=args.allow_download,
        output_dir=args.output_dir,
        resolution=args.resolution,
        action_horizon=args.action_horizon,
        raw_action_dim=args.raw_action_dim,
        max_action_dim=args.max_action_dim,
        domain_id=args.domain_id,
        action_schema=args.action_schema,
        action_denorm=args.action_denorm,
        action_stats_path=args.action_stats_path,
        action_history=args.action_history,
        action_state_stats_path=args.action_state_stats_path,
        action_state_stats_root=args.action_state_stats_root,
        conditioning_fps=args.conditioning_fps,
        sampler=args.sampler,
        guidance=args.guidance,
        num_steps=args.num_steps,
        shift=args.shift,
        use_ema_weights=args.use_ema_weights,
        seed=args.seed,
        deterministic_seed=args.deterministic_seed,
        action_clamp=args.action_clamp,
        decode_video=args.decode_video,
        model_input_debug_dir=args.model_input_debug_dir,
        model_input_debug_max=args.model_input_debug_max,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=policy.metadata,
    )
    try:
        logging.info("Cosmos3 RoboCasa policy server listening on %s:%d", args.host, args.port)
        server.serve_forever()
    finally:
        policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
