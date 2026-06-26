"""Run one Cosmos3-Nano-Policy-DROID forward from a StarVLA dataloader sample."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch

COSMOS_REPO = Path(os.environ.get("COSMOS_REPO", "/root/workspace/humanoid_pretrain/cosmos-framework"))
if COSMOS_REPO.exists() and str(COSMOS_REPO) not in sys.path:
    sys.path.insert(0, str(COSMOS_REPO))

DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent / "starvla_gr1_sample.pt"
DEFAULT_OUTPUT_DIR = Path("/tmp/cosmos3_forward_probe")
DEFAULT_CHECKPOINT_REPO = "nvidia/Cosmos3-Nano-Policy-DROID"
DEFAULT_DOMAIN_ID = 21
DEFAULT_FPS = 20
DEFAULT_MAX_ACTION_DIM = 64


def _as_image_uint8_tensor(image_uint8: Any) -> torch.Tensor:
    image = torch.as_tensor(image_uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image_uint8 [H,W,3], got {tuple(image.shape)}")
    if image.dtype != torch.uint8:
        image = image.clamp(0, 255).to(torch.uint8)
    return image.contiguous()


def _as_action_tensor(action: Any) -> torch.Tensor:
    action_tensor = torch.as_tensor(action, dtype=torch.float32)
    if action_tensor.ndim != 2:
        raise ValueError(f"Expected action [T,D], got {tuple(action_tensor.shape)}")
    return action_tensor.contiguous()


def build_cosmos3_policy_sample(
    *,
    image_uint8: Any,
    action: Any,
    prompt: str,
    domain_id: int = DEFAULT_DOMAIN_ID,
    conditioning_fps: int = DEFAULT_FPS,
    history_action: Any | None = None,
    viewpoint: str | None = None,
    idle_frames: int | None = None,
) -> dict[str, Any]:
    """Build the raw sample shape expected before Cosmos3 action transforms."""
    image = _as_image_uint8_tensor(image_uint8)
    action_tensor = _as_action_tensor(action)
    height, width = image.shape[:2]
    first_frame = image.permute(2, 0, 1).contiguous()
    video = first_frame.unsqueeze(1).repeat(1, action_tensor.shape[0] + 1, 1, 1).contiguous()

    sample: dict[str, Any] = {
        "ai_caption": prompt,
        "video": video,
        "action": action_tensor,
        "conditioning_fps": torch.tensor(conditioning_fps, dtype=torch.long),
        "mode": "policy",
        "domain_id": torch.tensor(domain_id, dtype=torch.long),
    }
    if history_action is not None:
        sample["history_action"] = _as_action_tensor(history_action)
    if viewpoint is not None:
        sample["viewpoint"] = viewpoint
    if idle_frames is not None:
        sample["idle_frames"] = torch.tensor(idle_frames, dtype=torch.long)
    return sample


def transform_cosmos3_policy_sample(
    sample: dict[str, Any],
    *,
    resolution: str = "256",
    max_action_dim: int = DEFAULT_MAX_ACTION_DIM,
    append_viewpoint_info: bool = False,
    append_idle_frames: bool = False,
) -> dict[str, Any]:
    from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

    transform = ActionTransformPipeline(
        max_action_dim=max_action_dim,
        cfg_dropout_rate=0.0,
        append_viewpoint_info=append_viewpoint_info,
        append_idle_frames=append_idle_frames,
        idle_frames_dropout=0.0,
    )
    return transform(sample, resolution=resolution)


def build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader

    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]
        else:
            data_batch[key] = [value]
    return data_batch


def resolve_checkpoint_path(
    checkpoint_path: str | None,
    checkpoint_repo: str,
    *,
    allow_download: bool,
    revision: str | None = None,
) -> str:
    if checkpoint_path:
        return checkpoint_path

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(checkpoint_repo, revision=revision, local_files_only=True)
    except Exception as local_exc:
        if not allow_download:
            raise RuntimeError(
                f"Checkpoint {checkpoint_repo!r} is not available in local HF cache. "
                "Re-run with --allow-download and HF_ENDPOINT=https://hf-mirror.com."
            ) from local_exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return snapshot_download(
        checkpoint_repo,
        revision=revision,
        token=token,
        local_files_only=False,
    )


def load_exported_starvla_sample(sample_path: Path) -> dict[str, Any]:
    return torch.load(sample_path, map_location="cpu", weights_only=False)


def _sequence_plan_dict(sample: dict[str, Any]) -> dict[str, Any]:
    sequence_plan = sample.get("sequence_plan")
    if hasattr(sequence_plan, "as_dict"):
        return sequence_plan.as_dict()
    return {}


def print_sample_summary(sample: dict[str, Any], transformed: dict[str, Any], data_batch: dict[str, Any]) -> None:
    print("transformed_ai_caption:", transformed["ai_caption"])
    print("video_shape:", tuple(transformed["video"].shape), "dtype=", transformed["video"].dtype)
    print("action_shape:", tuple(transformed["action"].shape), "dtype=", transformed["action"].dtype)
    print("raw_action_dim:", int(transformed["raw_action_dim"].item()))
    print("domain_id:", int(transformed["domain_id"].item()))
    print("image_size:", transformed["image_size"].tolist())
    print("sequence_plan:", _sequence_plan_dict(transformed))
    print("data_batch_keys:", sorted(data_batch.keys()))
    print("batch_video_shape:", tuple(data_batch["video"][0][0].shape))
    print("batch_action_shape:", tuple(data_batch["action"][0][0].shape))


def run_forward(
    *,
    transformed_sample: dict[str, Any],
    checkpoint_path: str,
    output_dir: Path,
    sampler: str,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
) -> dict[str, Any]:
    from cosmos_framework.inference.common.init import init_output_dir, init_script

    init_script()

    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.scripts.action_policy_server_utils import (
        disable_runtime_ema_for_frozen_config,
        maybe_init_distributed,
    )

    maybe_init_distributed()
    init_output_dir(output_dir)
    setup_args = OmniSetupOverrides.model_validate(
        {
            "checkpoint_path": checkpoint_path,
            "output_dir": str(output_dir),
            "sampler": sampler,
            "guardrails": False,
        }
    ).build_setup()
    setup_args = disable_runtime_ema_for_frozen_config(setup_args)

    print("checkpoint_path:", checkpoint_path)
    print("loading_model: start")
    pipe = OmniInference.create(setup_args)
    model = pipe.model
    model.eval()
    print("loading_model: done")
    print("model_max_action_dim:", getattr(model.config, "max_action_dim", None))

    data_batch = build_data_batch_from_sample(transformed_sample)
    print_sample_summary(transformed_sample, transformed_sample, data_batch)
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("cuda_mem_allocated_mb_before:", round(torch.cuda.memory_allocated() / 1024 / 1024, 2))

    with torch.inference_mode():
        samples = model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            seed=[seed],
            num_steps=num_steps,
            shift=shift,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("cuda_mem_allocated_mb_after:", round(torch.cuda.memory_allocated() / 1024 / 1024, 2))

    print("output_keys:", sorted(samples.keys()))
    for key, value in samples.items():
        if isinstance(value, list):
            shapes = [tuple(item.shape) if isinstance(item, torch.Tensor) else type(item).__name__ for item in value]
            print(f"output_{key}_list_shapes:", shapes)
        elif isinstance(value, torch.Tensor):
            print(f"output_{key}_shape:", tuple(value.shape), "dtype=", value.dtype)
        else:
            print(f"output_{key}_type:", type(value).__name__)

    action_output = samples.get("action")
    if isinstance(action_output, list) and action_output and isinstance(action_output[0], torch.Tensor):
        raw_dim = int(transformed_sample["raw_action_dim"].item())
        action = action_output[0].detach().float().cpu()
        print("pred_action_shape:", tuple(action.shape))
        print("pred_action_raw_shape:", tuple(action[:, :raw_dim].shape))
        print("pred_action_first_row_first_8:", action[0, : min(raw_dim, 8)].tolist())
        print("pred_action_raw_mean_std:", float(action[:, :raw_dim].mean()), float(action[:, :raw_dim].std()))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-repo", default=DEFAULT_CHECKPOINT_REPO)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", default="256")
    parser.add_argument("--max-action-dim", type=int, default=DEFAULT_MAX_ACTION_DIM)
    parser.add_argument("--domain-id", type=int, default=DEFAULT_DOMAIN_ID)
    parser.add_argument("--conditioning-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--sampler", default="unipc")
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--shift", type=float, default=5.0)
    args = parser.parse_args()

    payload = load_exported_starvla_sample(args.sample_path)
    sample = build_cosmos3_policy_sample(
        image_uint8=payload["image_uint8"],
        action=payload["action"],
        prompt=str(payload.get("prompt", "")),
        domain_id=args.domain_id,
        conditioning_fps=args.conditioning_fps,
    )
    transformed = transform_cosmos3_policy_sample(
        sample,
        resolution=args.resolution,
        max_action_dim=args.max_action_dim,
    )
    data_batch = build_data_batch_from_sample(transformed)
    print_sample_summary(sample, transformed, data_batch)

    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint_path,
        args.checkpoint_repo,
        allow_download=args.allow_download,
        revision=args.checkpoint_revision,
    )
    run_forward(
        transformed_sample=transformed,
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        sampler=args.sampler,
        guidance=args.guidance,
        seed=args.seed,
        num_steps=args.num_steps,
        shift=args.shift,
    )


if __name__ == "__main__":
    main()
