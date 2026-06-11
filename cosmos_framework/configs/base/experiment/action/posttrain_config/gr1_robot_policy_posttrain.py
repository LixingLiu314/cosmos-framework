# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GR1 robot-policy posttraining recipe for Cosmos3-Nano.

Policy mode: condition on language + first video frame, then supervise future
video and GR1 action tokens. Intended first smoke run: 2 GPUs.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data.dataloader import default_collate

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.action.datasets.gr1_lerobot_dataset import GR1LeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.data.vfm.data_packer import DataPacker
from cosmos_framework.data.vfm.data_packer_dataloader import DataPackerDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


class GR1RobotPolicyDataPacker(DataPacker):
    """Action policy DataPacker for GR1 LeRobot samples."""

    def __init__(
        self,
        tokenizer_config: Any,
        resolution: str = "256",
        max_action_dim: int = 64,
        spatial_compression: int = 16,
        temporal_compression: int = 4,
        patch_spatial: int = 2,
        cfg_dropout_rate: float = 0.1,
        append_idle_frames: bool = True,
        idle_frames_dropout: float = 0.05,
        format_prompt_as_json: bool = False,
    ) -> None:
        self.resolution = resolution
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression
        self.patch_spatial = patch_spatial
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            append_idle_frames=append_idle_frames,
            idle_frames_dropout=idle_frames_dropout,
            format_prompt_as_json=format_prompt_as_json,
        )
        self._debug_count = 0
        self._debug_limit = int(os.environ.get("GR1_DEBUG_LIMIT", "2"))

    @staticmethod
    def _summarize_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.float().min()) if value.numel() > 0 else None,
                "max": float(value.float().max()) if value.numel() > 0 else None,
            }
        if isinstance(value, list):
            return f"list[len={len(value)}]"
        return value

    def _debug_print(self, stage: str, data: dict) -> None:
        if os.environ.get("GR1_DEBUG_INPUT", "0") != "1":
            return
        if os.environ.get("RANK", "0") != "0":
            return
        if self._debug_count >= self._debug_limit:
            return
        summary = {key: self._summarize_value(value) for key, value in sorted(data.items())}
        print(f"[GR1_DEBUG_INPUT][{stage}] {summary}", flush=True)

    def sft_process_sample(self, item: dict) -> dict:
        item = dict(item)
        self._debug_print("raw_dataset_item", item)
        item["mode"] = "policy"
        if "state" in item:
            item["history_action"] = item["state"]
        sample = self.transform(item, resolution=self.resolution)
        self._debug_print("after_action_transform", sample)
        self._debug_count += 1
        return sample

    def compute_num_tokens(self, sample: dict) -> int:
        tokens = 1
        text = sample.get("text_token_ids")
        if isinstance(text, torch.Tensor):
            tokens += int(text.shape[0])

        video = sample.get("video")
        if isinstance(video, torch.Tensor):
            _, t, h, w = video.shape
            latent_h = math.ceil(h / (self.spatial_compression * self.patch_spatial))
            latent_w = math.ceil(w / (self.spatial_compression * self.patch_spatial))
            latent_t = 1 + (t - 1) // self.temporal_compression
            tokens += latent_h * latent_w * latent_t + 2

        action = sample.get("action")
        if isinstance(action, torch.Tensor):
            tokens += int(action.shape[0])
        return tokens

    def sft_collate_fn(self, samples: list[dict], max_len: int, ignore_label_id: int = -100) -> dict:
        list_of_list_keys = {"text_token_ids", "video", "action", "state"}
        list_keys = {"sequence_plan", "domain_id", "raw_action_dim", "image_size"}
        optional_drop_keys = set()
        batch: dict[str, Any] = {}
        keys = set().union(*(sample.keys() for sample in samples))

        for key in keys:
            if key in optional_drop_keys:
                continue
            values = [sample.get(key) for sample in samples]
            if any(value is None for value in values):
                continue
            if key in list_of_list_keys:
                batch[key] = [[value] for value in values]
            elif key in list_keys:
                batch[key] = values
            else:
                batch[key] = default_collate(values)

        if os.environ.get("GR1_DEBUG_INPUT", "0") == "1" and os.environ.get("RANK", "0") == "0":
            summary = {key: self._summarize_value(value[0][0] if key in list_of_list_keys else value[0] if key in list_keys and value else value) for key, value in sorted(batch.items())}
            print(f"[GR1_DEBUG_INPUT][collated_batch] batch_size={len(samples)} keys={sorted(batch.keys())} summary={summary}", flush=True)
        return batch


def _model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64
    cfg["num_embodiment_domains"] = 32
    cfg["resolution"] = "256"
    cfg["parallelism"]["data_parallel_shard_degree"] = 2
    cfg["parallelism"]["context_parallel_shard_degree"] = 1
    cfg["parallelism"]["cfg_parallel_shard_degree"] = 1
    cfg["compile"]["enabled"] = False
    cfg["ema"]["enabled"] = False
    return cfg


gr1_robot_policy_posttrain = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "local"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "generation"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="gr1_robot_policy",
            name="gr1_robot_policy_posttrain_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            wandb_mode="${oc.env:WANDB_MODE,online}",
        ),
        model=dict(config=_model_config()),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-6,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-5,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="AdamW",
            weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine",
            cycle_lengths=[10000],
            f_max=[1.0],
            f_min=[0.1],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[50],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=1000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(compile_after_iterations=3, enabled=False, warmup_resolutions=None),
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1),
                expert_heatmap=dict(every_n=1000),
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                norm_monitor=dict(
                    every_n=100,
                    layer_norm_only=False,
                    log_stat_wandb=True,
                    model_key=None,
                    save_s3=False,
                    step_size=1,
                    track_activations=True,
                ),
                param_count=dict(save_s3=False),
                sequence_packing_padding=dict(every_n=50),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                wandb_2x=dict(logging_iter_multipler=2, save_logging_iter_multipler=1, save_s3=False),
                wandb_val=dict(save_s3=False),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=True,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="${oc.env:BASE_CHECKPOINT_PATH,/root/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano}",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(enabled=False, export_every_n=1, hf_repo_id=None, upload_to_object_store=dict(bucket="", credentials="", enabled=False)),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(DataPackerDataLoader)(
            data_source=L(GR1LeRobotDataset)(
                root="${oc.env:GR1_DATA_ROOT,/root/workspace/mengya/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/}",
                chunk_length=16,
                mode="policy",
                viewpoint="ego_view",
            ),
            data_packer=L(GR1RobotPolicyDataPacker)(
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                resolution="256",
                max_action_dim=64,
                cfg_dropout_rate=0.1,
                append_idle_frames=True,
            ),
            max_tokens=999999,
            max_batch_size=64,
            pool_size=64,
            shuffle=True,
            seed=0,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(group="experiment", package="_global_", name="gr1_robot_policy_posttrain", node=gr1_robot_policy_posttrain)
