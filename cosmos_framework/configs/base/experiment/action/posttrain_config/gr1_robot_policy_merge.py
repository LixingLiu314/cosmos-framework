# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``gr1_robot_policy_merge`` — Cosmos3-Nano GR1 action-policy SFT recipe.

Re-based onto the new action dataloader stack (``PackingDataLoader`` +
``RankPartitionedDataLoader`` + ``ActionIterableShuffleDataset``), mirroring
``action_policy_droid_nano``. Feeds the GR1 LeRobot dataset (29D joint policy,
``ego_view``, per-dataset min/max normalization, ``use_state`` prepend) through
``ActionTransformPipeline``, and trains the generation + action heads from the
Cosmos3-Nano base.

Differences from the DROID recipe (GR1-specific):
  - 29D GR1 joint action (zero parts filtered), ego_view single camera
  - chunk_length=16  ->  encode_exact_durations=[17]
  - per-dataset normalization handled inside the dataset (transform normalizer off)
  - warm_up_steps=500, local checkpointing

Usage (1 node, 8 GPU)::

    GR1_DATA_ROOT=/path/to/gr1_lerobot_root \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/gr1_robot_policy_merge.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import get_action_gr1_sft_dataset

cs = ConfigStore.instance()


gr1_robot_policy_merge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "local"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "generation",  # online sampling: logs generated video/action to W&B every N steps
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="gr1_robot_policy",
            name="gr1_robot_policy_merge",
            wandb_mode="${oc.env:WANDB_MODE,online}",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, vision_gen=True, max_action_dim=64, num_embodiment_domains=32
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,  # GR1-tuned (v2/v3 policy); paired with max_samples_per_batch=128
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[60000],  # match max_iter
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            max_iter=60000,
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
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                # Online sampling viz (no S3 -> save_s3=False; samples logged to W&B).
                every_n_sample_reg=dict(every_n=2000, save_s3=False, do_x0_prediction=False),
                every_n_sample_ema=dict(every_n=2000, is_ema=True, save_s3=False, do_x0_prediction=False),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=True,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the base has no GR1-trained
            # action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="${oc.env:BASE_CHECKPOINT_PATH}",  # Cosmos3-Nano DCP dir
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=5000,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_gr1",
            max_samples_per_batch=128,  # per rank; reduce at launch on lower-memory GPUs
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=False,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # Shuffling handled by the dataset (iterable_shuffle=True): episode-order
                # shuffled, sequential-within-episode stream -> cross-rank batch
                # decorrelation with sequential reads (fixes grad-norm instability).
                datasets=dict(
                    gr1=dict(
                        ratio=1,
                        dataset=L(get_action_gr1_sft_dataset)(
                            root="${oc.env:GR1_DATA_ROOT}",
                            chunk_length=16,
                            # Policy-only task mode (predict actions + future video given
                            # first frame). "joint" would dilute each per-task loss.
                            mode="policy",
                            use_state=True,  # prepend initial observed state as conditioning frame
                            viewpoint="ego_view",
                            iterable_shuffle=True,  # rank x worker episode-shuffle stream
                            episode_shuffle_seed=42,
                            use_image_augmentation=True,  # random crop+rescale + color jitter
                            resolution="256",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            append_idle_frames=True,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


# GR1 model resolution (NANO default is 720).
gr1_robot_policy_merge["model"]["config"]["resolution"] = "256"

# chunk_length=16 -> 17 observation frames; pin the VAE encode duration to match.
gr1_robot_policy_merge["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]

# Uncap the packed-sequence length (NANO default 45056 caps + truncates long windows).
gr1_robot_policy_merge["model"]["config"]["max_num_tokens_after_packing"] = -1

# Weight the vision flow-matching loss 10x, balancing it against the action loss
# (action_loss_weight=10) so both heads train at comparable gradient magnitude.
gr1_robot_policy_merge["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0


for _item in [gr1_robot_policy_merge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
