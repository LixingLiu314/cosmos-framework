# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GR1 LeRobot dataset for Cosmos Action posttraining.

Built for the *new* action dataloader stack (``ActionSFTDataset`` +
``ActionIterableShuffleDataset`` + ``RankPartitionedDataLoader``), mirroring the
DROID recipe. The base ``__getitem__`` returns the raw sample dict consumed by
``ActionTransformPipeline`` (``video``/``action``/``ai_caption``/``mode``/
``domain_id``/``viewpoint``/``idle_frames``/``conditioning_fps``).

Action layout is 29D in RoboCasa order (left arm, right arm, left hand, right
hand, waist); constant-zero left/right leg and neck channels are excluded.
Per-dataset min/max normalization is read from each dataset's ``meta/stats.json``.

``use_state`` prepends the initial observed state as a *conditioning* (clean)
action frame -> action length becomes ``chunk+1 == video_length`` and
``build_sequence_plan_from_mode`` marks frame 0 as conditioning. The prepended
state is normalized with the *state* stats (``observation.state``) and the
commanded chunk with the *action* stats, matching how the RoboCasa eval server
feeds ``history_action`` at inference (train/inference parity).
"""

from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_normalization import normalize_action
from cosmos_framework.data.vfm.action.action_spec import Joint, build_action_spec
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import compute_idle_frames

Viewpoint = Literal["ego_view"]

_ACTIVE_GR1_PART_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
_ZERO_GR1_PARTS = frozenset({"left_leg", "right_leg", "neck"})
_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


class GR1LeRobotDataset(Dataset):
    """GR1 joint-action dataset backed by LeRobot parquet/video files.

    The action and state layouts are read from ``meta/modality.json`` and
    filtered to the non-zero tabletop control parts. The effective GR1 policy
    vector is 29D in RoboCasa order: left arm, right arm, left hand, right hand,
    and waist. Constant-zero left/right leg and neck channels are excluded.

    ``root`` may point either at a single LeRobot dataset (``meta/info.json``
    present) or at a parent directory of multiple datasets (``*/meta/info.json``),
    which are concatenated.
    """

    def __init__(
        self,
        root: str,
        fps: float | None = None,
        chunk_length: int = 16,
        mode: str = "joint",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "ego_view",
        use_state: bool = False,
        use_image_augmentation: bool = False,
    ) -> None:
        super().__init__()
        if viewpoint != "ego_view":
            raise NotImplementedError("GR1 LeRobot currently exposes only the ego_view camera.")

        self._root = Path(root)
        self._dataset_roots: list[Path] = []
        self._datasets: list[GR1LeRobotDataset | None] | None = None
        self._cumulative_sizes: list[int] = []

        expected_info_path = self._root / "meta" / "info.json"
        if not expected_info_path.exists():
            self._dataset_roots = sorted(path.parent.parent for path in self._root.glob("*/meta/info.json"))
            if not self._dataset_roots:
                raise FileNotFoundError(
                    f"No LeRobot datasets found under {self._root}. Expected either "
                    f"{expected_info_path} or */meta/info.json."
                )

            total = 0
            for dataset_root in self._dataset_roots:
                total += self._count_samples_for_root(dataset_root, int(chunk_length))
                self._cumulative_sizes.append(total)
            if total == 0:
                raise ValueError(f"No valid GR1 samples found under {self._root}.")
            self._datasets = [None] * len(self._dataset_roots)

            first_root = self._dataset_roots[0]
            self._info = json.loads((first_root / "meta" / "info.json").read_text())
            self._modality = json.loads((first_root / "meta" / "modality.json").read_text())
            self._stats = json.loads((first_root / "meta" / "stats.json").read_text())
            self._fps = float(fps if fps is not None else self._info.get("fps", 20.0))
            self._dt = 1.0 / self._fps
            self._chunk_length = int(chunk_length)
            self._mode = mode
            self._tolerance_s = float(tolerance_s)
            self._viewpoint = viewpoint
            self._use_state = bool(use_state)
            self._use_image_augmentation = bool(use_image_augmentation)
            self._image_augmentor: T.Compose | None = None
            self._domain_id = get_domain_id("gr1_lerobot")
            self._norm_stats = None
            self._state_stats = None
            self._action_parts = self._ordered_parts("action")
            self._state_parts = self._ordered_parts("state")
            self._action_key = self._single_original_key("action")
            self._state_key = self._single_original_key("state")
            self._video_key = self._video_key_from_modality(viewpoint)
            self._action_spec = build_action_spec(
                *(Joint(n=end - start, label=name) for name, start, end in self._action_parts)
            )
            return

        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._modality = json.loads((self._root / "meta" / "modality.json").read_text())
        self._stats = json.loads((self._root / "meta" / "stats.json").read_text())
        self._fps = float(fps if fps is not None else self._info.get("fps", 20.0))
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._use_state = bool(use_state)
        self._use_image_augmentation = bool(use_image_augmentation)
        self._image_augmentor: T.Compose | None = None
        self._domain_id = get_domain_id("gr1_lerobot")
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._state_stats: dict[str, torch.Tensor] | None = None

        self._action_parts = self._ordered_parts("action")
        self._state_parts = self._ordered_parts("state")
        self._action_key = self._single_original_key("action")
        self._state_key = self._single_original_key("state")
        self._video_key = self._video_key_from_modality(viewpoint)
        self._action_spec = build_action_spec(
            *(Joint(n=end - start, label=name) for name, start, end in self._action_parts)
        )

        self._episodes = self._load_episodes()
        self._tasks = self._load_tasks()
        # Compact per-episode window index (see _build_window_index): one parquet path
        # per episode + a cumulative window-count int64 array. Avoids a per-window
        # Python list, whose refcount churn would inflate fork/COW worker memory.
        self._episode_paths: list[Path] = []
        self._win_cumsum: np.ndarray = np.zeros(0, dtype=np.int64)
        self._row_cache_path: Path | None = None
        self._row_cache: list[dict[str, Any]] | None = None
        self._build_window_index()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value
        if self._datasets is not None:
            for dataset in self._datasets:
                if dataset is not None:
                    dataset.mode = value

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_dim(self) -> int:
        return self._action_spec.dim

    @property
    def action_names(self) -> list[str]:
        return self._action_spec.names

    @staticmethod
    def _count_samples_for_root(root: Path, chunk_length: int) -> int:
        episodes_jsonl = root / "meta" / "episodes.jsonl"
        if episodes_jsonl.exists():
            return sum(max(0, int(row.get("length", 0)) - chunk_length) for row in _read_jsonl(episodes_jsonl))

        episodes_dir = root / "meta" / "episodes"
        return sum(
            max(0, int(row.get("length", 0)) - chunk_length)
            for path in sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path, columns=["length"]).to_pylist()
        )

    def _ordered_parts(self, kind: str) -> list[tuple[str, int, int]]:
        raw_parts = {
            name: (name, int(item["start"]), int(item["end"]))
            for name, item in self._modality[kind].items()
            if name not in _ZERO_GR1_PARTS
        }
        ordered = [raw_parts[name] for name in _ACTIVE_GR1_PART_ORDER if name in raw_parts]
        ordered.extend(
            part
            for name, part in sorted(raw_parts.items(), key=lambda item: item[1][1])
            if name not in _ACTIVE_GR1_PART_ORDER
        )
        if not ordered:
            raise ValueError(f"GR1 {kind} modality has no active parts after filtering zero parts")
        return ordered

    def _single_original_key(self, kind: str) -> str:
        keys = {str(item["original_key"]) for item in self._modality[kind].values()}
        if len(keys) != 1:
            raise ValueError(f"GR1 {kind} modalities must share one original_key, got {sorted(keys)}")
        return next(iter(keys))

    def _video_key_from_modality(self, viewpoint: str) -> str:
        video_item = self._modality["video"][viewpoint]
        return str(video_item["original_key"])

    def _load_episodes(self) -> dict[int, dict[str, Any]]:
        jsonl_path = self._root / "meta" / "episodes.jsonl"
        if jsonl_path.exists():
            return {int(row["episode_index"]): row for row in _read_jsonl(jsonl_path)}

        episodes_dir = self._root / "meta" / "episodes"
        return {
            int(row["episode_index"]): row
            for path in sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }

    def _load_tasks(self) -> dict[int, str]:
        jsonl_path = self._root / "meta" / "tasks.jsonl"
        if jsonl_path.exists():
            return {int(row["task_index"]): str(row.get("task", "")) for row in _read_jsonl(jsonl_path)}
        parquet_path = self._root / "meta" / "tasks.parquet"
        return {int(row["task_index"]): str(row["task"]) for row in pq.read_table(parquet_path).to_pylist()}

    def _build_window_index(self) -> None:
        """Build the compact per-episode window index: one parquet path per episode
        (file) plus a cumulative window-count int64 array. The flat sample index maps
        to ``(episode, row_start)`` via ``np.searchsorted`` at access time, so we never
        materialize a per-window Python list (which would dirty fork/COW pages via
        refcount churn and grow worker RSS over training)."""
        paths: list[Path] = []
        counts: list[int] = []
        for path in sorted((self._root / "data").glob("chunk-*/*.parquet")):
            num_rows = pq.ParquetFile(path).metadata.num_rows
            n = int(num_rows) - self._chunk_length
            if n > 0:
                paths.append(path)
                counts.append(n)
        self._episode_paths = paths
        self._win_cumsum = np.cumsum(counts, dtype=np.int64) if counts else np.zeros(0, dtype=np.int64)

    def _load_episode_rows(self, path: Path) -> list[dict[str, Any]]:
        if self._row_cache_path == path and self._row_cache is not None:
            return self._row_cache
        rows = pq.read_table(path).to_pylist()
        rows.sort(key=lambda row: (int(row["episode_index"]), int(row.get("index", 0))))
        self._row_cache_path = path
        self._row_cache = rows
        return rows

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _make_sub_dataset(self, dataset_idx: int) -> "GR1LeRobotDataset":
        return GR1LeRobotDataset(
            root=str(self._dataset_roots[dataset_idx]),
            fps=self._fps,
            chunk_length=self._chunk_length,
            mode=self._mode,
            tolerance_s=self._tolerance_s,
            viewpoint=self._viewpoint,
            use_state=self._use_state,
            use_image_augmentation=self._use_image_augmentation,
        )

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode flat-index blocks ``(start, length)`` for the episode-shuffle
        stream. ``ActionIterableShuffleDataset`` shuffles the ORDER of these blocks
        and shards them disjointly across ranks, keeping windows *within* a block
        sequential (decorrelation without random-access I/O). One LeRobot parquet
        file == one episode, so consecutive same-file windows form one block."""
        if self._datasets is not None:
            blocks: list[tuple[int, int]] = []
            prev_size = 0
            for dataset_idx in range(len(self._dataset_roots)):
                sub = self._datasets[dataset_idx]
                if sub is None:
                    sub = self._make_sub_dataset(dataset_idx)
                    self._datasets[dataset_idx] = sub
                for start, length in sub.get_shuffle_blocks():
                    blocks.append((prev_size + start, length))
                prev_size += len(sub)
            return blocks

        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in self._win_cumsum.tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._datasets is not None:
            idx = int(idx)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(idx)
            dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
            prev_size = 0 if dataset_idx == 0 else self._cumulative_sizes[dataset_idx - 1]
            dataset = self._datasets[dataset_idx]
            if dataset is None:
                dataset = self._make_sub_dataset(dataset_idx)
                self._datasets[dataset_idx] = dataset
            return dataset[idx - prev_size]

        mode = self._choose_mode()
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = int(np.searchsorted(self._win_cumsum, idx, side="right"))
        prev = int(self._win_cumsum[file_idx - 1]) if file_idx > 0 else 0
        row_start = idx - prev
        episode_path = self._episode_paths[file_idx]
        rows = self._load_episode_rows(episode_path)
        observation_rows = rows[row_start : row_start + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]
        episode = self._episodes[int(observation_rows[0]["episode_index"])]

        video = self._load_video(episode, observation_rows)
        raw_action = self._extract_modal_vector(action_rows, self._action_key, self._action_parts)
        raw_state = (
            self._extract_modal_vector(observation_rows[:1], self._state_key, self._state_parts)
            if self._use_state
            else None
        )
        task = self._caption_for(observation_rows[0], episode)

        return self._build_result(mode=mode, video=video, action=raw_action, ai_caption=task, raw_state=raw_state)

    def _load_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frame_timestamps = [float(episode.get(f"videos/{self._video_key}/from_timestamp", 0.0)) + ts for ts in timestamps]
        video = decode_video_frames(self._video_path(episode, self._video_key), frame_timestamps, self._tolerance_s)
        if not self._use_image_augmentation:
            return video
        if self._image_augmentor is None:
            _, _, h, w = video.shape
            self._image_augmentor = T.Compose([
                T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                T.Resize((h, w), antialias=True),
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            ])
        # One sampled set of params applied uniformly across all frames (temporally
        # consistent), resampled per __getitem__.
        return self._image_augmentor(video)

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        episode_index = int(episode["episode_index"])
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(
                    f"videos/{video_key}/episode_chunk",
                    episode.get("data/chunk_index", episode_index // int(self._info.get("chunks_size", 1000))),
                ),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=episode_index,
            episode_chunk=chunk_idx,
            episode_file=episode_index,
            episode_index=episode_index,
        )
        return self._root / rel

    @staticmethod
    def _extract_modal_vector(
        rows: list[dict[str, Any]],
        original_key: str,
        parts: list[tuple[str, int, int]],
    ) -> torch.Tensor:
        values = np.asarray([row[original_key] for row in rows], dtype=np.float32)
        ordered = [values[:, start:end] for _, start, end in parts]
        return torch.from_numpy(np.concatenate(ordered, axis=-1)).float()

    def _caption_for(self, row: dict[str, Any], episode: dict[str, Any]) -> str:
        task = self._tasks.get(int(row.get("task_index", 0)), "")
        candidates = [task, episode.get("remarks", ""), episode.get("description", "")]
        candidates.extend(episode.get("tasks", []))
        candidates = [str(item) for item in candidates if str(item).strip()]
        if not candidates:
            return "GR1 robot manipulation task"
        caption = random.choice(candidates)
        return random.choice(caption.split(" | "))

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        raw_state: torch.Tensor | None = None,
        **extras: Any,
    ) -> dict[str, Any]:
        idle_frames = compute_idle_frames(
            action,
            self._action_spec,
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )
        normalized_action = normalize_action(action, "minmax", self._load_norm_stats())
        if raw_state is not None:
            # Prepend the initial observed state as a conditioning (clean) action
            # frame. Normalize with STATE stats to match the eval server's
            # history_action normalization (train/inference parity).
            normalized_state = normalize_action(raw_state, "minmax", self._load_state_stats())
            normalized_action = torch.cat([normalized_state, normalized_action], dim=0)
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is not None:
            return self._norm_stats
        self._norm_stats = self._stats_for_parts(self._action_key, self._action_parts)
        return self._norm_stats

    def _load_state_stats(self) -> dict[str, torch.Tensor]:
        if self._state_stats is not None:
            return self._state_stats
        self._state_stats = self._stats_for_parts(self._state_key, self._state_parts)
        return self._state_stats

    def _stats_for_parts(self, feature_key: str, parts: list[tuple[str, int, int]]) -> dict[str, torch.Tensor]:
        stats = self._stats[feature_key]
        output = {}
        for stat_key in ("min", "max"):
            values = np.asarray(stats[stat_key], dtype=np.float32)
            ordered = [values[start:end] for _, start, end in parts]
            output[stat_key] = torch.from_numpy(np.concatenate(ordered, axis=0)).float()
        return output

    def __len__(self) -> int:
        if self._datasets is not None:
            return self._cumulative_sizes[-1]
        return int(self._win_cumsum[-1]) if self._win_cumsum.size else 0
