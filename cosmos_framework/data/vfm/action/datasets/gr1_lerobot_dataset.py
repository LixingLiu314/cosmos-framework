# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GR1 LeRobot dataset for Cosmos Action posttraining.

A member of the ``ActionBaseDataset`` family (like ``DROIDLeRobotDataset``),
built for the new action dataloader stack (``ActionSFTDataset`` +
``ActionIterableShuffleDataset`` + ``RankPartitionedDataLoader``). ``__getitem__``
returns the raw sample dict consumed by ``ActionTransformPipeline``
(``video``/``action``/``ai_caption``/``mode``/``domain_id``/``viewpoint``/
``idle_frames``/``conditioning_fps``).

It reuses the base class for the common machinery (``_choose_mode``,
``_compute_idle_frames``, the property accessors, ``domain_id``/``action_names``)
and overrides only what the GR1 data format forces:

  * GR1 ships LeRobot **v2** meta (``episodes.jsonl``/``tasks.jsonl``) rather than
    the family's v3 parquet meta, and may point at a *parent* directory of many
    datasets (multi-root concat) -> custom ``__init__`` (cannot call
    ``super().__init__``, which reads ``meta/tasks.parquet`` and materializes every
    row as a dict).
  * Action/state layouts are read from ``meta/modality.json`` and filtered to the
    non-zero tabletop control parts -> 29D in RoboCasa order (left arm, right arm,
    left hand, right hand, waist); constant-zero left/right leg + neck excluded.
  * Per-dataset min/max normalization from each dataset's ``meta/stats.json`` for
    BOTH action and state (no class-level stats file) -> custom ``_build_result`` /
    ``_load_norm_stats``.
  * ``use_state`` prepends the initial observed state as a *conditioning* (clean)
    action frame -> action length becomes ``chunk+1 == video_length`` and
    ``build_sequence_plan_from_mode`` marks frame 0 as conditioning. The prepended
    state is normalized with the *state* stats and the commanded chunk with the
    *action* stats, matching how the RoboCasa eval server feeds ``history_action``
    at inference (train/inference parity).
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
from torchvision.transforms import functional as TF
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_normalization import normalize_action
from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Joint, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

Viewpoint = Literal["ego_view"]

_ACTIVE_GR1_PART_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
_ZERO_GR1_PARTS = frozenset({"left_leg", "right_leg", "neck"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


class GR1LeRobotDataset(ActionBaseDataset):
    """GR1 joint-action dataset backed by LeRobot parquet/video files.

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
        # We deliberately do NOT call super().__init__: ActionBaseDataset.__init__
        # reads v3 parquet meta (meta/episodes/*.parquet, meta/tasks.parquet) the GR1
        # v2 export lacks, and materializes every row as a Python dict. We set the
        # base-expected attributes directly and load GR1's jsonl/modality meta below.
        if viewpoint != "ego_view":
            raise NotImplementedError("GR1 LeRobot currently exposes only the ego_view camera.")

        # --- base-class attributes (read by inherited helpers/properties) ---
        self._root = Path(root)
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._pose_convention = "backward_framewise"
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = "gr1_lerobot"
        self._domain_id = get_domain_id(self._domain_name)
        self._action_normalization = None  # GR1 normalizes internally (per-dataset min/max)
        self._norm_stats: dict[str, torch.Tensor] | None = None
        self._sample_stride = 1

        # --- GR1-specific attributes ---
        self._use_state = bool(use_state)
        self._use_image_augmentation = bool(use_image_augmentation)
        self._image_augmentor: T.Compose | None = None
        self._state_stats: dict[str, torch.Tensor] | None = None
        self._dataset_roots: list[Path] = []
        self._datasets: list[GR1LeRobotDataset | None] | None = None
        self._cumulative_sizes: list[int] = []

        expected_info_path = self._root / "meta" / "info.json"
        if not expected_info_path.exists():
            # Multi-root: a parent directory of multiple LeRobot datasets.
            self._dataset_roots = sorted(path.parent.parent for path in self._root.glob("*/meta/info.json"))
            if not self._dataset_roots:
                raise FileNotFoundError(
                    f"No LeRobot datasets found under {self._root}. Expected either "
                    f"{expected_info_path} or */meta/info.json."
                )
            total = 0
            for dataset_root in self._dataset_roots:
                total += self._count_samples_for_root(dataset_root, self._chunk_length)
                self._cumulative_sizes.append(total)
            if total == 0:
                raise ValueError(f"No valid GR1 samples found under {self._root}.")
            self._datasets = [None] * len(self._dataset_roots)
            self._load_meta(self._dataset_roots[0], fps)
            return

        # Single-root dataset.
        self._load_meta(self._root, fps)
        self._episodes = self._load_episodes()
        self._tasks = self._load_tasks()
        self._episode_paths: list[Path] = []
        self._win_cumsum: np.ndarray = np.zeros(0, dtype=np.int64)
        self._row_cache_path: Path | None = None
        self._row_cache: list[dict[str, Any]] | None = None
        # Whole-episode decoded-frame cache: the episode-shuffle stream visits an
        # episode's windows sequentially, so we decode the episode's video ONCE and
        # slice each window from memory (~chunk_length x fewer frame decodes).
        self._video_cache_path: Path | None = None
        self._video_cache: torch.Tensor | None = None
        self._build_window_index()

    # ------------------------------------------------------------------ #
    # ActionBaseDataset ABC contract
    # ------------------------------------------------------------------ #
    @property
    def action_dim(self) -> int:
        return self._aspec.dim

    def _action_spec(self) -> ActionSpec:
        return self._aspec

    @classmethod
    def _stats_path(cls) -> Path:
        raise NotImplementedError(
            "GR1 normalizes from each dataset's meta/stats.json (action + state); "
            "there is no class-level stats file."
        )

    # GR1 overrides the base mode setter to fan out to lazily-built sub-datasets.
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

    # ------------------------------------------------------------------ #
    # Meta / index construction
    # ------------------------------------------------------------------ #
    def _load_meta(self, root: Path, fps: float | None) -> None:
        """Load info/modality/stats and derive the (filtered, reordered) 29D layout."""
        self._info = json.loads((root / "meta" / "info.json").read_text())
        self._modality = json.loads((root / "meta" / "modality.json").read_text())
        self._stats = json.loads((root / "meta" / "stats.json").read_text())
        self._fps = float(fps if fps is not None else self._info.get("fps", 20.0))
        self._dt = 1.0 / self._fps
        self._action_parts = self._ordered_parts("action")
        self._state_parts = self._ordered_parts("state")
        self._action_key = self._single_original_key("action")
        self._state_key = self._single_original_key("state")
        self._video_key = self._video_key_from_modality(self._viewpoint)
        self._aspec = build_action_spec(
            *(Joint(n=end - start, label=name) for name, start, end in self._action_parts)
        )

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

        blocks = []
        prev = 0
        for c in self._win_cumsum.tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    # ------------------------------------------------------------------ #
    # Sample construction
    # ------------------------------------------------------------------ #
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

        video = self._load_video(episode, episode_path, rows, row_start)
        raw_action = self._extract_modal_vector(action_rows, self._action_key, self._action_parts)
        raw_state = (
            self._extract_modal_vector(observation_rows[:1], self._state_key, self._state_parts)
            if self._use_state
            else None
        )
        task = self._caption_for(observation_rows[0], episode)

        return self._build_result(mode=mode, video=video, action=raw_action, ai_caption=task, raw_state=raw_state)

    def _load_video(
        self,
        episode: dict[str, Any],
        episode_path: Path,
        rows: list[dict[str, Any]],
        row_start: int,
    ) -> torch.Tensor:
        # Slice this window from the (cached) whole-episode decode, then augment.
        full = self._load_episode_video(episode, episode_path, rows)
        video = full[row_start : row_start + self._chunk_length + 1]
        if not self._use_image_augmentation:
            return video
        if self._image_augmentor is None:
            _, _, h, w = video.shape
            self._image_augmentor = T.Compose([
                T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                T.Resize((h, w), antialias=True),
                # hue is applied once per episode in _load_episode_video (it is ~95%
                # of the augmentation cost); here we keep the cheap per-window jitters.
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.0),
            ])
        # One sampled set of params applied uniformly across the window's frames
        # (temporally consistent), resampled per __getitem__.
        return self._image_augmentor(video)

    def _load_episode_video(
        self,
        episode: dict[str, Any],
        episode_path: Path,
        rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Decode and cache the whole episode's frames ([T, C, H, W]) once. Decoding
        every overlapping window separately re-decodes ~chunk_length frames per step;
        decoding the episode in one sequential pass and slicing windows is ~chunk_length
        x fewer decodes (the stream is sequential within an episode, so the cache hits
        for all but the first window)."""
        if self._video_cache_path == episode_path and self._video_cache is not None:
            return self._video_cache
        from_timestamp = float(episode.get(f"videos/{self._video_key}/from_timestamp", 0.0))
        frame_timestamps = [from_timestamp + float(row["timestamp"]) for row in rows]
        frames = decode_video_frames(self._video_path(episode, self._video_key), frame_timestamps, self._tolerance_s)
        if self._use_image_augmentation:
            # Per-episode hue jitter: one RGB<->HSV conversion for the whole episode
            # instead of per window (~95% of the augmentation cost). Color robustness
            # is a dataset-level property, so per-episode granularity is equivalent for
            # learning while ~chunk_length x cheaper. Re-sampled each (re)stream.
            frames = TF.adjust_hue(frames, random.uniform(-0.08, 0.08))
        self._video_cache_path = episode_path
        self._video_cache = frames
        return frames

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
        idle_frames = self._compute_idle_frames(action)
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

    # ------------------------------------------------------------------ #
    # Normalization (per-dataset, dual action/state stats)
    # ------------------------------------------------------------------ #
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
