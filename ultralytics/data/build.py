# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import (
    GroundingDataset,
    YOLODataset,
    YOLOMultiModalDataset,
)
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_2_0


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Yield batches from the underlying iterator.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite DataLoader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # not supported by earlier versions
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the DataLoader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration over a
    dataset without recreating the sampler.

    Attributes:
        sampler (torch.utils.data.Sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler that assigns contiguous batch-aligned chunks of the dataset to each GPU.

    Unlike PyTorch's DistributedSampler which distributes samples in a round-robin fashion (GPU 0 gets indices
    [0,2,4,...], GPU 1 gets [1,3,5,...]), this sampler gives each GPU contiguous batches of the dataset (GPU 0 gets
    batches [0,1,2,...], GPU 1 gets batches [k,k+1,...], etc.). This preserves any ordering or grouping in the original
    dataset, which is critical when samples are organized by similarity (e.g., images sorted by size to enable efficient
    batching without padding when using rect=True).

    The sampler handles uneven batch counts by distributing remainder batches to the first few ranks, ensuring all
    samples are covered exactly once across all GPUs.

    Args:
        dataset (Dataset): Dataset to sample from. Must implement __len__.
        num_replicas (int, optional): Number of distributed processes. Defaults to world size.
        batch_size (int, optional): Batch size used by dataloader. Defaults to dataset.batch_size or 1.
        rank (int, optional): Rank of current process. Defaults to current rank.
        shuffle (bool, optional): Whether to shuffle indices within each rank's chunk. Defaults to False. When True,
            shuffling is deterministic and controlled by set_epoch() for reproducibility.

    Examples:
        >>> # For validation with size-grouped images
        >>> sampler = ContiguousDistributedSampler(val_dataset, batch_size=32, shuffle=False)
        >>> loader = DataLoader(val_dataset, batch_size=32, sampler=sampler)
        >>> # For training with shuffling
        >>> sampler = ContiguousDistributedSampler(train_dataset, batch_size=32, shuffle=True)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize the sampler with dataset and distributed training parameters."""
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        # ensure all ranks have a sample if batch size >= total size; degenerates to round-robin sampler
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        """Calculate the start and end sample indices for this rank."""
        # Calculate which batches this rank handles
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas

        # This rank gets an extra batch if rank < remainder
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)

        # Calculate starting batch: base position + number of extra batches given to earlier ranks
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank

        # Convert batch indices to sample indices
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)

        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        """Generate indices for this rank's contiguous chunk of the dataset."""
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples in this rank's chunk."""
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler to ensure different shuffling patterns across epochs.

        Args:
            epoch (int): Epoch number to use as the random seed for shuffling.
        """
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
) -> Dataset:
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> Dataset:
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    """Create and return an InfiniteDataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker processes for data loading.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    # return InfiniteDataLoader(
    #     dataset=dataset,
    #     batch_size=batch,
    #     shuffle=shuffle and sampler is None,
    #     num_workers=nw,
    #     sampler=sampler,
    #     prefetch_factor=4 if nw > 0 else None,  # increase over default 2
    #     pin_memory=nd > 0 and pin_memory,
    #     collate_fn=getattr(dataset, "collate_fn", None),
    #     worker_init_fn=seed_worker,
    #     generator=generator,
    #     drop_last=drop_last and len(dataset) % batch != 0,
    # )
    use_oversample = True
    oversample_beta = 0.5
    oversample_strategy = 'mean'
    # train
    # if shuffle == True: 
    #     return InfiniteDataLoaderV2(
    #         dataset=dataset,
    #         batch_size=batch,
    #         shuffle=shuffle and sampler is None and not use_oversample,
    #         num_workers=nw,
    #         sampler=sampler,
    #         prefetch_factor=4 if nw > 0 else None,
    #         pin_memory=nd > 0 and pin_memory,
    #         collate_fn=getattr(dataset, "collate_fn", None),
    #         worker_init_fn=seed_worker,
    #         generator=generator,
    #         drop_last=drop_last and len(dataset) % batch != 0,
    #         oversample=use_oversample,
    #         oversample_beta=oversample_beta,
    #         oversample_strategy=oversample_strategy,
    #     )
    # else:
    #     return InfiniteDataLoader(
    #         dataset=dataset,
    #         batch_size=batch,
    #         shuffle=shuffle and sampler is None,
    #         num_workers=nw,
    #         sampler=sampler,
    #         prefetch_factor=4 if nw > 0 else None,  # increase over default 2
    #         pin_memory=nd > 0 and pin_memory,
    #         collate_fn=getattr(dataset, "collate_fn", None),
    #         worker_init_fn=seed_worker,
    #         generator=generator,
    #         drop_last=drop_last and len(dataset) % batch != 0,
    #     )
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )

def check_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
) -> tuple[Any, bool, bool, bool, bool, bool]:
    """Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Examples:
        Check a file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        Check a webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(".")[-1] in (
            IMG_FORMATS | VID_FORMATS
        )
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    """Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source for
            inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # DataLoader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset



def _extract_labels(dataset) -> List[np.ndarray]:
    """Extract class arrays from YOLO-style dataset.labels."""
    raw = getattr(dataset, "labels", None)
    if raw is None:
        raise AttributeError("Dataset must have a `labels` attribute.")
    out = []
    for entry in raw:
        if isinstance(entry, dict):
            cls = entry.get("cls", np.array([]))
        else:
            cls = entry
        out.append(np.asarray(cls).flatten())
    return out


# ---------------------------------------------------------------------------
# Diagnosis — gọi trước khi train
# ---------------------------------------------------------------------------

def diagnose_dataset(dataset, class_names: Optional[List[str]] = None) -> Dict:
    """
    In thống kê phân phối class. Gọi trước khi train để verify.

    Usage:
        diagnose_dataset(train_dataset, class_names=CLASS_NAMES)
    """
    labels = _extract_labels(dataset)
    nc = int(max(c for lb in labels for c in lb) + 1) if labels else 1

    class_counts = np.zeros(nc, dtype=np.int64)
    images_per_class = np.zeros(nc, dtype=np.int64)
    multi_class_images = 0

    for lb in labels:
        unique_cls = set(int(c) for c in lb)
        if len(unique_cls) > 1:
            multi_class_images += 1
        for c in lb:
            class_counts[int(c)] += 1
        for c in unique_cls:
            images_per_class[c] += 1

    freq = class_counts / np.maximum(class_counts.sum(), 1)
    weights = 1.0 / np.maximum(freq, 1e-8)
    weights /= weights.sum()

    print("\n" + "=" * 65)
    print(f"{'CLASS':<30} {'INSTANCES':>10} {'IMAGES':>10} {'WEIGHT':>10}")
    print("=" * 65)
    for i in range(nc):
        name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        print(f"{name:<30} {class_counts[i]:>10} {images_per_class[i]:>10} {weights[i]:>10.4f}")
    print("=" * 65)
    print(f"Total images       : {len(labels)}")
    print(f"Multi-class images : {multi_class_images} ({100*multi_class_images/max(len(labels),1):.1f}%)")
    print("=" * 65 + "\n")

    return dict(class_counts=class_counts, images_per_class=images_per_class, weights=weights)


def verify_batch_distribution(
    loader, num_batches: int = 50, class_names: Optional[List[str]] = None
):
    """
    Chạy num_batches rồi in phân phối class thực tế — dùng sau khi tạo loader.

    Usage:
        verify_batch_distribution(train_loader, num_batches=50, class_names=CLASS_NAMES)
    """
    counts = None
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        cls = batch["cls"].clone().to(torch.int).flatten()
        bc = torch.bincount(cls)
        if counts is None:
            counts = bc
        else:
            if bc.shape[0] > counts.shape[0]:
                counts = torch.cat([counts, counts.new_zeros(bc.shape[0] - counts.shape[0])])
            counts[: bc.shape[0]] += bc

    print(f"\nBatch distribution over {num_batches} batches:")
    print("=" * 55)
    for i, cnt in enumerate(counts):
        name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        bar = "█" * int(40 * cnt.item() / counts.max().item())
        print(f"{name:<30} {cnt.item():>6}  {bar}")
    print("=" * 55 + "\n")
    return counts


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class ClassAwareWeightedSampler(torch.utils.data.WeightedRandomSampler):
    """
    Oversample minority classes dựa trên inverse-frequency.

    Chiến lược tổng hợp weight khi 1 ảnh có nhiều class (strategy):
        'max'  → weight = max weight trong các class của ảnh  (aggressive)
        'mean' → weight = mean                                 (balanced, khuyến nghị)
        'sum'  → weight = sum                                  (ưu tiên ảnh đa dạng class)

    Args:
        dataset:            YOLO dataset với `labels`.
        num_samples:        Số samples/epoch. Mặc định = len(dataset).
        beta (float):       [0=uniform, 1=full inv-freq]. Mặc định 0.5.
        strategy (str):     'max' | 'mean' | 'sum'. Mặc định 'mean'.
        background_weight:  Weight cho ảnh không có foreground.
    """

    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        beta: float = 0.5,
        strategy: str = "mean",
        background_weight: Optional[float] = None,
    ):
        assert strategy in ("max", "mean", "sum"), \
            f"strategy must be 'max'|'mean'|'sum', got '{strategy}'"

        self.beta = beta
        self.strategy = strategy

        labels = _extract_labels(dataset)
        nc = int(max(c for lb in labels for c in lb) + 1) if labels else 1

        # ── Class weight ─────────────────────────────────────────────────────
        class_counts = np.zeros(nc, dtype=np.float64)
        for lb in labels:
            for c in lb:
                class_counts[int(c)] += 1

        class_counts = np.maximum(class_counts, 1)
        freq = class_counts / class_counts.sum()
        class_weights = 1.0 / (freq ** beta)
        class_weights /= class_weights.sum()
        self.class_weights_ = class_weights

        _bg_w = background_weight if background_weight is not None else float(class_weights.min())

        # ── Per-image weight ─────────────────────────────────────────────────
        sample_weights = np.zeros(len(labels), dtype=np.float64)
        for i, lb in enumerate(labels):
            if len(lb) == 0:
                sample_weights[i] = _bg_w
                continue
            ws = class_weights[[int(c) for c in lb]]
            if strategy == "max":
                sample_weights[i] = ws.max()
            elif strategy == "mean":
                sample_weights[i] = ws.mean()
            else:
                sample_weights[i] = ws.sum()

        super().__init__(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=num_samples or len(dataset),
            replacement=True,
        )

    def summary(self) -> Dict[str, float]:
        return {f"class_{i}": float(w) for i, w in enumerate(self.class_weights_)}


# ---------------------------------------------------------------------------
# InfiniteDataLoader
# ---------------------------------------------------------------------------

class InfiniteDataLoaderV2(dataloader.DataLoader):
    """
    InfiniteDataLoader với oversampling tích hợp.

    Extra kwargs:
        oversample (bool):           Default True.
        oversample_beta (float):     Default 0.5.
        oversample_strategy (str):   'max'|'mean'|'sum'. Default 'mean'.
    """

    def __init__(
        self,
        *args: Any,
        oversample: bool = True,
        oversample_beta: float = 0.5,
        oversample_strategy: str = "mean",
        **kwargs: Any,
    ):
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)

        if oversample and kwargs.get("sampler") is None:
            dataset = args[0] if args else kwargs["dataset"]
            sampler = ClassAwareWeightedSampler(
                dataset=dataset,
                num_samples=len(dataset),
                beta=oversample_beta,
                strategy=oversample_strategy,
            )
            kwargs["sampler"] = sampler
            kwargs["shuffle"] = False
            print(
                f"[InfiniteDataLoader] Oversampling ON | "
                f"beta={oversample_beta} | strategy='{oversample_strategy}'"
            )

        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()
        except Exception:
            pass

    def reset(self):
        self.iterator = self._get_iterator()