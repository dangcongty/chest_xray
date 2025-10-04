# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, List
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.utils.data.sampler
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import (
    GroundingDataset,
    YOLOConfidenceAwareDataset,
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
    """
    Dataloader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Create a sampler that repeats indefinitely.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # not supported by earlier versions

        if kwargs['mode'] == 'val':
            kwargs.pop('mode')
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        else:
            # Tạo balanced batch sampler
            batch_sampler = BalancedBatchSampler(
                batch_size=kwargs['batch_size'],
                drop_last=kwargs.get('drop_last', True),
                shuffle=kwargs.get('shuffle', True),
                bg_ratio=0.5,  # Có thể điều chỉnh: 0.3, 0.4, 0.5
                label_files=kwargs['dataset'].label_files
            )
            # Wrap với RepeatSampler cho infinite iteration
            repeat_sampler = _RepeatSampler(batch_sampler)
            # Khởi tạo DataLoader với batch_sampler
            kwargs.pop('mode')
            kwargs.pop('batch_size')
            kwargs.pop('drop_last')
            kwargs.pop('shuffle')
            super().__init__(
                batch_sampler=repeat_sampler,
                **kwargs
            )
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the dataloader is deleted."""
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

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler cân bằng 2 class với tỷ lệ bất đối xứng (vd: 1:15).
    
    Chiến lược:
    - Mỗi batch có tỷ lệ bg:obj = 50:50 (hoặc tùy chỉnh)
    - Class thiểu số được oversample (lặp lại)
    - Class đa số được sử dụng đầy đủ qua các epoch
    
    Args:
        batch_size: kích thước batch (nên là số chẵn)
        drop_last: bỏ batch cuối nếu không đủ samples
        shuffle: có shuffle indices không
        bg_ratio: tỷ lệ background trong batch (0.5 = 50%)
        label_txt: đường dẫn file train.txt (optional, mặc định datasets/process/train.txt)
    """
    
    def __init__(self, batch_size, drop_last=True, shuffle=True, bg_ratio=0.5, label_files=[], max_oversample_ratio=None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bg_ratio = bg_ratio
        self.label_files = label_files
        self.max_oversample_ratio = max_oversample_ratio  # Giới hạn tỷ lệ oversample (vd: 2.0 = lặp tối đa 2 lần)
        
        # Tính số samples mỗi class trong 1 batch
        self.bg_per_batch = int(batch_size * bg_ratio)
        self.obj_per_batch = batch_size - self.bg_per_batch
        
        # Scan và phân loại indices
        self.bg_indices, self.obj_indices = self._scan_labels()
        
        print(f"[INFO] Found {len(self.bg_indices)} background samples")
        print(f"[INFO] Found {len(self.obj_indices)} object samples")
        print(f"[INFO] Ratio bg:obj = 1:{len(self.obj_indices)/max(len(self.bg_indices), 1):.1f}")
        print(f"[INFO] Each batch: {self.bg_per_batch} bg + {self.obj_per_batch} obj")
        
        # Tính số batch dựa trên class có NHIỀU samples hơn
        # Để tận dụng hết data của class đa số
        if len(self.bg_indices) > len(self.obj_indices):
            # Background là đa số
            max_bg_batches = len(self.bg_indices) // self.bg_per_batch
            
            # Giới hạn oversample nếu cần
            if self.max_oversample_ratio:
                max_allowed = int(len(self.obj_indices) * self.max_oversample_ratio / self.obj_per_batch)
                self.num_batches = min(max_bg_batches, max_allowed)
            else:
                self.num_batches = max_bg_batches
        else:
            # Object là đa số
            max_obj_batches = len(self.obj_indices) // self.obj_per_batch
            
            # Giới hạn oversample nếu cần
            if self.max_oversample_ratio:
                max_allowed = int(len(self.bg_indices) * self.max_oversample_ratio / self.bg_per_batch)
                self.num_batches = min(max_obj_batches, max_allowed)
            else:
                self.num_batches = max_obj_batches
        
        # Tính tỷ lệ oversample thực tế
        minority_class = "obj" if len(self.bg_indices) > len(self.obj_indices) else "bg"
        minority_size = min(len(self.bg_indices), len(self.obj_indices))
        minority_per_batch = self.obj_per_batch if minority_class == "obj" else self.bg_per_batch
        actual_oversample = (self.num_batches * minority_per_batch) / minority_size if minority_size > 0 else 0
        
        print(f"[INFO] Total batches per epoch: {self.num_batches}")
        print(f"[INFO] Class {minority_class} (thiểu số) sẽ được oversample {actual_oversample:.2f}x")
    
    def _scan_labels(self):
        """Quét các file label và phân loại thành bg/obj indices"""
        bg_indices, obj_indices = [], []
        invalid_count = 0
        # QUAN TRỌNG: Dùng index gốc từ file train.txt
        for dataset_idx, img_path in enumerate(self.label_files):
            # Chuyển đổi từ image path sang label path
            label_path = img_path.replace('images', 'labels')
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG']:
                label_path = label_path.replace(ext, '.txt')
            
            if not label_path.endswith('.txt'):
                label_path += '.txt'
            
            # Kiểm tra file tồn tại
            if not os.path.exists(label_path):
                invalid_count += 1
                continue
            
            # Kiểm tra valid (không có bbox âm hoặc > 1)
            is_valid = True
            if os.path.getsize(label_path) > 0:
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                x_c, y_c, w, h = map(float, parts[1:5])
                                if w <= 0 or h <= 0 or w > 1 or h > 1:
                                    is_valid = False
                                    break
                except Exception as e:
                    is_valid = False
            
            if not is_valid:
                invalid_count += 1
                continue
            
            # Phân loại bg hoặc obj - SỬ DỤNG dataset_idx (index gốc)
            if os.path.getsize(label_path) == 0:
                bg_indices.append(dataset_idx)
            else:
                obj_indices.append(dataset_idx)
        
        if invalid_count > 0:
            print(f"[WARNING] Bỏ qua {invalid_count} samples không hợp lệ")
        
        return bg_indices, obj_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """Tạo các batch cân bằng với oversample class thiểu số"""
        # Xác định class nào là thiểu số
        is_bg_minority = len(self.bg_indices) < len(self.obj_indices)
        
        # Tạo pools
        bg_pool = self.bg_indices.copy()
        obj_pool = self.obj_indices.copy()
        
        if self.shuffle:
            random.shuffle(bg_pool)
            random.shuffle(obj_pool)
        
        # Oversample class thiểu số để đủ cho tất cả các batch
        if is_bg_minority:
            # Background là thiểu số, cần oversample
            required_bg = self.num_batches * self.bg_per_batch
            repeats = (required_bg // len(bg_pool)) + 1
            bg_pool_extended = []
            for _ in range(repeats):
                temp = bg_pool.copy()
                if self.shuffle:
                    random.shuffle(temp)
                bg_pool_extended.extend(temp)
            bg_pool = bg_pool_extended[:required_bg]
        else:
            # Object là thiểu số, cần oversample
            required_obj = self.num_batches * self.obj_per_batch
            repeats = (required_obj // len(obj_pool)) + 1
            obj_pool_extended = []
            for _ in range(repeats):
                temp = obj_pool.copy()
                if self.shuffle:
                    random.shuffle(temp)
                obj_pool_extended.extend(temp)
            obj_pool = obj_pool_extended[:required_obj]
        
        # Tạo các batch
        bg_ptr = 0
        obj_ptr = 0
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Lấy bg samples
            for _ in range(self.bg_per_batch):
                if bg_ptr >= len(bg_pool):
                    if self.drop_last:
                        return
                    bg_ptr = 0
                batch.append(bg_pool[bg_ptr])
                bg_ptr += 1
            
            # Lấy obj samples
            for _ in range(self.obj_per_batch):
                if obj_ptr >= len(obj_pool):
                    if self.drop_last:
                        return
                    obj_ptr = 0
                batch.append(obj_pool[obj_ptr])
                obj_ptr += 1
            
            # Shuffle trong batch để trộn bg và obj
            if self.shuffle:
                random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        """Trả về số lượng batches trong một epoch"""
        return self.num_batches



class _RepeatSampler:
    """
    Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration
    over a dataset without recreating the sampler.

    Attributes:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id: int):  # noqa
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
):
    """Build and return a YOLO dataset based on configuration parameters."""
    if cfg.use_conf_aware:
        dataset = YOLOConfidenceAwareDataset
    else:
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
):
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


def build_dataloader(dataset, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False, mode = 'val'):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.

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
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
        mode = mode,
    )


def check_source(source):
    """
    Check the type of input source and return corresponding flag values.

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


def load_inference_source(source=None, batch: int = 1, vid_stride: int = 1, buffer: bool = False, channels: int = 3):
    """
    Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | Path | torch.Tensor | PIL.Image | np.ndarray, optional): The input source for inference.
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

    # Dataloader
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
