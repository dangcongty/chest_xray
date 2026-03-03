# import os
# from glob import glob

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torchvision
# from ensemble_boxes import weighted_boxes_fusion
# from scipy.stats import multivariate_normal
# from tqdm import tqdm


# def bbox_to_gaussian_heatmap(bboxes, image_shape, sigma_factor=0.3):
#     """
#     Convert bounding boxes to Gaussian heatmap.
    
#     Args:
#         bboxes: List of bounding boxes in xyxy format [[x1, y1, x2, y2], ...]
#         image_shape: Tuple (height, width) of the output heatmap
#         sigma_factor: Controls the spread of Gaussian (relative to bbox size)
    
#     Returns:
#         heatmap: 2D numpy array with Gaussian distributions
#     """
#     height, width = image_shape
#     heatmap = np.zeros((height, width), dtype=np.float16)
    
#     # Create coordinate grid
#     y_coords, x_coords = np.mgrid[0:height, 0:width]
#     pos = np.dstack((x_coords, y_coords))
    
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox
        
#         # Calculate center and size
#         cx = (x1 + x2) / 2
#         cy = (y1 + y2) / 2
#         w = x2 - x1
#         h = y2 - y1
        
#         # Define mean (center of bbox)
#         mean = np.array([cx, cy])
        
#         # Define covariance based on bbox size
#         sigma_x = sigma_factor * w
#         sigma_y = sigma_factor * h
#         cov = np.array([[sigma_x**2, 0],
#                         [0, sigma_y**2]])
        
#         # Create Gaussian distribution
#         rv = multivariate_normal(mean, cov)
#         gaussian = rv.pdf(pos)
#         if gaussian.max() > 0:
#             gaussian = gaussian / gaussian.max()
#         # Add to heatmap (take maximum to handle overlaps)
#         heatmap += gaussian
    
#     return heatmap.clip(0, 1)


# os.makedirs('datasets/heatmap', exist_ok=True)

# for path in tqdm(glob('datasets/labels/*.txt')):
#     with open(path, 'r') as f:
#         label = f.readlines()

#     image_shape = (640, 640)  # height, width
#     if not len(label):
#         heatmap = np.zeros(image_shape)
#         continue

#     classes = [int(lb.strip().split()[0]) for lb in label]

#     box_xywh = np.array([np.array(lb.strip().split()[1:], dtype = float) for lb in label])
#     box_xyxy = np.zeros_like(box_xywh)
#     box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2]/2
#     box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3]/2
#     box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2]/2
#     box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3]/2

#     # box_xyxy = box_xyxy*image_shape[0]

#     boxes, scores, labels = weighted_boxes_fusion(
#         [box_xyxy.tolist()], 
#         [np.ones(box_xyxy.shape[0]).tolist()], 
#         [[0] * len(classes)],
#         weights=None,
#         iou_thr=0.5, skip_box_thr=0.0
#     )
#     boxes = boxes*image_shape[0]

#     # Generate heatmap
#     heatmap = bbox_to_gaussian_heatmap(boxes, image_shape, sigma_factor=0.3)

#     np.save(f'datasets/heatmap/{os.path.basename(path)[:-4]}.npy', heatmap)

    # # Visualization
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # # Plot 1: Heatmap with bounding boxes
    # axes[0].imshow(heatmap, cmap='hot', origin='upper')
    # for bbox in boxes:
    #     x1, y1, x2, y2 = bbox.astype(int)
    #     rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
    #                         fill=False, edgecolor='cyan', linewidth=2)
    #     axes[0].add_patch(rect)
    # axes[0].set_title('Gaussian Heatmap with Bounding Boxes')
    # axes[0].set_xlabel('X')
    # axes[0].set_ylabel('Y')

    # # Plot 2: 3D surface plot
    # from mpl_toolkits.mplot3d import Axes3D

    # ax = fig.add_subplot(122, projection='3d')
    # x = np.arange(0, image_shape[1], 5)
    # y = np.arange(0, image_shape[0], 5)
    # X, Y = np.meshgrid(x, y)
    # Z = heatmap[::5, ::5]
    # surf = ax.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)
    # ax.set_title('3D Heatmap Surface')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Density')
    # fig.colorbar(surf, ax=ax, shrink=0.5)

    # plt.tight_layout()
    # plt.show()



import os
from glob import glob

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm


def bbox_to_gaussian_heatmap_gpu(bboxes, image_shape, sigma_factor=0.3, device='cuda'):
    """
    GPU-accelerated conversion of bounding boxes to Gaussian heatmap.
    
    Args:
        bboxes: Tensor of bounding boxes in xyxy format [N, 4]
        image_shape: Tuple (height, width) of the output heatmap
        sigma_factor: Controls the spread of Gaussian (relative to bbox size)
        device: 'cuda' or 'cpu'
    
    Returns:
        heatmap: 2D tensor with Gaussian distributions
    """
    height, width = image_shape
    
    if len(bboxes) == 0:
        return torch.zeros((height, width), dtype=torch.float32, device=device)
    
    # Convert to tensor if needed
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.tensor(bboxes, dtype=torch.float32, device=device)
    else:
        bboxes = bboxes.to(device)
    
    # Create coordinate grid on GPU
    y_coords = torch.arange(height, dtype=torch.float32, device=device).view(-1, 1)
    x_coords = torch.arange(width, dtype=torch.float32, device=device).view(1, -1)
    
    # Calculate centers and sizes
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2  # [N]
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2  # [N]
    w = bboxes[:, 2] - bboxes[:, 0]  # [N]
    h = bboxes[:, 3] - bboxes[:, 1]  # [N]
    
    # Define sigmas based on bbox size
    sigma_x = sigma_factor * w  # [N]
    sigma_y = sigma_factor * h  # [N]
    
    # Vectorized Gaussian computation for all boxes
    # Expand dimensions for broadcasting
    cx = cx.view(-1, 1, 1)  # [N, 1, 1]
    cy = cy.view(-1, 1, 1)  # [N, 1, 1]
    sigma_x = sigma_x.view(-1, 1, 1)  # [N, 1, 1]
    sigma_y = sigma_y.view(-1, 1, 1)  # [N, 1, 1]
    
    # Compute Gaussian for all boxes at once
    dx = x_coords - cx  # [N, H, W]
    dy = y_coords - cy  # [N, H, W]
    
    gaussian = torch.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))
    
    # Normalize each Gaussian
    max_vals = gaussian.reshape(len(bboxes), -1).max(dim=1)[0].view(-1, 1, 1)
    gaussian = torch.where(max_vals > 0, gaussian / max_vals, gaussian)
    
    # Sum all Gaussians (handles overlaps)
    heatmap = gaussian.sum(dim=0)
    
    return heatmap.clamp(0, 1)


def process_batch_gpu(label_paths, image_shape=(640, 640), sigma_factor=0.15, device='cuda', batch_size=32):
    """
    Process multiple labels in batches on GPU for maximum efficiency.
    """
    results = {}
    
    for i in range(0, len(label_paths), batch_size):
        batch_paths = label_paths[i:i+batch_size]
        
        for path in batch_paths:
            with open(path, 'r') as f:
                label = f.readlines()
            
            if not len(label):
                heatmap = np.zeros(image_shape, dtype=np.float32)
                results[path] = heatmap
                continue
            
            classes = [int(lb.strip().split()[0]) for lb in label]
            
            box_xywh = np.array([np.array(lb.strip().split()[1:], dtype=float) for lb in label])
            box_xyxy = np.zeros_like(box_xywh)
            box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2]/2
            box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3]/2
            box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2]/2
            box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3]/2
            box_xyxy = box_xyxy.clip(0, 1)

            boxes, scores, labels_out = weighted_boxes_fusion(
                [box_xyxy.tolist()], 
                [np.ones(box_xyxy.shape[0]).tolist()], 
                [[0] * len(classes)],
                weights=None,
                iou_thr=0.9, skip_box_thr=0.0
            )
            boxes = boxes * image_shape[0]
            
            # Generate heatmap on GPU
            heatmap = bbox_to_gaussian_heatmap_gpu(boxes, image_shape, sigma_factor, device)
            results[path] = heatmap.cpu().numpy()
    
    return results


# Main processing
os.makedirs('datasets/heatmap_v2', exist_ok=True)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

label_paths = glob('datasets/process/labels/*.txt')

if device == 'cuda':
    # GPU batch processing
    print("Processing with GPU acceleration...")
    batch_size = 64  # Adjust based on your GPU memory
    
    for i in tqdm(range(0, len(label_paths), batch_size)):
        batch_paths = label_paths[i:i+batch_size]
        results = process_batch_gpu(batch_paths, batch_size=batch_size, device=device, sigma_factor = 0.1)
        
        # Save results
        for path, heatmap in results.items():
            np.save(f'datasets/heatmap_v2/{os.path.basename(path)[:-4]}.npy', heatmap)
else:
    # Fallback to CPU (still faster than original)
    print("CUDA not available, using CPU...")
    for path in tqdm(label_paths):
        with open(path, 'r') as f:
            label = f.readlines()
        
        image_shape = (640, 640)
        if not len(label):
            heatmap = np.zeros(image_shape)
            np.save(f'datasets/heatmap_v2/{os.path.basename(path)[:-4]}.npy', heatmap)
            continue
        
        classes = [int(lb.strip().split()[0]) for lb in label]
        
        box_xywh = np.array([np.array(lb.strip().split()[1:], dtype=float) for lb in label])
        box_xyxy = np.zeros_like(box_xywh)
        box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2]/2
        box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3]/2
        box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2]/2
        box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3]/2
        
        boxes, scores, labels_out = weighted_boxes_fusion(
            [box_xyxy.tolist()], 
            [np.ones(box_xyxy.shape[0]).tolist()], 
            [[0] * len(classes)],
            weights=None,
            iou_thr=0.5, skip_box_thr=0.0
        )
        boxes = boxes * image_shape[0]
        
        heatmap = bbox_to_gaussian_heatmap_gpu(boxes, image_shape, device='cpu')
        np.save(f'datasets/heatmap/{os.path.basename(path)[:-4]}.npy', heatmap.cpu().numpy())

print("Processing complete!")