from collections import defaultdict
from sklearn.model_selection import KFold
import os
import yaml
import numpy as np
from pathlib import Path

from ultralytics import YOLO

# ===================== CONFIG =====================
DATA_LIST = 'datasets/process/full_0bg.txt'
OUTPUT_DIR = 'datasets/process/k_folds'
N_FOLDS = 5
CONF_THRESHOLD = 0.25  # Ngưỡng confidence để detect
IOU_THRESHOLD = 0.5    # Ngưỡng IoU matching GT-Pred

# Noise scoring weights
WEIGHT_MISSING = 1.0   # GT không có prediction nào match
WEIGHT_LOW_CONF = 0.5  # Prediction có confidence thấp
WEIGHT_LOW_IOU = 0.8   # Prediction có IoU thấp với GT

# ===================== UTILS =====================
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def load_yolo_labels(label_path):
    """Load YOLO format labels: [class, x_center, y_center, width, height]"""
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, x, y, w, h = map(float, parts)
            # Convert to [x1, y1, x2, y2] format
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            labels.append({'class': int(cls), 'box': [x1, y1, x2, y2]})
    
    return labels

def calculate_noise_score(predictions, ground_truths):
    """
    Calculate noise score for a single image
    Higher score = more suspicious (likely noisy)
    
    Returns dict with detailed metrics
    """
    if len(ground_truths) == 0:
        return {
            'total_score': 0.0,
            'missing_gt': 0,
            'low_conf_pred': 0,
            'low_iou_match': 0,
            'extra_pred': 0
        }
    
    score = 0.0
    missing_gt = 0
    low_conf_pred = 0
    low_iou_match = 0
    extra_pred = 0
    
    # Extract predictions
    pred_boxes = []
    if hasattr(predictions, 'boxes') and predictions.boxes is not None:
        for box in predictions.boxes:
            pred_boxes.append({
                'box': box.xyxyn[0].cpu().numpy(),  # normalized coords
                'conf': float(box.conf[0]),
                'class': int(box.cls[0])
            })
    
    # 1. Check missing GT (GT không match với prediction nào)
    for gt in ground_truths:
        matched = False
        best_iou = 0
        
        for pred in pred_boxes:
            if pred['class'] == gt['class']:
                iou = calculate_iou(gt['box'], pred['box'])
                best_iou = max(best_iou, iou)
                if iou >= IOU_THRESHOLD:
                    matched = True
                    break
        
        if not matched:
            missing_gt += 1
            score += WEIGHT_MISSING
            
            # Nếu có IoU thấp nhưng vẫn cùng class → box lệch
            if best_iou > 0.1:
                low_iou_match += 1
                score += WEIGHT_LOW_IOU
    
    # 2. Check low confidence predictions
    for pred in pred_boxes:
        if pred['conf'] < 0.5:  # Confidence threshold
            low_conf_pred += 1
            score += WEIGHT_LOW_CONF * (1 - pred['conf'])
    
    # 3. Check extra predictions (không match GT nào)
    for pred in pred_boxes:
        matched = False
        for gt in ground_truths:
            if pred['class'] == gt['class']:
                iou = calculate_iou(pred['box'], gt['box'])
                if iou >= IOU_THRESHOLD:
                    matched = True
                    break
        
        if not matched and pred['conf'] > 0.5:
            extra_pred += 1
            # Extra predictions ít nghiêm trọng hơn (có thể là object bị miss label)
            score += 0.3
    
    # Normalize by number of GT boxes
    total_score = score / len(ground_truths) if len(ground_truths) > 0 else 0
    
    return {
        'total_score': total_score,
        'missing_gt': missing_gt,
        'low_conf_pred': low_conf_pred,
        'low_iou_match': low_iou_match,
        'extra_pred': extra_pred,
        'num_gt': len(ground_truths),
        'num_pred': len(pred_boxes)
    }

# ===================== MAIN =====================
def main():
    # OPTION 1: Load only TRAIN split (recommended)
    TRAIN_LIST = 'datasets/process/full_0bg.txt'  # Đổi path này
    
    with open(TRAIN_LIST, 'r') as f:
        data_paths = [line.strip() for line in f.readlines()]
    
    print(f"Running K-Fold on TRAIN set only: {len(data_paths)} images")
    print(f"This is safer as val set is usually cleaner\n")
    
    # Prepare K-Fold
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    noise_scores = defaultdict(list)
    
    template_yaml = {
        "train": "",
        "val": "",
        "nc": 14,
        "names": [
            "Phinh dong mach chu", "Xep phoi", "Voi hoa", "Tim to", "Dong dac",
            "Benh phoi ke (ILD)", "Tham nhiem", "Mo phoi", "Not/Khoi", 
            "Ton thuong khac", "Tran dich mang phoi", "Day mang phoi", 
            "Tran khi mang phoi", "Xo hoa phoi"
        ]
    }
    
    # K-Fold training
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_paths)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")
        
        fold_dir = Path(OUTPUT_DIR) / str(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train/val splits
        with open(fold_dir / 'train.txt', 'w') as f:
            for idx in train_idx:
                f.write(f'{data_paths[idx]}\n')
        
        with open(fold_dir / 'val.txt', 'w') as f:
            for idx in val_idx:
                f.write(f'{data_paths[idx]}\n')
        
        # Create dataset.yaml
        template_yaml["train"] = str('./train.txt')
        template_yaml["val"] = str('./val.txt')
        
        with open(fold_dir / 'dataset.yaml', "w") as f:
            yaml.dump(template_yaml, f, sort_keys=False, default_flow_style=False)
        
        # Train model
        print(f"Training fold {fold}...")
        # model = YOLO('yolo11n.yaml')
        # model.train(
        #     data=f'/media/ssd220/ty/xray/datasets/process/k_folds/{fold}/dataset.yaml',
        #     epochs=20,  # Số epoch thấp để tránh overfit vào noise
        #     imgsz=640,
        #     batch=16,
        #     cos_lr=True,
        #     patience=5,
            
        #     box=7.5, cls=0.5, dfl=1.5,
            
        #     device='cuda:0',
        #     name=f'k_folds/{fold}',
        #     plots=True,
            
        #     # Augmentation vừa phải
        #     mosaic=0.5,
        #     fliplr=0.5,
        #     scale=0.1,
        #     degrees=5,
        #     translate=0.1,
        #     flipud=0.5
        # )
        
        # Load best model
        best_model = YOLO(f'runs/detect/k_folds/{fold}/weights/best.pt')
        
        # Evaluate on validation set (data model chưa thấy)
        print(f"Evaluating fold {fold} on validation set...")
        for idx in val_idx:
            img_path = data_paths[idx]
            label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
            
            # Load ground truth
            ground_truths = load_yolo_labels(label_path)
            
            # Predict
            results = best_model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
            
            # Calculate noise score
            score_dict = calculate_noise_score(results[0], ground_truths)
            noise_scores[idx].append(score_dict)
    
    # ===================== ANALYSIS =====================
    print(f"\n{'='*50}")
    print("ANALYZING NOISE SCORES")
    print(f"{'='*50}")
    
    # Aggregate scores across folds
    aggregated_scores = {}
    for idx, score_list in noise_scores.items():
        avg_score = np.mean([s['total_score'] for s in score_list])
        std_score = np.std([s['total_score'] for s in score_list])
        
        aggregated_scores[idx] = {
            'mean_score': avg_score,
            'std_score': std_score,
            'img_path': data_paths[idx],
            'details': score_list
        }
    
    # Sort by mean score (descending)
    sorted_scores = sorted(aggregated_scores.items(), 
                          key=lambda x: x[1]['mean_score'], 
                          reverse=True)
    
    # Define thresholds for suspicious samples
    # Với 5000 ảnh, expect ~5-15% có vấn đề
    # Top 10% = 500 ảnh để review
    scores_list = [data['mean_score'] for _, data in sorted_scores]
    percentile_90 = np.percentile(scores_list, 90)  # Top 10%
    percentile_95 = np.percentile(scores_list, 95)  # Top 5%
    
    print(f"\nScore distribution:")
    print(f"  Mean: {np.mean(scores_list):.3f}")
    print(f"  Median: {np.median(scores_list):.3f}")
    print(f"  90th percentile: {percentile_90:.3f}")
    print(f"  95th percentile: {percentile_95:.3f}")
    
    # Use 90th percentile as threshold (top 10%)
    SUSPICIOUS_THRESHOLD = percentile_90
    
    suspicious_samples = []
    for idx, data in sorted_scores:
        if data['mean_score'] >= SUSPICIOUS_THRESHOLD:
            suspicious_samples.append({
                'idx': idx,
                'path': data['img_path'],
                'mean_score': data['mean_score'],
                'std_score': data['std_score']
            })
    
    # Save results
    output_file = Path(OUTPUT_DIR) / 'suspicious_samples.txt'
    with open(output_file, 'w') as f:
        f.write(f"# Suspicious samples (mean_score >= {SUSPICIOUS_THRESHOLD})\n")
        f.write(f"# Total: {len(suspicious_samples)} / {len(data_paths)}\n")
        f.write(f"# Format: idx | mean_score | std_score | path\n\n")
        
        for sample in suspicious_samples:
            f.write(f"{sample['idx']}\t{sample['mean_score']:.3f}\t"
                   f"{sample['std_score']:.3f}\t{sample['path']}\n")
    
    print(f"\n✓ Found {len(suspicious_samples)} suspicious samples "
          f"({len(suspicious_samples)/len(data_paths)*100:.1f}%)")
    print(f"✓ Results saved to: {output_file}")
    
    # Print top 20 most suspicious
    print(f"\nTop 20 most suspicious samples:")
    print(f"{'Idx':<8} {'Score':<8} {'Std':<8} Path")
    print("-" * 80)
    for sample in suspicious_samples[:20]:
        print(f"{sample['idx']:<8} {sample['mean_score']:<8.3f} "
              f"{sample['std_score']:<8.3f} {sample['path']}")

if __name__ == '__main__':
    main()