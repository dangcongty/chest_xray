import os
import shutil
from pathlib import Path


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in YOLO format [x_center, y_center, width, height]
    All values are normalized (0-1)
    """
    # Convert to corner coordinates
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou


def check_and_remap_overlaps(label_dir, output_dir=None, iou_threshold=0.5, unknown_class_id=999, backup=True):
    """
    Check for highly overlapping objects and remap them to 'unknown' class
    
    Args:
        label_dir: Directory containing YOLO format label files (.txt)
        output_dir: Directory to save modified labels (if None, overwrites original)
        iou_threshold: IoU threshold to consider boxes as overlapping (default: 0.5)
        unknown_class_id: Class ID to assign to overlapping objects (default: 999)
        backup: Whether to backup original files (default: True)
    
    Returns:
        Dictionary with statistics
    """
    label_dir = Path(label_dir)
    
    if output_dir is None:
        output_dir = label_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup if needed
    if backup and output_dir == label_dir:
        backup_dir = label_dir.parent / f"{label_dir.name}_backup"
        if not backup_dir.exists():
            shutil.copytree(label_dir, backup_dir)
            print(f"✓ Backup created at: {backup_dir}")
    
    stats = {
        'total_files': 0,
        'files_with_overlaps': 0,
        'total_boxes': 0,
        'remapped_boxes': 0,
        'overlap_pairs': []
    }
    
    # Process each label file
    for label_file in label_dir.glob('*.txt'):
        stats['total_files'] += 1
        
        # Read annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
        
        # Parse boxes
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append([class_id, x, y, w, h])
        
        stats['total_boxes'] += len(boxes)
        
        # Check for overlaps
        remapped_indices = set()
        file_has_overlap = False
        
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                # Calculate IoU
                iou = calculate_iou(boxes[i][1:5], boxes[j][1:5])
                
                if iou >= iou_threshold:
                    file_has_overlap = True
                    remapped_indices.add(i)
                    remapped_indices.add(j)
                    
                    stats['overlap_pairs'].append({
                        'file': label_file.name,
                        'box1_class': boxes[i][0],
                        'box2_class': boxes[j][0],
                        'iou': round(iou, 3)
                    })
        
        # Remap overlapping boxes to unknown class
        if remapped_indices:
            stats['files_with_overlaps'] += 1
            stats['remapped_boxes'] += len(remapped_indices)
            
            for idx in remapped_indices:
                boxes[idx][0] = unknown_class_id
        
        # Write output
        output_file = output_dir / label_file.name
        with open(output_file, 'w') as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")
    
    return stats


def print_report(stats):
    """Print a detailed report of the overlap detection"""
    print("\n" + "="*60)
    print("OVERLAP DETECTION REPORT")
    print("="*60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files with overlaps: {stats['files_with_overlaps']}")
    print(f"Total boxes: {stats['total_boxes']}")
    print(f"Remapped boxes: {stats['remapped_boxes']}")
    print(f"Total overlap pairs found: {len(stats['overlap_pairs'])}")
    
    if stats['overlap_pairs']:
        print("\n" + "-"*60)
        print("OVERLAP DETAILS (first 20):")
        print("-"*60)
        for i, pair in enumerate(stats['overlap_pairs'][:20]):
            print(f"{i+1}. {pair['file']}: Class {pair['box1_class']} ↔ Class {pair['box2_class']} (IoU: {pair['iou']})")
        
        if len(stats['overlap_pairs']) > 20:
            print(f"... and {len(stats['overlap_pairs']) - 20} more overlaps")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Configuration
    LABEL_DIR = "datasets/labels"  # Change this
    OUTPUT_DIR = "datasets/labels_unknown"  # Optional: None to overwrite
    IOU_THRESHOLD = 0.9  # Adjust based on your needs (0.3-0.7 typical)
    UNKNOWN_CLASS_ID = 15  # Choose an unused class ID
    
    # Run the checker
    stats = check_and_remap_overlaps(
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        iou_threshold=IOU_THRESHOLD,
        unknown_class_id=UNKNOWN_CLASS_ID,
        backup=True
    )
    
    # Print report
    print_report(stats)
    
    # Optionally save detailed report to file
    if stats['overlap_pairs']:
        with open('overlap_report.txt', 'w') as f:
            f.write("Detailed Overlap Report\n")
            f.write("="*60 + "\n")
            for pair in stats['overlap_pairs']:
                f.write(f"File: {pair['file']}, Classes: {pair['box1_class']} & {pair['box2_class']}, IoU: {pair['iou']}\n")
        print("✓ Detailed report saved to: overlap_report.txt")