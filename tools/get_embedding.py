import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

# from yolo_encoder import parse_model
# from ultralytics.nn.tasks import yaml_model_load

# =========================
# Custom Transform: Padding + Resize
# =========================
class PadToSquare:
    """Pad image to square while maintaining aspect ratio"""
    def __init__(self, fill=0):
        self.fill = fill
    
    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        
        max_side = max(w, h)
        pad_w = (max_side - w) // 2
        pad_h = (max_side - h) // 2
        
        padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
        return ImageOps.expand(img, padding, fill=self.fill)


# =========================
# Dataset
# =========================
class TripletImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_single=False, class_to_idx=None):
        """
        Args:
            root_dir: Path to dataset
            transform: Transform to apply
            return_single: If True, return single image + label (for validation)
            class_to_idx: Optional fixed class mapping (for consistency)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_single = return_single
        
        # Lấy danh sách classes và sort (numeric nếu có thể)
        classes_raw = os.listdir(root_dir)
        # Try to sort numerically if possible
        try:
            self.classes = sorted(classes_raw, key=lambda x: int(x))
        except ValueError:
            # Fallback to alphabetical if not all numeric
            self.classes = sorted(classes_raw)
        
        # Use provided mapping or create new one
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            print(f"📋 Using provided class mapping")
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            print(f"📋 Created new class mapping")
        
        # Print mapping for verification
        print("🗂️  Class mapping:")
        for cls, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            print(f"   '{cls}' → label {idx}")
        
        # Thu thập tất cả ảnh
        self.class_to_images = {}
        self.all_images = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            images = [
                os.path.join(cls_dir, img)
                for img in os.listdir(cls_dir)
                if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))
            ]
            
            if len(images) > 0:
                self.class_to_images[cls] = images
                label = self.class_to_idx[cls]
                self.all_images.extend([(cls, label, path) for path in images])
        
        if len(self.all_images) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"📁 Dataset loaded: {len(self.classes)} classes, {len(self.all_images)} images")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_cls, anchor_label, anchor_path = self.all_images[idx]
        
        if self.return_single:
            img = self._load_img(anchor_path)
            return img, anchor_label
        
        # Positive: cùng class với anchor (nhưng khác ảnh)
        positive_candidates = [p for p in self.class_to_images[anchor_cls] if p != anchor_path]
        if len(positive_candidates) == 0:
            positive_path = anchor_path
        else:
            positive_path = random.choice(positive_candidates)
        
        # Negative: khác class
        negative_classes = [c for c in self.classes if c != anchor_cls]
        if len(negative_classes) == 0:
            raise ValueError("Need at least 2 classes for triplet loss")
        negative_cls = random.choice(negative_classes)
        negative_path = random.choice(self.class_to_images[negative_cls])

        anchor = self._load_img(anchor_path)
        positive = self._load_img(positive_path)
        negative = self._load_img(negative_path)
        
        return anchor, positive, negative, anchor_label
    
    def _load_img(self, path):
        """Helper to load and transform image"""
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img) if self.transform else img
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            dummy = Image.new("RGB", (224, 224), (0, 0, 0))
            return self.transform(dummy) if self.transform else dummy


# =========================
# Model
# =========================
class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # cfg = 'ultralytics/cfg/models/11/yolo11-p2_enc.yaml'
        # yaml = yaml_model_load(cfg)  # cfg dict
        # ch = 3
        # yaml["channels"] = ch
        # self.backbone, _ = parse_model(yaml, ch=ch, verbose=False)  # mode

        embedding_dim = 512
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        # x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# =========================
# Loss
# =========================
def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    return torch.clamp(pos_dist - neg_dist + margin, min=0).mean()


# =========================
# Training
# =========================
def train_embedding(root_dir="datasets/classify", epochs=50, batch_size=32, lr=1e-4,
                    embedding_dim=512, val_split=0.2, ckpt_path="ckpt/best_embedding.pth"):
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Transforms
    train_tf = transforms.Compose([
        PadToSquare(fill=(0, 0, 0)),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        PadToSquare(fill=(0, 0, 0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset (creates mapping)
    print("\n" + "="*60)
    print("TRAINING DATASET")
    print("="*60)
    full_dataset = TripletImageDataset(root_dir, transform=train_tf)
    
    # Save class mapping for future use
    class_mapping = {
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': {idx: cls for cls, idx in full_dataset.class_to_idx.items()},
        'classes': full_dataset.classes
    }
    mapping_path = "ckpt/class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"\n💾 Saved class mapping to {mapping_path}")

    # Split train/val by indices
    indices = np.arange(len(full_dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(len(indices) * val_split)
    train_idx, val_idx = indices[split:], indices[:split]

    np.savez("ckpt/split_indices.npz", train_idx=train_idx, val_idx=val_idx)
    print(f"💾 Saved split indices: {len(train_idx)} train, {len(val_idx)} val")

    # Weighted sampler for class balance
    train_labels = [full_dataset.all_images[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create subsets with SAME mapping
    train_set = Subset(full_dataset, train_idx)
    val_dataset = TripletImageDataset(root_dir, transform=val_tf, 
                                      class_to_idx=full_dataset.class_to_idx)
    val_set = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)

    # Model & optimizer
    model = EmbeddingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=3, verbose=True)

    best_val = float('inf')
    patience, patience_cnt = 8, 0

    for ep in range(1, epochs + 1):
        # Training
        model.train()
        tr_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            a, p, n, _ = batch
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            loss = triplet_loss(model(a), model(p), model(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        
        tr_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                a, p, n, _ = batch
                a, p, n = a.to(device), p.to(device), n.to(device)
                loss = triplet_loss(model(a), model(p), model(n))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {ep:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'embedding_dim': embedding_dim,
                'class_to_idx': full_dataset.class_to_idx,
            }, ckpt_path)
            print(f"✅ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"⏹ Early stopping triggered after {ep} epochs")
                break

    print(f"\n🎯 Training complete! Best val_loss: {best_val:.4f}")
    return model


# =========================
# Helper: Load model for inference
# =========================
def load_model(model_path="ckpt/best_embedding.pth", embedding_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', embedding_dim)
    else:
        state_dict = checkpoint
    
    model = EmbeddingModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ Model loaded from {model_path} (embedding_dim={embedding_dim})")
    return model, device


# =========================
# Load class mapping
# =========================
def load_class_mapping(mapping_path="ckpt/class_mapping.json"):
    """Load saved class mapping"""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Class mapping not found at {mapping_path}. "
            "Please train the model first to generate the mapping."
        )
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Convert string keys back to int for idx_to_class
    mapping['idx_to_class'] = {int(k): v for k, v in mapping['idx_to_class'].items()}
    
    print("🗂️  Loaded class mapping:")
    for idx, cls in sorted(mapping['idx_to_class'].items()):
        print(f"   Label {idx} → '{cls}'")
    
    return mapping


# =========================
# Extract embeddings for a dataset
# =========================
def extract_embeddings(model, data_dir, device, class_to_idx, batch_size=64):
    """Extract embeddings and labels from a dataset with consistent mapping"""
    tf = transforms.Compose([
        PadToSquare(fill=(0, 0, 0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("\n" + "="*60)
    print("INFERENCE DATASET")
    print("="*60)
    # Use TripletImageDataset with fixed mapping
    dataset = TripletImageDataset(data_dir, transform=tf, return_single=True,
                                   class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
            emb = model(imgs.to(device))
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings).numpy()
    labels = torch.cat(all_labels).numpy()
    paths = np.array([img[2] for img in dataset.all_images])
    
    return embeddings, labels, paths, dataset.classes


# =========================
# t-SNE visualization
# =========================
def visualize_tsne(model_path="ckpt/best_embedding.pth", data_dir="datasets/classify",
                   embedding_dim=512, index_path="ckpt/split_indices.npz"):
    
    # Load class mapping
    class_mapping = load_class_mapping()
    
    # Load model
    model, device = load_model(model_path, embedding_dim)
    
    # Extract embeddings
    print("\n📊 Extracting embeddings for visualization...")
    embeddings, labels, paths, class_names = extract_embeddings(
        model, data_dir, device, class_mapping['class_to_idx']
    )
    
    # Load train/val split if exists
    splits = {}
    if os.path.exists(index_path):
        split_indices = np.load(index_path)
        train_idx = split_indices["train_idx"]
        val_idx = split_indices["val_idx"]
        
        splits['train'] = (embeddings[train_idx], labels[train_idx])
        splits['val'] = (embeddings[val_idx], labels[val_idx])
        splits['all'] = (embeddings, labels)
    else:
        print(f"⚠️  Split indices not found at {index_path}. Only plotting all data.")
        splits['all'] = (embeddings, labels)
    
    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    
    os.makedirs("tsne_plots", exist_ok=True)
    
    # Generate t-SNE plots
    splits.pop('train')
    splits.pop('val')
    for split_name, (X, y) in splits.items():
        print(f"\n🔄 Running t-SNE for {split_name} set ({len(X)} samples)...")
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1), 
                   random_state=42)
        X_2d = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 9))
        unique_labels = np.unique(y)
        
        for i, label in enumerate(unique_labels):
            idx = y == label
            color = colors[i % len(colors)]
            class_name = class_mapping['idx_to_class'][int(label)]
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                       s=20, alpha=0.7, color=color, 
                       label=f"{class_name} (label {label})", edgecolors='none')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, 
                  frameon=True, fancybox=True, shadow=True)
        plt.title(f"t-SNE Visualization ({split_name} set, {len(X)} samples)", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        output_path = f"tsne_plots/tsne_{split_name}.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {output_path}")
    
    print("\n🎯 All t-SNE plots saved in tsne_plots/")


# =========================
# Outlier detection & re-clustering
# =========================
def compute_centroids(embeddings, labels):
    """Compute centroid for each class"""
    centroids = {}
    for label in np.unique(labels):
        cluster_vectors = embeddings[labels == label]
        centroids[label] = cluster_vectors.mean(axis=0)
    return centroids

def find_outliers(embeddings, labels, centroids, threshold_std=2.0):
    """Find outliers based on distance to cluster centroid,
    only if they are closer to another cluster centroid."""
    outliers = []
    outlier_info = []
    
    for label in np.unique(labels):
        cluster_vectors = embeddings[labels == label]
        centroid = centroids[label]
        
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + threshold_std * std_dist
        
        cluster_outlier_indices = np.where(distances > threshold)[0]
        true_indices = np.where(labels == label)[0][cluster_outlier_indices]
        
        for idx, dist in zip(true_indices, distances[cluster_outlier_indices]):
            emb = embeddings[idx]
            
            # Distance to all other centroids
            other_dists = {
                lbl: np.linalg.norm(emb - c)
                for lbl, c in centroids.items() if lbl != label
            }
            
            # Find nearest other cluster
            nearest_lbl, nearest_dist = min(other_dists.items(), key=lambda x: x[1])
            
            # Only count as outlier if it's actually closer to another cluster
            if nearest_dist < dist:
                outliers.append(idx)
                outlier_info.append({
                    'index': idx,
                    'label': label,
                    'distance': dist,
                    'nearest_label': nearest_lbl,
                    'nearest_distance': nearest_dist,
                    'mean_dist': mean_dist,
                    'std_dist': std_dist,
                    'closer_to_other': True
                })
    
    return np.array(outliers), outlier_info



def re_cluster(embedding_dim=512, data_dir='datasets/classify', 
               model_path='ckpt/best_embedding.pth', threshold_std=2.0, top_k=3):
    """Find outliers and suggest alternative clusters"""
    
    # Load class mapping
    class_mapping = load_class_mapping()
    idx_to_class = class_mapping['idx_to_class']
    
    # Load model
    model, device = load_model(model_path, embedding_dim)
    
    # Extract embeddings
    print("\n📊 Extracting embeddings...")
    embeddings, labels, paths, class_names = extract_embeddings(
        model, data_dir, device, class_mapping['class_to_idx']
    )
    
    # Compute centroids
    print("\n🎯 Computing cluster centroids...")
    centroids = compute_centroids(embeddings, labels)
    
    # Find outliers
    print(f"\n🔍 Finding outliers (threshold={threshold_std} std)...")
    outlier_indices, outlier_info = find_outliers(embeddings, labels, centroids, threshold_std)
    
    print(f"\n🧩 Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(embeddings)*100:.2f}% of dataset)")
    
    if len(outlier_indices) == 0:
        print("✅ No outliers found! Dataset looks clean.")
        return
        print(f"\n📋 Analyzing outliers and suggesting top-{top_k} alternative clusters:\n")
    print("=" * 120)
    
    for info in outlier_info:
        idx = info['index']
        emb = embeddings[idx]
        original_label = int(info['label'])
        original_class = idx_to_class[original_label]
        
        nearest_label = int(info['nearest_label'])
        nearest_class = idx_to_class[nearest_label]
        
        print(f"\n📷 Image: {paths[idx]}")
        print(f"   Current folder: '{original_class}' (label {original_label})")
        print(f"   Distance to current centroid: {info['distance']:.4f}")
        print(f"   Distance to nearest centroid '{nearest_class}' (label {nearest_label}): {info['nearest_distance']:.4f}")
        print(f"   Cluster stats: mean={info['mean_dist']:.4f}, std={info['std_dist']:.4f}")
        print(f"   ⚠️ This image is closer to '{nearest_class}' than its own cluster centroid!")
        
        # Compute distance to all centroids again for top-k suggestions
        distances = {lbl: np.linalg.norm(emb - c) for lbl, c in centroids.items()}
        sorted_clusters = sorted(distances.items(), key=lambda x: x[1])
        nearest_clusters = [c for c in sorted_clusters if c[0] != original_label][:top_k]
        
        print(f"   Top-{top_k} alternative clusters:")
        for rank, (lbl, dist) in enumerate(nearest_clusters, start=1):
            lbl_int = int(lbl)
            class_name = idx_to_class[lbl_int]
            improvement = (info['distance'] - dist) / info['distance'] * 100
            print(f"      {rank}. '{class_name}' (label {lbl_int}): distance={dist:.4f} (↓ {improvement:.1f}% closer)")
        
        print("-" * 120)



# =========================
# Main
# =========================
if __name__ == '__main__':
    # Uncomment the function you want to run:
    
    # 1. Train the model
    train_embedding(
        root_dir="datasets/classify/trainset",
        epochs=50,
        batch_size=16,
        lr=1e-4,
        embedding_dim=512,
        val_split=0.2
    )
    
    # 2. Visualize embeddings with t-SNE
    visualize_tsne(
        model_path="ckpt/best_embedding.pth",
        data_dir="datasets/classify/trainset",
        embedding_dim=512
    )
    
    # 3. Find outliers and suggest re-clustering
    re_cluster(
        embedding_dim=512,
        data_dir='datasets/classify/trainset',
        model_path='ckpt/best_embedding.pth',
        threshold_std=2.0,
        top_k=3
    )