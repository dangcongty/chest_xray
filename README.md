## Ý tưởng chính
* Enhance vùng đặc trưng sử dụng heatmap và contrastive learning 

---

## Các file sửa
* Loader 50% là ảnh có background và 50% ko có background => hàm BalancedContiguousDistributedSampler ở dòng 119 trong file ultralytics/data/build.py
* Thêm head Heatmap => hàm Heatmap kế thừa Detect ở dòng 1238 file ultralytics/nn/modules/head.py
* Thêm layer ở Detect head => hàm Detect dòng 46 file ultralytics/nn/modules/head.py
* Thêm hàm HeatmapLoss ở dòng 960 file ultralytics/utils/loss.py

---

## Các nhiệm vụ cần làm
### ***Tiền xử lý dữ liệu***
* Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) toàn cục + cục bộ trên lung ROI (lung segmentation trước bằng U-Net nhẹ), giúp nổi bật tổn thương mờ như nốt nhỏ hoặc xơ hóa. Kết hợp Gaussian/Median Blur augmentation (sigma 0.5-1.5) để tăng robustness với noise X-quang.
* Auto-crop lung region: Sử dụng pre-trained lung segmentation (từ TorchXRayVision hoặc VinDr-CXR model) để crop chỉ vùng phổi, giảm background noise và tăng focus vào abnormality.
### ***Cải thiện Loss và Heatmap***
* Thay heatmap loss bằng Dice + Focal MSE: heatmap_loss = α * MSE(gauss) + β * Focal(heatmap) + γ * Dice(thresholded_heatmap, GT), với α=0.3, β=0.5, γ=0.2. Dice giúp handle overlap tốt hơn MSE thuần, focal vẫn giữ cho hard examples.
* Multi-scale heatmap supervision: Generate heatmap ở 3 scales (full, 0.5x, 2x crop) và supervise riêng, kết hợp pyramid pooling trong YOLO neck để capture small/large lesions.
* Giảm sigma xuống 0.15-0.18 nếu GT bbox tight; test Gaussian mixture cho multi-peak heatmap nếu abnormality diffuse (như edema).
### ***Data & Augmentation nâng cao***
* Class-balanced sampler + hard negative mining: Oversample rare classes (e.g., nodule <5%); mine negatives từ val set với high false positive.
* Mixup/CutMix variant cho medical: Mixup giữa normal/abnormal pairs với α=0.2, chỉ apply trên lung ROI để tránh artifact.
* Sử dụng VinDr-CXR dataset (18 classes, Việt Nam data) mix với CheXpert/NIH (label noise handling via uncertainty weighting).
### ***Model Architecture & Training***
| Hướng cải thiện              | Chi tiết implement                                                                                                                    | Dự kiến gain         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| YOLO variant lớn hơn         | Switch sang YOLOv11x hoặc RT-DETR-L (nhẹ hơn nhưng accurate hơn YOLOv10/11 trên medical); freeze backbone đầu train 10 epochs.        | +0.1-0.15 mAP        |
| Ensemble prediction          | Train 3 models (YOLOv11n + m + l), average bbox scores + NMS; hoặc Knowledge Distillation từ teacher DenseNet121 (pretrain CheXpert). | +0.08-0.12 mAP       |
| Test-time augmentation (TTA) | Flip, rotate ±10°, CLAHE variants; average heatmaps trước predict.                                                                    | +0.05 mAP nhanh      |
| Pseudo-labeling              | Sau 50 epochs, generate pseudo GT trên unlabeled data (threshold confidence 0.7), retrain 20 epochs.                                  | +0.1 mAP nếu data ít |

---

## Dữ liệu:
https://drive.google.com/file/d/1tTpiwLWyGwG_uRJeMfhcYTUvUJfqOSQH/view?usp=sharing

---

# UPDATE NOTES

## [Update 08/11/2025]
* Add Heatmap branch
* Add option 'heatmap' to dataset/dataloader/augmentation
* Add loss MSE for heatmap 
* Add Heatmap validation

## [Update 09/11/2025]
* Add visualize heatmap
* Handle inf heatmap loss/mse

## [Update 10/11/2025]
* Transfer YOLO weight to Heatmap
* Adjusting loss Heatmap to 'mean'
* Fix metrics Heatmap 
* Fix prediction Heatmap 
* Add HeatmapAttention

## [Update 12/11/2025]
* Change Attention layer to Transformer/ViT
* Add AdaptiveWing Loss


## [Update 13/11/2025]
* Add more layer to heatmap

## [Update 14/11/2025]
* Add sigmoid Heatmap
* Add focal loss
* Heatmap loss = mse + focal
* Sigma heatmap change from 0.3 -> 0.2
* Update valiation MSE

