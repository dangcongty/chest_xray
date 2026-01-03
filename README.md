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

## Ý tưởng chính
* Enhance vùng đặc trưng sử dụng heatmap và contrastive learning 

## Các file sửa
* Loader 50% là ảnh có background và 50% ko có background => hàm BalancedContiguousDistributedSampler ở dòng 119 trong file ultralytics/data/build.py
* Thêm head Heatmap => hàm Heatmap kế thừa Detect ở dòng 1238 file ultralytics/nn/modules/head.py
* Thêm layer ở Detect head => hàm Detect dòng 46 file ultralytics/nn/modules/head.py
* Thêm hàm HeatmapLoss ở dòng 960 file ultralytics/utils/loss.py


## Các nhiệm vụ cần làm 
* train lại baseline với data của a Nhân
