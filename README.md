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



####### => mAP 0.4x => triplet loss => train/val 3k bg => re-train 1k bg

            for idx, s in enumerate([4, 6, 10]):

                cls_ct_losses.append(ct_classify_loss_func(ct_classify[idx].squeeze(), labels))

                f = saved_feats[s]  # [B, C, H, W]

                # 1. Region pooling
                pool_f = F.adaptive_avg_pool2d(f, (20, 20))  # [B, C, 20, 20]

                # 2. Flatten
                pool_f = pool_f.flatten(1)  # [B, C*20*20]

                # 3. Normalize
                norm_f = F.normalize(pool_f, dim=1)

                # 4. Cosine similarity matrix
                cosine_sim = norm_f @ norm_f.T  # [B, B]

                triplet_losses = []

                # 5. Build triplets
                for i in range(cosine_sim.size(0)):
                    pos_idx = pos_mask[i]          # positives for anchor i
                    neg_idx = neg_mask[i]          # negatives for anchor i

                    if pos_idx.sum() == 0 or neg_idx.sum() == 0:
                        continue

                    pos_sim = cosine_sim[i][pos_idx]  # [P]
                    neg_sim = cosine_sim[i][neg_idx]  # [N]

                    # 6. Hard mining (recommended)
                    hardest_pos = pos_sim.min()       # lowest similarity
                    hardest_neg = neg_sim.max()       # highest similarity

                    ctloss = torch.clamp(
                        hardest_neg - hardest_pos + margin,
                        min=0.0
                    )
                    triplet_losses.append(ctloss)

                if len(triplet_losses) > 0:
                    ct_losses.append(torch.stack(triplet_losses).mean())
                else:
                    ct_losses.append(torch.zeros([], device=f.device))


=> Leak data background trong tập 3k và 1k => xóa bớt bg trong tập 3k rồi test thử
