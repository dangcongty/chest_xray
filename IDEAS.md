1. Chưa thể dùng CT tại cái box vì có quá nhiều box chồng lấn => nhiều class khác nhau cùng 1 vị trí => CT đẩy đặc trưng xa ra là ko hợp lý 
2. Do model đang miss-detect khá nhiều => Thêm contrastive loss 2 classes ở các bounding box thay vì toàn bộ | ý tưởng khác: Nếu có nhiều box cùng 1 vị trí, thay nhãn = abnormaly, nếu ko thì vẫn giữ nhãn cũ

(1), (2) => Sử dụng các box ở các hình khác nhau để so sánh thay vì chung 1 hình 


3. Thử để sau global CT sau lớp C2PSA

4. Upsample thêm 1 tầng nữa (/4, /8, /16, /32) => Done



5. Plot

6. Mất cân bằng dữ liệu trong 1 batch => CT khó học hơn => Sampler cho Dataloader => Xong sampler => upsampling cho abnormally 



################################################################3


Thêm conf-aware vào dataloader


################### Vấn đề ###########################3
1. Augmentation: Có nên fliplr, scale ko?