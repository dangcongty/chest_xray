1. Chưa thể dùng CT tại cái box vì có quá nhiều box chồng lấn => nhiều class khác nhau cùng 1 vị trí => CT đẩy đặc trưng xa ra là ko hợp lý 
2. Do model đang miss-detect khá nhiều => Thêm contrastive loss 2 classes ở các bounding box thay vì toàn bộ | ý tưởng khác: Nếu có nhiều box cùng 1 vị trí, thay nhãn = abnormaly, nếu ko thì vẫn giữ nhãn cũ












################################################################3


Thêm conf=aware vào dataloader
