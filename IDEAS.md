1. Chưa thể dùng CT tại cái box vì có quá nhiều box chồng lấn => nhiều class khác nhau cùng 1 vị trí => CT đẩy đặc trưng xa ra là ko hợp lý 
2. Do model đang miss-detect khá nhiều => Thêm contrastive loss 2 classes ở các bounding box thay vì toàn bộ | ý tưởng khác: Nếu có nhiều box cùng 1 vị trí, thay nhãn = abnormaly, nếu ko thì vẫn giữ nhãn cũ

(1), (2) => Sử dụng các box ở các hình khác nhau để so sánh thay vì chung 1 hình 


3. Thử để sau global CT sau lớp C2PSA

4. Upsample thêm 1 tầng nữa (/4, /8, /16, /32) => Done



5. Plot

6. Mất cân bằng dữ liệu trong 1 batch => CT khó học hơn => Sampler cho Dataloader => Xong sampler => upsampling cho abnormally 

7. Bỏ bớt background => mAP tăng => false detect khá nhiều


8. miss-detect nhiều + cosine score pos-neg cao => chưa phân biệt đc đâu là vùng background đâu là vùng có bệnh => tạo ROI => ko augmentation được do box bị thay đôi vị trí
=> Thay vì phải tạo box mỗi lần chạy => Thêm nhãn box ko chứa bệnh ra file xong load chung với dataset
=> Tạo nhãn box ko bệnh trước => vị trí box nofindings chưa hợp lý

=> Thử segment phổi => Từ vị trí vùng có bệnh chiếu qua bệnh nhân khác ko có bệnh để lấy box => Tạm dừng
=> Tạo box ở ảnh bệnh khác, ko quan tâm vùng tạo ra có bệnh ko 
=> Ở bước mosaic, có thông tin của 4 ảnh => map vùng có bệnh sang các hình khác. Điều kiện là các vùng sample ko phải cùng loại bệnh
==== Mosaic: Trước khi mosaic sẽ sample box từ các ảnh trong ảnh mosaic

9. resize về 640 ko ngon => tạo thêm loader cho background

################################################################3


Thêm conf-aware vào dataloader


################### Vấn đề ###########################3
1. Augmentation: Có nên fliplr, scale ko?


