# Nhật ký học Deep Learning

Bản tổng hợp này được viết từ nội dung code hiện có. Các mốc bên dưới mô tả kỹ thuật xuất hiện trong từng bài, không thay thế ghi chép thực nghiệm hoặc khẳng định kết quả huấn luyện.

| Mốc | Điều đã triển khai | Câu hỏi để tự kiểm chứng |
| --- | --- | --- |
| 01 · Linear regression | Dataset tổng hợp với `y = 2.435x + 1`, mô hình tuyến tính, MSE và SGD | Trọng số và bias học được gần giá trị tạo dữ liệu đến đâu? |
| 02 · MNIST CNN | Hai lớp convolution, pooling, classifier và đánh giá sau mỗi epoch | Train loss giảm có đi cùng cải thiện test accuracy không? |
| 03 · QuickDraw | Template trung bình cho 10 lớp; so sánh dot product với cosine trên 100 ảnh test | Độ lớn vector ảnh ảnh hưởng thứ hạng dot product như thế nào? |
| 04 · CIFAR-10 pickle | Đọc batch thô, nối ảnh/nhãn và reshape để hiển thị | Phân biệt layout CHW và HWC bằng cách nào? |
| 05 · Animal Dataset | Dataset ảnh RGB, nhãn theo thư mục và transforms | Ảnh và nhãn có giữ đúng thứ tự khi DataLoader shuffle không? |
| 06 · Torchvision Dataset | CIFAR-10 có sẵn, tensor và DataLoader nhiều worker | API có sẵn thay thế những phần nào của bài 4? |
| 07 · MLP | Flatten ảnh RGB 32×32, các lớp 512 → 256 → 10 | Vì sao đầu ra là 10 logits, và flatten làm mất thông tin không gian gì? |
| 08 · Train & evaluate | Import mô hình bài 7, Adam, CrossEntropyLoss, classification report | Lớp nào có recall thấp và có thể bị nhầm với lớp nào? |
| 09 · ResNet draft | Bottleneck 1×1 → 3×3 → 1×1, residual và downsample | Hai nhánh cần cùng shape ra sao trước khi cộng? Bản này còn cần hoàn thiện trước khi chạy. |
| 10 · CNN workflow | Năm conv blocks, augmentation, confusion matrix, TensorBoard và resume checkpoint | So sánh best/last checkpoint thế nào? Cần tách validation trước khi chọn mô hình bằng test. |
| 11 · CNN inference | Nạp trọng số bài 10, đọc ảnh OpenCV, chuyển RGB/CHW, softmax và hiển thị dự đoán | Tiền xử lý có khớp lúc train không? Xác suất softmax có phản ánh độ tin cậy không? |

## Trạng thái bằng chứng

- Có 11 file bài thực hành để đọc và đối chiếu trực tiếp.
- Bài 9 là bản phác thảo chưa hoàn chỉnh.
- Chưa lưu bảng metric hoặc thông tin môi trường được kiểm chứng trong tài liệu này.
- Mục tiêu tiếp theo: ghi lại thực nghiệm thật, thay vì chỉ liệt kê công cụ đã dùng.

## Mốc cập nhật · 2026-10-17

- Bài 10 thêm lật ngang ngẫu nhiên khi train, biểu đồ confusion matrix và lưu trạng thái optimizer để tiếp tục huấn luyện.
- Bài 11 đưa mô hình đã học vào luồng dự đoán một ảnh mới.
- Launcher TensorBoard chuyển sang thư mục `tensorboard/`.
- Đây là ngày ghi nhận phiên bản trong repo; chưa bổ sung metric hay xác nhận chạy huấn luyện/inference thành công.

## Mẫu ghi chép cho lần thực hành tiếp theo

Sao chép mục này và điền bằng những gì thực sự quan sát được. Giữ các ô chưa đo là “chưa đo”.

### Bài XX · Tên bài

| Nội dung | Ghi chép |
| --- | --- |
| Ngày thực hành | YYYY-MM-DD |
| Mục tiêu | Bài toán hoặc câu hỏi muốn giải quyết |
| Điểm mới so với bài trước | Kỹ thuật hoặc cách tổ chức mới |
| Dataset & split | Nguồn, số mẫu, train/validation/test |
| Môi trường | Python, PyTorch/Torchvision, CPU/GPU |
| Cấu hình | Seed, epochs, batch size, learning rate, optimizer |
| Kết quả | Loss, metric, thời gian chạy; ghi rõ tập được đánh giá |
| Bằng chứng | Đường dẫn hình, bảng kết quả hoặc commit |
| Điều hiểu rõ hơn | Nhận xét của bản thân sau khi thực hành |
| Hạn chế | Lỗi còn gặp hoặc điều chưa kiểm chứng |
| Bước tiếp theo | Một cải tiến cụ thể để thử |

Không cần viết lại các bài cũ cho đẹp để thể hiện tiến bộ. Những điều chưa hoàn thiện cũng là mốc hữu ích nếu được ghi rõ và đối chiếu với bài sau.
