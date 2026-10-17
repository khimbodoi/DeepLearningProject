# Deep Learning · Learning Journey

**11 bài thực hành · Python & PyTorch · Từ mô hình đầu tiên đến CNN phân loại ảnh**

Đây là nơi mình lưu lại quá trình học Deep Learning bằng cách tự viết và thử nghiệm từng bài. Các phiên bản code được lưu qua Git để nhìn lại mình đã tiến bộ ở đâu và còn cần hoàn thiện điều gì.

> Điểm nhấn của repo là sự phát triển qua từng bài: hiểu dữ liệu → xây mô hình → huấn luyện → đánh giá → theo dõi thực nghiệm.

## Lộ trình thực hành

| Bài | Nội dung | Kỹ năng thể hiện trong code |
| :--: | --- | --- |
| [01](bai1_linear_regression.py) | Hồi quy tuyến tính | Custom Dataset, `nn.Linear`, MSE, SGD và dự đoán sau huấn luyện |
| [02](bai2_mnist_cnn.py) | CNN nhận dạng chữ số MNIST | Convolution, pooling, chuẩn hóa ảnh, Adam và test accuracy |
| [03](bai3_quickdraw.py) | Template matching trên QuickDraw | Ảnh trung bình, dot product, cosine similarity và trực quan hóa; baseline không dùng neural network |
| [04](bai4_cifar10_pickle.py) | Đọc CIFAR-10 từ pickle | Tải dữ liệu, đọc batch, tự viết Dataset và chuyển thứ tự chiều ảnh |
| [05](bai5_animal_dataset.py) | Custom Dataset ảnh động vật | Đọc ảnh RGB từ thư mục, gán nhãn, transforms và DataLoader |
| [06](bai6_cifar10_torchvision.py) | CIFAR-10 với Torchvision | Dataset có sẵn, `ToTensor` và nạp dữ liệu theo batch |
| [07](bai7_cifar10_difine_aNN.py) | Định nghĩa MLP cho CIFAR-10 | `nn.Module`, flatten, các lớp fully connected và forward pass |
| [08](bai8_train_neuralnetwork.py) | Huấn luyện MLP | Tái sử dụng mô hình bài 7, train/eval và báo cáo precision, recall, F1 |
| [09](bai9_luyentapvietmotmohinhbatki.py) | Phác thảo ResNet bottleneck | Nhánh residual, convolution 1×1 / 3×3 và downsampling; **đang viết dở** |
| [10](bai10_buildCNNmodel.py) | CNN phân loại 10 loài động vật | Augmentation, confusion matrix, TensorBoard, checkpoint và tiếp tục huấn luyện |
| [11](bai11_trainCNN.py) | Dự đoán ảnh với CNN đã huấn luyện | Nạp checkpoint, tiền xử lý ảnh, softmax và hiển thị dự đoán bằng OpenCV |

Số thứ tự là thứ tự bài thực hành, không phải lịch sử commit hay bằng chứng về thời gian hoàn thành.

## Những bước phát triển có thể nhìn thấy

- **Bài 1–3:** từ học tham số bằng gradient sang xử lý ảnh; thử một baseline dựa trên độ tương đồng để hiểu vai trò của chuẩn hóa.
- **Bài 4–6:** tập trung vào dữ liệu, từ đọc định dạng thô đến Dataset tùy biến và API có sẵn của Torchvision.
- **Bài 7–8:** tách định nghĩa mô hình khỏi chương trình huấn luyện, rồi bổ sung đánh giá theo từng lớp.
- **Bài 9–11:** thử kiến trúc residual, bổ sung augmentation và confusion matrix vào quy trình CNN, rồi nạp checkpoint để dự đoán ảnh mới.

Xem [nhật ký học tập](docs/LEARNING_JOURNAL.md) để theo dõi từng mốc và ghi kết quả cho các bài tiếp theo.

## Bắt đầu

Chạy lệnh từ thư mục gốc của repo. Tạo môi trường riêng bằng Python có hỗ trợ các thư viện bên dưới:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision numpy matplotlib opencv-python pillow scikit-learn tqdm tensorboard
```

Lệnh trên là cách cài cơ bản; repo hiện chưa có lockfile hoặc môi trường phiên bản đã kiểm chứng. Nếu dùng GPU, cần bản PyTorch tương thích với môi trường CUDA của máy. Các bài huấn luyện 1, 2, 8 và 10 tự chọn CUDA khi khả dụng, còn lại dùng CPU.

Ví dụ chạy từng bài:

```powershell
python bai1_linear_regression.py
python bai2_mnist_cnn.py
python bai3_quickdraw.py
python bai8_train_neuralnetwork.py
```

MNIST, CIFAR-10 ở các bài dùng Torchvision và QuickDraw có logic tải dữ liệu. Lần chạy đầu cần mạng; bài QuickDraw cũng mở các biểu đồ Matplotlib.

### Dữ liệu ảnh động vật

Bài 10 đọc `./animal` với cấu trúc sau. Mỗi thư mục `train` và `test` cần đủ 10 thư mục nhãn:

```text
animal/
├── train/
│   ├── cat/
│   ├── dog/
│   └── ...
└── test/
    ├── cat/
    ├── dog/
    └── ...
```

Thứ tự nhãn trong code: `cat`, `dog`, `cow`, `horse`, `chicken`, `squirrel`, `sheep`, `elephant`, `spider`, `butterfly`. Dataset này cần được chuẩn bị riêng; repo không kèm ảnh huấn luyện.

```powershell
python bai10_buildCNNmodel.py --root ./animal --epochs 15 --batch-size 64 --trained_models ./trained_models/run01 --logging ./tensorboard/run01
python -m tensorboard.main --logdir ./tensorboard --port 6006
```

Tạo thư mục cha `trained_models/` trước khi dùng đường dẫn con như ví dụ trên. Mở TensorBoard tại `http://localhost:6006`. Bài 10 ghi `Train/Loss`, `Val/Accuracy` và `confusion_matrix`; lưu `last_cnn.pt` và `best_cnn.pt` gồm epoch, trọng số mô hình và trạng thái optimizer. Có thể truyền `--checkpoint` để tiếp tục huấn luyện; `best_acc` chỉ được lưu trong best checkpoint.

Bài 11 thực hiện inference dù tên file là `bai11_trainCNN.py`:

```powershell
python bai11_trainCNN.py --image-path "C:/duong-dan/anh.jpg" --checkpoint ./trained_models/run01/best_cnn.pt
```

Dùng đường dẫn ảnh tuyệt đối trên Windows để thay thế thư mục `test_dir` cố định trong code. Chương trình mở cửa sổ OpenCV và in nhãn dự đoán cùng xác suất softmax; xác suất này chưa được hiệu chỉnh độ tin cậy.

**Lưu ý khi chạy code hiện tại:**

- Bài 4 có `input()` khi đọc từng batch và mở cửa sổ OpenCV.
- Phần chạy trực tiếp của bài 5 dùng đường dẫn Windows `D:\Download\data`.
- Bài 7 đưa toàn bộ 50.000 ảnh qua mô hình cùng lúc, có thể cần nhiều RAM; bài 8 dùng DataLoader theo batch.
- Bài 9 chưa chạy được: còn lỗi thụt lề, chưa có `_make_layer`, stem/forward của ResNet và tham chiếu `BottleneckBlock` chưa được định nghĩa.
- Bài 10 nhận `--root`, `--image-size`, `--epochs`, `--batch-size`, `--logging`, `--trained_models` và `--checkpoint`. Classifier vẫn cố định đầu vào `128 × 7 × 7`; giữ kích thước ảnh mặc định 224 cho bài 10 và 11.
- Bài 10 **xóa thư mục log được chỉ định nếu nó đã tồn tại** và ghi đè checkpoint cùng tên. Dùng thư mục log và model mới cho mỗi thực nghiệm, kể cả khi resume.
- Bài 11 cần checkpoint dạng dictionary có khóa `model`, tương thích phiên bản hiện tại của bài 10; ảnh thử và trọng số không được kèm trong repo.
- Bài 10 chọn checkpoint bằng accuracy trên tập `test`; đây chưa phải quy trình tách riêng train/validation/test.

## Cách đọc kết quả

Repo thể hiện những kỹ thuật đã được triển khai, chưa khẳng định các bài đều đã chạy thành công trong một môi trường chung. Chưa có bộ kết quả kèm seed, phiên bản thư viện và cấu hình máy để đối chiếu; vì vậy README không công bố accuracy cụ thể.

Khi có lần chạy mới, ghi dataset/split, cấu hình, loss, metric và nhận xét vào nhật ký. Với bài phân loại, theo dõi thêm precision, recall và F1 theo lớp để hiểu mô hình ngoài một con số accuracy.

## Quy ước lưu trữ

Giữ nguyên tên file; lưu các lần cập nhật code bằng commit để bảo toàn dấu vết học tập. Tài liệu bổ sung nằm trong `docs/`. Dataset, cache, TensorBoard logs, trọng số, môi trường ảo và tài liệu CV cá nhân được bỏ qua bởi `.gitignore`.

Bài tiếp theo sẽ tiếp tục được thêm theo số thứ tự, cùng một mục nhật ký ngắn: **mục tiêu → điều đã làm → kết quả quan sát → bài học → bước tiếp theo**.
