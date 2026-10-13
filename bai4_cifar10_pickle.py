from torch.utils.data import Dataset
import os
import urllib.request
import tarfile
import pickle
import numpy as np
import cv2

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def ensure_cifar10(download_dir=None):
    """
    Tự động kiểm tra file; nếu chưa có thì tải về và giải nén.
    Trả về đường dẫn tuyệt đối đến thư mục chứa các file batch.
    """
    if download_dir is None:
        download_dir = os.path.dirname(os.path.abspath(__file__))
    extracted_dir = os.path.join(download_dir, "cifar-10-batches-py")
    tar_path = os.path.join(download_dir, "cifar-10-python.tar.gz")
    # Kiểm tra xem dữ liệu đã được giải nén và tồn tại file data_batch_1 chưa
    if not os.path.exists(os.path.join(extracted_dir, "data_batch_1")):
        # Nếu chưa có file nén tar.gz thì tiến hành tải về
        if not os.path.exists(tar_path):
            print("Đang tải CIFAR-10...")
            urllib.request.urlretrieve(CIFAR_URL, tar_path)
            print("Tải hoàn tất.")

        print("Đang giải nén...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print("Giải nén hoàn tất.")

    return extracted_dir 

class MyDataset(Dataset):
    def __init__(self, root, train=True):
        if train:
            datafiles = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            datafiles = [os.path.join(root, "test_batch")]
        
        self.images = []
        self.labels = []
        for file in datafiles:
            with open(file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                input(data.keys())
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


if __name__ == "__main__":
    root_path = ensure_cifar10()
    dataset = MyDataset(root=root_path, train=True)
    
    img, label = dataset.__getitem__(3234)
    img = np.reshape(img, (3, 32, 32)).transpose(1, 2, 0)  # Chuyển đổi hình ảnh từ (C, H, W) sang (H, W, C)
    label = np.reshape(label, (1,))
    print(img.shape)
    print(label)

    cv2.imshow("Image", img)
    cv2.waitKey(0)