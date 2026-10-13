"""
QuickDraw Classification - Template Matching
So sánh giữa Tích vô hướng thuần túy (Dot Product) và Cosine Similarity (Chuẩn hóa)
Chạy trực tiếp không cần tải thủ công dataset.
"""

import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# HÀM HỖ TRỢ: Tự động nạp hoặc tải dữ liệu từ Google Cloud QuickDraw
# ==============================================================================
def load_or_download_category(category_name, num_samples=2500):
    """
    Tự động load dữ liệu:
    1. Nếu file 'full_numpy_bitmap_{category_name}.npy' đã có trong thư mục -> Đọc trực tiếp.
    2. Nếu chưa có -> Tự động tải online trực tiếp từ Google Cloud Storage về máy.
    """
    filename = f"./full_numpy_bitmap_{category_name}.npy"
    
    # 1. Kiểm tra nếu file đã có sẵn
    if os.path.exists(filename):
        print(f"-> Đang đọc dữ liệu cục bộ: {filename}")
        data = np.load(filename).astype(np.float32)
        return data[:num_samples] if len(data) >= num_samples else data

    # 2. Nếu chưa có, tải tự động một phần dữ liệu (Range Request) để tiết kiệm thời gian
    print(f"-> Đang tự động tải category '{category_name}' từ Google Storage...")
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category_name}.npy"
    
    headers = {'Range': f"bytes=80-{80 + num_samples * 784 - 1}"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            raw_bytes = response.read()
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((num_samples, 784))
        np.save(filename, arr)
        print(f"   Đã tải và lưu {filename} thành công! Shape: {arr.shape}")
        return arr.astype(np.float32)
    except Exception as e:
        print(f"   Lỗi khi tải {category_name}: {e}")
        raise

def cosine_similarity(v1, v2):
    """Hàm tính độ tương đồng Cosine giữa 2 vector đã được chuẩn hóa độ dài"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# ==============================================================================
# BƯỚC 1: Chọn 1 category chính và phân chia train / test
# ==============================================================================
selected_category = "bicycle"
images = load_or_download_category(selected_category, num_samples=2500)
print(f"\n[Bước 1] Kích thước dữ liệu '{selected_category}': {images.shape}")

train_images = images[:-10]  # Lấy tất cả ảnh, trừ 10 ảnh cuối làm bộ train
test_images = images[-10:]   # 10 ảnh cuối làm bộ test
print(f"Số lượng ảnh train: {train_images.shape[0]}, ảnh test: {test_images.shape[0]}")

# ==============================================================================
# BƯỚC 2: Tính bức ảnh trung bình của bộ training (28x28 pixel)
# ==============================================================================
avg_image = np.mean(train_images, axis=0).reshape(28, 28)
print(f"[Bước 2] Đã tính ảnh trung bình. Kích thước avg_image: {avg_image.shape}")

# ==============================================================================
# BƯỚC 3: Visualize bức ảnh trung bình
# ==============================================================================
print(f"[Bước 3] Hiển thị bức ảnh trung bình của category '{selected_category}'...")
plt.figure(figsize=(3.5, 3.5))
plt.title(f"Average image of '{selected_category}'", fontsize=12)
plt.imshow(avg_image, cmap='gray')
plt.axis('off')
plt.show()

# ==============================================================================
# BƯỚC 4: Chọn 1 ảnh test và tính tích vô hướng (Dot Product) với avg_image
# ==============================================================================
index = 4  # Chọn index = 4 từ 10 ảnh test
test_image = test_images[index]

# 1. Tích vô hướng thông thường (như trong bài giảng)
score_dot = np.dot(test_image.flatten(), avg_image.flatten())

# 2. Cosine similarity (sau khi đã chuẩn hóa)
score_cos = cosine_similarity(test_image.flatten(), avg_image.flatten())

print(f"\n[Bước 4] Điểm số của ảnh test[{index}] ({selected_category}) với chính nó:")
print(f"  - Tích vô hướng thô (Dot Product): {score_dot:,.2f}")
print(f"  - Cosine Similarity (Chuẩn hóa) : {score_cos:.4f}")

# ==============================================================================
# BƯỚC 5: Lặp lại cho 10 categories và so sánh điểm số
# ==============================================================================
print("\n" + "="*70)
print("[Bước 5] Huấn luyện Template Matching trên 10 categories...")
categories = [
    'bicycle', 'apple', 'banana', 'cat', 'fish', 
    'star', 'tree', 'car', 'house', 'clock'
]

weights = {}
all_test_images = {}

for cat in categories:
    cat_images = load_or_download_category(cat, num_samples=2500)
    cat_train = cat_images[:-10]
    cat_test = cat_images[-10:]
    
    # Vector trọng số chính là ảnh trung bình dạng phẳng (784,)
    weights[cat] = np.mean(cat_train, axis=0)
    all_test_images[cat] = cat_test

# --- SO SÁNH 2 PHƯƠNG PHÁP TRÊN ẢNH TEST (bicycle, index=4) ---
test_cat_true = selected_category
dot_scores = {}
cos_scores = {}

for cat in categories:
    dot_scores[cat] = float(np.dot(test_image.flatten(), weights[cat].flatten()))
    cos_scores[cat] = float(cosine_similarity(test_image.flatten(), weights[cat].flatten()))

sorted_dot = sorted(dot_scores.items(), key=lambda x: x[1], reverse=True)
sorted_cos = sorted(cos_scores.items(), key=lambda x: x[1], reverse=True)

print(f"\n--- BẢNG SO SÁNH KẾT QUẢ DỰ ĐOÁN (ẢNH TEST: {test_cat_true.upper()}, INDEX={index}) ---")
print(f"{'Hạng':<5} | {'[1] DOT PRODUCT (THÔ)':<28} | {'[2] COSINE SIMILARITY (CHUẨN HÓA)':<30}")
print("-" * 75)
for i in range(len(categories)):
    cat_d, sc_d = sorted_dot[i]
    cat_c, sc_c = sorted_cos[i]
    
    mark_d = " (*)" if cat_d == test_cat_true else "    "
    mark_c = " (*)" if cat_c == test_cat_true else "    "
    
    print(f"{i+1:<5} | {cat_d:<10} {sc_d:>12,.1f}{mark_d} | {cat_c:<10} {sc_c:>14.4f}{mark_c}")

print("-" * 75)
print(f"(*) là nhãn thực tế ('{test_cat_true}')")
print(f"=> Dự đoán theo Dot Product : '{sorted_dot[0][0]}' (Top 1) -> {'ĐÚNG' if sorted_dot[0][0] == test_cat_true else 'SAI'}")
print(f"=> Dự đoán theo Cosine Sim  : '{sorted_cos[0][0]}' (Top 1) -> {'ĐÚNG' if sorted_cos[0][0] == test_cat_true else 'SAI'}")

# --- ĐÁNH GIÁ ACCURACY TRÊN TOÀN BỘ 100 ẢNH TEST ---
acc_dot = 0
acc_cos = 0
total = 0

for cat in categories:
    for img in all_test_images[cat]:
        # Dot product
        scores_d = {c: np.dot(img.flatten(), weights[c]) for c in categories}
        if max(scores_d, key=scores_d.get) == cat:
            acc_dot += 1
            
        # Cosine similarity
        scores_c = {c: cosine_similarity(img.flatten(), weights[c]) for c in categories}
        if max(scores_c, key=scores_c.get) == cat:
            acc_cos += 1
            
        total += 1

print(f"\n--- ĐỘ CHÍNH XÁC TRÊN TOÀN BỘ TẬP TEST (100 ẢNH) ---")
print(f"  - Tích vô hướng thô (Dot Product): {acc_dot}/{total} ({acc_dot/total*100:.1f}%)")
print(f"  - Cosine Similarity (Chuẩn hóa) : {acc_cos}/{total} ({acc_cos/total*100:.1f}%)")

# ==============================================================================
# BƯỚC 6 (Optional): Visualize 10 trọng số (avg_image) trên lưới 2x5
# ==============================================================================
print("\n[Bước 6] Đang hiển thị lưới 2x5 ảnh trung bình (Weights) của 10 categories...")
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("10 Class Weights (Average Images) in QuickDraw", fontsize=15)

for idx, cat in enumerate(categories):
    ax = axes[idx // 5, idx % 5]
    weight_img = weights[cat].reshape(28, 28)
    ax.imshow(weight_img, cmap='gray')
    ax.set_title(f"{idx+1}. {cat}", fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.show()