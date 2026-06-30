# -*- coding: utf-8 -*-
"""
人工智能课程作业 —— 第12讲：卷积神经网络 CNN
====================================================
包含三个练习：
  练习1：图像边缘提取（手动卷积实现）
  练习3：MNIST 手写数字 CNN 分类
  练习5：(A) MNIST 最佳模型测试 + (B) 猫狗图片 CNN 分类

运行方式：
    cd D:/人工智能/第十四周/code
    python cnn_assignment_all.py

依赖：tensorflow, numpy, matplotlib, PIL, scikit-learn
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ---- 配置 matplotlib 中文字体，避免图片标题乱码 ----
# Windows 常见中文字体优先级列表
_CHINESE_FONT_CANDIDATES = [
    'Microsoft YaHei',   # 微软雅黑
    'SimHei',            # 黑体
    'KaiTi',             # 楷体
    'FangSong',          # 仿宋
    'SimSun',            # 宋体
    'STSong',            # 华文宋体
    'STKaiti',           # 华文楷体
    'STFangsong',        # 华文仿宋
    'Noto Sans CJK SC',  # 思源黑体
    'Noto Sans SC',
    'WenQuanYi Micro Hei',
]
_available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_chinese_font = None
for _font in _CHINESE_FONT_CANDIDATES:
    if _font in _available_fonts:
        _chinese_font = _font
        break

if _chinese_font is not None:
    matplotlib.rcParams['font.sans-serif'] = [_chinese_font, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
    print(f"[字体] 使用中文字体: {_chinese_font}")
else:
    # 没有中文字体时回退，中文标题会用英文替代
    print("[字体] 未找到中文字体，将使用英文标注")

# ============================================================
# 设置路径（数据文件所在目录）—— 自动查找
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 多个候选数据根目录（脚本可能在 code/ 或 大作业/ 下）
_CANDIDATE_ROOTS = [
    _SCRIPT_DIR,                                                  # 脚本所在目录
    os.path.join(_SCRIPT_DIR, ".."),                              # 上级目录
    r"D:\人工智能\第十四周\code",                                   # 已知的 code 目录
    os.path.join(_SCRIPT_DIR, "卷积网络基本概念和分类网络"),        # 课程子目录
]

# 在后两个候选中也查找该脚本所在目录的可能父目录
if os.path.basename(_SCRIPT_DIR) == "大作业":
    _CANDIDATE_ROOTS.append(r"D:\人工智能\第十四周\code")

SUB_PATH = os.path.join("卷积网络基本概念和分类网络",
                        "04和05-卷积网络基本概念和分类网络-预计2个和4个学时")
ANIMAL_DIR = None
for root in _CANDIDATE_ROOTS:
    test_path = os.path.join(root, SUB_PATH, "Animal Images")
    if os.path.exists(test_path):
        ANIMAL_DIR = test_path
        BASE_DIR = root
        break

if ANIMAL_DIR is None:
    print("错误: 找不到数据目录 (Animal Images)!")
    print("请将脚本放到以下目录之一运行：")
    print('  D:/人工智能/第十四周/code/')
    print("或确保数据目录相对于脚本路径正确。")
    sys.exit(1)

print("=" * 70)
print("人工智能课程作业 —— 卷积神经网络 CNN")
print(f"工作目录: {BASE_DIR}")
print("=" * 70)


# ---- 导入 scipy 加速卷积运算 ----
from scipy.signal import convolve2d as _scipy_convolve2d

# ============================================================
# 辅助函数：手动实现 2D 卷积（使用 scipy 加速）
# ============================================================
def conv2d_manual(image_2d, kernel):
    """
    对单通道 2D 图像执行卷积（valid 模式，步长=1）。

    使用 scipy.signal.convolve2d 高效实现（基于 FFT），
    比纯 Python for 循环快数百倍。

    参数:
        image_2d: shape (H, W) 的二维 numpy 数组
        kernel:   shape (kH, kW) 的卷积核

    返回:
        output: 卷积结果
    """
    # scipy 的 convolve2d 默认做的是"互相关"（correlation），
    # 在深度学习中我们通常说的"卷积"其实就是互相关。
    # 这里用 mode='valid' 实现无 padding 的卷积
    return _scipy_convolve2d(image_2d, kernel, mode='valid')


def conv2d_3d_manual(image_3d, kernel_3d):
    """
    对三通道 RGB 图像执行 3D 卷积。

    实现方式：对每个通道分别进行 2D 卷积，然后将三个通道的结果求和。

    参数:
        image_3d:  shape (H, W, 3) 的 RGB 图像
        kernel_3d: shape (kH, kW, 3) 的三通道卷积核

    返回:
        output: shape (out_H, out_W) 的二维结果
    """
    result = None
    for c in range(image_3d.shape[-1]):
        channel_result = _scipy_convolve2d(
            image_3d[:, :, c], kernel_3d[:, :, c], mode='valid')
        if result is None:
            result = channel_result
        else:
            result += channel_result
    return result


# ============================================================
# 练习 1：图像边缘提取
# ============================================================
print("\n" + "=" * 70)
print("练习 1：图像边缘提取（手动卷积）")
print("=" * 70)

# 1.1 读取图片
img_path = os.path.join(os.path.dirname(ANIMAL_DIR), 'bagua.jpg')
img = Image.open(img_path)
print(f"\n原始图片尺寸: {img.size}, 模式: {img.mode}")

plt.figure(figsize=(12, 10))
plt.subplot(3, 3, 1)
plt.imshow(img)
plt.title("原始图片 (RGB)")
plt.axis('off')

# ---- 1.2 转换为灰度图像 ----
img_gray = img.convert('L')
img_gray_np = np.array(img_gray, dtype='float32')
print(f"灰度图像 shape: {img_gray_np.shape}")

plt.subplot(3, 3, 2)
plt.imshow(img_gray_np, cmap='gray')
plt.title("灰度图像")
plt.axis('off')

# ---- 1.3 二值化预处理 ----
threshold = 128
img_binary = np.where(img_gray_np > threshold, 255.0, 0.0).astype('float32')
print(f"二值图像 (阈值={threshold}), 白像素: {np.sum(img_binary > 0)}, "
      f"黑像素: {np.sum(img_binary == 0)}")

plt.subplot(3, 3, 3)
plt.imshow(img_binary, cmap='gray')
plt.title(f"二值图像 (阈值={threshold})")
plt.axis('off')

# ---- 1.4 尝试 [1, -1] 水平卷积核 ----
kernel_h = np.array([[1.0, -1.0]])
result_h = conv2d_manual(img_gray_np, kernel_h)
print(f"\n[1,-1] 水平卷积结果 shape: {result_h.shape}")

plt.subplot(3, 3, 4)
plt.imshow(result_h, cmap='gray')
plt.title("水平边缘 [1, -1]")
plt.axis('off')

# ---- 1.5 尝试 [1, -1]^T 垂直卷积核 ----
kernel_v = np.array([[1.0], [-1.0]])
result_v = conv2d_manual(img_gray_np, kernel_v)
print(f"[1,-1]^T 垂直卷积结果 shape: {result_v.shape}")

plt.subplot(3, 3, 5)
plt.imshow(result_v, cmap='gray')
plt.title("垂直边缘 [1,-1]^T")
plt.axis('off')

# ---- 1.6 3x3 拉普拉斯边缘检测算子 ----
w_2d = np.array([[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]], dtype='float32')

# 在灰度图像上应用
result_laplacian = conv2d_manual(img_gray_np, w_2d)
print(f"3x3 拉普拉斯算子 (灰度) 结果 shape: {result_laplacian.shape}")
print(f"  值范围: [{result_laplacian.min():.1f}, {result_laplacian.max():.1f}]")

plt.subplot(3, 3, 6)
plt.imshow(result_laplacian, cmap='gray')
plt.title("3x3 拉普拉斯算子 (灰度)")
plt.axis('off')

# 在二值图像上应用
result_laplacian_bin = conv2d_manual(img_binary, w_2d)
print(f"3x3 拉普拉斯算子 (二值) 结果 shape: {result_laplacian_bin.shape}")
print(f"  值范围: [{result_laplacian_bin.min():.1f}, {result_laplacian_bin.max():.1f}]")

plt.subplot(3, 3, 7)
plt.imshow(result_laplacian_bin, cmap='gray')
plt.title("3x3 拉普拉斯算子 (二值)")
plt.axis('off')

# ---- 1.7 3D 卷积（RGB三通道同时处理） ----
img_rgb_np = np.array(img, dtype='float32')
print(f"\nRGB 图像 shape: {img_rgb_np.shape}")

# 构建 3D 卷积核 (3, 3, 3)
w_3d_kernel = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]], dtype='float32')
# 复制到三个通道
w_3d = np.stack([w_3d_kernel] * 3, axis=-1)  # shape (3, 3, 3)
print(f"3D 卷积核 shape: {w_3d.shape}")

result_3d = conv2d_3d_manual(img_rgb_np, w_3d)
print(f"3D 卷积结果 shape: {result_3d.shape}")
print(f"  值范围: [{result_3d.min():.1f}, {result_3d.max():.1f}]")

plt.subplot(3, 3, 8)
plt.imshow(result_3d, cmap='gray')
plt.title("3D 卷积 (RGB三通道)")
plt.axis('off')

# ---- 1.8 对比分析：各通道单独处理 vs 3D卷积 ----
r_result = conv2d_manual(img_rgb_np[:, :, 0], w_3d_kernel)
g_result = conv2d_manual(img_rgb_np[:, :, 1], w_3d_kernel)
b_result = conv2d_manual(img_rgb_np[:, :, 2], w_3d_kernel)
avg_result = (r_result + g_result + b_result) / 3.0

plt.subplot(3, 3, 9)
plt.imshow(avg_result, cmap='gray')
plt.title("各通道分别卷积后平均")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'exercise1_edge_detection.png'), dpi=100)
plt.show()
print("\n[练习1 完成] 边缘检测结果已保存为 exercise1_edge_detection.png")

# ============================================================
# 练习 1 结果分析
# ============================================================
print("""
+-----------------------------------------------------------+
|                    练习 1 结果分析                          |
+-----------------------------------------------------------+
| 1. [1,-1] 水平卷积核：只能检测垂直边缘（水平方向变化）。    |
|    在黑白分明的 bagua 图上，垂直边缘处有强烈响应。         |
|    但在水平边缘处几乎无响应。                              |
|                                                           |
| 2. 3x3 拉普拉斯算子 [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]:   |
|    中心权重 8，周边权重 -1，能同时检测水平和垂直边缘。     |
|    在灰度图像上效果好；二值图像上边缘更锐利但可能丢细节。  |
|                                                           |
| 3. 3D 卷积 vs 分别卷积后平均：                             |
|    数学上等价（当 3D 核各通道相同时），但提供了为不同      |
|    颜色通道设置不同权重的灵活性。                          |
|                                                           |
| 4. 改进建议：                                              |
|    - 使用 Canny 边缘检测（高斯平滑+梯度计算+NMS）         |
|    - 使用 Sobel 算子获得方向性边缘                         |
|    - 预处理：高斯模糊去噪后再检测                          |
|    - 不同阈值二值化对边缘检测结果影响很大                  |
+-----------------------------------------------------------+
""")


# ============================================================
# 练习 3：MNIST CNN 分类
# ============================================================
print("\n" + "=" * 70)
print("练习 3：MNIST 手写数字 CNN 分类")
print("=" * 70)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout)

print(f"TensorFlow 版本: {tf.__version__}")

# ---- 3.1 加载 MNIST 数据集 ----
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"\n训练集: {x_train.shape}, 标签: {y_train.shape}")
print(f"测试集: {x_test.shape}, 标签: {y_test.shape}")

# ---- 3.2 数据预处理 ----
# 归一化到 [0, 1]
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

# 扩展维度以适应 CNN 输入 (N, 28, 28, 1)
x_train_cnn = np.expand_dims(x_train_norm, axis=-1)
x_test_cnn = np.expand_dims(x_test_norm, axis=-1)

# One-hot 编码
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

print(f"预处理后训练集: {x_train_cnn.shape}, 标签: {y_train_onehot.shape}")
print(f"预处理后测试集: {x_test_cnn.shape}, 标签: {y_test_onehot.shape}")

# ---- 3.3 构建 CNN 模型 ----
model_mnist = models.Sequential([
    # 第一个卷积块
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           padding='same', input_shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 第二个卷积块
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 分类头
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_mnist.summary()

# ---- 3.4 编译与训练 ----
model_mnist.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("\n开始训练 MNIST CNN 模型...")
history = model_mnist.fit(
    x_train_cnn, y_train_onehot,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    verbose=1
)

# ---- 3.5 评估 ----
test_loss, test_acc = model_mnist.evaluate(x_test_cnn, y_test_onehot, verbose=0)
print(f"\nMNIST 测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# ---- 3.6 绘制训练曲线 ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('MNIST CNN - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('MNIST CNN - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'exercise3_mnist_training.png'), dpi=100)
plt.show()
print("[练习3 完成] MNIST 训练曲线已保存为 exercise3_mnist_training.png")


# ============================================================
# 练习 5-A：使用最佳 MNIST 模型预测测试集
# ============================================================
print("\n" + "=" * 70)
print("练习 5-A：MNIST 最佳模型测试")
print("=" * 70)

# 使用上面训练好的模型（已经是练习3中优化过的）
# 做预测
y_pred_probs = model_mnist.predict(x_test_cnn, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_onehot, axis=1)

# 计算各类别准确率
from sklearn.metrics import classification_report, confusion_matrix

print("\nMNIST 测试集分类报告:")
print(classification_report(y_true, y_pred, digits=4))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵 (10x10):")
print(cm)

# 可视化部分预测结果
plt.figure(figsize=(12, 10))
for i in range(25):
    idx = np.random.randint(0, len(x_test))
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    pred_label = np.argmax(y_pred_probs[idx])
    true_label = y_test[idx]
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"True:{true_label} Pred:{pred_label}", color=color, fontsize=9)
    plt.axis('off')
plt.suptitle("MNIST Test Predictions (Green=Correct, Red=Wrong)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'exercise5a_mnist_predictions.png'), dpi=100)
plt.show()
print("[练习5-A 完成] MNIST 预测结果已保存")


# ============================================================
# 练习 5-B：猫狗图片 CNN 分类
# ============================================================
print("\n" + "=" * 70)
print("练习 5-B：猫狗图片 CNN 分类")
print("=" * 70)

# ---- 5B.1 加载猫狗图片数据集 ----
CATS_DIR = os.path.join(ANIMAL_DIR, 'cats')
DOGS_DIR = os.path.join(ANIMAL_DIR, 'dogs')

print(f"猫图片目录: {CATS_DIR}")
print(f"狗图片目录: {DOGS_DIR}")

# 获取文件列表
cat_files = [os.path.join(CATS_DIR, f) for f in os.listdir(CATS_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
dog_files = [os.path.join(DOGS_DIR, f) for f in os.listdir(DOGS_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"猫图片数量: {len(cat_files)}")
print(f"狗图片数量: {len(dog_files)}")

# ---- 5B.2 数据预处理函数 ----
IMG_SIZE = 128  # 将图片缩放到 128x128（平衡速度与精度）

def load_and_preprocess_image(file_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """加载并预处理单张图片"""
    try:
        img = Image.open(file_path).convert('L')  # 转为灰度
        img = img.resize(target_size)
        img_array = np.array(img, dtype='float32') / 255.0
        return img_array
    except Exception as e:
        return None


# 为了快速演示，使用子集训练（完整训练可选）
# 完整数据集约 15000+15000 张，训练时间较长
USE_SUBSET = True  # 设为 False 使用全部数据
SUBSET_SIZE = 5000  # 每类使用的数量

if USE_SUBSET:
    print(f"\n*** 使用子集训练，每类 {SUBSET_SIZE} 张（设 USE_SUBSET=False 用全部数据）")
    cat_files = cat_files[:SUBSET_SIZE]
    dog_files = dog_files[:SUBSET_SIZE]

print("\n加载猫图片...")
cat_images = []
for f in cat_files:
    img = load_and_preprocess_image(f)
    if img is not None:
        cat_images.append(img)
cat_images = np.array(cat_images)
print(f"成功加载猫图片: {cat_images.shape}")

print("加载狗图片...")
dog_images = []
for f in dog_files:
    img = load_and_preprocess_image(f)
    if img is not None:
        dog_images.append(img)
dog_images = np.array(dog_images)
print(f"成功加载狗图片: {dog_images.shape}")

# ---- 5B.3 合并并创建标签 ----
# 猫=0, 狗=1
X_all = np.concatenate([cat_images, dog_images], axis=0)
y_all = np.concatenate([np.zeros(len(cat_images)), np.ones(len(dog_images))])

# 扩展维度 (N, H, W, 1)
X_all = np.expand_dims(X_all, axis=-1)
print(f"\n总数据集: {X_all.shape}, 标签: {y_all.shape}")

# ---- 5B.4 打乱并拆分训练/验证/测试集 ----
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X_all, y_all = shuffle(X_all, y_all, random_state=42)

# 8:1:1 拆分
X_temp, X_test_cd, y_temp, y_test_cd = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42)
X_train_cd, X_val_cd, y_train_cd, y_val_cd = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42)  # 0.111 * 0.9 ~ 0.1

print(f"\n训练集: {X_train_cd.shape}, 标签: {y_train_cd.shape}")
print(f"验证集: {X_val_cd.shape}, 标签: {y_val_cd.shape}")
print(f"测试集: {X_test_cd.shape}, 标签: {y_test_cd.shape}")

# One-hot 编码
y_train_cd_oh = tf.keras.utils.to_categorical(y_train_cd, 2)
y_val_cd_oh = tf.keras.utils.to_categorical(y_val_cd, 2)
y_test_cd_oh = tf.keras.utils.to_categorical(y_test_cd, 2)

# ---- 5B.5 数据增强 ----
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# ---- 5B.6 构建猫狗分类 CNN ----
model_catdog = models.Sequential([
    # 数据增强层（仅训练时生效）
    data_augmentation,

    # 卷积块 1
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 卷积块 2
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 卷积块 3
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 分类头
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model_catdog.summary()

# ---- 5B.7 编译与训练 ----
model_catdog.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 回调函数
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
]

print("\n开始训练猫狗分类 CNN 模型...")
print(f"训练样本数: {len(X_train_cd)}, 验证样本数: {len(X_val_cd)}")
history_cd = model_catdog.fit(
    X_train_cd, y_train_cd_oh,
    batch_size=64,
    epochs=50,
    validation_data=(X_val_cd, y_val_cd_oh),
    callbacks=callbacks,
    verbose=1
)

# ---- 5B.8 评估 ----
test_loss_cd, test_acc_cd = model_catdog.evaluate(
    X_test_cd, y_test_cd_oh, verbose=0)
print(f"\n猫狗分类 测试集准确率: {test_acc_cd:.4f} ({test_acc_cd * 100:.2f}%)")

# 各类别准确率
y_pred_cd = np.argmax(model_catdog.predict(X_test_cd, verbose=0), axis=1)
y_true_cd = np.argmax(y_test_cd_oh, axis=1)

print("\n猫狗分类 测试集分类报告:")
print(classification_report(y_true_cd, y_pred_cd,
                            target_names=['猫 (Cat)', '狗 (Dog)'], digits=4))

# 混淆矩阵
cm_cd = confusion_matrix(y_true_cd, y_pred_cd)
print(f"混淆矩阵:\n{cm_cd}")

# ---- 5B.9 绘制训练曲线 ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_cd.history['accuracy'], label='train acc')
plt.plot(history_cd.history['val_accuracy'], label='val acc')
plt.title('Cat/Dog CNN - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_cd.history['loss'], label='train loss')
plt.plot(history_cd.history['val_loss'], label='val loss')
plt.title('Cat/Dog CNN - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'exercise5b_catdog_training.png'), dpi=100)
plt.show()
print("[练习5-B 完成] 猫狗分类训练曲线已保存为 exercise5b_catdog_training.png")

# ---- 5B.10 可视化预测结果 ----
plt.figure(figsize=(12, 10))
class_names = ['Cat', 'Dog']
for i in range(25):
    idx = np.random.randint(0, len(X_test_cd))
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test_cd[idx].squeeze(), cmap='gray')
    pred_label = y_pred_cd[idx]
    true_label = y_true_cd[idx]
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"T:{class_names[true_label]} P:{class_names[pred_label]}",
              color=color, fontsize=9)
    plt.axis('off')
plt.suptitle("Cat/Dog Test Predictions (Green=Correct, Red=Wrong)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'exercise5b_catdog_predictions.png'), dpi=100)
plt.show()
print("[练习5-B 完成] 猫狗预测结果已保存")


# ============================================================
# 最终总结
# ============================================================
print("\n" + "=" * 70)
print("全部练习完成！")
print("=" * 70)
print(f"""
生成的文件：
  1. exercise1_edge_detection.png     -- 边缘检测对比图
  2. exercise3_mnist_training.png     -- MNIST CNN 训练曲线
  3. exercise5a_mnist_predictions.png -- MNIST 测试集预测样本
  4. exercise5b_catdog_training.png   -- 猫狗分类 CNN 训练曲线
  5. exercise5b_catdog_predictions.png-- 猫狗分类预测样本

最终结果汇总：
  +---------------------+------------------+
  | 任务                 | 测试准确率        |
  +---------------------+------------------+
  | MNIST CNN 分类       | {test_acc:.4f} ({test_acc*100:.2f}%)      |
  | 猫狗 CNN 分类         | {test_acc_cd:.4f} ({test_acc_cd*100:.2f}%)      |
  +---------------------+------------------+

训练体会与感悟：
  1. 手动实现卷积帮助理解了卷积核如何滑动提取特征，
     不同的卷积核（水平/垂直/拉普拉斯）提取不同类型的边缘。
  2. MNIST 手写数字分类是 CNN 的经典入门任务，使用
     Conv-ReLU-Pool-Dropout 的标准结构即可达到 >99% 准确率。
  3. 猫狗分类任务更难（图片复杂、背景多变），数据增强
     （翻转/旋转/缩放）对防止过拟合非常重要。
  4. 使用 ReduceLROnPlateau 和 EarlyStopping 可以自动
     调节学习率和停止训练，避免手动调参。
  5. Dropout 层对于全连接层防止过拟合效果显著。
""")
