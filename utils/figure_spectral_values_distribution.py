import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 定义图像文件夹路径
folder_path = '/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/images'
# folder_path = '/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/images'

# 获取目录下所有的 .tif 或 .png 图像 (Vaihingen 数据集通常是 .tif)
image_paths = glob.glob(os.path.join(folder_path, '*.tif'))
image_paths.extend(glob.glob(os.path.join(folder_path, '*.png')))

if not image_paths:
    print(f"在 {folder_path} 目录下未找到图像，请检查路径是否正确。")
else:
    # 增量计算初始化，避免将所有图像载入内存
    total_pixels = 0
    sum_vals = None
    sum_sq_vals = None

    print(f"共找到 {len(image_paths)} 张图像，开始计算光谱统计信息...")

    for path in image_paths:
        # cv2.IMREAD_UNCHANGED 保证能够正确读取 16-bit 深度或多通道遥感图像
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
            
        # 如果是单通道图，补充通道维度以统一处理逻辑
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        h, w, c = img.shape
        num_pixels = h * w
        
        # 根据图像的通道数初始化累加器
        if sum_vals is None:
            sum_vals = np.zeros(c, dtype=np.float64)
            sum_sq_vals = np.zeros(c, dtype=np.float64)
            
        # 转为 float64 以防止在平方或求和时发生数据溢出
        img_float = img.astype(np.float64)
        
        # 按空间维度 (H, W) 对每个通道求和与平方和
        sum_vals += np.sum(img_float, axis=(0, 1))
        sum_sq_vals += np.sum(img_float ** 2, axis=(0, 1))
        total_pixels += num_pixels

    # 2. 计算均值和标准差 (Standard Deviation)
    mean_vals = sum_vals / total_pixels
    
    # 方差公式: Variance = E(X^2) - (E(X))^2
    variance = (sum_sq_vals / total_pixels) - (mean_vals ** 2)
    variance = np.maximum(variance, 0)  # 修正因浮点精度导致的极小负数异常
    std_vals = np.sqrt(variance)

    # 在终端输出计算结果
    print("-" * 40)
    print("Spectral Values Distribution:")
    xticks = ["IR", "R", "G"]
    for i in range(len(mean_vals)):
        # 提示：Vaihingen 数据集通道通常对应 IR, R, G
        print(f"Band/Channel {i}: Mean = {mean_vals[i]:.2f}, Std = {std_vals[i]:.2f}")
    print("-" * 40)

    # 3. 仿照 Figure 3(b) 绘制均值和标准差光谱统计图
    plt.figure(figsize=(8, 6))
    
    channels = [f'{xticks[i]} ' for i in range(len(mean_vals))]
    
    # 设置常用的遥感波段可视化颜色 (可根据实际 IR-R-G 进行调整)
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd'][:len(mean_vals)]

    # 绘制带有误差棒的条形图
    bars = plt.bar(channels, mean_vals, yerr=std_vals, capsize=8, 
                   color=colors, alpha=0.75, edgecolor='black', 
                   width=0.4, linewidth=1.2)

    # 完善图表信息
    plt.ylim(0, np.max(mean_vals + std_vals) * 1.2)  # 动态设置 y 轴范围以适应数据
    plt.xlabel('Spectral Bands', fontsize=12, fontweight='bold')
    plt.ylabel('Spectral Values (Pixel Intensity)', fontsize=12, fontweight='bold')
    plt.title('Spectral Statistics (Mean & Standard Deviation)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 在柱状图上方标注出具体的 Mean 和 Std 数值
    for bar, m, s in zip(bars, mean_vals, std_vals):
        y_offset = m + s + (np.max(mean_vals) * 0.05) # 动态设置文本高度以避免重叠
        plt.text(bar.get_x() + bar.get_width() / 2, y_offset, 
                 f'Mean: {m:.1f}\nStd: {s:.1f}', 
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    
    # 保存与展示
    save_path = '/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/assets/spectral_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已成功保存至当前运行目录: {save_path}")
    plt.show()