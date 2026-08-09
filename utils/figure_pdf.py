import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_single_file(file_path):
    """
    函数 1：读取单个 txt 文件，统计总词数、词表大小，并返回所有句子的长度列表和词汇集合。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return [], set()

    sentence_lengths = []
    all_words = []
    
    # 正则表达式：用于匹配开头的图像文件名（如 train_886_0.tif 或 png 等）并将其剔除
    img_prefix_pattern = re.compile(r'^\S+\.(tif|tiff|png|jpg|jpeg)\s+')

    print(f"正在读取并统计文件: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 1. 移除图像文件名前缀，只保留后面的描述文本
            clean_text = img_prefix_pattern.sub('', line)
            
            # 2. 分词（转换为小写以确保词表统计的准确性）
            words = clean_text.lower().split()
            if words:
                sentence_lengths.append(len(words))
                all_words.extend(words)

    # 计算指标
    total_words = len(all_words)
    vocabulary = set(all_words)
    vocabulary_size = len(vocabulary)
    
    # 打印数值结果
    print("-" * 40)
    print(f"文件: {os.path.basename(file_path)}")
    print(f"Total Words (总词数): {total_words}")
    print(f"Vocabulary Size (词表大小): {vocabulary_size}")
    print("-" * 40)
    
    return sentence_lengths, vocabulary


def plot_vrsbench_style_pdf(sentence_lengths, save_path="sentence_length_pdf.pdf"):
    """
    函数 2：仿照 VRSBench 论文风格，绘制文本长度的概率密度函数 (PDF) 分布图，并保存为 PDF/PNG 格式。
    """
    if not sentence_lengths:
        print("没有文本长度数据，无法绘图。")
        return

    print(f"正在绘制 VRSBench 风格分布图，准备保存至: {save_path}")
    
    # 设置学术论文常用的白色网格背景风格
    sns.set_theme(style="whitegrid")
    
    # 初始化画布大小
    plt.figure(figsize=(6.5, 4.8))
    
    # 绘制直方图 (Histogram) 和核密度曲线 (KDE) 
    # stat="probability" 会将纵轴设为概率密度，binwidth=1 适合离散的单词长度统计
    # color1 = '#1f77b4' # 蓝色，代表第一组数据
    # color2 = '#d62728' # 红色，代表第二组数据
    sns.histplot(
        sentence_lengths, 
        stat="probability", 
        # kde=True, 
        color="#d62728",     # 学术界经典的深蓝色调
        alpha=0.6, 
        binwidth=1, 
        edgecolor="white",
        kde_kws={"bw_adjust": 1.2} # 稍微平滑一下 KDE 曲线
    )
    
    # 仿照 VRSBench 等遥感多模态论文的图表美化配置
    # plt.title("Probability Density Function (PDF) of Sentence Lengths", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Sentence Length (Number of Words)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    
    # 优化布局，防止标签切边
    plt.tight_layout()
    
    # 保存图像（支持 .pdf 或 .png 后缀）
    plt.savefig(save_path, dpi=300)
    print("图表保存成功！")
    plt.show()


# ==========================================
# 使用示例（你可以根据需要修改下面的路径）
# ==========================================
if __name__ == "__main__":
    
    # 替换为你实际的单文件路径
    # target_file = "/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/output_phrase_train_simple.txt"
    # target_file = "/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/output_phrase_train_standard.txt"
    # target_file = "/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/output_phrase_train_complex.txt"


    # target_file = "/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_test_simple.txt"
    # target_file = "/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_test_standard.txt"
    target_file = "/home/icclab/Documents/lqw/DatasetMMF/PotsdamRef/output_phrase_test_complex.txt"
    
    # 1. 统计单个文件
    lengths, vocab = analyze_single_file(target_file)
    # lengths1, vocab = analyze_single_file(target_file)
    # lengths2, vocab = analyze_single_file(target_file)
    # lengths = lengths1 + lengths2
    
    # 2. 仿照 VRSBench 绘图（你可以保存为 .pdf 矢量图，方便直接放进 LaTeX 论文中）
    output_pdf_path = "/home/icclab/Documents/lqw/Referring_Segmentation/ReferringSegFra/assets/output_phrase_test_complex.png"
    plot_vrsbench_style_pdf(lengths, save_path=output_pdf_path)