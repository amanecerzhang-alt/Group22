import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

# --- 全局绘图配置 ---
# 设置中文字体为黑体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_cleaning_tools():
    """
    加载停用词工具：包括外部词表和自定义黑名单
    """
    stopwords = set()
    
    # 构建哈工大停用词文件的绝对路径
    hit_path = os.path.join(current_dir, 'stopwords_hit.txt')
    
    # 读取外部停用词表
    if os.path.exists(hit_path):
        with open(hit_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
        print(f"成功加载停用词表: {hit_path}")
    else:
        print(f"警告：未找到 {hit_path}")
    
    # 针对诈骗话术自定义的动态黑名单：过滤无意义口语、人名及地名
    dynamic_blacklist = {
        '很多', '喜欢', '感觉', '真的', '自己', '知道', '可以', '觉得', 
        '我们', '这个', '这么', '那么', '什么', '就是', '还是', '怎么',
        '非常', '因为', '所以', '如果', '其实', '现在', '时候', '只是',
        '比较', '这种', '一个', '没有', '不是', '这样', '那个', '这里',
        '大家', '一些', '已经', '或者', '开始', '加上', '那么', '怎么',
        '不要', '上午', '下午', '晚上', '明天', '昨天', '现在', '刚刚', 
        '早上', '早晨', '需要', '瓜达拉哈拉', '大卫', '小东', '刘洋', '小涵', '阿图罗', '海伦', '土耳其',
        '布里斯班', '清晨', '黄昏', '刘诗雅', '凌晨', '忽忽', '不会', '使用', '纽约', '贴图', '利雅得',
        '我会', '时间', '叶欣', '布里斯班', '土耳其', '利雅得', '今天'
    }
    stopwords.update(dynamic_blacklist)
    return stopwords

# 初始化全局停用词库
STOPWORDS = load_cleaning_tools()

def clean_text_pure_chinese(text):
    """
    严格清洗逻辑：
    1. 必须是中文
    2. 不能是单字（排除“的”、“了”等）
    3. 不能在停用词表或动态黑名单中
    4. 剔除所有包含数字、英文、符号的非纯中文词汇
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    # 使用 jieba 进行中文分词
    words = jieba.lcut(text)
    
    cleaned = []
    for word in words:
        word = word.strip()
        
        # 长度过滤：排除单字
        if len(word) <= 1:
            continue
        
        # 停用词过滤
        if word in STOPWORDS:
            continue
       
        # 正则表达式过滤：如果词汇中包含非汉字字符（英文、数字、标点），则剔除
        if re.search(r'[^\u4e00-\u9fa5]', word):
            continue
            
        cleaned.append(word)
    return cleaned

def process_and_visualize(full_path, title_prefix):
    """
    数据处理与可视化主函数：生成条形图和词云图
    """
    print(f"\n>>> 正在分析：{title_prefix}")
    
    try:
        # 读取 Excel 文件
        df = pd.read_excel(full_path)
        
        # 自动识别包含“话术”或“内容”关键字的列
        target_cols = [c for c in df.columns if any(k in str(c) for k in ['话术', '内容'])]
        
        all_words = []
        # 遍历目标列进行文本清洗
        for col in target_cols:
            for content in df[col].dropna():
                all_words.extend(clean_text_pure_chinese(content))
        
        if not all_words:
            print(f"{title_prefix} 未提取到有效中文词汇。")
            return

        # 统计词频
        counter = Counter(all_words)
        # 获取频率最高的 10 个词
        top_10 = counter.most_common(10)
        words, counts = zip(*top_10)

        # --- 绘图 1：水平条形图 ---
        plt.figure(figsize=(10, 8))
        # words[::-1] 确保频率最高的词排在最上方
        plt.barh(words[::-1], counts[::-1], color='steelblue', edgecolor='black')
        plt.title(f'{title_prefix} - 核心词频 Top 10', fontsize=15)
        plt.xlabel('出现次数')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # --- 绘图 2：词云图 ---
        wc = WordCloud(
            font_path='simhei.ttf', # 指定中文字体文件
            width=1000, height=600, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(counter)

        plt.figure(figsize=(12, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off') # 隐藏坐标轴
        plt.title(f'{title_prefix} - 诈骗特征词云', fontsize=15)
        plt.show()

    except Exception as e:
        print(f"处理出错：{e}")

if __name__ == "__main__":
    # 定义待处理的文件列表及对应的阶段标题
    files = [
        ("_合并_01_准备与引流(1).xlsx", "阶段1：准备与引流"),
        ("_合并_02_建立信任与诱导(1).xlsx", "阶段2：建立信任与诱导"),
        ("_合并_03_交易与收割(1).xlsx", "阶段3：交易与收割")
    ]

    # 循环处理每个阶段的文件
    for file_name, prefix in files:
        full_path = os.path.join(current_dir, file_name)
        
        if os.path.exists(full_path):
            process_and_visualize(full_path, prefix)
        else:
            print(f"找不到文件：{full_path}")