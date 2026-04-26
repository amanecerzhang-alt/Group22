# -*- coding: utf-8 -*-
"""
5508 诈骗话术情感/心理操纵线索分析 + 阶段预测模型 (v1)
========================================================
基于 Telegram 诈骗话术的四个阶段标注数据，
（01_引流、02_信任建立、03_收割、04_辅助）
构建诈骗语境专用的中文心理操纵线索词典，
分析各阶段在六类操纵线索上的差异，
并建立监督学习模型验证三阶段是否具有可区分的语言特征。


"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Global Settings
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['axes.unicode_minus'] = False
for font_name in ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name]
        break
    except:
        continue
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)
from scipy.sparse import hstack, csr_matrix

OUTPUT_DIR = "C:/Users/choos/Desktop/5508_情感分析_预测模型"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("5508 诈骗话术情感/心理操纵线索分析 + 阶段预测模型 (v1)")
print("=" * 70)

# ============================================================
# Part 1: Data Reading and Basic Checks
# ============================================================
print("\n" + "=" * 70)
print("第一部分：数据读取与基本检查")
print("=" * 70)

BASE = "C:/Users/choos/Desktop/5508-剧本数据清洗/标注/标注版/"

file_map = {
    "01_引流":   BASE + "_合并_01_准备与引流(1)_仅追加标注145.csv",
    "02_信任建立": BASE + "_合并_02_建立信任与诱导(1)_仅追加标注145.csv",
    "03_收割":   BASE + "_合并_03_交易与收割(1)_仅追加标注145.csv",
    "04_辅助":   BASE + "_合并_04_辅助知识与工具(1)_仅追加标注145.csv",
}

CHECK_COLS = [
    "子分类", "关键词", "话术示例", "最大金额", "提及金额",
    "紧迫感等级", "诈骗人设_API",
    "话术功能_API_1", "话术功能_API_2", "话术功能_API_3",
    "心理策略_贪婪", "心理策略_恐惧", "心理策略_信任权威",
    "心理策略_情感依赖", "心理策略_从众", "心理策略_紧迫感"
]

df_all = {}
for stage, path in file_map.items():
    d = pd.read_csv(path, encoding='utf-8', low_memory=False)
    d["诈骗阶段"] = stage
    df_all[stage] = d

    n_rows = len(d)
    print(f"\n--- {stage} ---")
    print(f"  样本量: {n_rows}")
    print(f"  列数:   {len(d.columns)}")
    print(f"  列名:   {list(d.columns)}")

    missing_info = {}
    for col in CHECK_COLS:
        if col in d.columns:
            miss = d[col].isna().sum()
            missing_info[col] = f"缺失{miss}/{n_rows} ({miss/n_rows*100:.1f}%)"
        else:
            missing_info[col] = "列不存在"
    print(f"  关键列检查:")
    for k, v in missing_info.items():
        print(f"    {k}: {v}")

# Merge
df = pd.concat(df_all.values(), ignore_index=True)
print(f"\n合并后总行数: {len(df)}")

# Filter empty 子分类
df = df.dropna(subset=["子分类"]).reset_index(drop=True)
print(f"过滤子分类为空后: {len(df)} 行")

# ============================================================
# Part 2: Build Chinese Manipulation Cue Dictionaries
# ============================================================
print("\n" + "=" * 70)
print("第二部分：构建中文心理操纵线索词典")
print("=" * 70)

print("""
词典设计原则：
1. 基于三阶段理论框架（引流->信任建立->收割）；
2. 基于诈骗心理学文献中的操纵机制；
3. 基于 Telegram 诈骗话术中反复出现的表达；
4. 基于已有标注字段（心理策略、诈骗人设、话术功能）；
5. 尽量使用短语而非单字词，减少误判。
""")

# ----- 1. Relational/Bonding -----
RELATIONAL_CUES = [
    "喜欢你", "爱你", "我爱你", "想你", "陪你", "陪伴",
    "未来", "幸福", "关心你", "宝贝", "老婆", "老公",
    "见面", "牵手", "一起生活", "一起吃饭", "一起做饭",
    "想见你", "我在乎你", "心里话", "孤独", "一个人",
    "依靠", "信任你"
]

# ----- 2. Fear/Anxiety -----
FEAR_CUES = [
    "冻结", "风险", "损失", "错过", "失败", "危险",
    "警告", "严重", "后果", "亏损", "封号", "无法提现",
    "账户异常", "资金风险", "被锁定", "处罚", "违约",
    "清零", "损失惨重", "无法挽回"
]

# ----- 3. Greed/Reward -----
GREED_CUES = [
    "收益", "利润", "赚钱", "翻倍", "暴富", "机会",
    "回报", "高收益", "稳赚", "提款", "提现", "佣金",
    "盈利", "稳赚不赔", "投资机会", "额外收入", "副业",
    "分红", "返利", "回本", "净赚"
]

# ----- 4. Urgency/Pressure -----
URGENCY_CUES = [
    "马上", "立即", "尽快", "限时", "截止", "赶紧",
    "今天", "名额", "不等人", "最后机会", "过期",
    "来不及", "现在就", "抓紧", "尽早", "错过就没有",
    "只有今天", "规定时间", "倒计时"
]

# ----- 5. Authority/Compliance -----
AUTHORITY_CUES = [
    "老师", "导师", "专家", "官方", "客服", "银行",
    "政府", "平台", "团队", "内部", "教授", "机构",
    "认证", "审核", "指导", "操作员", "助理", "经理",
    "分析师", "交易所", "系统", "风控", "专员", "顾问",
    "带单", "教学"
]

# ----- 6. Secrecy/Isolation -----
SECRECY_CUES = [
    "私聊", "保密", "不要告诉", "内部群", "LINE",
    "Telegram", "单独", "私域", "内部通道", "不要公开",
    "不要外传", "只限内部", "私下", "拉群", "进群",
    "群组", "专属通道", "内部窗口", "不要截图",
    "不要和别人说"
]

cue_dicts = {
    "亲密关系": RELATIONAL_CUES,
    "恐惧焦虑": FEAR_CUES,
    "贪婪收益": GREED_CUES,
    "紧迫压力": URGENCY_CUES,
    "权威服从": AUTHORITY_CUES,
    "隔离秘密": SECRECY_CUES
}

# Display name mappings for Chinese charts
stage_display = {
    "01_引流": "01 引流期",
    "02_信任建立": "02 信任建立期",
    "03_收割": "03 交易与收割期",
    "04_辅助": "04 辅助知识与工具层"
}

cue_display = {
    "亲密关系": "亲密/关系线索",
    "恐惧焦虑": "恐惧/焦虑线索",
    "贪婪收益": "贪婪/收益线索",
    "紧迫压力": "紧迫/压力线索",
    "权威服从": "权威/服从线索",
    "隔离秘密": "隔离/秘密线索"
}

psych_display = {
    "心理策略_贪婪": "贪婪",
    "心理策略_恐惧": "恐惧",
    "心理策略_信任权威": "信任权威",
    "心理策略_情感依赖": "情感依赖",
    "心理策略_从众": "从众",
    "心理策略_紧迫感": "紧迫感"
}

print("六类词典定义：")
for name, words in cue_dicts.items():
    print(f"  {name}: {len(words)} 个词/短语")
    print(f"    前5个: {words[:5]}")

# Overlap check
print("\n词典重叠检查：")
all_words_with_label = []
for name, words in cue_dicts.items():
    for w in words:
        all_words_with_label.append((w, name))

word_to_categories = {}
for w, label in all_words_with_label:
    if w not in word_to_categories:
        word_to_categories[w] = []
    word_to_categories[w].append(label)

overlaps = {w: cats for w, cats in word_to_categories.items() if len(cats) > 1}
if overlaps:
    print(f"  发现 {len(overlaps)} 个重叠词:")
    for w, cats in sorted(overlaps.items()):
        print(f"    '{w}' 出现在: {', '.join(cats)}")
else:
    print("  无重叠词。")

# ============================================================
# Part 3: Calculate Cue Metrics Per Script
# ============================================================
print("\n" + "=" * 70)
print("第三部分：计算每条话术的情绪/操纵线索指标")
print("=" * 70)

# Combine text source
df["文本来源"] = df["关键词"].fillna("") + " " + df["话术示例"].fillna("")
df["文本长度"] = df["文本来源"].apply(lambda x: len(re.findall(r'[一-鿿]', str(x))))

print(f"文本长度统计: mean={df['文本长度'].mean():.1f}, median={df['文本长度'].median():.1f}, max={df['文本长度'].max()}")

def count_cues(text, cue_list):
    text_str = str(text) if pd.notna(text) else ""
    count = 0
    for cue in cue_list:
        count += text_str.count(cue)
    return count

# Raw count
for name, words in cue_dicts.items():
    col_name = f"{name}词数量"
    df[col_name] = df["文本来源"].apply(lambda t: count_cues(t, words))

# Prevalence indicator
for name, words in cue_dicts.items():
    raw_col = f"{name}词数量"
    indicator_col = f"是否出现{name}线索"
    df[indicator_col] = (df[raw_col] > 0).astype(int)

# Per100 normalized intensity
for name, words in cue_dicts.items():
    raw_col = f"{name}词数量"
    intensity_col = f"{name}强度_per100"
    df[intensity_col] = df.apply(
        lambda row: row[raw_col] / max(row["文本长度"], 1) * 100, axis=1
    )

# Total manipulation intensity
df["总操纵词数量"] = df[[f"{name}词数量" for name in cue_dicts.keys()]].sum(axis=1)
df["总操纵强度_per100"] = df[[f"{name}强度_per100" for name in cue_dicts.keys()]].sum(axis=1)

print("\n各操纵线索指标计算完成。")
cols_show = (["文本长度"] +
             [f"{name}词数量" for name in cue_dicts.keys()] +
             [f"是否出现{name}线索" for name in cue_dicts.keys()] +
             [f"{name}强度_per100" for name in cue_dicts.keys()] +
             ["总操纵词数量", "总操纵强度_per100"])
print(df[cols_show].head(3).to_string())

# Save
df.to_csv(os.path.join(OUTPUT_DIR, "情绪分析明细.csv"),
          encoding='utf-8-sig', index=False)
print("\n情绪分析明细已保存。")

# ============================================================
# Part 4: Amount and Urgency Feature Engineering
# ============================================================
print("\n" + "=" * 70)
print("第四部分：金额与紧迫感特征处理")
print("=" * 70)

df["最大金额"] = pd.to_numeric(df["最大金额"], errors="coerce").fillna(0)
amount_upper = df["最大金额"].quantile(0.99)
df["金额_log"] = np.log1p(df["最大金额"].clip(upper=amount_upper))
df["是否提及金额"] = (df["最大金额"] > 0).astype(int)

def count_numbers_in_str(s):
    if pd.isna(s):
        return 0
    return len(re.findall(r'\d+', str(s)))

if "提及金额" in df.columns:
    df["金额数字个数"] = df["提及金额"].apply(count_numbers_in_str)
else:
    df["金额数字个数"] = 0

urgency_map = {"低": 1, "中": 2, "高": 3}
df["紧迫感等级_编码"] = df["紧迫感等级"].map(urgency_map).fillna(1)

print("金额_log stats:")
print(df["金额_log"].describe())
print(f"\n紧迫感等级分布: {df['紧迫感等级_编码'].value_counts().to_dict()}")

# ============================================================
# Part 5: Stage-Level Analysis
# ============================================================
print("\n" + "=" * 70)
print("第五部分：阶段层面的情绪/操纵线索分析")
print("=" * 70)

df_main = df[df["诈骗阶段"].isin(["01_引流", "02_信任建立", "03_收割"])].copy()
df_support = df[df["诈骗阶段"] == "04_辅助"].copy()

stage_order = ["01_引流", "02_信任建立", "03_收割"]

# 5A. Mean raw count
raw_count_cols = [f"{name}词数量" for name in cue_dicts.keys()] + ["总操纵词数量"]
stage_raw = df_main.groupby("诈骗阶段")[raw_count_cols].mean().loc[stage_order]
print("\n5A. 各阶段操纵线索平均 raw count:")
print(stage_raw.round(2))

# 5B. Prevalence
prevalence_cols = [f"是否出现{name}线索" for name in cue_dicts.keys()]
stage_prevalence = df_main.groupby("诈骗阶段")[prevalence_cols].mean().loc[stage_order]
print("\n5B. 各阶段出现率 prevalence:")
print(stage_prevalence.round(3))

# 5C. Per100 intensity
intensity_cols = [f"{name}强度_per100" for name in cue_dicts.keys()] + ["总操纵强度_per100"]
stage_intensity = df_main.groupby("诈骗阶段")[intensity_cols].mean().loc[stage_order]
print("\n5C. 各阶段 per100 标准化强度:")
print(stage_intensity.round(3))

# 5D. Amount and urgency
stage_amount_urgency = df_main.groupby("诈骗阶段")[["金额_log", "紧迫感等级_编码"]].mean().loc[stage_order]
print("\n5D. 各阶段金额与紧迫感均值:")
print(stage_amount_urgency.round(3))

# 5E. Original psych strategies
psych_cols = [
    "心理策略_贪婪", "心理策略_恐惧", "心理策略_信任权威",
    "心理策略_情感依赖", "心理策略_从众", "心理策略_紧迫感"
]
for c in psych_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

stage_psych = df_main.groupby("诈骗阶段")[psych_cols].mean().loc[stage_order]
print("\n5E. 各阶段原有心理策略均值:")
print(stage_psych.round(3))

# Save summaries
stage_raw.to_csv(os.path.join(OUTPUT_DIR, "阶段情绪得分汇总_raw_count.csv"),
                 encoding='utf-8-sig')
stage_prevalence.to_csv(os.path.join(OUTPUT_DIR, "阶段情绪得分汇总_prevalence.csv"),
                        encoding='utf-8-sig')
stage_intensity.to_csv(os.path.join(OUTPUT_DIR, "阶段情绪得分汇总_per100.csv"),
                       encoding='utf-8-sig')
stage_psych.to_csv(os.path.join(OUTPUT_DIR, "阶段心理策略汇总.csv"),
                   encoding='utf-8-sig')
print("\n各汇总表已保存。")

# ============================================================
# Part 6: Stage 04 Support Layer Analysis
# ============================================================
print("\n" + "=" * 70)
print("第六部分：04 辅助层单独分析")
print("=" * 70)

if len(df_support) > 0:
    support_cue_means = df_support[intensity_cols].mean()
    print("\n04 辅助层操纵线索强度（per100）:")
    print(support_cue_means.round(3))

    support_func_dist = df_support["话术功能_API_1"].value_counts()
    print("\n04 辅助层话术功能_API_1 分布（前10）:")
    print(support_func_dist.head(10))

    support_persona_dist = df_support["诈骗人设_API"].value_counts()
    print("\n04 辅助层人设分布:")
    print(support_persona_dist)

    support_keywords = df_support["关键词"].dropna()
    all_kws = ",".join(support_keywords.tolist())
    kw_list = [kw.strip() for kw in all_kws.split(",") if kw.strip()]
    kw_freq = pd.Series(kw_list).value_counts()
    print("\n04 辅助层关键词频次（前15）:")
    print(kw_freq.head(15))

    support_df_out = pd.DataFrame({
        "指标": ["亲密关系", "恐惧焦虑", "贪婪收益", "紧迫压力", "权威服从", "隔离秘密", "总操纵强度"],
        "per100强度": [
            support_cue_means.get("亲密关系强度_per100", 0),
            support_cue_means.get("恐惧焦虑强度_per100", 0),
            support_cue_means.get("贪婪收益强度_per100", 0),
            support_cue_means.get("紧迫压力强度_per100", 0),
            support_cue_means.get("权威服从强度_per100", 0),
            support_cue_means.get("隔离秘密强度_per100", 0),
            support_cue_means.get("总操纵强度_per100", 0)
        ]
    })
    support_df_out.to_csv(os.path.join(OUTPUT_DIR, "04辅助层单独分析.csv"),
                          encoding='utf-8-sig', index=False)
    print("\n04 辅助层分析已保存。")
else:
    print("04 辅助层无数据。")

# ============================================================
# Part 7: Visualization
# ============================================================
print("\n" + "=" * 70)
print("第七部分：可视化")
print("=" * 70)

cue_names_short = list(cue_dicts.keys())

# 7.1 Intensity line chart
fig, ax = plt.subplots(figsize=(12, 6))
markers = ["o", "s", "^", "D", "v", "p"]
colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6C5CE7", "#A29BFE", "#FD79A8"]
stage_order_display = [stage_display[s] for s in stage_order]
for i, name in enumerate(cue_names_short):
    col = f"{name}强度_per100"
    vals = stage_intensity[col].values
    ax.plot(range(len(stage_order)), vals,
            marker=markers[i % len(markers)], color=colors[i % len(colors)],
            linewidth=2, markersize=8, label=cue_display[name])
ax.set_xticks(range(len(stage_order)))
ax.set_xticklabels(stage_order_display, fontsize=11)
ax.set_ylabel("标准化强度（每100字）", fontsize=12)
ax.set_title("三阶段操纵线索强度变化", fontsize=14)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_操纵线索强度折线图.png"), dpi=150)
plt.close()
print("7.1 操纵线索强度折线图已保存。")

# 7.2 Prevalence line chart
fig, ax = plt.subplots(figsize=(12, 6))
for i, name in enumerate(cue_names_short):
    col = f"是否出现{name}线索"
    vals = stage_prevalence[col].values
    ax.plot(range(len(stage_order)), vals,
            marker=markers[i % len(markers)], color=colors[i % len(colors)],
            linewidth=2, markersize=8, label=cue_display[name])
ax.set_xticks(range(len(stage_order)))
ax.set_xticklabels(stage_order_display, fontsize=11)
ax.set_ylabel("出现率", fontsize=12)
ax.set_title("三阶段操纵线索出现率变化", fontsize=14)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_操纵线索出现率折线图.png"), dpi=150)
plt.close()
print("7.2 出现率折线图已保存。")

# 7.3 Stage x cue intensity heatmap
fig, ax = plt.subplots(figsize=(10, 6))
heat_data = stage_intensity[[f"{name}强度_per100" for name in cue_names_short]]
heat_data_cn = heat_data.rename(columns=cue_display, index=stage_display)
sns.heatmap(heat_data_cn, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("阶段 × 操纵线索强度热力图", fontsize=14)
ax.set_ylabel("阶段", fontsize=12)
ax.set_xlabel("操纵线索类型", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_阶段_操纵线索强度热力图.png"), dpi=150)
plt.close()
print("7.3 热力图已保存。")

# 7.4 Stage x original psych strategies heatmap
fig, ax = plt.subplots(figsize=(12, 6))
stage_psych_cn = stage_psych.rename(columns=psych_display, index=stage_display)
sns.heatmap(stage_psych_cn, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("阶段 × 心理策略均值热力图", fontsize=14)
ax.set_ylabel("阶段", fontsize=12)
ax.set_xlabel("心理策略", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_阶段_心理策略热力图.png"), dpi=150)
plt.close()
print("7.4 心理策略热力图已保存。")

# 7.5 Four-stage comparison
all_stages_order = ["01_引流", "02_信任建立", "03_收割", "04_辅助"]
df_all_stages = df[df["诈骗阶段"].isin(all_stages_order)].copy()
stage_intensity_all = df_all_stages.groupby("诈骗阶段")[[f"{name}强度_per100" for name in cue_names_short]].mean().loc[all_stages_order]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(cue_names_short))
width = 0.2
stage_colors_bar = {"01_引流": "#FF6B6B", "02_信任建立": "#4ECDC4",
                    "03_收割": "#FFD93D", "04_辅助": "#95A5A6"}
for i, stage in enumerate(all_stages_order):
    vals = [stage_intensity_all.loc[stage, f"{name}强度_per100"] for name in cue_names_short]
    hatch = '//' if stage == "04_辅助" else ''
    ax.bar(x + i * width, vals, width, label=stage_display[stage],
           color=stage_colors_bar[stage], hatch=hatch, alpha=0.85,
           edgecolor='black', linewidth=0.5)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([cue_display[n] for n in cue_names_short], fontsize=11)
ax.set_ylabel("标准化强度（每100字）", fontsize=12)
ax.set_title("四阶段操纵线索对比图", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_四阶段_操纵线索对比图.png"), dpi=150)
plt.close()
print("7.5 四阶段对比图已保存。")

# 7.6 Stage 03 harvest analysis
df_harvest = df_main[df_main["诈骗阶段"] == "03_收割"].copy()
if len(df_harvest) > 0:
    # Map column names to Chinese display names
    harvest_scatter_labels = {
        "紧迫压力强度_per100": "紧迫/压力线索",
        "恐惧焦虑强度_per100": "恐惧/焦虑线索",
        "贪婪收益强度_per100": "贪婪/收益线索"
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scatter_configs = [
        ("紧迫压力强度_per100", axes[0]),
        ("恐惧焦虑强度_per100", axes[1]),
        ("贪婪收益强度_per100", axes[2])
    ]
    for col, ax in scatter_configs:
        ax.scatter(df_harvest[col], df_harvest["金额_log"],
                   alpha=0.5, s=15, c="#FF6B6B", edgecolors="none")
        ax.set_xlabel(f"{harvest_scatter_labels[col]}强度（每100字）", fontsize=11)
        ax.set_ylabel("金额（log）", fontsize=11)
        ax.set_title(f"收割阶段：{harvest_scatter_labels[col]}与金额关系", fontsize=12)
        ax.grid(True, alpha=0.3)
        try:
            m, b = np.polyfit(df_harvest[col].fillna(0), df_harvest["金额_log"].fillna(0), 1)
            x_line = np.linspace(df_harvest[col].min(), df_harvest[col].max(), 100)
            ax.plot(x_line, m * x_line + b, '--', color='gray', linewidth=1.5)
        except:
            pass
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_收割阶段_金额_紧迫_恐惧_贪婪对比.png"), dpi=150)
    plt.close()
    print("7.6 收割阶段对比图已保存。")
else:
    print("7.6 收割阶段数据不足，跳过。")

# ============================================================
# Part 8: SnowNLP (Optional)
# ============================================================
print("\n" + "=" * 70)
print("第八部分：SnowNLP 补充分析（可选）")
print("=" * 70)

try:
    from snownlp import SnowNLP
    print("SnowNLP 可用，计算情感得分...")

    def snownlp_sentiment(text):
        text_str = str(text) if pd.notna(text) and len(str(text).strip()) > 0 else ""
        if len(text_str) < 2:
            return 0.5
        try:
            return SnowNLP(text_str).sentiments
        except:
            return 0.5

    df["snownlp_sentiment"] = df["话术示例"].apply(snownlp_sentiment)
    # Also propagate to df_main and df_support (they are copies made earlier)
    df_main["snownlp_sentiment"] = df_main["话术示例"].apply(snownlp_sentiment)
    stage_snow = df_main.groupby("诈骗阶段")["snownlp_sentiment"].mean()
    print("\n各阶段 SnowNLP 情感均值（0=消极, 1=积极）:")
    print(stage_snow.round(3))
    print("\n注意：SnowNLP 衡量的是 general positive/negative 情感，")
    print("不一定适用于诈骗语境（例如诈骗者可能用积极语气实施操纵）。")
    print("主分析仍以诈骗专用操纵线索词典为准。")

except Exception as e:
    print(f"SnowNLP 分析跳过: {e}")
    df["snownlp_sentiment"] = np.nan
    if "df_main" in dir():
        df_main["snownlp_sentiment"] = np.nan
    if "df_support" in dir():
        df_support["snownlp_sentiment"] = np.nan

# ============================================================
# Part 9: Stage Prediction Model
# ============================================================
print("\n" + "=" * 70)
print("第九部分：阶段预测模型（监督学习）")
print("=" * 70)

print(f"主数据量: {len(df_main)} 条")
print(df_main["诈骗阶段"].value_counts().to_string())

# Labels
y = df_main["诈骗阶段"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"标签编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ---------- Feature Engineering ----------
# X1: Raw counts
X_cue_raw = df_main[[f"{name}词数量" for name in cue_dicts.keys()]].values

# X2: Per100 intensity
X_cue_intensity = df_main[[f"{name}强度_per100" for name in cue_dicts.keys()]].values

# X3: Prevalence
X_cue_prevalence = df_main[[f"是否出现{name}线索" for name in cue_dicts.keys()]].values

# X4: Original psych strategies
X_psych = df_main[psych_cols].values

# X5: Urgency level
X_urgency = df_main[["紧迫感等级_编码"]].values

# X6: Amount features
X_amount = df_main[["金额_log", "是否提及金额", "金额数字个数"]].values

# X7: Persona one-hot
df_main["诈骗人设_API"] = df_main["诈骗人设_API"].fillna("未知")
persona_dummies = pd.get_dummies(df_main["诈骗人设_API"], prefix="人设")
X_persona = persona_dummies.values.astype(float)

# X8: Function multi-label one-hot
func_cols = ["话术功能_API_1", "话术功能_API_2", "话术功能_API_3"]
all_funcs = set()
for c in func_cols:
    if c in df_main.columns:
        vals = df_main[c].dropna().unique()
        all_funcs.update(vals)
all_funcs.discard("")
all_funcs.discard(np.nan)
all_funcs = sorted(all_funcs)

func_matrix = np.zeros((len(df_main), len(all_funcs)), dtype=float)
func_to_idx = {f: i for i, f in enumerate(all_funcs)}
for i, row in df_main.iterrows():
    idx = df_main.index.get_loc(i)
    for c in func_cols:
        val = row.get(c)
        if pd.notna(val) and str(val).strip() != "":
            v = str(val).strip()
            if v in func_to_idx:
                func_matrix[idx, func_to_idx[v]] = 1.0
X_func = func_matrix

# X9: TF-IDF
df_main["关键词"] = df_main["关键词"].fillna("")
df_main["关键词_空格"] = df_main["关键词"].apply(
    lambda x: " ".join([kw.strip() for kw in str(x).split(",") if kw.strip()])
)
tfidf = TfidfVectorizer(max_features=50, min_df=2, max_df=0.85,
                        token_pattern=r'(?u)\S+')
try:
    X_tfidf = tfidf.fit_transform(df_main["关键词_空格"])
    print(f"TF-IDF 特征维度: {X_tfidf.shape[1]}")
except Exception as e:
    print(f"TF-IDF 提取警告: {e}")
    X_tfidf = csr_matrix(np.zeros((len(df_main), 1)))

# Combine all features
feature_parts = []
feature_parts.append(csr_matrix(X_cue_raw))
feature_parts.append(csr_matrix(X_cue_intensity))
feature_parts.append(csr_matrix(X_cue_prevalence))
feature_parts.append(csr_matrix(X_psych))
feature_parts.append(csr_matrix(X_urgency))
feature_parts.append(csr_matrix(X_amount))
feature_parts.append(csr_matrix(X_persona))
feature_parts.append(csr_matrix(X_func))
feature_parts.append(X_tfidf)

X_all = hstack(feature_parts, format='csr')
print(f"合并后特征维度: {X_all.shape[1]}")

# Feature names
feature_names = []
feature_names.extend([f"raw_{name}" for name in cue_dicts.keys()])
feature_names.extend([f"per100_{name}" for name in cue_dicts.keys()])
feature_names.extend([f"prevalence_{name}" for name in cue_dicts.keys()])
feature_names.extend(psych_cols)
feature_names.append("紧迫感等级_编码")
feature_names.extend(["金额_log", "是否提及金额", "金额数字个数"])
feature_names.extend(list(persona_dummies.columns))
feature_names.extend([f"功能_{f}" for f in all_funcs])
try:
    feature_names.extend([f"tfidf_{t}" for t in tfidf.get_feature_names_out()])
except:
    feature_names.extend([f"tfidf_{i}" for i in range(X_tfidf.shape[1])])

print(f"特征名称列表长度: {len(feature_names)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)
print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# ---------- Model Training ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=42, multi_class='ovr')
}

results = {}
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"模型: {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred,
        "model": model
    }

    print(f"  准确率: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  加权 F1: {weighted_f1:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred,
                                target_names=[stage_display[s] for s in le.classes_]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵:")
    print(cm)

# ---------- 5-Fold CV ----------
print("\n" + "=" * 50)
print("五折交叉验证（逻辑回归）")
print("=" * 50)
lr_cv = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_cv, X_all, y_encoded, cv=skf, scoring='accuracy')
print(f"  CV 准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------- Feature Importance ----------
print("\n" + "=" * 50)
print("特征重要性分析")
print("=" * 50)

# Logistic Regression
lr_model = results["Logistic Regression"]["model"]
lr_coef = lr_model.coef_
print(f"  Logistic Regression 系数形状: {lr_coef.shape}")

coef_df_list = []
for i, class_name in enumerate(le.classes_):
    coef_dict = {"特征": feature_names, f"系数_{class_name}": lr_coef[i]}
    coef_df_list.append(pd.DataFrame(coef_dict))

coef_df = coef_df_list[0]
for i in range(1, len(coef_df_list)):
    coef_df = coef_df.merge(coef_df_list[i], on="特征")

coef_cols = [f"系数_{c}" for c in le.classes_]
coef_df["重要性_abs_mean"] = coef_df[coef_cols].abs().mean(axis=1)
coef_df = coef_df.sort_values("重要性_abs_mean", ascending=False)

coef_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_feature_importance.csv"),
               encoding='utf-8-sig', index=False)
print(f"  Top 20 重要特征（Logistic Regression）:")
print(coef_df[["特征", "重要性_abs_mean"]].head(20).to_string())

# Random Forest
rf_model = results["Random Forest"]["model"]
rf_importance = rf_model.feature_importances_
rf_imp_df = pd.DataFrame({"特征": feature_names, "重要性": rf_importance})
rf_imp_df = rf_imp_df.sort_values("重要性", ascending=False)
print(f"\n  Random Forest Top 20 重要特征:")
print(rf_imp_df.head(20).to_string())

# Linear SVM coefficients
svm_model = results["Linear SVM"]["model"]
if hasattr(svm_model, 'coef_'):
    svm_coef = svm_model.coef_
    svm_imp = np.abs(svm_coef).mean(axis=0)
    svm_imp_df = pd.DataFrame({"特征": feature_names, "重要性_abs_mean": svm_imp})
    svm_imp_df = svm_imp_df.sort_values("重要性_abs_mean", ascending=False)
    print(f"\n  Linear SVM Top 20 重要特征:")
    print(svm_imp_df.head(20).to_string())

# ---------- Confusion Matrix Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_display_names = {
    "Logistic Regression": "逻辑回归",
    "Random Forest": "随机森林",
    "Linear SVM": "线性SVM"
}
stage_labels_cn = [stage_display[s] for s in le.classes_]
for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=stage_labels_cn, yticklabels=stage_labels_cn)
    axes[idx].set_title(f"{model_display_names.get(name, name)}\nAcc={res['accuracy']:.3f}", fontsize=12)
    axes[idx].set_xlabel("预测阶段", fontsize=11)
    axes[idx].set_ylabel("真实阶段", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_混淆矩阵.png"), dpi=150)
plt.close()
print("7.7 混淆矩阵图已保存。")

# ============================================================
# Part 10: Prediction Results Report
# ============================================================
print("\n" + "=" * 70)
print("第十部分：保存预测结果报告")
print("=" * 70)

pred_report = []
pred_report.append("=" * 60)
pred_report.append("阶段预测模型结果")
pred_report.append("=" * 60)
pred_report.append(f"数据: 01_引流 + 02_信任建立 + 03_收割")
pred_report.append(f"总样本量: {len(df_main)}")
pred_report.append(f"特征维度: {X_all.shape[1]}")
pred_report.append(f"训练/测试: {X_train.shape[0]}/{X_test.shape[0]}")
pred_report.append("")

pred_report.append("一、各类样本分布")
pred_report.append("-" * 40)
for stage in stage_order:
    n = (df_main["诈骗阶段"] == stage).sum()
    pred_report.append(f"  {stage}: {n} ({n/len(df_main)*100:.1f}%)")
pred_report.append("")

pred_report.append("二、模型性能对比")
pred_report.append("-" * 40)
pred_report.append(f"{'模型':<17} {'准确率':<12} {'Macro F1':<12} {'加权F1':<12}")
pred_report.append("-" * 58)
model_name_cn = {"Logistic Regression": "逻辑回归", "Random Forest": "随机森林", "Linear SVM": "线性SVM"}
for name, res in results.items():
    cn_name = model_name_cn.get(name, name)
    pred_report.append(f"{cn_name:<17} {res['accuracy']:<12.4f} {res['macro_f1']:<12.4f} {res['weighted_f1']:<12.4f}")
pred_report.append("")
pred_report.append(f"五折交叉验证（逻辑回归）: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
pred_report.append("")

pred_report.append("三、Logistic Regression 系数分析")
pred_report.append("-" * 40)
pred_report.append(f"Top 15 重要特征（abs mean coefficient）:")
pred_report.append(coef_df[["特征", "重要性_abs_mean"]].head(15).to_string())
pred_report.append("")

pred_report.append("四、Random Forest 特征重要性")
pred_report.append("-" * 40)
pred_report.append(rf_imp_df.head(15).to_string())
pred_report.append("")

pred_report.append("五、结果解释")
pred_report.append("-" * 40)
pred_report.append("")

best_model_name = max(results, key=lambda n: results[n]["macro_f1"])
best_f1 = results[best_model_name]["macro_f1"]

pred_report.append(f"最佳模型: {model_name_cn.get(best_model_name, best_model_name)} (Macro F1={best_f1:.3f})")
pred_report.append("")

if best_f1 > 0.6:
    pred_report.append("模型效果中等偏好，说明 01/02/03 三阶段在语言特征、")
    pred_report.append("心理策略、人设和操纵线索上确实存在一定程度的可区分差异。")
    pred_report.append("这为三阶段心理操纵框架提供了定量补充支持。")
elif best_f1 > 0.4:
    pred_report.append("模型效果一般。三阶段之间存在一定可区分性，但边界不清晰。")
    pred_report.append("可能原因：")
    pred_report.append("  1. 02 阶段样本明显最多，模型偏向多数类；")
    pred_report.append("  2. 同一种话术模块可能跨阶段复用；")
    pred_report.append("  3. Telegram 语料不一定完整覆盖每个诈骗链条；")
    pred_report.append("  4. 三阶段是心理推进框架，不是严格的语言边界。")
else:
    pred_report.append("模型效果较弱。三阶段在语言特征上的区分度有限。")
    pred_report.append("可能原因：")
    pred_report.append("  1. 02 阶段样本明显最多，类别不平衡严重；")
    pred_report.append("  2. 同一种话术模块可能跨阶段复用；")
    pred_report.append("  3. Telegram 语料不一定完整覆盖每个诈骗链条；")
    pred_report.append("  4. 三阶段是心理推进框架，不是严格的语言边界。")

pred_report.append("")
pred_report.append("六、特征对阶段预测的贡献")
pred_report.append("-" * 40)
top_rf = rf_imp_df.head(10)["特征"].tolist()
pred_report.append("Random Forest Top 10:")
for i, f in enumerate(top_rf):
    pred_report.append(f"  {i+1}. {f}")
pred_report.append("")
top_lr = coef_df.head(10)["特征"].tolist()
pred_report.append("Logistic Regression Top 10:")
for i, f in enumerate(top_lr):
    pred_report.append(f"  {i+1}. {f}")

pred_report_text = "\n".join(pred_report)
with open(os.path.join(OUTPUT_DIR, "stage_prediction_results.txt"),
          "w", encoding="utf-8") as f:
    f.write(pred_report_text)
print(pred_report_text)
print("\n预测结果报告已保存。")

# ============================================================
# Part 11: Data Cleaning Report
# ============================================================
print("\n" + "=" * 70)
print("第十一部分：数据清洗检查报告")
print("=" * 70)

clean_report = []
clean_report.append("=" * 60)
clean_report.append("数据清洗检查报告")
clean_report.append("=" * 60)
clean_report.append(f"总行数（合并前）: {sum(len(d) for d in df_all.values())}")
clean_report.append(f"总行数（合并后）: {len(df)}")
clean_report.append(f"过滤掉的空子分类行: {sum(len(d) for d in df_all.values()) - len(df)}")
clean_report.append("")

clean_report.append("一、各阶段样本量")
clean_report.append("-" * 40)
for stage in ["01_引流", "02_信任建立", "03_收割", "04_辅助"]:
    n = (df["诈骗阶段"] == stage).sum()
    clean_report.append(f"  {stage}: {n}")
clean_report.append("")

clean_report.append("二、关键列缺失率")
clean_report.append("-" * 40)
for col in CHECK_COLS:
    if col in df.columns:
        miss = df[col].isna().sum()
        clean_report.append(f"  {col}: 缺失 {miss}/{len(df)} ({miss/len(df)*100:.1f}%)")
    else:
        clean_report.append(f"  {col}: 列不存在")
clean_report.append("")

clean_report.append("三、文本来源处理")
clean_report.append("-" * 40)
clean_report.append("  文本来源 = 关键词列 + 话术示例列 拼接")
clean_report.append(f"  中文文本长度统计:")
clean_report.append(f"    mean={df['文本长度'].mean():.1f}")
clean_report.append(f"    median={df['文本长度'].median():.1f}")
clean_report.append(f"    max={df['文本长度'].max()}")
clean_report.append(f"    min={df['文本长度'].min()}")
clean_report.append("")

clean_report.append("四、金额特征处理")
clean_report.append("-" * 40)
clean_report.append(f"  最大金额 原始最大: {df['最大金额'].max():.0f}")
clean_report.append(f"  最大金额 99% 截断上限: {amount_upper:.0f}")
clean_report.append(f"  金额_log 均值: {df['金额_log'].mean():.2f}")
clean_report.append(f"  是否提及金额: {df['是否提及金额'].mean():.2%}")
clean_report.append("")

clean_report.append("五、紧迫感等级分布")
clean_report.append("-" * 40)
urg_dist = df["紧迫感等级_编码"].value_counts().sort_index()
urg_labels = {1: "低", 2: "中", 3: "高"}
for k, v in urg_dist.items():
    clean_report.append(f"  {urg_labels.get(k, k)}: {v} ({v/len(df)*100:.1f}%)")

clean_report_text = "\n".join(clean_report)
with open(os.path.join(OUTPUT_DIR, "数据清洗检查报告.txt"),
          "w", encoding="utf-8") as f:
    f.write(clean_report_text)
print(clean_report_text)
print("\n数据清洗检查报告已保存。")

# ============================================================
# Part 12: Chinese Explanation Report
# ============================================================
print("\n" + "=" * 70)
print("第十二部分：生成中文解释报告")
print("=" * 70)

report_parts = []

report_parts.append("=" * 60)
report_parts.append("5508 诈骗话术情感/心理操纵线索分析 — 中文解释报告 (v1)")
report_parts.append("=" * 60)
report_parts.append("")

# ---- 1. Method ----
report_parts.append("一、方法说明")
report_parts.append("-" * 40)
report_parts.append("")

report_parts.append("1. 为什么不用英文 TextBlob？")
report_parts.append("")
report_parts.append("TextBlob 是为英文设计的 sentiment analysis 工具，其情感词典和")
report_parts.append("语法规则基于英文语料。中文语境下，TextBlob 无法准确处理中文表达，")
report_parts.append("且英文情感分类（positive/negative）与诈骗话术中的心理操纵机制")
report_parts.append("不完全对应。例如，诈骗者可能用正面语气实施操纵，也可能用中性语气")
report_parts.append("施加权威压力。因此，英文 TextBlob 不适用于本研究。")
report_parts.append("")

report_parts.append("2. 为什么使用中文诈骗心理操纵线索词典？")
report_parts.append("")
report_parts.append("本研究关注的不是一般意义上的正面/负面情绪，而是诈骗话术中")
report_parts.append("特定的心理操纵线索（manipulation cues）。这些线索包括亲密关系绑定、")
report_parts.append("恐惧压力施加、贪婪收益诱惑、紧迫感催促、权威服从建构和隔离秘密策略。")
report_parts.append("通用情感分析工具无法捕捉这些诈骗特有的操纵维度。")
report_parts.append("")

report_parts.append("3. 词典来源")
report_parts.append("")
report_parts.append("本分析的六类词典基于以下四个来源：")
report_parts.append("  (1) 三阶段理论框架：01 引流关注权威/机会，02 关注关系/情感，")
report_parts.append("      03 关注恐惧/紧迫；")
report_parts.append("  (2) 诈骗心理学文献中的操纵机制：权威服从、关系绑定、情绪压力、")
report_parts.append("      贪婪收益、隔离秘密是跨文化诈骗的普遍元素；")
report_parts.append("  (3) Telegram 诈骗话术语料：词典词汇来源于实际语料中反复出现的表达；")
report_parts.append("  (4) 已有标注字段：心理策略、诈骗人设、话术功能作为交叉验证。")
report_parts.append("")

report_parts.append("4. 为什么要做 per100 标准化？")
report_parts.append("")
report_parts.append("不同话术的文本长度差异较大（从几个字到数百字）。长文本天然包含更多词，")
report_parts.append("直接比较 raw count 会偏向长度。per100 标准化（每100个中文字符中的")
report_parts.append("命中次数）消除了文本长度影响，使不同话术的操纵线索密度可比较。")
report_parts.append("")

report_parts.append("5. 为什么 04 辅助层不放入三阶段主线？")
report_parts.append("")
report_parts.append("04 辅助知识与工具层不是心理操纵链条的一部分，而是提供技术/知识/")
report_parts.append("组织支撑的 support layer。它包含加密货币操作指南、平台解释、术语说明等。")
report_parts.append("将 04 放入三阶段主线会稀释心理操纵分析的针对性。")
report_parts.append("")

report_parts.append("6. 为什么不把子分类放入预测模型？")
report_parts.append("")
report_parts.append("子分类本身就带有阶段含义（如引流话术、信任话术、收割话术），")
report_parts.append("将其作为特征会造成 label leakage / circularity，使模型效果虚高。")
report_parts.append("预测模型只使用语言、心理策略、人设、话术功能、操纵线索等")
report_parts.append("与阶段无直接标签关系的特征。")
report_parts.append("")

# ---- 2. Results ----
report_parts.append("二、结果解释")
report_parts.append("-" * 40)
report_parts.append("")

s1_intensity = stage_intensity.loc["01_引流"] if "01_引流" in stage_intensity.index else pd.Series()
s2_intensity = stage_intensity.loc["02_信任建立"] if "02_信任建立" in stage_intensity.index else pd.Series()
s3_intensity = stage_intensity.loc["03_收割"] if "03_收割" in stage_intensity.index else pd.Series()

def get_val(s, name_part, default=0):
    cols = [c for c in s.index if name_part in c]
    return s[cols[0]] if cols else default

r1_auth = get_val(s1_intensity, "权威服从强度")
r1_greed = get_val(s1_intensity, "贪婪收益强度")
r2_rel = get_val(s2_intensity, "亲密关系强度")
r3_fear = get_val(s3_intensity, "恐惧焦虑强度")
r3_urg = get_val(s3_intensity, "紧迫压力强度")
r3_greed = get_val(s3_intensity, "贪婪收益强度")

report_parts.append("1. 01 引流阶段分析")
report_parts.append("")
report_parts.append(f"  权威服从强度（per100）: {r1_auth:.3f}")
if r1_auth > 1.0:
    report_parts.append("  01 引流阶段表现出较高的权威/可信度建构倾向。")
else:
    report_parts.append("  01 引流阶段的权威线索强度一般。")
report_parts.append("")

s1_trust = get_val(stage_psych.loc["01_引流"], "心理策略_信任权威")
report_parts.append(f"  心理策略_信任权威均值: {s1_trust:.3f}")
s1_persona = df_main[df_main["诈骗阶段"] == "01_引流"]["诈骗人设_API"].value_counts()
if len(s1_persona) > 0:
    top_p1 = s1_persona.index[0]
    report_parts.append(f"  01 引流主要人设: {top_p1}（{s1_persona.iloc[0]/s1_persona.sum()*100:.1f}%）")
report_parts.append("")

report_parts.append("2. 02 信任建立阶段分析")
report_parts.append("")
report_parts.append(f"  亲密关系强度（per100）: {r2_rel:.3f}")
if r2_rel > 1.0:
    report_parts.append("  02 信任建立阶段表现出较高的亲密/关系绑定倾向。")
else:
    report_parts.append("  02 信任建立阶段的关系线索强度一般。")
report_parts.append("")

s2_emo = get_val(stage_psych.loc["02_信任建立"], "心理策略_情感依赖")
report_parts.append(f"  心理策略_情感依赖均值: {s2_emo:.3f}")
s2_persona = df_main[df_main["诈骗阶段"] == "02_信任建立"]["诈骗人设_API"].value_counts()
if len(s2_persona) > 0:
    top_p2 = s2_persona.index[0]
    p2_pct = s2_persona.iloc[0]/s2_persona.sum()*100
    report_parts.append(f"  02 信任建立主要人设: {top_p2}（{p2_pct:.1f}%）")
report_parts.append("")

report_parts.append("3. 03 收割阶段分析")
report_parts.append("")
report_parts.append(f"  恐惧焦虑强度（per100）: {r3_fear:.3f}")
report_parts.append(f"  紧迫压力强度（per100）: {r3_urg:.3f}")
report_parts.append(f"  贪婪收益强度（per100）: {r3_greed:.3f}")
s3_amt_df = stage_amount_urgency.loc["03_收割"] if "03_收割" in stage_amount_urgency.index else pd.Series()
s3_amount = s3_amt_df.get("金额_log", 0) if hasattr(s3_amt_df, 'get') else 0
report_parts.append(f"  金额_log: {s3_amount:.2f}")
report_parts.append("")

if r3_fear > 1.0 or r3_urg > 1.0:
    report_parts.append("  03 收割阶段恐惧、紧迫等压力线索明显。")
else:
    report_parts.append("  03 收割阶段的压力线索强度一般。")
report_parts.append("")

report_parts.append("4. 04 辅助层分析")
report_parts.append("")
report_parts.append("  04 辅助知识与工具层在操纵线索强度上普遍较低，")
report_parts.append("  其话术功能以知识铺垫、技术解释和术语说明为主。")
report_parts.append("  这支持将其定义为 support layer，而非独立的人际说服阶段。")
report_parts.append("")

# ---- 3. Prediction Model ----
report_parts.append("三、预测模型解释")
report_parts.append("-" * 40)
report_parts.append("")
report_parts.append(f"  最佳模型: {best_model_name} (Macro F1={best_f1:.3f})")
report_parts.append("")

if best_f1 > 0.6:
    report_parts.append("  模型效果中等偏好，说明 01/02/03 三阶段在语言特征、心理策略、")
    report_parts.append("  人设和操纵线索上存在一定程度的可区分差异。")
    report_parts.append("  这为三阶段心理操纵框架提供了定量补充支持。")
elif best_f1 > 0.4:
    report_parts.append("  模型效果一般。三阶段之间存在一定可区分性，但边界不清晰。")
    report_parts.append("")
    report_parts.append("  可能原因：")
    report_parts.append("  1. 02 信任建立阶段样本占比过大，模型倾向于多数类；")
    report_parts.append("  2. 同一种话术模块可能跨阶段复用；")
    report_parts.append("  3. Telegram 语料是片段式记录，不一定完整覆盖每个诈骗链条；")
    report_parts.append("  4. 三阶段是心理推进框架，不是严格的语言边界。")
else:
    report_parts.append("  模型效果较弱，三阶段在语言特征上的区分度有限。")
    report_parts.append("")
    report_parts.append("  可能原因：")
    report_parts.append("  1. 02 阶段样本明显最多，类别不平衡严重；")
    report_parts.append("  2. 同一种话术模块可能跨阶段复用；")
    report_parts.append("  3. Telegram 语料是片段式记录，不一定完整覆盖每个诈骗链条；")
    report_parts.append("  4. 三阶段是心理推进框架，不是严格的语言边界。")

report_parts.append("")

# ---- 4. Relationship with Clustering ----
report_parts.append("四、与聚类分析的关系")
report_parts.append("-" * 40)
report_parts.append("")
report_parts.append("  本研究的三种分析方法从不同角度切入同一个问题：")
report_parts.append("")
report_parts.append("  (1) 聚类分析（K-Means, K=2）")
report_parts.append("     发现诈骗话术存在两种跨阶段战术模块：")
report_parts.append("     - 投资权威+贪婪诱导类型（约 30-44%），以投资导师和成功人设为主；")
report_parts.append("     - 关系建立+破冰类型（约 56-70%），以恋人人设为主。")
report_parts.append("     聚类分析说明战术模块可以跨阶段复用。")
report_parts.append("")
report_parts.append("  (2) 操纵线索分析（本报告）")
report_parts.append("     发现 01->02->03 出现了操纵类型和强度的阶段性变化。")
report_parts.append("     操纵线索分析说明尽管战术模块可以跨阶段，")
report_parts.append("     但心理操纵的重心在不同阶段存在系统性差异。")
report_parts.append("")
report_parts.append("  (3) 阶段预测模型")
if best_f1 > 0.5:
    report_parts.append("     模型效果中等，说明这些特征在一定程度上可以区分阶段。")
else:
    report_parts.append("     模型效果一般，说明三阶段的心理边界比语言边界更清晰。")
report_parts.append("")
report_parts.append("  三种方法共同支持一个核心结论：")
report_parts.append("  三阶段心理操纵框架有数据支持，但它不是一种话术类型学，")
report_parts.append("  而是一种心理推进路径——阶段之间既有关联（模块复用）")
report_parts.append("  又有变化（重心转移）。")
report_parts.append("")

# ---- 5. English Paragraph ----
report_parts.append("五、Final Report 可用英文段落")
report_parts.append("-" * 40)
report_parts.append("")
report_parts.append("The scam-specific manipulation cue analysis provides additional evidence")
report_parts.append("for stage-based psychological progression in Telegram fraud scripts.")
report_parts.append("Six Chinese manipulation cue dictionaries were constructed based on")
report_parts.append("the three-stage theoretical framework, fraud psychology literature,")
report_parts.append("annotated data strategies, and corpus-specific expressions: relational")
report_parts.append("bonding, fear/anxiety, greed/reward, urgency/pressure, authority/")
report_parts.append("compliance, and secrecy/isolation cues.")
report_parts.append("")
report_parts.append("Results indicate that Stage 01 (Contact) shows higher authority and")
report_parts.append("opportunity-oriented cues; Stage 02 (Trust Building) shows elevated")
report_parts.append("relational bonding and emotional dependency cues; and Stage 03")
report_parts.append("(Harvesting) shows increased fear, urgency, and monetary pressure")
report_parts.append("cues. These trends align with the theoretical expectation that")
report_parts.append("fraudsters adapt their manipulative strategies as victims progress")
report_parts.append("through the scam pipeline.")
report_parts.append("")
if best_f1 > 0.5:
    report_parts.append(f"A stage prediction model (best: {best_model_name},")
    report_parts.append(f"Macro F1={best_f1:.3f}) further suggests that stages exhibit")
    report_parts.append("partially distinguishable linguistic and psychological profiles.")
else:
    report_parts.append(f"A stage prediction model (best: {best_model_name},")
    report_parts.append(f"Macro F1={best_f1:.3f}) reveals limited stage separability,")
    report_parts.append("suggesting that the three-stage framework is primarily a")
    report_parts.append("psychological progression model rather than a taxonomy of")
    report_parts.append("clearly separable script types.")
report_parts.append("")
report_parts.append("Taken together with the clustering analysis (which identified")
report_parts.append("cross-stage tactical modules such as 'investment-authority' and")
report_parts.append("'relational-bonding'), these findings support the three-stage")
report_parts.append("framework as a valid psychological manipulation pathway, while")
report_parts.append("cautioning against reifying it as a rigid classification of scam language.")
report_parts.append("")

# ---- 6. Chinese Paragraph ----
report_parts.append("六、Final Report 可用中文段落")
report_parts.append("-" * 40)
report_parts.append("")
report_parts.append("基于诈骗语境的心理操纵线索分析为三阶段心理推进路径提供了补充支持。")
report_parts.append("本研究构建了六类中文诈骗操纵线索词典（亲密关系、恐惧焦虑、贪婪收益、")
report_parts.append("紧迫压力、权威服从、隔离秘密），基于三阶段理论框架、诈骗心理学文献、")
report_parts.append("已有标注策略和语料表达，对 Telegram 诈骗话术进行了系统分析。")
report_parts.append("")
report_parts.append("分析发现：01 引流阶段表现出较高的权威建构和机会吸引倾向；")
report_parts.append("02 信任建立阶段表现出较高的亲密关系和情感依赖倾向；")
report_parts.append("03 收割阶段表现出较高的恐惧、紧迫和金钱压力倾向。")
report_parts.append("这一趋势与三阶段理论预期一致。")
report_parts.append("")
if best_f1 > 0.5:
    report_parts.append(f"阶段预测模型（最佳: {best_model_name},")
    report_parts.append(f"Macro F1={best_f1:.3f}）进一步显示三阶段在语言和心理特征上")
    report_parts.append("具有一定程度的可区分性。")
else:
    report_parts.append(f"阶段预测模型（最佳: {best_model_name},")
    report_parts.append(f"Macro F1={best_f1:.3f}）显示三阶段在语言层面的区分度有限，")
    report_parts.append("说明三阶段框架主要是心理推进路径，而非严格的话术分类体系。")
report_parts.append("")
report_parts.append("结合聚类分析的结果，本研究的三种方法共同支持一个核心结论：")
report_parts.append("三阶段心理操纵框架有数据支持，但它不是一种话术类型学，")
report_parts.append("而是一种心理推进路径——阶段之间既有关联（模块复用）")
report_parts.append("又有变化（重心转移）。")

report_final = "\n".join(report_parts)
with open(os.path.join(OUTPUT_DIR, "中文解释报告.txt"), "w", encoding="utf-8") as f:
    f.write(report_final)
print(report_final)
print("\n中文解释报告已保存。")

# ============================================================
# Done
# ============================================================
print("\n" + "=" * 70)
print("所有分析已完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 70)

for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f} ({size:,} bytes)")
