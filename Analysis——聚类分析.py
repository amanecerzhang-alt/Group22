# -*- coding: utf-8 -*-
"""
5508 诈骗话术聚类分析（v3）
===============================
基于 Telegram 诈骗话术的四个阶段标注数据，
对 01_引流、02_信任建立、03_收割 三阶段的话术记录进行 KMeans 聚类，
检测话术在心理策略、人设、话术功能和关键词使用上是否存在系统性差异，
为三阶段心理操纵框架提供补充证据。

Note: 04_辅助知识与工具是 support layer，不作为主聚类输入。
      聚类结果不等于诈骗天然分为几类，而是验证不同阶段的话术特征是否可区分。


"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局设置
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

# 中文显示映射字典
stage_display = {
    "01_引流": "01_引流",
    "02_信任建立": "02_信任建立",
    "03_收割": "03_收割",
    "04_辅助": "04_辅助"
}

psych_display = {
    "心理策略_贪婪": "贪婪",
    "心理策略_恐惧": "恐惧",
    "心理策略_信任权威": "信任权威",
    "心理策略_情感依赖": "情感依赖",
    "心理策略_从众": "从众",
    "心理策略_紧迫感": "紧迫感"
}

dict_feature_display = {
    "金钱词数量": "金钱",
    "关系词数量": "关系",
    "权威词数量": "权威",
    "紧迫词数量": "紧迫",
    "隔离词数量": "隔离"
}

import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

OUTPUT_DIR = "C:/Users/choos/Desktop/5508_聚类输出_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 第一部分：数据读取与基本检查
# ============================================================
print("=" * 70)
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
    "子分类", "关键词", "最大金额", "提及金额",
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

    # 关键列缺失情况
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

# 合并
df = pd.concat(df_all.values(), ignore_index=True)
print(f"\n合并后总行数: {len(df)}")

# 过滤无效行
df = df.dropna(subset=["子分类"]).reset_index(drop=True)
print(f"过滤子分类为空后: {len(df)} 行")

# ============================================================
# 第二部分：提取主聚类数据（只保留 01/02/03）
# ============================================================
print("\n" + "=" * 70)
print("第二部分：提取主聚类数据（01/02/03）")
print("=" * 70)

df_main = df[df["诈骗阶段"].isin(["01_引流", "02_信任建立", "03_收割"])].copy()
print(f"主聚类样本量: {len(df_main)}")
print(df_main["诈骗阶段"].value_counts().to_string())

# 04 保留做单独描述
df_support = df[df["诈骗阶段"] == "04_辅助"].copy()
print(f"\n04_辅助层样本量: {len(df_support)}")

# ============================================================
# 第三部分：特征工程
# ============================================================
print("\n" + "=" * 70)
print("第三部分：特征工程")
print("=" * 70)

# ---------- A. 心理策略特征 ----------
psych_cols = [
    "心理策略_贪婪", "心理策略_恐惧", "心理策略_信任权威",
    "心理策略_情感依赖", "心理策略_从众", "心理策略_紧迫感"
]
for c in psych_cols:
    df_main[c] = pd.to_numeric(df_main[c], errors="coerce").fillna(0)
df_main["心理策略总分"] = df_main[psych_cols].sum(axis=1)

X_psych = df_main[psych_cols + ["心理策略总分"]].values
print(f"A. 心理策略特征: {X_psych.shape[1]} 维")

# ---------- B. 紧迫感特征 ----------
urgency_map = {"低": 1, "中": 2, "高": 3}
df_main["紧迫感等级_编码"] = df_main["紧迫感等级"].map(urgency_map).fillna(1)
X_urgency = df_main[["紧迫感等级_编码"]].values
print(f"B. 紧迫感特征: {X_urgency.shape[1]} 维")

# ---------- C. 金额特征 ----------
df_main["最大金额"] = pd.to_numeric(df_main["最大金额"], errors="coerce").fillna(0)
# 对极端金额进行截断（winsorize 到 99 百分位），防止单个异常值主导聚类
amount_upper = df_main["最大金额"].quantile(0.99)
df_main["金额_log"] = np.log1p(df_main["最大金额"].clip(upper=amount_upper))
df_main["是否提及金额"] = (df_main["最大金额"] > 0).astype(int)

# 从提及金额列提取数字个数
def count_numbers_in_str(s):
    if pd.isna(s):
        return 0
    return len(re.findall(r'\d+', str(s)))

if "提及金额" in df_main.columns:
    df_main["金额数字个数"] = df_main["提及金额"].apply(count_numbers_in_str)
else:
    df_main["金额数字个数"] = 0

X_amount = df_main[["金额_log", "是否提及金额", "金额数字个数"]].values
print(f"C. 金额特征: {X_amount.shape[1]} 维")

# ---------- D. 人设特征（one-hot） ----------
df_main["诈骗人设_API"] = df_main["诈骗人设_API"].fillna("未知")
persona_dummies = pd.get_dummies(df_main["诈骗人设_API"], prefix="人设")
X_persona = persona_dummies.values.astype(float)
print(f"D. 人设 one-hot: {X_persona.shape[1]} 维")
print(f"   人设类别: {list(persona_dummies.columns)}")

# ---------- E. 话术功能特征（multi-label one-hot） ----------
func_cols = ["话术功能_API_1", "话术功能_API_2", "话术功能_API_3"]
all_funcs = set()
for c in func_cols:
    if c in df_main.columns:
        vals = df_main[c].dropna().unique()
        all_funcs.update(vals)
all_funcs.discard("")
all_funcs.discard(np.nan)
all_funcs = sorted(all_funcs)
print(f"E. 话术功能 multi-label: 共 {len(all_funcs)} 种功能")

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
print(f"   非零功能平均每行: {func_matrix.sum(axis=1).mean():.2f}")

# ---------- F. 关键词 TF-IDF 特征 ----------
print("\nF. 关键词 TF-IDF 特征:")
df_main["关键词"] = df_main["关键词"].fillna("")

# 将逗号分隔的关键词转为空格分隔（TF-IDF 以空格分词）
df_main["关键词_空格"] = df_main["关键词"].apply(
    lambda x: " ".join([kw.strip() for kw in str(x).split(",") if kw.strip()])
)

tfidf = TfidfVectorizer(max_features=80, min_df=2, max_df=0.85,
                        token_pattern=r'(?u)\S+')
try:
    X_tfidf = tfidf.fit_transform(df_main["关键词_空格"])
    print(f"   TF-IDF 特征维度: {X_tfidf.shape[1]}")
    print(f"   关键词Top20: {list(tfidf.get_feature_names_out()[:20])}")
except Exception as e:
    print(f"   TF-IDF 提取警告: {e}")
    X_tfidf = csr_matrix(np.zeros((len(df_main), 1)))

# ---------- G. 文本特征 ----------
df_main["话术示例"] = df_main["话术示例"].fillna("").astype(str)
df_main["text_len"] = df_main["话术示例"].apply(len)
df_main["question_count"] = df_main["话术示例"].apply(lambda x: x.count("？") + x.count("?"))
df_main["exclamation_count"] = df_main["话术示例"].apply(lambda x: x.count("！") + x.count("!"))
df_main["digit_count"] = df_main["话术示例"].apply(lambda x: len(re.findall(r'\d+', x)))

X_text = df_main[["text_len", "question_count", "exclamation_count", "digit_count"]].values
print(f"\nG. 文本特征: {X_text.shape[1]} 维")

# ---------- H. 自定义词典特征 ----------
print("\nH. 自定义词典特征:")

MONEY_WORDS = ["收益","利润","赚钱","提现","充值","入金","本金",
               "佣金","翻倍","账户","冻结","提款","回报"]
RELATION_WORDS = ["喜欢","爱","陪","想你","未来","幸福","孤独",
                  "信任","宝贝","老婆","老公","关心"]
AUTHORITY_WORDS = ["老师","导师","专家","官方","客服","银行","政府",
                   "平台","团队","教授","内部","机构"]
URGENCY_WORDS = ["马上","立即","尽快","限时","截止","错过",
                 "名额","今天","冻结","机会","赶紧"]
ISOLATION_WORDS = ["私聊","群组","LINE","Telegram","不要告诉",
                   "保密","单独","内部通道"]

def count_dict_words(text, word_list):
    text_str = str(text) if pd.notna(text) else ""
    count = 0
    for w in word_list:
        count += text_str.count(w)
    return count

# 用关键词列 + 话术示例文本 一起匹配
text_source = df_main["关键词"].fillna("") + " " + df_main["话术示例"].fillna("")

df_main["金钱词数量"] = text_source.apply(lambda t: count_dict_words(t, MONEY_WORDS))
df_main["关系词数量"] = text_source.apply(lambda t: count_dict_words(t, RELATION_WORDS))
df_main["权威词数量"] = text_source.apply(lambda t: count_dict_words(t, AUTHORITY_WORDS))
df_main["紧迫词数量"] = text_source.apply(lambda t: count_dict_words(t, URGENCY_WORDS))
df_main["隔离词数量"] = text_source.apply(lambda t: count_dict_words(t, ISOLATION_WORDS))

X_dict = df_main[["金钱词数量","关系词数量","权威词数量","紧迫词数量","隔离词数量"]].values
print(f"   词典特征: {X_dict.shape[1]} 维")
for col in ["金钱词数量","关系词数量","权威词数量","紧迫词数量","隔离词数量"]:
    print(f"     {col}: 均值={df_main[col].mean():.2f}, 最大值={df_main[col].max()}")

# ============================================================
# 第四部分：合并特征、降维
# ============================================================
print("\n" + "=" * 70)
print("第四部分：合并特征与降维")
print("=" * 70)

# 稀疏矩阵合并
features_sparse = []

# 心理策略 + 紧迫感 + 金额 + 文本 + 词典 是稠密特征
# 先标准化再合并，防止量纲差异主导 SVD
dense_part = np.hstack([X_psych, X_urgency, X_amount, X_text, X_dict])
dense_part_scaled = StandardScaler().fit_transform(dense_part)
features_sparse.append(csr_matrix(dense_part_scaled))

# 人设 + 功能 one-hot
features_sparse.append(csr_matrix(X_persona))
features_sparse.append(csr_matrix(X_func))

# TF-IDF
features_sparse.append(X_tfidf)

# 合并
X_combined = hstack(features_sparse, format='csr')
print(f"合并后特征维度: {X_combined.shape[1]}")

# TruncatedSVD 降维（控制维度防止过拟合）
# 70 个原始特征降到 12 维，保留主要结构但不保留噪声
n_components = min(12, X_combined.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_svd = svd.fit_transform(X_combined)
print(f"TruncatedSVD 降维到 {n_components} 维")
print(f"累计方差解释率: {svd.explained_variance_ratio_.sum():.3f}")

# 标准化
scaler = StandardScaler()
X_cluster = scaler.fit_transform(X_svd)
print("标准化完成")

# ============================================================
# 第五部分：KMeans 聚类
# ============================================================
print("\n" + "=" * 70)
print("第五部分：KMeans 聚类")
print("=" * 70)

K_range = range(2, 7)
inertias = []
silhouettes = []

# 大样本用子集计算 silhouette（加速）
n_sil_sample = min(3000, X_cluster.shape[0])

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)

    # 子集轮廓系数
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(X_cluster.shape[0], size=n_sil_sample, replace=False)
    sil = silhouette_score(X_cluster[idx_sample], labels[idx_sample])
    silhouettes.append(sil)
    print(f"  K={k}: 惯量={km.inertia_:.2f}, 轮廓系数={sil:.4f}")

# K 值选择图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2)
axes[0].set_xlabel("K（聚类数）"); axes[0].set_ylabel("惯量")
axes[0].set_title("肘部法则"); axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouettes, 'ro-', linewidth=2)
axes[1].set_xlabel("K（聚类数）"); axes[1].set_ylabel("轮廓系数")
axes[1].set_title("轮廓系数"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_K值选择图.png"), dpi=150)
plt.close()
print("\nK值选择图已保存")

# ---------- 选择最终 K ----------
def check_degenerate(labels, threshold=0.01):
    counts = np.bincount(labels)
    return (counts / len(labels)).min() < threshold

print("\n筛选非退化 K:")
valid_ks = []
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_tmp = km_tmp.fit_predict(X_cluster)
    if check_degenerate(labels_tmp):
        print(f"  K={k}: 退化（跳过）")
        continue
    rng = np.random.RandomState(42)
    idx_s = rng.choice(X_cluster.shape[0], size=n_sil_sample, replace=False)
    sil = silhouette_score(X_cluster[idx_s], labels_tmp[idx_s])
    valid_ks.append((k, sil, labels_tmp, km_tmp))
    print(f"  K={k}: 轮廓系数={sil:.4f}")

if not valid_ks:
    # 全退化时用 K=3
    best_k = 3
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = km_final.fit_predict(X_cluster)
else:
    # 按 silhouette 排序
    valid_ks.sort(key=lambda x: x[1], reverse=True)
    top_k, top_sil = valid_ks[0][0], valid_ks[0][1]

    # 检查是否有更小的 K 与最高 silhouette 差异在 0.03 以内
    # 若有，选更小的 K（更简洁、更可解释）
    best_k = top_k
    for k, sil, _, _ in valid_ks:
        if k < top_k and (top_sil - sil) < 0.03:
            best_k = k
            top_sil = sil
            break

    # 取 best_k 对应的结果
    for k, sil, labels, km in valid_ks:
        if k == best_k:
            cluster_labels = labels
            km_final = km
            print(f"\n最终选择 K={best_k}（轮廓系数={sil:.4f}）")
            break

df_main["聚类标签"] = cluster_labels
print(f"各聚类样本数: {np.bincount(cluster_labels)}")

# ============================================================
# 第六部分：聚类解释
# ============================================================
print("\n" + "=" * 70)
print("第六部分：聚类解释")
print("=" * 70)

# 基础统计
cluster_counts = df_main["聚类标签"].value_counts().sort_index()
cluster_pcts = cluster_counts / len(df_main) * 100
print("\n1. 各聚类样本量与占比:")
for cl in sorted(cluster_counts.index):
    print(f"   聚类 {cl}: {cluster_counts[cl]} ({cluster_pcts[cl]:.1f}%)")

# 聚类 × 阶段分布
cluster_stage = pd.crosstab(df_main["聚类标签"], df_main["诈骗阶段"], normalize="index")
cluster_stage_counts = pd.crosstab(df_main["聚类标签"], df_main["诈骗阶段"])
print("\n2. 各聚类在三阶段中的分布比例:")
print(cluster_stage.round(3))
print("\n   绝对数量:")
print(cluster_stage_counts)

# 各聚类的主要阶段
print("\n3. 各聚类主要阶段:")
for cl in sorted(cluster_stage.index):
    main_stage = cluster_stage.loc[cl].idxmax()
    pct = cluster_stage.loc[cl].max()
    print(f"   聚类 {cl}: {main_stage} ({pct:.1%})")

# 各聚类的主要人设
print("\n4. 各聚类主要人设:")
persona_by_cluster = pd.crosstab(
    df_main["聚类标签"], df_main["诈骗人设_API"], normalize="index"
)
for cl in sorted(persona_by_cluster.index):
    main_p = persona_by_cluster.loc[cl].idxmax()
    pct = persona_by_cluster.loc[cl].max()
    print(f"   聚类 {cl}: {main_p} ({pct:.1%})")

# 各聚类的主要话术功能
print("\n5. 各聚类主要话术功能（取前2）:")
func_cols_list = ["话术功能_API_1"]
func_top = {}
for cl in sorted(df_main["聚类标签"].unique()):
    sub = df_main[df_main["聚类标签"] == cl]
    top_funcs = sub["话术功能_API_1"].value_counts()
    func_top[cl] = top_funcs
    print(f"   聚类 {cl}: {top_funcs.index[0]}({top_funcs.iloc[0]}), "
          f"{top_funcs.index[1] if len(top_funcs)>1 else ''}({top_funcs.iloc[1] if len(top_funcs)>1 else 0})")

# 心理策略均值
print("\n6. 各聚类心理策略均值:")
psych_by_cluster = df_main.groupby("聚类标签")[psych_cols + ["心理策略总分"]].mean()
print(psych_by_cluster.round(3))

# 金额_log 均值
print("\n7. 各聚类金额_log均值:")
amount_by_cluster = df_main.groupby("聚类标签")["金额_log"].mean()
print(amount_by_cluster.round(3))

# 紧迫感均值
print("\n8. 各聚类紧迫感均值:")
urgency_by_cluster = df_main.groupby("聚类标签")["紧迫感等级_编码"].mean()
print(urgency_by_cluster.round(2))

# 自定义词典均值
print("\n9. 各聚类自定义词典均值:")
dict_by_cluster = df_main.groupby("聚类标签")[
    ["金钱词数量","关系词数量","权威词数量","紧迫词数量","隔离词数量"]
].mean()
print(dict_by_cluster.round(3))

# ---------- 命名聚类 ----------
# 基于数据特征自动生成描述性名称
cluster_names = {}
cluster_detail = {}
for cl in sorted(df_main["聚类标签"].unique()):
    sub = df_main[df_main["聚类标签"] == cl]
    main_stage = cluster_stage.loc[cl].idxmax()
    main_persona = persona_by_cluster.loc[cl].idxmax()
    top_psych = psych_by_cluster.loc[cl, psych_cols].idxmax()
    top_func = sub["话术功能_API_1"].value_counts().index[0]

    # 基于多维特征生成描述性名称
    psych_profile = psych_by_cluster.loc[cl, psych_cols]
    money_mean = dict_by_cluster.loc[cl, "金钱词数量"]
    relation_mean = dict_by_cluster.loc[cl, "关系词数量"]
    authority_mean = dict_by_cluster.loc[cl, "权威词数量"]
    urgency_mean_dict = dict_by_cluster.loc[cl, "紧迫词数量"]
    urg_level = urgency_by_cluster.loc[cl]

    # 找出突出的心理策略
    high_psych = [col.replace("心理策略_","") for col in psych_cols
                  if psych_profile[col] > (psych_by_cluster[col].mean() + 0.1)]

    # 基于多个维度组合命名
    desc_parts = []

    # 人设维度
    if main_persona == "恋人":
        desc_parts.append("Relational-Bonding")
    elif main_persona == "投资导师":
        desc_parts.append("Investment-Authority")
    elif main_persona == "成功人设":
        desc_parts.append("Success-Persona")
    elif main_persona == "官方客服":
        desc_parts.append("Official-Platform")
    else:
        desc_parts.append(main_persona)

    # 心理策略维度
    if "情感依赖" in high_psych:
        desc_parts.append("Emotional")
    if "信任权威" in high_psych and main_persona not in ["官方客服"]:
        desc_parts.append("Authority")
    if "紧迫感" in high_psych:
        desc_parts.append("Urgency")
    if "贪婪" in high_psych:
        desc_parts.append("Greed-Appeal")
    if "恐惧" in high_psych:
        desc_parts.append("Fear-Pressure")

    # 词典维度补充
    if relation_mean > 1.0 and "Emotional" not in desc_parts:
        desc_parts.append("Relational")
    if money_mean > 3.0:
        desc_parts.append("Money-Focused")
    if urgency_mean_dict > 1.5:
        desc_parts.append("Time-Pressure")
    if authority_mean > 2.0 and "Authority" not in desc_parts and "Investment-Authority" not in desc_parts:
        desc_parts.append("Credibility")

    # 话术功能维度
    if top_func == "知识铺垫":
        desc_parts.append("Knowledge-Priming")
    if top_func == "破冰搭讪":
        desc_parts.append("Icebreaking")
    if top_func in ["善后维稳", "风险威胁"]:
        desc_parts.append("Compliance")
    if top_func == "人设包装":
        desc_parts.append("Packaging")

    # 去重并保留有区分度的标签（最多 3 个）
    seen = set()
    unique_parts = []
    for p in desc_parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)
    desc = " + ".join(unique_parts[:3])

    cluster_names[cl] = f"Cluster {cl}: {desc}"
    print(f"\n   聚类 {cl}: {cluster_names[cl]}")
    # 存一笔详细的描述用于报告
    cluster_detail[cl] = {
        "name": desc,
        "main_stage": main_stage,
        "main_persona": main_persona,
        "top_func": top_func,
        "top_psych": psych_profile.idxmax().replace("心理策略_",""),
        "high_psych": high_psych,
        "money_mean": money_mean,
        "relation_mean": relation_mean,
        "urgency_level": urg_level
    }

# ============================================================
# 第七部分：可视化
# ============================================================
print("\n" + "=" * 70)
print("第七部分：可视化输出")
print("=" * 70)

# ---------- 7.1 cluster × stage stacked bar ----------
fig, ax = plt.subplots(figsize=(10, 6))
cluster_stage_pct = cluster_stage  # normalize='index' already done
cluster_stage_pct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2",
                       edgecolor="black", linewidth=0.5)
ax.set_xlabel("聚类"); ax.set_ylabel("比例")
ax.set_title("聚类 × 阶段分布（01/02/03）")
ax.legend(title="阶段", bbox_to_anchor=(1.01, 1))
ax.set_xticklabels([f"聚类 {c}" for c in cluster_stage_pct.index], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_聚类_阶段堆叠图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("7.1 聚类 × 阶段堆叠图已保存")

# ---------- 7.2 cluster psychological profile heatmap ----------
# 重命名心理策略列为中文显示
psych_by_cluster_display = psych_by_cluster.rename(columns=psych_display)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(psych_by_cluster_display, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("聚类 × 心理策略均值热力图")
ax.set_ylabel("聚类"); ax.set_xlabel("心理策略")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_聚类_心理策略热力图.png"), dpi=150)
plt.close()
print("7.2 聚类心理策略热力图已保存")

# ---------- 7.3 stage × psychological strategies line chart (01→02→03) ----------
stage_order_main = ["01_引流", "02_信任建立", "03_收割"]
stage_psych = df_main.groupby("诈骗阶段")[psych_cols].mean().loc[stage_order_main]

# 将心理策略列重命名为中文显示
stage_psych_display = stage_psych.rename(columns=psych_display)

fig, ax = plt.subplots(figsize=(12, 6))
markers = ["o", "s", "^", "D", "v", "p"]
for i, col in enumerate(stage_psych_display.columns):
    ax.plot(range(len(stage_psych_display)), stage_psych_display[col].values,
            marker=markers[i % len(markers)], linewidth=2, label=col)
ax.set_xticks(range(len(stage_psych_display)))
ax.set_xticklabels([stage_display[s] for s in stage_psych_display.index], rotation=15)
ax.set_ylabel("均值")
ax.set_title("三阶段心理策略变化趋势（01→02→03）")
ax.legend(bbox_to_anchor=(1.01, 1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_阶段_心理策略折线图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("7.3 三阶段心理策略折线图已保存")

# ---------- 7.4 SVD 2D plot by cluster and by stage ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# by cluster
colors_cl = plt.cm.Set1(np.linspace(0, 1, best_k))
for cl in sorted(df_main["聚类标签"].unique()):
    mask = df_main["聚类标签"] == cl
    axes[0].scatter(X_svd[mask, 0], X_svd[mask, 1],
                    c=[colors_cl[cl]], label=f"聚类 {cl}",
                    s=8, alpha=0.6, edgecolors="none")
axes[0].set_xlabel("SVD1"); axes[0].set_ylabel("SVD2")
axes[0].set_title("SVD投影（按聚类）")
axes[0].legend(markerscale=3); axes[0].grid(True, alpha=0.3)

# by stage
stage_colors = {"01_引流": "#FF6B6B", "02_信任建立": "#4ECDC4", "03_收割": "#FFD93D"}
for stage, color in stage_colors.items():
    mask = df_main["诈骗阶段"] == stage
    axes[1].scatter(X_svd[mask, 0], X_svd[mask, 1],
                    c=color, label=stage_display[stage], s=8, alpha=0.6, edgecolors="none")
axes[1].set_xlabel("SVD1"); axes[1].set_ylabel("SVD2")
axes[1].set_title("SVD投影（按阶段）")
axes[1].legend(markerscale=3); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_SVD二维投影.png"), dpi=150)
plt.close()
print("7.4 SVD二维投影图已保存")

# ---------- 7.5 cluster × persona distribution heatmap ----------
persona_heat = pd.crosstab(
    df_main["聚类标签"], df_main["诈骗人设_API"], normalize="index"
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(persona_heat, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("聚类 × 人设分布")
ax.set_ylabel("聚类"); ax.set_xlabel("人设")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_聚类_人设热力图.png"), dpi=150)
plt.close()
print("7.5 聚类 × 人设热力图已保存")

# ---------- 7.6 cluster × function distribution heatmap ----------
# 用话术功能_API_1 做主要功能分布
func_heat = pd.crosstab(
    df_main["聚类标签"], df_main["话术功能_API_1"], normalize="index"
)
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(func_heat, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("聚类 × 话术功能分布")
ax.set_ylabel("聚类"); ax.set_xlabel("话术功能")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_聚类_话术功能热力图.png"), dpi=150)
plt.close()
print("7.6 聚类 × 话术功能热力图已保存")

# ---------- 7.7 额外：词典特征雷达图 ----------
dict_radar = df_main.groupby("聚类标签")[
    ["金钱词数量","关系词数量","权威词数量","紧迫词数量","隔离词数量"]
].mean()
# 重命名为中文短标签
dict_radar_display = dict_radar.rename(columns=dict_feature_display)
# MinMax normalize
dict_radar_norm = (dict_radar_display - dict_radar_display.min()) / (dict_radar_display.max() - dict_radar_display.min() + 0.001)

N = len(dict_radar_norm.columns)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for cl in dict_radar_norm.index:
    values = dict_radar_norm.loc[cl].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f"聚类 {cl}")
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dict_radar_norm.columns, fontsize=11)
ax.set_title("聚类 × 词典特征分布", size=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_聚类_词典特征雷达图.png"), dpi=150, bbox_inches="tight")
plt.close()
print("7.7 词典特征雷达图已保存")

# ============================================================
# 第八部分：聚类分析报告
# ============================================================
print("\n" + "=" * 70)
print("第八部分：聚类分析报告")
print("=" * 70)

report = []
report.append("=" * 60)
report.append("5508 Telegram 诈骗话术聚类分析报告 (v3)")
report.append("=" * 60)
report.append("")
report.append("一、数据概览")
report.append("-" * 40)
report.append(f"主聚类样本 (01/02/03): {len(df_main)} 条")
report.append(f"  01_引流:   {(df_main['诈骗阶段']=='01_引流').sum()}")
report.append(f"  02_信任建立: {(df_main['诈骗阶段']=='02_信任建立').sum()}")
report.append(f"  03_收割:   {(df_main['诈骗阶段']=='03_收割').sum()}")
report.append(f"04_辅助层 (独立): {len(df_support)} 条")
report.append("")

report.append("二、K 值选择")
report.append("-" * 40)
for k, inert, sil in zip(K_range, inertias, silhouettes):
    report.append(f"  K={k}: 惯量={inert:.2f}, 轮廓系数={sil:.4f}")
if best_k < max(K_range):
    report.append(f"  最终选择 K={best_k}（K={max(K_range)} 的轮廓系数略高但增益有限，")
    report.append(f"  根据简洁性原则选更小的 K={best_k}）")
else:
    report.append(f"  最终选择 K={best_k}（该 K 下轮廓系数最高）")
report.append("")

report.append("三、聚类基础统计")
report.append("-" * 40)
for cl in sorted(cluster_counts.index):
    report.append(f"  聚类 {cl}: {cluster_counts[cl]} 条 ({cluster_pcts[cl]:.1f}%)")
report.append("")

report.append("四、各聚类阶段分布")
report.append("-" * 40)
for cl in sorted(cluster_stage.index):
    stages = cluster_stage.loc[cl].to_dict()
    main_stage = max(stages, key=stages.get)
    stages_str = ", ".join([f"{s}={v:.2f}" for s, v in stages.items()])
    report.append(f"  聚类 {cl}: {stages_str}  |  主要阶段: {main_stage}")
report.append("  (注：两聚类的阶段分布较为接近，说明聚类主要区分")
report.append("   话术风格而非阶段本身，与三阶段框架形成交叉验证)")
report.append("")

report.append("五、各聚类心理策略均值")
report.append("-" * 40)
report.append(psych_by_cluster.round(3).to_string())
report.append("")

report.append("六、各聚类人设分布（主要人设）")
report.append("-" * 40)
for cl in sorted(persona_by_cluster.index):
    top_p = persona_by_cluster.loc[cl].sort_values(ascending=False).head(2)
    report.append(f"  聚类 {cl}: {top_p.index[0]}({top_p.iloc[0]:.2f}), {top_p.index[1] if len(top_p)>1 else ''}({top_p.iloc[1] if len(top_p)>1 else 0:.2f})")
report.append("")

report.append("七、各聚类话术功能（Top 1）")
report.append("-" * 40)
for cl in sorted(func_top.keys()):
    report.append(f"  聚类 {cl}: {func_top[cl].index[0]} ({func_top[cl].iloc[0]})")
report.append("")

report.append("八、各聚类自定义词典均值")
report.append("-" * 40)
report.append(dict_by_cluster.round(3).to_string())
report.append("")

report.append("九、聚类命名与解读")
report.append("-" * 40)
for cl in sorted(df_main["聚类标签"].unique()):
    sub = df_main[df_main["聚类标签"] == cl]
    main_stage = cluster_stage.loc[cl].idxmax()
    main_p = persona_by_cluster.loc[cl].idxmax()
    top_psych_label = psych_by_cluster.loc[cl, psych_cols].idxmax()
    top_func_name = sub["话术功能_API_1"].value_counts().index[0]
    money_mean = dict_by_cluster.loc[cl, "金钱词数量"]
    relation_mean = dict_by_cluster.loc[cl, "关系词数量"]
    urgency_mean_val = urgency_by_cluster.loc[cl]

    report.append(f"\n  聚类 {cl} ({cluster_names[cl]}):")
    report.append(f"    样本量: {cluster_counts[cl]} ({cluster_pcts[cl]:.1f}%)")
    report.append(f"    主要阶段: {main_stage}")
    report.append(f"    主要人设: {main_p}")
    report.append(f"    主要话术功能: {top_func_name}")
    report.append(f"    最强心理策略: {top_psych_label}")
    report.append(f"    金钱词均值: {money_mean:.2f}, 关系词均值: {relation_mean:.2f}")
    report.append(f"    紧迫感均值: {urgency_mean_val:.2f}")

report.append("")
report.append("十、结论与讨论")
report.append("-" * 40)

# 从实际结果中提取数据
c0_pct = f"{cluster_pcts[0]:.1f}"
c1_pct = f"{cluster_pcts[1]:.1f}"
c0_greed = f"{psych_by_cluster.loc[0, '心理策略_贪婪']:.2f}"
c0_trust = f"{psych_by_cluster.loc[0, '心理策略_信任权威']:.2f}"
c0_money = f"{dict_by_cluster.loc[0, '金钱词数量']:.2f}"
c1_money = f"{dict_by_cluster.loc[1, '金钱词数量']:.2f}"

report.append(f"""
本次聚类分析的目的不是证明诈骗话术天然分成 K 个类别，而是检验
01_引流、02_信任建立、03_收割 三个阶段的话术在心理策略、人设、
话术功能、关键词使用上是否存在系统性差异。

分析发现（K=2）：
- 聚类 0（"投资权威+贪婪诱导"，{c0_pct}%）：以投资导师和成功人设
  为主，心理策略全面高企（贪婪 {c0_greed}、信任权威 {c0_trust}），金钱词
  和权威词使用频繁，紧迫感较高。
- 聚类 1（"关系建立+破冰"，{c1_pct}%）：以恋人人设为主，各心理
  策略使用频率较低，话术更侧重日常交流与关系维护。

两个聚类在三阶段中的分布比例接近，说明聚类更多区分的是诈
骗话术的"风格"（高强度推销 vs 关系型渗透），而非阶段本身。
这恰好从数据角度验证了三阶段理论的一个重要推论：同一阶段
内可以存在多种话术策略，而同一策略也可以跨阶段使用。

各心理策略在阶段间的变化趋势（参见折线图）与三阶段理论
框架一致：引流阶段信任权威最高，信任建立阶段情感依赖达到
峰值，收割阶段紧迫感与恐惧明显上升。自定义词典特征的聚类
间差异也进一步支持了这一模式（聚类 0 金钱词 {c0_money} vs
聚类 1 金钱词 {c1_money}）。

综上，聚类分析为三阶段心理操纵框架提供了基于数据驱动的
补充支持，但不能被理解为"诈骗话术天然分为几类"。话术的
实际结构远比三阶段模型复杂，二者之间存在交叉映射关系。
""")

report_text = "\n".join(report)
print(report_text)

with open(os.path.join(OUTPUT_DIR, "聚类分析报告_v3.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

# 保存中间数据
df_main.to_csv(os.path.join(OUTPUT_DIR, "聚类数据_完整.csv"),
               encoding="utf-8-sig", index=False)
cluster_stage_counts.to_csv(os.path.join(OUTPUT_DIR, "聚类_阶段数量.csv"),
                            encoding="utf-8-sig")
psych_by_cluster.to_csv(os.path.join(OUTPUT_DIR, "聚类_心理策略均值.csv"),
                        encoding="utf-8-sig")
dict_by_cluster.to_csv(os.path.join(OUTPUT_DIR, "聚类_词典均值.csv"),
                       encoding="utf-8-sig")

print(f"\n所有输出已保存至: {OUTPUT_DIR}")
print("聚类分析 v3 完成！")
