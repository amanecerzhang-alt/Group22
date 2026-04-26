import pandas as pd
import matplotlib.pyplot as plt

# --- 全局绘图配置 ---
# 设置中文字体为黑体，确保中文标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 

# --- 操作 1: 诈骗人设演变折线图 ---
# 读取存储人设占比数据的 CSV 文件
df1 = pd.read_csv('板块1_诈骗人设演变.csv')
# 将“诈骗阶段”设为索引，以便在横坐标显示阶段名称；plot(marker='o') 用于绘制带圆点的折线图
df1.set_index('诈骗阶段').plot(marker='o', figsize=(10, 6))
plt.title('不同诈骗阶段下的人设占比演变')
plt.ylabel('出现频率 (%)')
# 开启网格线，方便观察占比数值
plt.grid(True)
plt.show()

# --- 操作 2: 金额与紧迫感双轴关联图 ---
# 读取存储金额与心理策略比例的数据文件
df2 = pd.read_csv('板块2_金额与心理策略演变.csv')

# 数据清洗：使用 .str.replace 移除单位（元），并用 .astype(float) 转换为数值类型以便绘图
df2['最大提取金额'] = df2['最大提取金额'].str.replace(' 元', '', regex=False).astype(float)
# 数据清洗：移除百分号（%）并转换为浮点数
df2['心理策略_紧迫感'] = df2['心理策略_紧迫感'].str.replace('%', '', regex=False).astype(float)

# 创建画布和第一个坐标轴（ax1）
fig, ax1 = plt.subplots(figsize=(10, 6))

# 在 ax1 上绘制条形图（Bar Chart），展示不同阶段的金额变化
ax1.bar(df2['诈骗阶段'], df2['最大提取金额'], color='skyblue', label='最大提取金额', alpha=0.7)
ax1.set_ylabel('金额 (元)', color='blue')

# 【关键步骤】创建共享横轴的第二个纵坐标轴（ax2）
ax2 = ax1.twinx()
# 在 ax2 上绘制折线图，展示紧迫感策略的使用频率；marker='D' 表示使用菱形标记点
ax2.plot(df2['诈骗阶段'], df2['心理策略_紧迫感'], color='red', marker='D', label='紧迫感飙升', linewidth=2)
ax2.set_ylabel('紧迫感百分比 (%)', color='red')

# 设置图表标题
plt.title('金额暴涨与紧迫感心理策略的关联')
# 自动调整布局，防止标签重叠
fig.tight_layout()
plt.show()