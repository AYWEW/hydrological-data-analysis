import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

script_dir = Path(__file__).parent
file_path = script_dir / '水文数据.xls'
output_dir = script_dir / 'plots'
output_dir.mkdir(exist_ok=True)

# 读取并重命名列
df = pd.read_excel(file_path, header=1)
name_map = {
    'year': '年份', 'month': '月份', 'day': '日期',
    'tmax': '最高气温', 'tmin': '最低气温', 'tavg': '平均气温',
    'rh': '相对湿度', 'ah': '绝对湿度', 'vpd': '饱和差',
    'wind_dir': '最多风向', 'wind_max': '最大风速', 'wind_avg': '平均风速',
    'sun': '日照时数', 'rain': '降雨量', 'evap': '蒸发量'
}
rename_dict = {}
for eng, chn in name_map.items():
    for col in df.columns:
        if chn in str(col):
            rename_dict[col] = eng
            break
df.rename(columns=rename_dict, inplace=True)
keep_cols = list(name_map.keys())
df = df[[c for c in keep_cols if c in df.columns]].copy()

# 年/月/日向下填充
for col in ['year', 'month', 'day']:
    if col in df.columns:
        df[col] = df[col].ffill().astype(int)

# 数值列转换
num_cols = [c for c in df.columns if c not in ['year', 'month', 'day', 'wind_dir']]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 物理范围过滤
range_rules = {
    'tmax': (-20, 45), 'tmin': (-40, 35), 'tavg': (-30, 40),
    'rh': (0, 100), 'ah': (0, 50), 'vpd': (0, 20),
    'wind_max': (0, 30), 'wind_avg': (0, 20),
    'sun': (0, 24), 'rain': (0, 200), 'evap': (0, 15)
}
for col, (low, high) in range_rules.items():
    if col in df.columns:
        df[col] = df[col].where(df[col].between(low, high))

# IQR 统计过滤
def remove_outliers_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.where((series >= Q1 - factor * IQR) & (series <= Q3 + factor * IQR))

for col in num_cols:
    df[col] = remove_outliers_iqr(df[col], factor=3.0)

# 插值（降雨量除外，限制连续缺失长度）
MAX_GAP = 5   # 连续缺失超过5天则不填充，留空
interp_cols = [c for c in num_cols if c != 'rain']
df[interp_cols] = df[interp_cols].interpolate(
    method='linear',
    limit=MAX_GAP,
    limit_direction='both'
)

# 风向填充
if 'wind_dir' in df.columns:
    mode_val = df['wind_dir'].mode()
    if not mode_val.empty:
        df['wind_dir'] = df['wind_dir'].fillna(mode_val[0])

# 日期列
df['日期时间'] = pd.to_datetime({
    'year': df['year'], 'month': df['month'], 'day': df['day']
})
df = df.sort_values('日期时间').reset_index(drop=True)

# 绘图（所有数值要素）
plot_cols = interp_cols + ['rain']
for col in plot_cols:
    plt.figure(figsize=(12, 5))
    plt.plot(df['日期时间'], df[col], linewidth=0.8, color='steelblue')
    plt.title(f'{col} 年内变化')
    plt.xlabel('日期'); plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f'折线_{col}.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    df.boxplot(column=col, by='month', grid=False, showmeans=True,
               meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    plt.title(f'{col} 各月分布箱线图')
    plt.suptitle(''); plt.xlabel('月份'); plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(output_dir / f'箱线_{col}.png', dpi=150)
    plt.close()

print(f"✅ 全部图表已保存至 {output_dir}")