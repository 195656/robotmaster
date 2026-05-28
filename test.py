import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据准备
voltage = [2, 4, 6, 8, 10]
magnetic_electric = [350, 790, 1220, 1730, 2230]
photoelectric = [360, 860, 1130, 1800, 2270]
hall = [360, 850, 1320, 1780, 1910]

# 频率表数据
frequency_data = [
    [35, 38, 38],
    [79, 85, 85],
    [112, 133, 132],
    [173, 179, 180],
    [223, 198, 227]
]

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制传感器读数图
ax1.plot(voltage, magnetic_electric, 'o-', linewidth=2, markersize=8, label='磁电传感器')
ax1.plot(voltage, photoelectric, 's-', linewidth=2, markersize=8, label='光电传感器')
ax1.plot(voltage, hall, '^-', linewidth=2, markersize=8, label='霍尔传感器')

ax1.set_xlabel('电压 (V)', fontsize=12)
ax1.set_ylabel('传感器读数', fontsize=12)
ax1.set_title('电压与传感器读数关系', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 11)

# 绘制频率数据图
frequency_array = np.array(frequency_data)
ax2.plot(voltage, frequency_array[:, 0], 'o-', linewidth=2, markersize=8, label='磁电频率')
ax2.plot(voltage, frequency_array[:, 1], 's-', linewidth=2, markersize=8, label='光电频率')
ax2.plot(voltage, frequency_array[:, 2], '^-', linewidth=2, markersize=8, label='霍尔频率')

ax2.set_xlabel('电压 (V)', fontsize=12)
ax2.set_ylabel('频率', fontsize=12)
ax2.set_title('电压与频率关系', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 11)

plt.tight_layout()
plt.show()