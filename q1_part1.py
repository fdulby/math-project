"""
第一部分：数据读取与可视化
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================
# 配置参数
# ==========================================

class CONFIG:
    """全局配置类"""
    
    ALGORITHM = 1  # 0=遗传, 1=模拟退火, 2=蚁群
    
    CROWD_WEIGHTS = {
        '普通': [0.25, 0.25, 0.25, 0.25],
        '亲子': [0.10, 0.20, 0.40, 0.30],
        '情侣': [0.40, 0.40, 0.10, 0.10]
    }
    
    DATE_QUEUE_MULTIPLIER = {
        '工作日': 0.7,
        '双休日': 1.0,
        '节假日': 1.5
    }
    
    WALK_SPEED_MULTIPLIER = 1.0
    GMM_PEAK_INTENSITY_SCALE = 1.0
    GMM_PEAK_WIDTH_SCALE = 1.0
    
    WEIGHT_QUEUE_TIME = 0.1
    WEIGHT_WAIT_TIME = 0.15
    WEIGHT_WALK_TIME = 0.05
    WEIGHT_OVERTIME = 5.0
    WEIGHT_MISSED_SHOW = 1000.0
    WEIGHT_DIVERSITY = 2.0
    
    PARK_OPEN_TIME = 0
    PARK_CLOSE_TIME = 1260
    START_TIME = 0
    
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 200
    GA_CROSSOVER_RATE = 0.8
    GA_MUTATION_RATE = 0.2
    GA_ELITE_SIZE = 10
    
    SA_INITIAL_TEMP = 1000
    SA_COOLING_RATE = 0.995
    SA_MAX_ITERATIONS = 5000
    
    ACO_ANT_COUNT = 50
    ACO_ITERATIONS = 100
    ACO_ALPHA = 1.0
    ACO_BETA = 2.0
    ACO_RHO = 0.5
    ACO_Q = 100
    
    FIGURE_DPI = 100
    FIGURE_SIZE = (12, 8)


# ==========================================
# 数据读取
# ==========================================

def load_projects_from_csv(csv_file: str) -> Tuple[Dict, pd.DataFrame]:
    """从CSV读取项目信息（所有项目都有时间窗）"""
    df = pd.read_csv(csv_file)
    
    project_info = {}
    
    for idx, row in df.iterrows():
        proj_id = int(row['项目ID'])
        
        info = {
            'name': row['项目名称'],
            'duration': float(row['游玩时长（分钟）']),
            'type': 'show' if row['是否演出'] == '是' else 'normal',
            'features': [
                float(row['刺激度']),
                float(row['沉浸度']),
                float(row['互动度']),
                float(row['休闲度'])
            ]
        }
        
        # 时间窗（所有项目都有）
        if pd.notna(row['时间窗开始']) and pd.notna(row['时间窗结束']):
            info['time_window'] = (float(row['时间窗开始']), float(row['时间窗结束']))
        else:
            info['time_window'] = (CONFIG.PARK_OPEN_TIME, CONFIG.PARK_CLOSE_TIME)
        
        # 动态排队参数（仅普通项目）
        if info['type'] == 'normal':
            info['base_q'] = float(row['基础排队'])
            peaks = []
            if pd.notna(row['峰值1强度']):
                peaks.append((
                    float(row['峰值1强度']) * CONFIG.GMM_PEAK_INTENSITY_SCALE,
                    float(row['峰值1时间']),
                    float(row['峰值1宽度']) * CONFIG.GMM_PEAK_WIDTH_SCALE
                ))
            if pd.notna(row['峰值2强度']):
                peaks.append((
                    float(row['峰值2强度']) * CONFIG.GMM_PEAK_INTENSITY_SCALE,
                    float(row['峰值2时间']),
                    float(row['峰值2宽度']) * CONFIG.GMM_PEAK_WIDTH_SCALE
                ))
            info['peaks'] = peaks
        
        project_info[proj_id] = info
    
    return project_info, df


# ==========================================
# 可视化函数
# ==========================================

def plot_crowd_radar(crowd_type: str, save_path: str = 'radar.png'):
    """绘制人物画像雷达图"""
    weights = CONFIG.CROWD_WEIGHTS[crowd_type]
    categories = ['刺激度', '沉浸度', '互动度', '休闲度']
    
    values = weights + [weights[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label=crowd_type, color='#FF6B6B')
    ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylim(0, 0.5)
    ax.set_title(f'{crowd_type}游客偏好画像', fontsize=18, pad=20, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 雷达图已保存: {save_path}")
    plt.show()


def print_config_summary(crowd_type: str, date_type: str):
    """打印配置摘要"""
    algorithm_names = ['遗传算法', '模拟退火', '蚁群算法']
    
    print("=" * 70)
    print(" " * 20 + "迪士尼乐园路线优化系统")
    print("=" * 70)
    print(f"\n【人群类型】 {crowd_type}")
    print(f"【日期类型】 {date_type}")
    print(f"【优化算法】 {algorithm_names[CONFIG.ALGORITHM]}")
    print(f"【闭园时间】 {CONFIG.PARK_CLOSE_TIME}分钟 ({CONFIG.PARK_CLOSE_TIME/60:.1f}小时)")
    print(f"【排队系数】 {CONFIG.DATE_QUEUE_MULTIPLIER[date_type]}x")
    print(f"【行走速度】 {CONFIG.WALK_SPEED_MULTIPLIER}x")
    print("\n【目标函数权重】")
    print(f"  排队惩罚: {CONFIG.WEIGHT_QUEUE_TIME}")
    print(f"  等待惩罚: {CONFIG.WEIGHT_WAIT_TIME}")
    print(f"  步行惩罚: {CONFIG.WEIGHT_WALK_TIME}")
    print(f"  超时惩罚: {CONFIG.WEIGHT_OVERTIME}")
    print(f"  错过演出: {CONFIG.WEIGHT_MISSED_SHOW}")
    print(f"  多样性奖励: {CONFIG.WEIGHT_DIVERSITY}")
    print("=" * 70 + "\n")


def visualize_projects_map(project_info: Dict, save_path: str = 'projects_map.png'):
    """可视化项目分布（简化版地图）"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 按类型分组
    normal_projects = [(id, info) for id, info in project_info.items() if info['type'] == 'normal']
    show_projects = [(id, info) for id, info in project_info.items() if info['type'] == 'show']
    
    # 绘制普通项目
    for i, (proj_id, info) in enumerate(normal_projects):
        x = (i % 5) * 2
        y = (i // 5) * 2
        ax.scatter(x, y, s=300, c='#4ECDC4', marker='o', edgecolors='black', linewidths=2, zorder=5)
        ax.text(x, y, str(proj_id), ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.5, info['name'], ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 绘制演出项目
    for i, (proj_id, info) in enumerate(show_projects):
        x = (i % 3) * 3 + 1
        y = 8 + (i // 3) * 1.5
        ax.scatter(x, y, s=400, c='#FF6B6B', marker='s', edgecolors='black', linewidths=2, zorder=5)
        ax.text(x, y, str(proj_id), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(x, y-0.5, info['name'], ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # 图例
    ax.scatter([], [], s=300, c='#4ECDC4', marker='o', edgecolors='black', linewidths=2, label='普通项目')
    ax.scatter([], [], s=400, c='#FF6B6B', marker='s', edgecolors='black', linewidths=2, label='演出项目')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 12)
    ax.set_title('迪士尼乐园项目分布图', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('区域 X', fontsize=12)
    ax.set_ylabel('区域 Y', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 项目分布图已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 测试第一部分
    project_info, df = load_projects_from_csv('projects_data.csv')
    print(f"✓ 成功读取 {len(project_info)} 个项目\n")
    
    # 选择人群和日期
    selected_crowd = '普通'
    selected_date = '工作日'
    
    # 打印配置
    print_config_summary(selected_crowd, selected_date)
    
    # 生成可视化
    plot_crowd_radar(selected_crowd, f'radar_{selected_crowd}.png')
    visualize_projects_map(project_info, 'projects_map.png')
    
    print("\n✓ 第一部分完成！")
