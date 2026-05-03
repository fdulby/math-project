"""
迪士尼乐园路线优化系统 - 第二问
基于实时排队数据的动态路径规划
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================
# 配置模块
# ==========================================

class CONFIG:
    """全局配置类"""
    
    CROWD_WEIGHTS = {
        '普通': [0.25, 0.25, 0.25, 0.25],
        '亲子': [0.10, 0.20, 0.40, 0.30],
        '情侣': [0.40, 0.40, 0.10, 0.10]
    }
    
    PARK_OPEN_TIME = 540  # 9:00 (分钟)
    PARK_CLOSE_TIME = 1260  # 21:00 (分钟)
    START_TIME = 540
    
    # 权重参数
    WEIGHT_QUEUE_TIME = 0.1
    WEIGHT_WAIT_TIME = 0.1
    WEIGHT_WALK_TIME = 0.1
    WEIGHT_UTILITY = 1.0
    
    # 路径配置
    DATA_DIR = '../data'
    OUTPUT_DIR = '../Q2-test'
    
    FIGURE_DPI = 100


# ==========================================
# 数据加载
# ==========================================

def load_queue_data(csv_file: str) -> pd.DataFrame:
    """加载实时排队数据"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"✓ 成功读取排队数据: {len(df)} 条记录")
    print(f"  场景: {df['scenario'].unique()}")
    print(f"  项目数: {df['project_id'].nunique()}")
    print(f"  时间范围: {df['time_min'].min()}-{df['time_min'].max()}分钟")
    return df


def load_projects_data(csv_file: str) -> pd.DataFrame:
    """加载项目基础数据"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"✓ 成功读取项目数据: {len(df)} 个项目")
    return df


def load_distance_matrix(csv_file: str) -> np.ndarray:
    """加载距离矩阵"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    df['ID_num'] = df['ID'].astype(int)
    df = df.sort_values('ID_num').reset_index(drop=True)
    
    walk_cols = [col for col in df.columns if col.startswith('walk_to_')]
    distance_matrix = np.zeros((len(df), len(df)))
    
    for i in range(len(df)):
        row_data = df.iloc[i]
        for j, col in enumerate(walk_cols):
            distance_matrix[i, j] = float(row_data[col])
    
    print(f"✓ 成功读取距离矩阵: {distance_matrix.shape}")
    return distance_matrix


# ==========================================
# 核心算法：贪心策略
# ==========================================

def get_realtime_queue(queue_df: pd.DataFrame, scenario: str, 
                      project_id: str, current_time: int) -> float:
    """获取实时排队时间（10分钟粒度，线性插值）"""
    # 找到最接近的时间点
    project_data = queue_df[
        (queue_df['scenario'] == scenario) & 
        (queue_df['project_id'] == project_id)
    ]
    
    if len(project_data) == 0:
        return 0.0
    
    # 找到当前时间前后的数据点
    before = project_data[project_data['time_min'] <= current_time]
    after = project_data[project_data['time_min'] >= current_time]
    
    if len(before) == 0:
        return after.iloc[0]['realtime_wait_min']
    if len(after) == 0:
        return before.iloc[-1]['realtime_wait_min']
    
    # 线性插值
    t1 = before.iloc[-1]['time_min']
    q1 = before.iloc[-1]['realtime_wait_min']
    t2 = after.iloc[0]['time_min']
    q2 = after.iloc[0]['realtime_wait_min']
    
    if t1 == t2:
        return q1
    
    queue_time = q1 + (q2 - q1) * (current_time - t1) / (t2 - t1)
    return max(0.0, queue_time)


def calculate_utility(project_info: Dict, crowd_type: str) -> float:
    """计算项目效用"""
    w = np.array(CONFIG.CROWD_WEIGHTS[crowd_type])
    features = np.array(project_info['features'])
    return np.dot(features, w) * 10


def greedy_route_planning(queue_df: pd.DataFrame, projects_df: pd.DataFrame,
                         distance_matrix: np.ndarray, scenario: str,
                         crowd_type: str) -> Tuple[List, Dict]:
    """
    贪心策略路径规划
    
    策略：每次选择"效用/总时间"比值最大的项目
    """
    current_time = CONFIG.START_TIME
    current_pos = 0  # 入口
    visited = set([0])  # 已访问（入口）
    route = []
    timeline = []
    
    total_utility = 0.0
    total_queue_time = 0.0
    total_walk_time = 0.0
    total_play_time = 0.0
    
    # 构建项目信息字典
    project_dict = {}
    for idx, row in projects_df.iterrows():
        if row['项目ID'] == 0:
            continue
        project_dict[row['项目ID']] = {
            'name': row['项目名称'],
            'duration': row['游玩时长（分钟）'],
            'features': [row['刺激度'], row['沉浸度'], row['互动度'], row['休闲度']],
            'project_id': f"P{row['项目ID']:02d}"
        }
    
    while current_time < CONFIG.PARK_CLOSE_TIME:
        best_project = None
        best_score = -float('inf')
        best_info = None
        
        # 评估所有未访问的项目
        for proj_id, proj_info in project_dict.items():
            if proj_id in visited:
                continue
            
            # 计算到达该项目的时间
            walk_time = distance_matrix[current_pos, proj_id]
            arrive_time = current_time + walk_time
            
            # 如果到达时间超过闭园，跳过
            if arrive_time >= CONFIG.PARK_CLOSE_TIME:
                continue
            
            # 获取实时排队时间
            queue_time = get_realtime_queue(queue_df, scenario, 
                                           proj_info['project_id'], arrive_time)
            
            # 计算完成时间
            play_time = proj_info['duration']
            finish_time = arrive_time + queue_time + play_time
            
            # 如果完成时间超过闭园，跳过
            if finish_time > CONFIG.PARK_CLOSE_TIME:
                continue
            
            # 计算效用
            utility = calculate_utility(proj_info, crowd_type)
            
            # 计算总时间成本
            total_time = walk_time + queue_time + play_time
            
            # 贪心评分：效用/时间
            if total_time > 0:
                score = utility / total_time
            else:
                score = utility
            
            if score > best_score:
                best_score = score
                best_project = proj_id
                best_info = {
                    'walk_time': walk_time,
                    'arrive_time': arrive_time,
                    'queue_time': queue_time,
                    'play_time': play_time,
                    'finish_time': finish_time,
                    'utility': utility,
                    'name': proj_info['name']
                }
        
        # 如果没有可选项目，结束
        if best_project is None:
            break
        
        # 访问该项目
        visited.add(best_project)
        route.append(best_project)
        
        # 更新统计
        total_walk_time += best_info['walk_time']
        total_queue_time += best_info['queue_time']
        total_play_time += best_info['play_time']
        total_utility += best_info['utility']
        
        # 记录时间线
        timeline.append({
            '项目': best_info['name'],
            '到达时间': f"{int(best_info['arrive_time']//60):02d}:{int(best_info['arrive_time']%60):02d}",
            '排队(分钟)': round(best_info['queue_time'], 1),
            '游玩(分钟)': best_info['play_time'],
            '离开时间': f"{int(best_info['finish_time']//60):02d}:{int(best_info['finish_time']%60):02d}",
            '效用': round(best_info['utility'], 2)
        })
        
        # 更新当前状态
        current_time = best_info['finish_time']
        current_pos = best_project
    
    # 返回入口
    if current_pos != 0:
        walk_back = distance_matrix[current_pos, 0]
        if current_time + walk_back <= CONFIG.PARK_CLOSE_TIME:
            current_time += walk_back
            total_walk_time += walk_back
    
    result = {
        'route': route,
        'timeline': timeline,
        'total_utility': round(total_utility, 2),
        'total_queue_time': round(total_queue_time, 1),
        'total_walk_time': round(total_walk_time, 1),
        'total_play_time': round(total_play_time, 1),
        'total_time': round(current_time - CONFIG.START_TIME, 1),
        'visited_count': len(route)
    }
    
    return route, result


# ==========================================
# 可视化
# ==========================================

def plot_timeline(result: Dict, scenario: str, crowd_type: str, save_path: str):
    """绘制时间线甘特图"""
    df = pd.DataFrame(result['timeline'])
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, row in df.iterrows():
        # 解析时间
        arrive_h, arrive_m = map(int, row['到达时间'].split(':'))
        leave_h, leave_m = map(int, row['离开时间'].split(':'))
        
        arrive_min = arrive_h * 60 + arrive_m
        leave_min = leave_h * 60 + leave_m
        duration = leave_min - arrive_min
        
        ax.barh(i, duration, left=arrive_min, height=0.6, 
               color='#4ECDC4', edgecolor='black', linewidth=1.5)
        ax.text(arrive_min + duration/2, i, row['项目'], 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{i+1}" for i in range(len(df))])
    ax.set_xlabel('时间（分钟）', fontsize=12)
    ax.set_ylabel('项目序号', fontsize=12)
    ax.set_title(f'Q2-游览路线时间线\n{scenario} - {crowd_type}游客', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 时间线图已保存: {save_path}")
    plt.close()


def plot_queue_comparison(queue_df: pd.DataFrame, scenario: str, 
                         route: List, projects_df: pd.DataFrame, save_path: str):
    """绘制路线中项目的排队时间变化"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for proj_id in route[:10]:  # 只显示前10个项目
        proj_name = projects_df[projects_df['项目ID'] == proj_id]['项目名称'].values[0]
        proj_code = f"P{proj_id:02d}"
        
        data = queue_df[
            (queue_df['scenario'] == scenario) & 
            (queue_df['project_id'] == proj_code)
        ]
        
        if len(data) > 0:
            ax.plot(data['time_min'], data['realtime_wait_min'], 
                   label=proj_name, marker='o', markersize=3)
    
    ax.set_xlabel('时间（分钟）', fontsize=12)
    ax.set_ylabel('排队时间（分钟）', fontsize=12)
    ax.set_title(f'Q2-项目排队时间变化\n{scenario}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 排队对比图已保存: {save_path}")
    plt.close()


# ==========================================
# 主函数
# ==========================================

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" " * 20 + "迪士尼乐园路线优化系统 - 第二问")
    print(" " * 15 + "基于实时排队数据的动态路径规划")
    print("=" * 70 + "\n")
    
    # 加载数据
    print("=" * 70)
    print("数据加载")
    print("=" * 70 + "\n")
    
    queue_df = load_queue_data(f"{CONFIG.DATA_DIR}/new_simulated_queue_10min(1).csv")
    projects_df = load_projects_data(f"{CONFIG.DATA_DIR}/updated_projects_data.csv")
    distance_matrix = load_distance_matrix(f"{CONFIG.DATA_DIR}/poi.csv")
    
    # 选择场景和人群
    scenario = '工作日'
    crowd_type = '普通'
    
    print(f"\n{'='*70}")
    print(f"运行场景: {scenario} - {crowd_type}游客")
    print(f"{'='*70}\n")
    
    # 运行贪心算法
    route, result = greedy_route_planning(queue_df, projects_df, distance_matrix,
                                         scenario, crowd_type)
    
    # 打印结果
    print(f"\n{'='*70}")
    print("优化结果")
    print(f"{'='*70}\n")
    
    print(f"【访问项目数】 {result['visited_count']} 个")
    print(f"【总效用】 {result['total_utility']:.2f}")
    print(f"【总耗时】 {result['total_time']:.1f}分钟")
    print(f"【游玩时间】 {result['total_play_time']:.1f}分钟")
    print(f"【排队时间】 {result['total_queue_time']:.1f}分钟")
    print(f"【步行时间】 {result['total_walk_time']:.1f}分钟\n")
    
    print("【推荐路线】")
    for i, item in enumerate(result['timeline'], 1):
        print(f"  {i}. {item['项目']}")
    
    print(f"\n【详细时间线】")
    df_timeline = pd.DataFrame(result['timeline'])
    print(df_timeline.to_string(index=False))
    
    # 生成可视化
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    timeline_path = f"{CONFIG.OUTPUT_DIR}/Q2-timeline-{scenario}-{crowd_type}.png"
    plot_timeline(result, scenario, crowd_type, timeline_path)
    
    queue_path = f"{CONFIG.OUTPUT_DIR}/Q2-queue-{scenario}-{crowd_type}.png"
    plot_queue_comparison(queue_df, scenario, route, projects_df, queue_path)
    
    print(f"\n{'='*70}")
    print("✓ 优化完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
