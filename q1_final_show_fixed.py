"""
迪士尼乐园路线优化系统 - 最终版本
修复所有bug，规范化文件路径
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================
# 配置模块
# ==========================================

class CONFIG:
    """全局配置类"""
    ALGORITHM = 0  # 0=遗传, 1=模拟退火, 2=蚁群
    
    CROWD_WEIGHTS = {
        '普通': [2.5, 2.5, 2.5, 2.5],
        '亲子': [1, 2, 4, 3],
        '情侣': [4, 4, 1, 1]
    }
    
    DATE_QUEUE_MULTIPLIER = {'工作日': 0.7, '双休日': 1.0, '节假日': 1.5}
    WALK_SPEED_MULTIPLIER = 1.0
    GMM_PEAK_INTENSITY_SCALE = 1.0
    GMM_PEAK_WIDTH_SCALE = 1.0
    
    WEIGHT_QUEUE_TIME = 0.2
    WEIGHT_WAIT_TIME = 0.2
    WEIGHT_WALK_TIME = 0.1
    WEIGHT_OVERTIME = 5.0
    WEIGHT_MISSED_SHOW = 1000.0
    WEIGHT_DIVERSITY = 1.0
    
    PARK_OPEN_TIME = 0
    PARK_CLOSE_TIME = 720  # 12小时
    START_TIME = 0

    # 模拟退火参数
    SA_INITIAL_TEMP = 1000
    SA_COOLING_RATE = 0.995
    SA_MAX_ITERATIONS = 5000

    # 遗传算法参数
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 200
    GA_CROSSOVER_RATE = 0.8
    GA_MUTATION_RATE = 0.2
    GA_ELITE_SIZE = 10

    # 蚁群算法参数
    ACO_ANT_COUNT = 50
    ACO_ITERATIONS = 100
    ACO_ALPHA = 1.0
    ACO_BETA = 2.0
    ACO_RHO = 0.5
    ACO_Q = 100

    FIGURE_DPI = 100
    FIGURE_SIZE = (12, 8)

    # 路径配置（基于脚本所在目录的绝对路径）
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Q1-test')

    START_NODE = 0
    END_NODE = 27

    # 若需输出“总步行距离/m”，需把步行时间换算为距离
    # 这里取园区平均步速 70 m/min，可按需要调整
    WALK_SPEED_M_PER_MIN = 80.0

    # 用于把相对分钟转换成实际结束时刻（例如 09:00 开园）
    DISPLAY_OPEN_CLOCK_MIN = 9 * 60


# ==========================================
# 第一部分：数据读取与可视化
# ==========================================

def load_projects_from_csv(csv_file: str) -> Tuple[Dict, pd.DataFrame]:
    """从CSV读取项目信息"""
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
            ],
            'lat': float(row['纬度']),
            'lon': float(row['经度'])
        }
        
        # 时间窗（所有项目都有）
        if pd.notna(row['时间窗开始']) and pd.notna(row['时间窗结束']):
            info['time_window'] = (float(row['时间窗开始']), float(row['时间窗结束']))
        else:
            info['time_window'] = (CONFIG.PARK_OPEN_TIME, CONFIG.PARK_CLOSE_TIME)
        
        # 动态排队参数（仅普通项目，入口除外）
        if info['type'] == 'normal' and proj_id not in [CONFIG.START_NODE, CONFIG.END_NODE]:
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


def load_distance_matrix(csv_file: str) -> Tuple[np.ndarray, Dict]:
    """
    从poi.csv读取距离矩阵（修复版：严格排序）
    
    返回:
        distance_matrix: n×n矩阵
        poi_mapping: POI ID到矩阵索引的映射
    """
    df = pd.read_csv(csv_file)
    
    # 关键修复：按ID严格排序
    df['ID_num'] = df['ID'].astype(int)
    df = df.sort_values('ID_num').reset_index(drop=True)
    
    n = len(df)
    
    # 提取步行时间列（包括ENTRY和EXIT）
    walk_cols = [col for col in df.columns if col.startswith('walk_to_')]
    
    # 验证列数是否正确
    expected_cols = n  # 应该有n个目标列
    if len(walk_cols) != expected_cols:
        print(f"警告：步行时间列数({len(walk_cols)})与POI数量({n})不匹配")
    
    # 按照ID顺序提取距离矩阵
    # walk_to_ENTRY, walk_to_P01, ..., walk_to_P26, walk_to_EXIT
    # 对应索引: 0, 1, ..., 26, 27 (如果EXIT=ENTRY则是0)
    
    # 构建n×n矩阵
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        row_data = df.iloc[i]
        for j, col in enumerate(walk_cols):
            distance_matrix[i, j] = float(row_data[col])
    
    # POI ID映射
    poi_mapping = {}
    for idx, row in df.iterrows():
        poi_id = int(row['ID_num'])
        poi_mapping[poi_id] = idx
    
    print(f"✓ 距离矩阵加载完成: {distance_matrix.shape}")
    print(f"  POI映射: {poi_mapping}")
    
    return distance_matrix, poi_mapping


def plot_crowd_radar(crowd_type: str, save_path: str):
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
    plt.close()


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
    print("=" * 70 + "\n")


# ==========================================
# 第二部分：核心算法
# ==========================================

def calculate_utility_scores(project_info: Dict, crowd_type: str = '普通') -> Dict:
    """计算效用得分"""
    w = np.array(CONFIG.CROWD_WEIGHTS[crowd_type])
    for proj_id, info in project_info.items():
        if proj_id in [CONFIG.START_NODE, CONFIG.END_NODE]:  # 入口不计算效用
            info['utility'] = 0.0
            continue
        features = np.array(info['features'])
        utility = np.dot(features, w) * 10
        info['utility'] = round(utility, 2)
    return project_info


def get_dynamic_queue_time(t_current: float, base_q: float, peaks: List[Tuple]) -> float:
    """GMM动态排队时间"""
    q_time = float(base_q)
    for A, mu, sigma in peaks:
        q_time += A * np.exp(-((t_current - mu) ** 2) / (2 * sigma ** 2))
    return max(0.0, q_time)


def evaluate_route(route: List[int], distance_matrix: np.ndarray, project_info: Dict,
                  start_time: float = None, start_node: int = 0, 
                  end_node: Optional[int] = 0, return_to_end: bool = True) -> Dict:
    """
    评估路线（修复版：正确处理返回终点）
    
    修复bug：
    1. 如果走到出口不会超时，那就走过去并计入步行时间
    2. 如果走到出口会超时，那就不走（停在最后一个项目）
    """
    if start_time is None:
        start_time = CONFIG.START_TIME
    
    dist_matrix = distance_matrix * CONFIG.WALK_SPEED_MULTIPLIER
    current_time = float(start_time)
    current_node = start_node
    
    total_utility = 0.0
    total_queue_time = 0.0
    total_wait_time = 0.0
    total_walk_time = 0.0
    missed_shows = 0
    closed_projects = 0
    overtime = 0.0
    timeline_log = []
    feasible = True
    visited_projects = []
    
    if len(route) == 0:
        # 空路线：直接返回终点
        if return_to_end and end_node is not None:
            walk_time = dist_matrix[start_node, end_node]
            if current_time + walk_time <= CONFIG.PARK_CLOSE_TIME:
                current_time += walk_time
                total_walk_time += walk_time
        return {
            'final_score': 0.0, 'total_utility': 0.0, 'total_time': current_time - start_time,
            'feasible': True, 'timeline_log': [], 'route': route,
            'visited_projects': [], 'visited_count': 0
        }
    
    # 遍历路线中的项目
    for next_node in route:
        if next_node not in project_info or next_node == 0:  # 跳过入口
            continue
        
        p_info = project_info[next_node]
        walk_time = dist_matrix[current_node, next_node]
        arrive_time = current_time + walk_time
        
        # 提前终止条件1：在路上就超过闭园时间
        if arrive_time >= CONFIG.PARK_CLOSE_TIME:
            break
        
        total_walk_time += walk_time
        
        name = p_info['name']
        node_type = p_info['type']
        play_duration = p_info['duration']
        utility_score = p_info['utility']
        time_window_start, time_window_end = p_info['time_window']
        
        wait_time = 0.0
        queue_time = 0.0
        gained_utility = 0.0
        status = 'completed'
        leave_time = arrive_time
        
        # 检查项目是否开放
        if arrive_time < time_window_start:
            wait_time = time_window_start - arrive_time
            total_wait_time += wait_time
            arrive_time = time_window_start
            status = 'waited_for_open'
        
        if node_type == 'show':
            # 演出类项目采用“场次区间 [a,b]”建模：
            # 1) 到达时刻 >= b，视为错过演出；
            # 2) 到达时刻处于 (a,b) 内，只能观看剩余部分；
            # 3) 演出结束时刻固定控制在 b。
            if arrive_time >= time_window_end:
                missed_shows += 1
                status = 'missed_show'
                leave_time = arrive_time
                gained_utility = 0.0
            else:
                actual_start = max(arrive_time, time_window_start)
                full_show_duration = max(time_window_end - time_window_start, 1e-6)
                watched_duration = max(0.0, time_window_end - actual_start)
                watched_ratio = min(1.0, watched_duration / full_show_duration)

                # 若中途到场，仅获得剩余场次对应的部分效用
                gained_utility = utility_score * watched_ratio
                leave_time = time_window_end

                # 区分完整观看与部分观看，便于结果解释与可视化
                if actual_start > time_window_start:
                    status = 'partial_show'
        else:
            if arrive_time > time_window_end:
                closed_projects += 1
                status = 'closed'
                leave_time = arrive_time
                gained_utility = 0.0
            else:
                queue_time = get_dynamic_queue_time(arrive_time, p_info['base_q'], p_info['peaks'])
                total_queue_time += queue_time
                actual_start = arrive_time + queue_time
                leave_time = actual_start + play_duration
                gained_utility = utility_score
        
        # 提前终止条件2：游玩结束后超过闭园时间
        if leave_time > CONFIG.PARK_CLOSE_TIME:
            overtime = leave_time - CONFIG.PARK_CLOSE_TIME
            feasible = False
            total_utility += gained_utility
            timeline_log.append({
                '项目': name, '到达': round(arrive_time, 1), '排队': round(queue_time, 1),
                '等待': round(wait_time, 1), '离开': round(leave_time, 1),
                '效用': round(gained_utility, 2), '状态': status + '_overtime'
            })
            visited_projects.append(next_node)
            current_time = leave_time
            current_node = next_node
            break
        
        total_utility += gained_utility
        timeline_log.append({
            '项目': name, '到达': round(arrive_time, 1), '排队': round(queue_time, 1),
            '等待': round(wait_time, 1), '离开': round(leave_time, 1),
            '效用': round(gained_utility, 2), '状态': status
        })
        
        visited_projects.append(next_node)
        current_time = leave_time
        current_node = next_node
    
    # 修复bug：正确处理返回终点
    if return_to_end and end_node is not None and current_node != end_node:
        walk_time = dist_matrix[current_node, end_node]
        arrive_exit_time = current_time + walk_time
        
        if arrive_exit_time <= CONFIG.PARK_CLOSE_TIME:
            # 不会超时，走到出口
            current_time = arrive_exit_time
            total_walk_time += walk_time
        else:
            # 会超时，不走了（停在最后一个项目）
            # current_time保持不变，不增加步行时间
            pass
    
    # 计算综合得分
    time_cost = (CONFIG.WEIGHT_QUEUE_TIME * total_queue_time +
                CONFIG.WEIGHT_WAIT_TIME * total_wait_time +
                CONFIG.WEIGHT_WALK_TIME * total_walk_time)
    overtime_penalty = CONFIG.WEIGHT_OVERTIME * (overtime ** 2) if overtime > 0 else 0
    missed_penalty = CONFIG.WEIGHT_MISSED_SHOW * (missed_shows + closed_projects)
    
    visited_types = [project_info[node]['type'] for node in visited_projects if node in project_info]
    diversity_bonus = CONFIG.WEIGHT_DIVERSITY * len(set(visited_types)) if visited_types else 0
    
    final_score = total_utility - time_cost - overtime_penalty - missed_penalty + diversity_bonus
    
    return {
        'final_score': round(final_score, 2),
        'total_utility': round(total_utility, 2),
        'total_time': round(current_time - start_time, 1),
        'total_queue': round(total_queue_time, 1),
        'total_wait': round(total_wait_time, 1),
        'total_walk': round(total_walk_time, 1),
        'missed_shows': missed_shows,
        'closed_projects': closed_projects,
        'overtime': round(overtime, 1),
        'feasible': feasible,
        'timeline_log': timeline_log,
        'route': route,
        'visited_projects': visited_projects,
        'visited_count': len(visited_projects)
    }


def simulated_annealing(project_ids: List[int], distance_matrix: np.ndarray,
                       project_info: Dict, **kwargs) -> Tuple[List[int], Dict]:
    """模拟退火算法（固定随机种子以保证结果可复现）"""
    # 固定随机种子，确保每次运行结果一致
    random.seed(42)
    np.random.seed(42)

    current_route = random.sample(project_ids, len(project_ids))
    current_result = evaluate_route(current_route, distance_matrix, project_info, **kwargs)
    current_score = current_result['final_score']
    
    best_route = current_route.copy()
    best_result = current_result
    best_score = current_score
    
    temperature = CONFIG.SA_INITIAL_TEMP
    iteration = 0
    
    print(f"初始得分: {current_score:.2f}")
    
    while temperature > 0.1 and iteration < CONFIG.SA_MAX_ITERATIONS:
        new_route = current_route.copy()
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        
        new_result = evaluate_route(new_route, distance_matrix, project_info, **kwargs)
        new_score = new_result['final_score']
        delta = new_score - current_score
        
        if delta > 0:
            current_route = new_route
            current_score = new_score
            current_result = new_result
            if new_score > best_score:
                best_route = new_route.copy()
                best_result = new_result
                best_score = new_score
                if iteration % 500 == 0:
                    print(f"迭代 {iteration}: 发现更优解 {best_score:.2f}")
        else:
            accept_prob = np.exp(delta / temperature)
            if random.random() < accept_prob:
                current_route = new_route
                current_score = new_score
                current_result = new_result
        
        temperature *= CONFIG.SA_COOLING_RATE
        iteration += 1
    
    print(f"最终得分: {best_score:.2f}, 总迭代: {iteration}\n")
    return best_route, best_result


# ==========================================
# 第三部分：可视化（改进版）
# ==========================================

def plot_route_timeline(result: Dict, save_path: str):
    """绘制路线时间线甘特图"""
    df = pd.DataFrame(result['timeline_log'])
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'completed': '#4ECDC4', 'waited_for_open': '#FFE66D', 
             'partial_show': '#7DCEA0', 'missed_show': '#FF6B6B', 'closed': '#95A5A6',
             'completed_overtime': '#FF8C42', 'partial_show_overtime': '#58D68D'}
    
    for i, row in df.iterrows():
        start = row['到达']
        duration = row['离开'] - row['到达']
        color = colors.get(row['状态'], '#4ECDC4')
        ax.barh(i, duration, left=start, height=0.6, color=color, 
               edgecolor='black', linewidth=1.5)
        ax.text(start + duration/2, i, row['项目'], ha='center', va='center',
               fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{i+1}" for i in range(len(df))])
    ax.set_xlabel('时间（分钟）', fontsize=12)
    ax.set_ylabel('项目序号', fontsize=12)
    ax.set_title('游览路线时间线', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['completed'], label='正常完成'),
        Patch(facecolor=colors['waited_for_open'], label='等待开放'),
        Patch(facecolor=colors['partial_show'], label='部分观看演出'),
        Patch(facecolor=colors['missed_show'], label='错过演出'),
        Patch(facecolor=colors['closed'], label='已关闭'),
        Patch(facecolor=colors['completed_overtime'], label='超时完成')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 时间线图已保存: {save_path}")
    plt.close()


def plot_route_map_with_timeline(route: List[int], project_info: Dict, 
                                 timeline_log: List[Dict], crowd_type: str, 
                                 date_type: str, save_path: str):
    """
    绘制路线地图（改进版：左侧显示时间-地点链条）
    
    文件命名：Q1-游客类型-日期类型.png
    """
    fig = plt.figure(figsize=(18, 10))
    
    # 左侧：时间-地点链条
    ax_timeline = plt.subplot(1, 3, 1)
    ax_timeline.axis('off')
    
    # 绘制时间链条
    timeline_text = f"【{crowd_type}游客 - {date_type}】\n\n游览时间线：\n\n"
    for i, log in enumerate(timeline_log, 1):
        arrive_h = int(log['到达'] // 60)
        arrive_m = int(log['到达'] % 60)
        leave_h = int(log['离开'] // 60)
        leave_m = int(log['离开'] % 60)
        timeline_text += f"{i}. {log['项目']}\n"
        timeline_text += f"   {arrive_h:02d}:{arrive_m:02d} → {leave_h:02d}:{leave_m:02d}\n"
        if log['排队'] > 0:
            timeline_text += f"   排队: {log['排队']:.0f}分钟\n"
        if log['等待'] > 0:
            timeline_text += f"   等待: {log['等待']:.0f}分钟\n"
        timeline_text += "\n"
    
    ax_timeline.text(0.1, 0.95, timeline_text, transform=ax_timeline.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 右侧：地图
    ax_map = plt.subplot(1, 3, (2, 3))
    
    # 提取所有项目的坐标
    all_lats = [info['lat'] for info in project_info.values() if info['lat']]
    all_lons = [info['lon'] for info in project_info.values() if info['lon']]
    
    # 绘制所有项目（灰色）
    ax_map.scatter(all_lons, all_lats, c='lightgray', s=100, alpha=0.5, zorder=1)
    
    # 绘制路线中的项目（彩色）
    route_lons = [project_info[pid]['lon'] for pid in route if pid in project_info]
    route_lats = [project_info[pid]['lat'] for pid in route if pid in project_info]
    
    # 绘制路线连线
    ax_map.plot(route_lons, route_lats, 'b-', linewidth=2, alpha=0.6, zorder=2)
    
    # 绘制路线节点（彩色渐变）
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route)))
    for i, (pid, lon, lat) in enumerate(zip(route, route_lons, route_lats)):
        ax_map.scatter(lon, lat, c=[colors[i]], s=200, edgecolors='black', linewidths=2, zorder=3)
        ax_map.text(lon, lat, str(i+1), ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
    
    ax_map.set_xlabel('经度', fontsize=12)
    ax_map.set_ylabel('纬度', fontsize=12)
    ax_map.set_title(f'迪士尼乐园游览路线图\n{crowd_type}游客 - {date_type}', 
                    fontsize=16, fontweight='bold')
    ax_map.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 路线图已保存: {save_path}")
    plt.close()


# ==========================================
# 第四部分：主函数
# ==========================================

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("第一部分：数据读取与可视化")
    print("=" * 70 + "\n")
    
    # 确保输出目录存在
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    # 读取项目数据
    project_file = os.path.join(CONFIG.DATA_DIR, 'projects_data.csv')
    project_info, df_projects = load_projects_from_csv(project_file)
    print(f"✓ 成功读取 {len(project_info)} 个项目（包含入口）")
    
    # 读取距离矩阵
    poi_file = os.path.join(CONFIG.DATA_DIR, 'poi.csv')
    distance_matrix, poi_mapping = load_distance_matrix(poi_file)
    
    # 选择人群和日期
    selected_crowd = '普通'
    selected_date = '工作日'
    
    print_config_summary(selected_crowd, selected_date)
    
    # 生成雷达图
    radar_path = os.path.join(CONFIG.OUTPUT_DIR, f'radar_{selected_crowd}.png')
    plot_crowd_radar(selected_crowd, radar_path)
    
    print("\n" + "=" * 70)
    print("第二部分：运行优化算法")
    print("=" * 70 + "\n")
    
    # 计算效用
    project_info = calculate_utility_scores(project_info, selected_crowd)
    
    # 调整排队参数
    multiplier = CONFIG.DATE_QUEUE_MULTIPLIER[selected_date]
    for proj_id, info in project_info.items():
        if info['type'] == 'normal' and proj_id not in [CONFIG.START_NODE, CONFIG.END_NODE]:
            info['base_q'] *= multiplier
            info['peaks'] = [(A * multiplier, mu, sigma) for A, mu, sigma in info['peaks']]
    
    # 项目列表（排除入口0）
    project_ids = [pid for pid in project_info.keys() if pid not in [CONFIG.START_NODE, CONFIG.END_NODE]]
    
    # 运行优化（起点和终点都是0=米奇大街）
    best_route, best_result = simulated_annealing(
        project_ids=project_ids,
        distance_matrix=distance_matrix,
        project_info=project_info,
        start_node=CONFIG.START_NODE,
        end_node=CONFIG.END_NODE,
        return_to_end=True
    )
    
    print("\n" + "=" * 70)
    print("第三部分：结果展示与可视化")
    print("=" * 70 + "\n")
    
    print(f"【综合得分】 {best_result['final_score']:.2f}")
    print(f"【总效用】 {best_result['total_utility']:.2f}")
    print(f"【实际访问】 {best_result['visited_count']}/{len(project_ids)} 个项目")
    print(f"【总耗时】 {best_result['total_time']:.1f}分钟 ({best_result['total_time']/60:.1f}小时)")
    print(f"【排队时间】 {best_result['total_queue']:.1f}分钟")
    print(f"【等待时间】 {best_result['total_wait']:.1f}分钟")
    print(f"【步行时间】 {best_result['total_walk']:.1f}分钟")
    print(f"【错过演出】 {best_result['missed_shows']}场")
    print(f"【关闭项目】 {best_result['closed_projects']}个")
    print(f"【超时情况】 {best_result['overtime']:.1f}分钟")
    print(f"【可行性】 {'是' if best_result['feasible'] else '否'}\n")
    
    print("【推荐路线】（按访问顺序）")
    visited = best_result['visited_projects']
    for i, proj_id in enumerate(visited, 1):
        print(f"  {i}. {project_info[proj_id]['name']}")
    
    if len(visited) < len(best_route):
        print(f"\n  注：原计划{len(best_route)}个项目，实际完成{len(visited)}个")
        print(f"      未访问的项目：{len(best_route) - len(visited)}个（因闭园时间限制）")
    
    print("\n【详细时间线】")
    df_timeline = pd.DataFrame(best_result['timeline_log'])
    print(df_timeline.to_string(index=False))
    
    # 生成可视化（新命名格式）
    timeline_path = os.path.join(CONFIG.OUTPUT_DIR, 
                                 f'Q1-timeline-{selected_crowd}-{selected_date}.png')
    plot_route_timeline(best_result, timeline_path)
    
    route_map_path = os.path.join(CONFIG.OUTPUT_DIR, 
                                  f'Q1-{selected_crowd}-{selected_date}.png')
    plot_route_map_with_timeline(visited, project_info, best_result['timeline_log'],
                                 selected_crowd, selected_date, route_map_path)
    
    print("\n" + "=" * 70)
    # 保存最优路径供Q2使用
    algorithm_names = ['遗传算法', '模拟退火', '蚁群算法']
    save_optimal_route(best_route, best_result, selected_date, selected_crowd,
                      algorithm_names[CONFIG.ALGORITHM], CONFIG.OUTPUT_DIR)
    print("✓ 优化完成！所有结果已保存到 Q1-test 文件夹")
    print("=" * 70)



def save_optimal_route(route: List[int], result: Dict, scenario: str, crowd_type: str,
                      algorithm: str, output_dir: str):
    """
    保存最优路径供Q2使用（修正版）

    修正：保存实际执行的路径（visited_projects）和完整的timeline
    而不是理想的26个项目计划

    保存格式：JSON（便于跨语言读取）
    保存位置：Q1-test/routes/route_{algorithm}_{scenario}_{crowd_type}.json
    """
    import json

    routes_dir = os.path.join(output_dir, 'routes')
    os.makedirs(routes_dir, exist_ok=True)

    filename = f"route_{algorithm}_{scenario}_{crowd_type}.json"
    filepath = os.path.join(routes_dir, filename)

    # 保存实际执行的路径和timeline
    route_data = {
        'algorithm': algorithm,
        'scenario': scenario,
        'crowd_type': crowd_type,
        'planned_route': route,  # 原始计划路径（26个项目）
        'executed_route': result['visited_projects'],  # 实际执行路径（10个左右）
        'timeline': result['timeline_log'],  # 完整的时间线
        'final_score': result['final_score'],
        'total_utility': result['total_utility'],
        'visited_count': result['visited_count'],
        'total_time': result['total_time']
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(route_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 最优路径已保存: {filepath}")
    print(f"  计划路径: {len(route)} 个项目")
    print(f"  实际执行: {result['visited_count']} 个项目")

if __name__ == "__main__":
    main()


