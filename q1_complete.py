"""
迪士尼乐园路线优化系统 - 完整版（修正版）
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


class CONFIG:
    """全局配置类"""
    ALGORITHM = 1
    CROWD_WEIGHTS = {
        '普通': [0.25, 0.25, 0.25, 0.25],
        '亲子': [0.10, 0.20, 0.40, 0.30],
        '情侣': [0.40, 0.40, 0.10, 0.10]
    }
    DATE_QUEUE_MULTIPLIER = {'工作日': 0.7, '双休日': 1.0, '节假日': 1.5}
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
    SA_INITIAL_TEMP = 1000
    SA_COOLING_RATE = 0.995
    SA_MAX_ITERATIONS = 5000
    FIGURE_DPI = 100
    FIGURE_SIZE = (12, 8)


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
            'features': [float(row['刺激度']), float(row['沉浸度']), 
                        float(row['互动度']), float(row['休闲度'])]
        }
        
        if pd.notna(row['时间窗开始']) and pd.notna(row['时间窗结束']):
            info['time_window'] = (float(row['时间窗开始']), float(row['时间窗结束']))
        else:
            info['time_window'] = (CONFIG.PARK_OPEN_TIME, CONFIG.PARK_CLOSE_TIME)
        
        if info['type'] == 'normal':
            info['base_q'] = float(row['基础排队'])
            peaks = []
            if pd.notna(row['峰值1强度']):
                peaks.append((float(row['峰值1强度']) * CONFIG.GMM_PEAK_INTENSITY_SCALE,
                            float(row['峰值1时间']), 
                            float(row['峰值1宽度']) * CONFIG.GMM_PEAK_WIDTH_SCALE))
            if pd.notna(row['峰值2强度']):
                peaks.append((float(row['峰值2强度']) * CONFIG.GMM_PEAK_INTENSITY_SCALE,
                            float(row['峰值2时间']),
                            float(row['峰值2宽度']) * CONFIG.GMM_PEAK_WIDTH_SCALE))
            info['peaks'] = peaks
        
        project_info[proj_id] = info
    
    return project_info, df


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


def calculate_utility_scores(project_info: Dict, crowd_type: str = '普通') -> Dict:
    """计算效用得分"""
    w = np.array(CONFIG.CROWD_WEIGHTS[crowd_type])
    for proj_id, info in project_info.items():
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
                  end_node: Optional[int] = None, return_to_end: bool = False) -> Dict:
    """评估路线"""
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
    
    if len(route) == 0:
        if return_to_end and end_node is not None:
            walk_time = dist_matrix[start_node, end_node]
            current_time += walk_time
            total_walk_time += walk_time
        return {
            'final_score': 0.0, 'total_utility': 0.0, 'total_time': current_time - start_time,
            'feasible': True, 'timeline_log': [], 'route': route
        }
    
    # 记录实际访问的项目
    visited_projects = []

    for next_node in route:
        if next_node not in project_info:
            continue

        p_info = project_info[next_node]
        walk_time = dist_matrix[current_node, next_node]
        arrive_time = current_time + walk_time

        # 关键修改1：如果在路上就已经超过闭园时间，直接终止
        if arrive_time >= CONFIG.PARK_CLOSE_TIME:
            # print(f"  [提前终止] 前往 {p_info['name']} 的路上已超过闭园时间 ({arrive_time:.1f} >= {CONFIG.PARK_CLOSE_TIME})")
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

        # 检查是否错过项目（但不终止循环，继续尝试下一个）
        if arrive_time > time_window_end:
            if node_type == 'show':
                missed_shows += 1
                status = 'missed_show'
            else:
                closed_projects += 1
                status = 'closed'
            leave_time = arrive_time
            gained_utility = 0.0
        else:
            # 项目开放中，计算游玩时间
            if node_type == 'show':
                wait_time += max(0.0, time_window_start - arrive_time)
                total_wait_time += max(0.0, time_window_start - arrive_time)
                actual_start = max(arrive_time, time_window_start)
                leave_time = actual_start + play_duration
                gained_utility = utility_score
            else:
                queue_time = get_dynamic_queue_time(arrive_time, p_info['base_q'], p_info['peaks'])
                total_queue_time += queue_time
                actual_start = arrive_time + queue_time
                leave_time = actual_start + play_duration
                gained_utility = utility_score

        # 关键修改2：如果游玩结束后超过闭园时间，记录但终止循环
        if leave_time > CONFIG.PARK_CLOSE_TIME:
            overtime = leave_time - CONFIG.PARK_CLOSE_TIME
            feasible = False
            # print(f"  [提前终止] {name} 游玩结束时超过闭园时间 ({leave_time:.1f} > {CONFIG.PARK_CLOSE_TIME})")

            # 记录这个项目（虽然超时但已经玩了）
            total_utility += gained_utility
            timeline_log.append({
                '项目': name, '到达': round(arrive_time, 1), '排队': round(queue_time, 1),
                '等待': round(wait_time, 1), '离开': round(leave_time, 1),
                '效用': round(gained_utility, 2), '状态': status + '_overtime'
            })
            visited_projects.append(next_node)
            current_time = leave_time
            current_node = next_node
            break  # 终止循环，不再访问后续项目

        # 正常完成项目
        total_utility += gained_utility
        timeline_log.append({
            '项目': name, '到达': round(arrive_time, 1), '排队': round(queue_time, 1),
            '等待': round(wait_time, 1), '离开': round(leave_time, 1),
            '效用': round(gained_utility, 2), '状态': status
        })

        visited_projects.append(next_node)
        current_time = leave_time
        current_node = next_node
    
    # 返回终点（只有在没超时的情况下才考虑返回）
    if return_to_end and end_node is not None and current_node != end_node:
        walk_time = dist_matrix[current_node, end_node]
        # 检查返回路上是否超时
        if current_time + walk_time <= CONFIG.PARK_CLOSE_TIME:
            current_time += walk_time
            total_walk_time += walk_time
        else:
            # 返回路上超时，不返回了
            pass

    # 计算综合得分（关键修改：移除对未访问项目的惩罚）
    time_cost = (CONFIG.WEIGHT_QUEUE_TIME * total_queue_time +
                CONFIG.WEIGHT_WAIT_TIME * total_wait_time +
                CONFIG.WEIGHT_WALK_TIME * total_walk_time)

    # 超时惩罚：只对实际超时的情况惩罚，且惩罚减轻
    overtime_penalty = CONFIG.WEIGHT_OVERTIME * (overtime ** 2) if overtime > 0 else 0

    # 错过项目惩罚：只对实际尝试但错过的项目惩罚（不包括未访问的项目）
    # 注意：这里的missed_shows和closed_projects只统计了实际到达但无法游玩的项目
    missed_penalty = CONFIG.WEIGHT_MISSED_SHOW * (missed_shows + closed_projects)

    # 多样性奖励：基于实际访问的项目
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
        'visited_projects': visited_projects,  # 新增：实际访问的项目列表
        'visited_count': len(visited_projects)  # 新增：实际访问的项目数量
    }


def simulated_annealing(project_ids: List[int], distance_matrix: np.ndarray, 
                       project_info: Dict, **kwargs) -> Tuple[List[int], Dict]:
    """模拟退火算法"""
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


def plot_route_timeline(result: Dict, save_path: str = 'timeline.png'):
    """绘制路线时间线甘特图"""
    df = pd.DataFrame(result['timeline_log'])
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'completed': '#4ECDC4', 'waited_for_open': '#FFE66D', 
             'missed_show': '#FF6B6B', 'closed': '#95A5A6'}
    
    for i, row in df.iterrows():
        start = row['到达']
        duration = row['离开'] - row['到达']
        color = colors.get(row['状态'], '#4ECDC4')
        ax.barh(i, duration, left=start, height=0.6, color=color, 
               edgecolor='black', linewidth=1.5)
        ax.text(start + duration/2, i, row['项目'], ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{i+1}" for i in range(len(df))])
    ax.set_xlabel('时间（分钟）', fontsize=12)
    ax.set_ylabel('项目序号', fontsize=12)
    ax.set_title('游览路线时间线', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['completed'], label='正常完成'),
                      Patch(facecolor=colors['waited_for_open'], label='等待开放'),
                      Patch(facecolor=colors['missed_show'], label='错过演出'),
                      Patch(facecolor=colors['closed'], label='已关闭')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 时间线图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("第一部分：数据读取与可视化")
    print("=" * 70 + "\n")
    
    project_info, df = load_projects_from_csv('projects_data.csv')
    print(f"✓ 成功读取 {len(project_info)} 个项目\n")
    
    selected_crowd = '普通'
    selected_date = '工作日'
    
    print_config_summary(selected_crowd, selected_date)
    plot_crowd_radar(selected_crowd, f'radar_{selected_crowd}.png')
    
    print("\n" + "=" * 70)
    print("第二部分：运行优化算法")
    print("=" * 70 + "\n")
    
    project_info = calculate_utility_scores(project_info, selected_crowd)
    
    multiplier = CONFIG.DATE_QUEUE_MULTIPLIER[selected_date]
    for proj_id, info in project_info.items():
        if info['type'] == 'normal':
            info['base_q'] *= multiplier
            info['peaks'] = [(A * multiplier, mu, sigma) for A, mu, sigma in info['peaks']]
    
    n = len(project_info) + 1
    distance_matrix = np.random.randint(5, 20, size=(n, n))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    project_ids = list(project_info.keys())
    
    best_route, best_result = simulated_annealing(
        project_ids=project_ids, distance_matrix=distance_matrix,
        project_info=project_info, start_node=0, end_node=0, return_to_end=True
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
    if len(df_timeline) > 10:
        print(f"... (共{len(df_timeline)}条记录)")
    
    plot_route_timeline(best_result, 'timeline.png')
    
    print("\n" + "=" * 70)
    print("✓ 优化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
