"""
迪士尼乐园路线优化系统 - 问题1
支持不同人群（普通游客、亲子游、情侣游）在不同日期（工作日、双休日、节假日）的路线规划
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional


# ==========================================
# 第一部分：数据读取与预处理
# ==========================================

def load_projects_from_csv(csv_file: str) -> Tuple[Dict, pd.DataFrame]:
    """
    从CSV文件读取项目信息

    返回:
        project_info: 项目信息字典
        df_projects: 原始DataFrame
    """
    df = pd.read_csv(csv_file)
    print(f"=== 成功读取 {len(df)} 个项目 ===\n")

    project_info = {}

    for idx, row in df.iterrows():
        proj_id = int(row['项目ID'])

        info = {
            'name': row['项目名称'],
            'duration': float(row['游玩时长']),
            'type': 'show' if row['是否演出'] == '是' else 'normal',
            'features': [
                float(row['刺激度']),
                float(row['沉浸度']),
                float(row['互动度']),
                float(row['休闲度'])
            ]
        }

        # 时间窗（仅演出类）
        if info['type'] == 'show':
            info['time_window'] = (float(row['时间窗开始']), float(row['时间窗结束']))
        else:
            # 动态排队参数
            info['base_q'] = float(row['基础排队'])
            peaks = []
            if pd.notna(row['峰值1强度']):
                peaks.append((
                    float(row['峰值1强度']),
                    float(row['峰值1时间']),
                    float(row['峰值1宽度'])
                ))
            if pd.notna(row['峰值2强度']):
                peaks.append((
                    float(row['峰值2强度']),
                    float(row['峰值2时间']),
                    float(row['峰值2宽度'])
                ))
            info['peaks'] = peaks

        project_info[proj_id] = info

    return project_info, df


def calculate_utility_scores(project_info: Dict, crowd_type: str = '普通') -> Dict:
    """
    计算每个项目对特定人群的效用得分

    参数:
        project_info: 项目信息字典
        crowd_type: 人群类型 ['普通', '亲子', '情侣']

    返回:
        包含效用得分的项目信息字典
    """
    # 人群偏好权重矩阵 W (4维 -> 3类人群)
    weights = {
        '普通': [0.25, 0.25, 0.25, 0.25],  # 刺激、沉浸、互动、休闲
        '亲子': [0.10, 0.20, 0.40, 0.30],
        '情侣': [0.40, 0.40, 0.10, 0.10]
    }

    if crowd_type not in weights:
        raise ValueError(f"不支持的人群类型: {crowd_type}")

    w = np.array(weights[crowd_type])

    # 计算每个项目的效用得分
    for proj_id, info in project_info.items():
        features = np.array(info['features'])
        utility = np.dot(features, w) * 10  # 放大到0-10分
        info['utility'] = round(utility, 2)

    return project_info


# ==========================================
# 第二部分：动态排队时间模型
# ==========================================

def get_dynamic_queue_time(t_current: float, base_q: float, peaks: List[Tuple]) -> float:
    """
    计算特定时刻的排队时间（多峰高斯混合模型）

    参数:
        t_current: 当前时刻（以开园为0的分钟数）
        base_q: 基础排队时间
        peaks: 高斯峰参数列表 [(A1, mu1, sigma1), ...]

    返回:
        预计排队时间（分钟）
    """
    q_time = float(base_q)

    for A, mu, sigma in peaks:
        if sigma <= 0:
            raise ValueError(f"sigma必须为正数，当前sigma={sigma}")
        q_time += A * np.exp(-((t_current - mu) ** 2) / (2 * sigma ** 2))

    return max(0.0, q_time)


# ==========================================
# 第三部分：路线评估函数（改进版）
# ==========================================

def evaluate_route(
    route: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    start_time: float = 0,
    start_node: int = 0,
    end_node: Optional[int] = None,
    park_close_time: float = 720,
    return_to_end: bool = False
) -> Dict:
    """
    评估路线的综合得分（改进版目标函数）

    参数:
        route: 项目序列 [1, 3, 5, ...]
        distance_matrix: n×n距离矩阵（分钟）
        project_info: 项目信息字典
        start_time: 开始时刻（分钟，以开园为0）
        start_node: 起点节点
        end_node: 终点节点
        park_close_time: 闭园时间（分钟）
        return_to_end: 是否返回终点

    返回:
        包含得分、时间线等信息的字典
    """
    current_time = float(start_time)
    current_node = start_node

    total_utility = 0.0
    total_queue_time = 0.0
    total_wait_time = 0.0
    total_walk_time = 0.0
    missed_shows = 0
    overtime = 0.0

    timeline_log = []
    feasible = True

    # 空路线处理
    if len(route) == 0:
        if return_to_end and end_node is not None:
            walk_time = distance_matrix[start_node, end_node]
            current_time += walk_time
            total_walk_time += walk_time
        return {
            'final_score': 0.0,
            'total_utility': 0.0,
            'total_time': current_time - start_time,
            'feasible': True,
            'timeline_log': []
        }

    # 逐个项目推演
    for next_node in route:
        if next_node not in project_info:
            continue

        p_info = project_info[next_node]
        walk_time = distance_matrix[current_node, next_node]
        arrive_time = current_time + walk_time
        total_walk_time += walk_time

        name = p_info['name']
        node_type = p_info['type']
        play_duration = p_info['duration']
        utility_score = p_info['utility']

        wait_time = 0.0
        queue_time = 0.0
        gained_utility = 0.0

        # 演出类项目
        if node_type == 'show':
            E_start, L_end = p_info['time_window']

            if arrive_time > L_end:
                # 错过演出
                missed_shows += 1
                leave_time = arrive_time
            else:
                # 等待演出开始
                wait_time = max(0.0, E_start - arrive_time)
                total_wait_time += wait_time
                actual_start = arrive_time + wait_time
                leave_time = actual_start + play_duration
                gained_utility = utility_score

        # 普通项目
        else:
            queue_time = get_dynamic_queue_time(
                arrive_time,
                p_info['base_q'],
                p_info['peaks']
            )
            total_queue_time += queue_time
            actual_start = arrive_time + queue_time
            leave_time = actual_start + play_duration
            gained_utility = utility_score

        # 检查是否超时
        if leave_time > park_close_time:
            overtime = max(overtime, leave_time - park_close_time)
            feasible = False

        total_utility += gained_utility

        timeline_log.append({
            '项目': name,
            '到达': round(arrive_time, 1),
            '排队': round(queue_time, 1),
            '等待': round(wait_time, 1),
            '离开': round(leave_time, 1),
            '效用': round(gained_utility, 2)
        })

        current_time = leave_time
        current_node = next_node

    # 返回终点
    if return_to_end and end_node is not None and current_node != end_node:
        walk_time = distance_matrix[current_node, end_node]
        current_time += walk_time
        total_walk_time += walk_time

    # 计算综合得分（改进版目标函数）
    time_cost = 0.1 * total_queue_time + 0.15 * total_wait_time + 0.05 * total_walk_time
    overtime_penalty = 5 * (overtime ** 2) if overtime > 0 else 0
    missed_penalty = 1000 * missed_shows

    # 项目类型多样性奖励
    project_types = [project_info[node]['type'] for node in route if node in project_info]
    diversity_bonus = len(set(project_types)) * 2

    final_score = (
        total_utility
        - time_cost
        - overtime_penalty
        - missed_penalty
        + diversity_bonus
    )

    return {
        'final_score': round(final_score, 2),
        'total_utility': round(total_utility, 2),
        'total_time': round(current_time - start_time, 1),
        'total_queue': round(total_queue_time, 1),
        'total_wait': round(total_wait_time, 1),
        'total_walk': round(total_walk_time, 1),
        'missed_shows': missed_shows,
        'overtime': round(overtime, 1),
        'feasible': feasible,
        'timeline_log': timeline_log
    }


# ==========================================
# 第四部分：模拟退火优化算法
# ==========================================

def simulated_annealing(
    project_ids: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    start_node: int = 0,
    end_node: Optional[int] = None,
    park_close_time: float = 720,
    initial_temp: float = 1000,
    cooling_rate: float = 0.995,
    max_iterations: int = 5000,
    return_to_end: bool = False
) -> Tuple[List[int], Dict]:
    """
    模拟退火算法寻找最优路线

    参数:
        project_ids: 可选项目ID列表
        distance_matrix: 距离矩阵
        project_info: 项目信息
        其他参数同evaluate_route

    返回:
        best_route: 最优路线
        best_result: 最优结果
    """
    # 初始化：随机路线
    current_route = random.sample(project_ids, len(project_ids))
    current_result = evaluate_route(
        current_route, distance_matrix, project_info,
        start_node=start_node, end_node=end_node,
        park_close_time=park_close_time, return_to_end=return_to_end
    )
    current_score = current_result['final_score']

    best_route = current_route.copy()
    best_result = current_result
    best_score = current_score

    temperature = initial_temp
    iteration = 0

    print(f"=== 开始模拟退火优化 ===")
    print(f"初始得分: {current_score:.2f}\n")

    while temperature > 0.1 and iteration < max_iterations:
        # 生成邻域解：随机交换两个项目
        new_route = current_route.copy()
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        # 评估新路线
        new_result = evaluate_route(
            new_route, distance_matrix, project_info,
            start_node=start_node, end_node=end_node,
            park_close_time=park_close_time, return_to_end=return_to_end
        )
        new_score = new_result['final_score']

        # 计算接受概率
        delta = new_score - current_score

        if delta > 0:
            # 更好的解，直接接受
            current_route = new_route
            current_score = new_score
            current_result = new_result

            # 更新最优解
            if new_score > best_score:
                best_route = new_route.copy()
                best_result = new_result
                best_score = new_score
                print(f"迭代 {iteration}: 发现更优解 {best_score:.2f}")

        else:
            # 较差的解，以一定概率接受
            accept_prob = np.exp(delta / temperature)
            if random.random() < accept_prob:
                current_route = new_route
                current_score = new_score
                current_result = new_result

        # 降温
        temperature *= cooling_rate
        iteration += 1

        # 每500次迭代输出进度
        if iteration % 500 == 0:
            print(f"迭代 {iteration}: 当前温度={temperature:.2f}, 最优得分={best_score:.2f}")

    print(f"\n=== 优化完成 ===")
    print(f"最终得分: {best_score:.2f}")
    print(f"总迭代次数: {iteration}")

    return best_route, best_result


# ==========================================
# 第五部分：主函数
# ==========================================

def main():
    """主函数：演示完整流程"""

    # 1. 读取CSV数据
    print("=" * 60)
    print("步骤1：读取项目数据")
    print("=" * 60)
    project_info, df = load_projects_from_csv('projects_data.csv')

    # 2. 手动输入距离矩阵（n×n）
    print("\n" + "=" * 60)
    print("步骤2：输入距离矩阵")
    print("=" * 60)
    print("请输入距离矩阵（包含入口节点0和所有项目节点）")
    print("示例格式：7×7矩阵（0=入口, 1-6=项目）\n")

    # 示例距离矩阵（可替换为实际数据）
    distance_matrix = np.array([
        [0, 10, 15, 12, 8, 20, 18],   # 从入口到各项目
        [10, 0, 8, 10, 12, 15, 20],   # 项目1到其他
        [15, 8, 0, 6, 10, 12, 18],    # 项目2到其他
        [12, 10, 6, 0, 8, 10, 15],    # 项目3到其他
        [8, 12, 10, 8, 0, 14, 16],    # 项目4到其他
        [20, 15, 12, 10, 14, 0, 8],   # 项目5到其他
        [18, 20, 18, 15, 16, 8, 0]    # 项目6到其他
    ])

    print(f"距离矩阵维度: {distance_matrix.shape}")
    print(f"矩阵预览:\n{distance_matrix}\n")

    # 3. 选择人群类型
    print("=" * 60)
    print("步骤3：选择人群类型和日期类型")
    print("=" * 60)

    crowd_types = ['普通', '亲子', '情侣']
    date_types = ['工作日', '双休日', '节假日']

    print(f"人群类型: {crowd_types}")
    print(f"日期类型: {date_types}\n")

    # 示例：为普通游客在工作日规划路线
    selected_crowd = '普通'
    selected_date = '工作日'

    print(f"当前选择: {selected_crowd} - {selected_date}\n")

    # 4. 计算效用得分
    print("=" * 60)
    print("步骤4：计算项目效用得分")
    print("=" * 60)
    project_info = calculate_utility_scores(project_info, selected_crowd)

    print("项目效用得分:")
    for proj_id, info in project_info.items():
        print(f"  {proj_id}. {info['name']}: {info['utility']:.2f}分")

    # 5. 设置优化参数
    print("\n" + "=" * 60)
    print("步骤5：运行模拟退火优化")
    print("=" * 60)

    # 根据日期类型调整排队参数（工作日人少，节假日人多）
    date_multipliers = {
        '工作日': 0.7,
        '双休日': 1.0,
        '节假日': 1.5
    }

    multiplier = date_multipliers[selected_date]
    for proj_id, info in project_info.items():
        if info['type'] == 'normal':
            info['base_q'] *= multiplier
            info['peaks'] = [(A * multiplier, mu, sigma) for A, mu, sigma in info['peaks']]

    print(f"排队时间调整系数: {multiplier}x\n")

    # 可选项目列表（排除入口节点0）
    project_ids = list(project_info.keys())

    # 运行优化
    best_route, best_result = simulated_annealing(
        project_ids=project_ids,
        distance_matrix=distance_matrix,
        project_info=project_info,
        start_node=0,
        end_node=0,
        park_close_time=720,  # 12小时 = 720分钟
        initial_temp=1000,
        cooling_rate=0.995,
        max_iterations=5000,
        return_to_end=True
    )

    # 6. 输出结果
    print("\n" + "=" * 60)
    print("最优路线规划结果")
    print("=" * 60)

    print(f"\n人群类型: {selected_crowd}")
    print(f"日期类型: {selected_date}")
    print(f"综合得分: {best_result['final_score']:.2f}")
    print(f"总效用: {best_result['total_utility']:.2f}")
    print(f"总耗时: {best_result['total_time']:.1f}分钟")
    print(f"排队时间: {best_result['total_queue']:.1f}分钟")
    print(f"等待时间: {best_result['total_wait']:.1f}分钟")
    print(f"步行时间: {best_result['total_walk']:.1f}分钟")
    print(f"错过演出: {best_result['missed_shows']}场")
    print(f"超时: {best_result['overtime']:.1f}分钟")
    print(f"可行性: {'是' if best_result['feasible'] else '否'}")

    print(f"\n推荐路线:")
    for i, proj_id in enumerate(best_route, 1):
        print(f"  {i}. {project_info[proj_id]['name']}")

    print(f"\n详细时间线:")
    df_timeline = pd.DataFrame(best_result['timeline_log'])
    print(df_timeline.to_string(index=False))


if __name__ == "__main__":
    main()








