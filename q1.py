"""
迪士尼乐园路线优化系统 - 问题1
支持不同人群（普通游客、亲子游、情侣游）在不同日期（工作日、双休日、节假日）的路线规划
包含三种优化算法：遗传算法、模拟退火、蚁群算法
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
# 配置模块 - 所有超参数集中管理
# ==========================================

class CONFIG:
    """全局配置类 - 集中管理所有超参数"""

    # ========== 算法选择 ==========
    ALGORITHM = 1  # 0=遗传算法, 1=模拟退火, 2=蚁群算法

    # ========== 人群偏好权重矩阵 ==========
    # 格式: [刺激度, 沉浸度, 互动度, 休闲度]
    CROWD_WEIGHTS = {
        '普通': [0.25, 0.25, 0.25, 0.25],
        '亲子': [0.10, 0.20, 0.40, 0.30],
        '情侣': [0.40, 0.40, 0.10, 0.10]
    }

    # ========== 日期类型排队系数 ==========
    DATE_QUEUE_MULTIPLIER = {
        '工作日': 0.7,
        '双休日': 1.0,
        '节假日': 1.5
    }

    # ========== 人的行走速度系数 ==========
    WALK_SPEED_MULTIPLIER = 1.0  # 1.0=正常速度, >1加快, <1减慢

    # ========== GMM动态排队模型参数 ==========
    # 每个项目的峰值参数可在CSV中定义，这里是全局调整系数
    GMM_PEAK_INTENSITY_SCALE = 1.0  # 峰值强度缩放
    GMM_PEAK_WIDTH_SCALE = 1.0      # 峰值宽度缩放

    # ========== 目标函数权重参数 ==========
    # Score = Utility - (α×Queue + β×Wait + γ×Walk) - δ×Overtime² - ε×MissedShows + ζ×Diversity
    WEIGHT_QUEUE_TIME = 0.1      # α: 排队时间惩罚系数
    WEIGHT_WAIT_TIME = 0.15      # β: 等待时间惩罚系数
    WEIGHT_WALK_TIME = 0.05      # γ: 步行时间惩罚系数
    WEIGHT_OVERTIME = 5.0        # δ: 超时惩罚系数（二次）
    WEIGHT_MISSED_SHOW = 1000.0  # ε: 错过演出惩罚
    WEIGHT_DIVERSITY = 2.0       # ζ: 项目多样性奖励

    # ========== 时间约束 ==========
    PARK_OPEN_TIME = 0           # 开园时间（分钟，以0为基准）
    PARK_CLOSE_TIME = 720        # 闭园时间（分钟，12小时）
    START_TIME = 0               # 游客开始游玩时间

    # ========== 遗传算法参数 ==========
    GA_POPULATION_SIZE = 100     # 种群大小
    GA_GENERATIONS = 200         # 迭代代数
    GA_CROSSOVER_RATE = 0.8      # 交叉概率
    GA_MUTATION_RATE = 0.2       # 变异概率
    GA_ELITE_SIZE = 10           # 精英保留数量

    # ========== 模拟退火参数 ==========
    SA_INITIAL_TEMP = 1000       # 初始温度
    SA_COOLING_RATE = 0.995      # 降温速率
    SA_MAX_ITERATIONS = 5000     # 最大迭代次数

    # ========== 蚁群算法参数 ==========
    ACO_ANT_COUNT = 50           # 蚂蚁数量
    ACO_ITERATIONS = 100         # 迭代次数
    ACO_ALPHA = 1.0              # 信息素重要程度
    ACO_BETA = 2.0               # 启发函数重要程度
    ACO_RHO = 0.5                # 信息素挥发系数
    ACO_Q = 100                  # 信息素强度

    # ========== 可视化参数 ==========
    FIGURE_DPI = 100             # 图片分辨率
    FIGURE_SIZE = (12, 8)        # 图片尺寸
    MAP_ZOOM = 15                # 地图缩放级别

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 60)
        print("当前配置参数")
        print("=" * 60)
        algorithm_names = ['遗传算法', '模拟退火', '蚁群算法']
        print(f"优化算法: {algorithm_names[cls.ALGORITHM]}")
        print(f"闭园时间: {cls.PARK_CLOSE_TIME}分钟")
        print(f"行走速度系数: {cls.WALK_SPEED_MULTIPLIER}x")
        print(f"目标函数权重: 排队={cls.WEIGHT_QUEUE_TIME}, 等待={cls.WEIGHT_WAIT_TIME}, "
              f"步行={cls.WEIGHT_WALK_TIME}")
        print("=" * 60 + "\n")


# ==========================================
# 第一部分：数据读取与预处理
# ==========================================

def load_projects_from_csv(csv_file: str) -> Tuple[Dict, pd.DataFrame]:
    """从CSV文件读取项目信息（包含经纬度）"""
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
            ],
            'lat': float(row['纬度']) if '纬度' in row and pd.notna(row['纬度']) else None,
            'lon': float(row['经度']) if '经度' in row and pd.notna(row['经度']) else None
        }

        # 时间窗（仅演出类）
        if info['type'] == 'show':
            info['time_window'] = (float(row['时间窗开始']), float(row['时间窗结束']))
        else:
            # 动态排队参数（应用全局缩放）
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


def calculate_utility_scores(project_info: Dict, crowd_type: str = '普通') -> Dict:
    """计算每个项目对特定人群的效用得分"""
    if crowd_type not in CONFIG.CROWD_WEIGHTS:
        raise ValueError(f"不支持的人群类型: {crowd_type}")

    w = np.array(CONFIG.CROWD_WEIGHTS[crowd_type])

    for proj_id, info in project_info.items():
        features = np.array(info['features'])
        utility = np.dot(features, w) * 10  # 放大到0-10分
        info['utility'] = round(utility, 2)

    return project_info


def get_dynamic_queue_time(t_current: float, base_q: float, peaks: List[Tuple]) -> float:
    """计算特定时刻的排队时间（GMM模型）"""
    q_time = float(base_q)

    for A, mu, sigma in peaks:
        if sigma <= 0:
            raise ValueError(f"sigma必须为正数，当前sigma={sigma}")
        q_time += A * np.exp(-((t_current - mu) ** 2) / (2 * sigma ** 2))

    return max(0.0, q_time)


def evaluate_route(
    route: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    start_time: float = None,
    start_node: int = 0,
    end_node: Optional[int] = None,
    return_to_end: bool = False
) -> Dict:
    """评估路线的综合得分"""
    if start_time is None:
        start_time = CONFIG.START_TIME

    # 应用行走速度系数
    dist_matrix = distance_matrix * CONFIG.WALK_SPEED_MULTIPLIER

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

    if len(route) == 0:
        if return_to_end and end_node is not None:
            walk_time = dist_matrix[start_node, end_node]
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
        walk_time = dist_matrix[current_node, next_node]
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
                missed_shows += 1
                leave_time = arrive_time
            else:
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
        if leave_time > CONFIG.PARK_CLOSE_TIME:
            overtime = max(overtime, leave_time - CONFIG.PARK_CLOSE_TIME)
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
        walk_time = dist_matrix[current_node, end_node]
        current_time += walk_time
        total_walk_time += walk_time

    # 计算综合得分（使用CONFIG中的权重）
    time_cost = (
        CONFIG.WEIGHT_QUEUE_TIME * total_queue_time +
        CONFIG.WEIGHT_WAIT_TIME * total_wait_time +
        CONFIG.WEIGHT_WALK_TIME * total_walk_time
    )
    overtime_penalty = CONFIG.WEIGHT_OVERTIME * (overtime ** 2) if overtime > 0 else 0
    missed_penalty = CONFIG.WEIGHT_MISSED_SHOW * missed_shows

    # 项目类型多样性奖励
    project_types = [project_info[node]['type'] for node in route if node in project_info]
    diversity_bonus = CONFIG.WEIGHT_DIVERSITY * len(set(project_types))

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
        'timeline_log': timeline_log,
        'route': route
    }


# ==========================================
# 第二部分：优化算法
# ==========================================

# ---------- 1. 模拟退火算法 ----------
def simulated_annealing(
    project_ids: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    start_node: int = 0,
    end_node: Optional[int] = None,
    return_to_end: bool = False
) -> Tuple[List[int], Dict]:
    """模拟退火算法"""
    current_route = random.sample(project_ids, len(project_ids))
    current_result = evaluate_route(
        current_route, distance_matrix, project_info,
        start_node=start_node, end_node=end_node, return_to_end=return_to_end
    )
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

        new_result = evaluate_route(
            new_route, distance_matrix, project_info,
            start_node=start_node, end_node=end_node, return_to_end=return_to_end
        )
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
                if iteration % 100 == 0:
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


# ---------- 2. 遗传算法 ----------
def genetic_algorithm(
    project_ids: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    start_node: int = 0,
    end_node: Optional[int] = None,
    return_to_end: bool = False
) -> Tuple[List[int], Dict]:
    """遗传算法"""

    def create_individual():
        return random.sample(project_ids, len(project_ids))

    def fitness(route):
        result = evaluate_route(
            route, distance_matrix, project_info,
            start_node=start_node, end_node=end_node, return_to_end=return_to_end
        )
        return result['final_score']

    def crossover(parent1, parent2):
        """顺序交叉（OX）"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]

        pointer = end
        for gene in parent2[end:] + parent2[:end]:
            if gene not in child:
                if pointer >= size:
                    pointer = 0
                child[pointer] = gene
                pointer += 1

        return child

    def mutate(route):
        """交换变异"""
        if random.random() < CONFIG.GA_MUTATION_RATE:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # 初始化种群
    population = [create_individual() for _ in range(CONFIG.GA_POPULATION_SIZE)]
    best_route = None
    best_score = float('-inf')

    print(f"初始种群大小: {CONFIG.GA_POPULATION_SIZE}")

    for generation in range(CONFIG.GA_GENERATIONS):
        # 评估适应度
        fitness_scores = [(ind, fitness(ind)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # 更新最优解
        if fitness_scores[0][1] > best_score:
            best_route = fitness_scores[0][0].copy()
            best_score = fitness_scores[0][1]
            if generation % 20 == 0:
                print(f"代数 {generation}: 发现更优解 {best_score:.2f}")

        # 精英保留
        new_population = [ind for ind, _ in fitness_scores[:CONFIG.GA_ELITE_SIZE]]

        # 选择、交叉、变异
        while len(new_population) < CONFIG.GA_POPULATION_SIZE:
            # 轮盘赌选择
            total_fitness = sum(max(0, score) for _, score in fitness_scores)
            if total_fitness == 0:
                parent1, parent2 = random.sample(population, 2)
            else:
                weights = [max(0, score) / total_fitness for _, score in fitness_scores]
                parent1, parent2 = random.choices(
                    [ind for ind, _ in fitness_scores],
                    weights=weights,
                    k=2
                )

            # 交叉
            if random.random() < CONFIG.GA_CROSSOVER_RATE:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # 变异
            child = mutate(child)
            new_population.append(child)

        population = new_population

    best_result = evaluate_route(
        best_route, distance_matrix, project_info,
        start_node=start_node, end_node=end_node, return_to_end=return_to_end
    )

    print(f"最终得分: {best_score:.2f}, 总代数: {CONFIG.GA_GENERATIONS}\n")
    return best_route, best_result


# ---------- 3. 蚁群算法 ----------
def ant_colony_optimization(
    project_ids: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    start_node: int = 0,
    end_node: Optional[int] = None,
    return_to_end: bool = False
) -> Tuple[List[int], Dict]:
    """蚁群算法"""
    n = len(project_ids)

    # 初始化信息素矩阵
    pheromone = np.ones((n, n))

    # 启发函数：基于效用和距离
    heuristic = np.zeros((n, n))
    for i, id_i in enumerate(project_ids):
        for j, id_j in enumerate(project_ids):
            if i != j:
                utility = project_info[id_j]['utility']
                dist = distance_matrix[id_i, id_j] + 1e-6
                heuristic[i, j] = utility / dist

    best_route = None
    best_score = float('-inf')

    print(f"蚂蚁数量: {CONFIG.ACO_ANT_COUNT}")

    for iteration in range(CONFIG.ACO_ITERATIONS):
        all_routes = []
        all_scores = []

        # 每只蚂蚁构建路线
        for ant in range(CONFIG.ACO_ANT_COUNT):
            route = []
            visited = set()
            current_idx = random.randint(0, n - 1)

            route.append(project_ids[current_idx])
            visited.add(current_idx)

            # 构建完整路线
            while len(visited) < n:
                # 计算转移概率
                probabilities = []
                candidates = []

                for next_idx in range(n):
                    if next_idx not in visited:
                        tau = pheromone[current_idx, next_idx] ** CONFIG.ACO_ALPHA
                        eta = heuristic[current_idx, next_idx] ** CONFIG.ACO_BETA
                        probabilities.append(tau * eta)
                        candidates.append(next_idx)

                # 轮盘赌选择
                if sum(probabilities) == 0:
                    next_idx = random.choice(candidates)
                else:
                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()
                    next_idx = np.random.choice(candidates, p=probabilities)

                route.append(project_ids[next_idx])
                visited.add(next_idx)
                current_idx = next_idx

            # 评估路线
            result = evaluate_route(
                route, distance_matrix, project_info,
                start_node=start_node, end_node=end_node, return_to_end=return_to_end
            )
            score = result['final_score']

            all_routes.append(route)
            all_scores.append(score)

            # 更新最优解
            if score > best_score:
                best_route = route.copy()
                best_score = score
                best_result = result

        # 信息素挥发
        pheromone *= (1 - CONFIG.ACO_RHO)

        # 信息素更新
        for route, score in zip(all_routes, all_scores):
            if score > 0:  # 只有正得分才留下信息素
                delta_pheromone = CONFIG.ACO_Q * score
                for i in range(len(route) - 1):
                    idx_i = project_ids.index(route[i])
                    idx_j = project_ids.index(route[i + 1])
                    pheromone[idx_i, idx_j] += delta_pheromone

        if iteration % 10 == 0:
            print(f"迭代 {iteration}: 当前最优 {best_score:.2f}")

    print(f"最终得分: {best_score:.2f}, 总迭代: {CONFIG.ACO_ITERATIONS}\n")
    return best_route, best_result


# ---------- 算法选择器 ----------
def optimize_route(
    project_ids: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    algorithm: int = None,
    **kwargs
) -> Tuple[List[int], Dict]:
    """根据配置选择优化算法"""
    if algorithm is None:
        algorithm = CONFIG.ALGORITHM

    algorithms = {
        0: ('遗传算法', genetic_algorithm),
        1: ('模拟退火', simulated_annealing),
        2: ('蚁群算法', ant_colony_optimization)
    }

    if algorithm not in algorithms:
        raise ValueError(f"不支持的算法编号: {algorithm}")

    name, func = algorithms[algorithm]
    print(f"=== 使用 {name} 进行优化 ===")

    return func(project_ids, distance_matrix, project_info, **kwargs)


# ==========================================
# 第三部分：可视化模块
# ==========================================

def plot_crowd_radar(crowd_type: str, save_path: str = 'crowd_radar.png'):
    """绘制人物画像雷达图"""
    weights = CONFIG.CROWD_WEIGHTS[crowd_type]
    categories = ['刺激度', '沉浸度', '互动度', '休闲度']

    # 闭合雷达图
    values = weights + [weights[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label=crowd_type)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 0.5)
    ax.set_title(f'{crowd_type}游客偏好画像', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"雷达图已保存: {save_path}")
    plt.close()


def plot_route_map(
    route: List[int],
    project_info: Dict,
    start_node: int = 0,
    save_path: str = 'route_map.png'
):
    """绘制路线地图（基于经纬度）"""
    # 提取坐标
    lats = []
    lons = []
    names = []

    # 起点（假设入口坐标）
    if start_node == 0:
        # 使用第一个项目附近作为入口
        first_proj = project_info[route[0]]
        if first_proj['lat'] and first_proj['lon']:
            lats.append(first_proj['lat'] + 0.001)
            lons.append(first_proj['lon'] - 0.001)
            names.append('入口')

    # 路线中的项目
    for proj_id in route:
        info = project_info[proj_id]
        if info['lat'] and info['lon']:
            lats.append(info['lat'])
            lons.append(info['lon'])
            names.append(info['name'])

    if len(lats) < 2:
        print("警告：经纬度数据不足，无法绘制地图")
        return

    # 绘制地图
    fig, ax = plt.subplots(figsize=CONFIG.FIGURE_SIZE)

    # 绘制路线
    ax.plot(lons, lats, 'b-', linewidth=2, alpha=0.6, label='游览路线')

    # 绘制项目点
    ax.scatter(lons[1:], lats[1:], c='red', s=100, zorder=5, label='项目')
    ax.scatter(lons[0], lats[0], c='green', s=150, marker='s', zorder=5, label='入口')

    # 标注项目名称
    for i, (lon, lat, name) in enumerate(zip(lons, lats, names)):
        ax.annotate(
            f"{i}. {name}" if i > 0 else name,
            (lon, lat),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    ax.set_title('迪士尼乐园游览路线图', fontsize=16)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"路线图已保存: {save_path}")
    plt.close()


def compare_algorithms(
    project_ids: List[int],
    distance_matrix: np.ndarray,
    project_info: Dict,
    **kwargs
):
    """对比三种算法的性能"""
    print("\n" + "=" * 60)
    print("算法性能对比")
    print("=" * 60 + "\n")

    results = {}
    algorithms = {
        0: '遗传算法',
        1: '模拟退火',
        2: '蚁群算法'
    }

    for alg_id, alg_name in algorithms.items():
        print(f"\n--- {alg_name} ---")
        route, result = optimize_route(
            project_ids, distance_matrix, project_info,
            algorithm=alg_id, **kwargs
        )
        results[alg_name] = {
            'score': result['final_score'],
            'utility': result['total_utility'],
            'time': result['total_time'],
            'route': route
        }

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(results.keys())
    scores = [results[n]['score'] for n in names]
    utilities = [results[n]['utility'] for n in names]
    times = [results[n]['time'] for n in names]

    axes[0].bar(names, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('综合得分对比', fontsize=14)
    axes[0].set_ylabel('得分')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(names, utilities, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_title('总效用对比', fontsize=14)
    axes[1].set_ylabel('效用')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[2].set_title('总耗时对比', fontsize=14)
    axes[2].set_ylabel('时间（分钟）')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=CONFIG.FIGURE_DPI)
    print(f"\n算法对比图已保存: algorithm_comparison.png")
    plt.close()

    # 打印对比表格
    print("\n" + "=" * 60)
    print("算法性能汇总")
    print("=" * 60)
    df_compare = pd.DataFrame(results).T
    df_compare = df_compare[['score', 'utility', 'time']]
    df_compare.columns = ['综合得分', '总效用', '总耗时(分钟)']
    print(df_compare.to_string())
    print("=" * 60 + "\n")

    return results


# ==========================================
# 第四部分：主函数
# ==========================================

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("迪士尼乐园路线优化系统")
    print("=" * 60 + "\n")

    # 打印配置
    CONFIG.print_config()

    # 1. 读取数据
    print("=" * 60)
    print("步骤1：读取项目数据")
    print("=" * 60)
    project_info, df = load_projects_from_csv('projects_data.csv')

    # 2. 输入距离矩阵
    print("\n" + "=" * 60)
    print("步骤2：配置距离矩阵")
    print("=" * 60)

    # 示例距离矩阵（7×7：0=入口, 1-6=项目）
    distance_matrix = np.array([
        [0, 10, 15, 12, 8, 20, 18],
        [10, 0, 8, 10, 12, 15, 20],
        [15, 8, 0, 6, 10, 12, 18],
        [12, 10, 6, 0, 8, 10, 15],
        [8, 12, 10, 8, 0, 14, 16],
        [20, 15, 12, 10, 14, 0, 8],
        [18, 20, 18, 15, 16, 8, 0]
    ])
    print(f"距离矩阵维度: {distance_matrix.shape}\n")

    # 3. 选择人群和日期
    print("=" * 60)
    print("步骤3：选择人群类型和日期类型")
    print("=" * 60)

    selected_crowd = '普通'  # 可选: '普通', '亲子', '情侣'
    selected_date = '工作日'  # 可选: '工作日', '双休日', '节假日'

    print(f"人群类型: {selected_crowd}")
    print(f"日期类型: {selected_date}\n")

    # 4. 计算效用
    print("=" * 60)
    print("步骤4：计算项目效用得分")
    print("=" * 60)
    project_info = calculate_utility_scores(project_info, selected_crowd)

    print("项目效用得分:")
    for proj_id, info in project_info.items():
        print(f"  {proj_id}. {info['name']}: {info['utility']:.2f}分")

    # 5. 调整排队参数
    print("\n" + "=" * 60)
    print("步骤5：调整排队参数")
    print("=" * 60)

    multiplier = CONFIG.DATE_QUEUE_MULTIPLIER[selected_date]
    for proj_id, info in project_info.items():
        if info['type'] == 'normal':
            info['base_q'] *= multiplier
            info['peaks'] = [(A * multiplier, mu, sigma) for A, mu, sigma in info['peaks']]

    print(f"排队时间调整系数: {multiplier}x\n")

    # 6. 运行优化
    print("=" * 60)
    print("步骤6：运行优化算法")
    print("=" * 60 + "\n")

    project_ids = list(project_info.keys())

    # 选择单个算法或对比所有算法
    run_comparison = False  # 设为True可对比三种算法

    if run_comparison:
        results = compare_algorithms(
            project_ids=project_ids,
            distance_matrix=distance_matrix,
            project_info=project_info,
            start_node=0,
            end_node=0,
            return_to_end=True
        )
        # 使用最优算法的结果
        best_alg = max(results.items(), key=lambda x: x[1]['score'])
        best_route = best_alg[1]['route']
        best_result = evaluate_route(
            best_route, distance_matrix, project_info,
            start_node=0, end_node=0, return_to_end=True
        )
    else:
        best_route, best_result = optimize_route(
            project_ids=project_ids,
            distance_matrix=distance_matrix,
            project_info=project_info,
            start_node=0,
            end_node=0,
            return_to_end=True
        )

    # 7. 输出结果
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

    # 8. 生成可视化
    print("\n" + "=" * 60)
    print("步骤7：生成可视化结果")
    print("=" * 60 + "\n")

    # 人物画像雷达图
    plot_crowd_radar(selected_crowd, f'radar_{selected_crowd}.png')

    # 路线地图
    plot_route_map(best_route, project_info, start_node=0, save_path='route_map.png')

    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()








