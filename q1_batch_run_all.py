"""
批量运行所有27种情况
3种人群类型 × 3种日期类型 × 3种优化算法 = 27种组合
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

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, SCRIPT_DIR)

from q1_final_show_fixed import (
    CONFIG, load_projects_from_csv, load_distance_matrix,
    calculate_utility_scores, evaluate_route,
    plot_crowd_radar, plot_route_map_with_timeline, plot_route_timeline
)


def simulated_annealing(project_ids: List[int], distance_matrix: np.ndarray,
                       project_info: Dict, **kwargs) -> Tuple[List[int], Dict]:
    current_route = random.sample(project_ids, len(project_ids))
    current_result = evaluate_route(current_route, distance_matrix, project_info, **kwargs)
    current_score = current_result['final_score']

    best_route = current_route.copy()
    best_result = current_result
    best_score = current_score

    temperature = CONFIG.SA_INITIAL_TEMP
    iteration = 0

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
        else:
            accept_prob = np.exp(delta / temperature)
            if random.random() < accept_prob:
                current_route = new_route
                current_score = new_score
                current_result = new_result

        temperature *= CONFIG.SA_COOLING_RATE
        iteration += 1

    return best_route, best_result


def genetic_algorithm(project_ids: List[int], distance_matrix: np.ndarray,
                     project_info: Dict, **kwargs) -> Tuple[List[int], Dict]:
    population_size = CONFIG.GA_POPULATION_SIZE
    generations = CONFIG.GA_GENERATIONS

    population = [random.sample(project_ids, len(project_ids)) for _ in range(population_size)]

    def fitness(route):
        result = evaluate_route(route, distance_matrix, project_info, **kwargs)
        return result['final_score']

    best_route = None
    best_score = float('-inf')

    for _ in range(generations):
        fitness_scores = [(ind, fitness(ind)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        if fitness_scores[0][1] > best_score:
            best_route = fitness_scores[0][0].copy()
            best_score = fitness_scores[0][1]

        new_population = [ind for ind, _ in fitness_scores[:CONFIG.GA_ELITE_SIZE]]

        while len(new_population) < population_size:
            total_fitness = sum(max(0, score) for _, score in fitness_scores)
            if total_fitness == 0:
                parent1, parent2 = random.sample(population, 2)
            else:
                probs = [max(0, score) / total_fitness for _, score in fitness_scores]
                parent1 = random.choices([ind for ind, _ in fitness_scores], weights=probs)[0]
                parent2 = random.choices([ind for ind, _ in fitness_scores], weights=probs)[0]

            if random.random() < CONFIG.GA_CROSSOVER_RATE:
                size = len(parent1)
                start, end = sorted(random.sample(range(size), 2))
                child = [None] * size
                child[start:end] = parent1[start:end]
                pointer = 0
                for gene in parent2:
                    if gene not in child:
                        while child[pointer] is not None:
                            pointer += 1
                        child[pointer] = gene
            else:
                child = parent1.copy()

            if random.random() < CONFIG.GA_MUTATION_RATE:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]

            new_population.append(child)

        population = new_population

    best_result = evaluate_route(best_route, distance_matrix, project_info, **kwargs)
    return best_route, best_result


def ant_colony_optimization(project_ids: List[int], distance_matrix: np.ndarray,
                           project_info: Dict, **kwargs) -> Tuple[List[int], Dict]:
    n = len(project_ids)
    pheromone = np.ones((n, n))

    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                utility_i = project_info[project_ids[i]]['utility']
                utility_j = project_info[project_ids[j]]['utility']
                dist = distance_matrix[project_ids[i], project_ids[j]]
                heuristic[i, j] = (utility_i + utility_j) / (dist + 1)

    best_route = None
    best_score = float('-inf')

    for _ in range(CONFIG.ACO_ITERATIONS):
        all_routes = []
        all_scores = []

        for _ in range(CONFIG.ACO_ANT_COUNT):
            route = []
            visited = set()
            current = random.randint(0, n - 1)
            route.append(project_ids[current])
            visited.add(current)

            while len(visited) < n:
                probs = []
                candidates = []
                for next_node in range(n):
                    if next_node not in visited:
                        tau = pheromone[current, next_node] ** CONFIG.ACO_ALPHA
                        eta = heuristic[current, next_node] ** CONFIG.ACO_BETA
                        probs.append(tau * eta)
                        candidates.append(next_node)

                if sum(probs) == 0:
                    next_node = random.choice(candidates)
                else:
                    probs = np.array(probs) / sum(probs)
                    next_node = np.random.choice(candidates, p=probs)

                route.append(project_ids[next_node])
                visited.add(next_node)
                current = next_node

            result = evaluate_route(route, distance_matrix, project_info, **kwargs)
            score = result['final_score']
            all_routes.append(route)
            all_scores.append(score)

            if score > best_score:
                best_route = route.copy()
                best_score = score

        pheromone *= (1 - CONFIG.ACO_RHO)
        for route, score in zip(all_routes, all_scores):
            if score > 0:
                for i in range(len(route) - 1):
                    idx_i = project_ids.index(route[i])
                    idx_j = project_ids.index(route[i + 1])
                    pheromone[idx_i, idx_j] += CONFIG.ACO_Q * score

    best_result = evaluate_route(best_route, distance_matrix, project_info, **kwargs)
    return best_route, best_result

def format_end_time(total_time_min: float) -> str:
    """把在园总时长转换成实际结束时刻（默认 09:00 开园）"""
    end_clock_min = CONFIG.DISPLAY_OPEN_CLOCK_MIN + total_time_min
    hh = int(end_clock_min // 60)
    mm = int(round(end_clock_min % 60))
    return f"{hh:02d}:{mm:02d}"


def estimate_walk_distance(total_walk_min: float) -> float:
    """由总步行时间估算总步行距离（米）"""
    return round(total_walk_min * CONFIG.WALK_SPEED_M_PER_MIN, 1)

def run_single_case(crowd_type: str, date_type: str, algorithm_name: str,
                   project_info: Dict, distance_matrix: np.ndarray,
                   output_dir: str) -> Dict:
    print(f"\n运行：{crowd_type} - {date_type} - {algorithm_name}")

    import copy
    project_info_copy = copy.deepcopy(project_info)
    project_info_copy = calculate_utility_scores(project_info_copy, crowd_type)

    multiplier = CONFIG.DATE_QUEUE_MULTIPLIER[date_type]
    for proj_id, info in project_info_copy.items():
        if info['type'] == 'normal' and proj_id not in [CONFIG.START_NODE, CONFIG.END_NODE]:
            info['base_q'] *= multiplier
            info['peaks'] = [(A * multiplier, mu, sigma) for A, mu, sigma in info['peaks']]

    project_ids = [pid for pid in project_info_copy.keys() if pid not in [CONFIG.START_NODE, CONFIG.END_NODE]]

    if algorithm_name == '模拟退火':
        best_route, best_result = simulated_annealing(
            project_ids, distance_matrix, project_info_copy,
            start_node=CONFIG.START_NODE, end_node=CONFIG.END_NODE, return_to_end=True
        )
    elif algorithm_name == '遗传算法':
        best_route, best_result = genetic_algorithm(
            project_ids, distance_matrix, project_info_copy,
            start_node=CONFIG.START_NODE, end_node=CONFIG.END_NODE, return_to_end=True
        )
    else:
        best_route, best_result = ant_colony_optimization(
            project_ids, distance_matrix, project_info_copy,
            start_node=CONFIG.START_NODE, end_node=CONFIG.END_NODE, return_to_end=True
        )

    play_time = sum([project_info_copy[pid]['duration'] for pid in best_result['visited_projects']])
    print(f"  得分: {best_result['final_score']:.2f}, 访问: {best_result['visited_count']}个")

    visited = best_result['visited_projects']

    radar_path = os.path.join(output_dir, f'radar_{crowd_type}.png')
    if not os.path.exists(radar_path):
        plot_crowd_radar(crowd_type, radar_path)

    timeline_path = os.path.join(output_dir, f'Q1-timeline-{crowd_type}-{date_type}.png')
    plot_route_timeline(best_result, timeline_path)

    route_map_path = os.path.join(output_dir, f'Q1-{crowd_type}-{date_type}.png')
    plot_route_map_with_timeline(
        visited, project_info_copy, best_result['timeline_log'],
        crowd_type, date_type, route_map_path
    )

    return {
        'algorithm': algorithm_name,
        'crowd_type': crowd_type,
        'date_type': date_type,

        # 原始字段
        'score': best_result['final_score'],
        'utility': best_result['total_utility'],
        'visited_count': best_result['visited_count'],
        'total_time': best_result['total_time'],
        'play_time': play_time,
        'queue_time': best_result['total_queue'],
        'walk_time': best_result['total_walk'],
        'wait_time': best_result['total_wait'],

        # 论文表需要的字段
        'project_count': best_result['visited_count'],
        'walk_distance_m': estimate_walk_distance(best_result['total_walk']),
        'walk_and_wait_time': round(best_result['total_walk'] + best_result['total_wait'], 1),
        'queue_time_min': best_result['total_queue'],
        'play_time_min': play_time,
        'in_park_time_min': best_result['total_time'],
        'net_utility': best_result['final_score'],
        'end_clock': format_end_time(best_result['total_time']),

        'route': [project_info_copy[pid]['name'] for pid in visited]
    }


def generate_comparison_charts(df_summary: pd.DataFrame, output_dir: str):
    crowd_types = df_summary['人群类型'].unique()
    date_types = df_summary['日期类型'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(date_types))
    width = 0.25

    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        scores = [data[data['日期类型'] == date]['综合得分'].values[0] for date in date_types]
        ax.bar(x + i * width, scores, width, label=crowd)

    ax.set_xlabel('日期类型', fontsize=12)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('不同人群和日期的综合得分对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(date_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1-对比图-得分.png'),
                dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        counts = [data[data['日期类型'] == date]['访问项目数'].values[0] for date in date_types]
        ax.bar(x + i * width, counts, width, label=crowd)

    ax.set_xlabel('日期类型', fontsize=12)
    ax.set_ylabel('访问项目数', fontsize=12)
    ax.set_title('不同人群和日期的访问项目数对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(date_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1-对比图-访问项目数.png'),
                dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    combinations, play_times, queue_times, walk_times = [], [], [], []

    for crowd in crowd_types:
        for date in date_types:
            data = df_summary[(df_summary['人群类型'] == crowd) & (df_summary['日期类型'] == date)]
            combinations.append(f"{crowd}\n{date}")
            play_times.append(data['游玩时间(分钟)'].values[0])
            queue_times.append(data['排队时间(分钟)'].values[0])
            walk_times.append(data['步行时间(分钟)'].values[0])

    x_pos = np.arange(len(combinations))
    ax.bar(x_pos, play_times, label='游玩时间', color='#4ECDC4')
    ax.bar(x_pos, queue_times, bottom=play_times, label='排队时间', color='#FF6B6B')
    ax.bar(x_pos, walk_times, bottom=np.array(play_times) + np.array(queue_times),
           label='步行时间', color='#FFE66D')

    ax.set_xlabel('人群类型 - 日期类型', fontsize=12)
    ax.set_ylabel('时间（分钟）', fontsize=12)
    ax.set_title('不同情况下的时间分布对比（游玩:排队:行走）', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(combinations, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1-对比图-时间分布.png'),
                dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def generate_algorithm_comparison(all_results: List[Dict], output_dir: str):
    df = pd.DataFrame(all_results)
    algorithms = ['模拟退火', '遗传算法', '蚁群算法']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['综合得分', '访问项目数', '总耗时(分钟)']
    titles = ['综合得分对比', '访问项目数对比', '总耗时对比']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        data_by_algo = [df[df['算法'] == algo][metric].mean() for algo in algorithms]
        bars = ax.bar(algorithms, data_by_algo, color=['#FF6B6B', '#4ECDC4', '#FFE66D'])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1-算法对比-总览.png'),
                dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(algorithms))
    width = 0.25

    play_avg = [df[df['算法'] == algo]['游玩时间(分钟)'].mean() for algo in algorithms]
    queue_avg = [df[df['算法'] == algo]['排队时间(分钟)'].mean() for algo in algorithms]
    walk_avg = [df[df['算法'] == algo]['步行时间(分钟)'].mean() for algo in algorithms]

    ax.bar(x, play_avg, width, label='游玩时间', color='#4ECDC4')
    ax.bar(x, queue_avg, width, bottom=play_avg, label='排队时间', color='#FF6B6B')
    ax.bar(x, walk_avg, width, bottom=np.array(play_avg) + np.array(queue_avg),
           label='步行时间', color='#FFE66D')

    ax.set_xlabel('优化算法', fontsize=12)
    ax.set_ylabel('平均时间（分钟）', fontsize=12)
    ax.set_title('三种算法的平均时间分布对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Q1-算法对比-时间分布.png'),
                dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print("✓ 算法横向对比图已保存")


def main():
    print("\n" + "=" * 70)
    print(" " * 10 + "迪士尼乐园路线优化系统 - 全量批量运行")
    print("=" * 70)
    print("\n将运行所有27种情况：")
    print("  人群类型: 普通、亲子、情侣")
    print("  日期类型: 工作日、双休日、节假日")
    print("  优化算法: 模拟退火、遗传算法、蚁群算法")
    print("  总计: 3 × 3 × 3 = 27 种组合\n")

    base_output_dir = CONFIG.OUTPUT_DIR
    algorithms = ['模拟退火', '遗传算法', '蚁群算法']
    os.makedirs(base_output_dir, exist_ok=True)

    for algo in algorithms:
        os.makedirs(os.path.join(base_output_dir, algo), exist_ok=True)

    print("=" * 70)
    print("加载数据")
    print("=" * 70)

    project_file = os.path.join(CONFIG.DATA_DIR, 'projects_data.csv')
    poi_file = os.path.join(CONFIG.DATA_DIR, 'poi.csv')

    if not os.path.exists(project_file):
        raise FileNotFoundError(f"找不到项目数据文件：{project_file}")
    if not os.path.exists(poi_file):
        raise FileNotFoundError(f"找不到距离矩阵文件：{poi_file}")

    project_info, _ = load_projects_from_csv(project_file)
    print(f"✓ 成功读取 {len(project_info)} 个项目")

    distance_matrix, _ = load_distance_matrix(poi_file)

    crowd_types = ['普通', '亲子', '情侣']
    date_types = ['工作日', '双休日', '节假日']

    all_results = []
    total_cases = len(crowd_types) * len(date_types) * len(algorithms)
    current_case = 0

    for algorithm in algorithms:
        print(f"\n{'=' * 70}")
        print(f"算法: {algorithm}")
        print(f"{'=' * 70}")

        algo_dir = os.path.join(base_output_dir, algorithm)
        algo_results = []

        for crowd_type in crowd_types:
            for date_type in date_types:
                current_case += 1
                print(f"\n进度: {current_case}/{total_cases}")

                result = run_single_case(
                    crowd_type, date_type, algorithm,
                    project_info, distance_matrix, algo_dir
                )
                algo_results.append(result)
                all_results.append(result)

        df_algo = pd.DataFrame(algo_results)
        df_algo = df_algo[['crowd_type', 'date_type', 'score', 'utility',
                           'visited_count', 'total_time', 'play_time',
                           'queue_time', 'walk_time']]
        df_algo.columns = ['人群类型', '日期类型', '综合得分', '总效用',
                           '访问项目数', '总耗时(分钟)', '游玩时间(分钟)',
                           '排队时间(分钟)', '步行时间(分钟)']

        csv_path = os.path.join(algo_dir, f'Q1-汇总结果-{algorithm}.csv')
        df_algo.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ {algorithm} 汇总表格已保存: {csv_path}")

        df_paper = pd.DataFrame(algo_results)
        df_paper = df_paper[
            ['crowd_type', 'date_type', 'project_count', 'walk_distance_m',
             'walk_and_wait_time', 'queue_time_min', 'play_time_min',
             'in_park_time_min', 'net_utility', 'end_clock']
        ]
        df_paper.columns = [
            '游客类型', '日期情景', '项目数量', '总步行距离/m',
            '总步行及候场时间/min', '总排队时间/min', '总游玩时间/min',
            '总在园时间/min', '综合净效用', '结束时刻'
        ]

        paper_csv_path = os.path.join(algo_dir, f'Q1-论文汇总表-{algorithm}.csv')
        df_paper.to_csv(paper_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ {algorithm} 论文版汇总表已保存: {paper_csv_path}")

        generate_comparison_charts(df_algo, algo_dir)
        print(f"✓ {algorithm} 对比图已生成")

    print(f"\n{'=' * 70}")
    print("生成算法横向对比")
    print(f"{'=' * 70}")
    generate_algorithm_comparison(all_results, base_output_dir)

    df_all = pd.DataFrame(all_results)
    df_all = df_all[['algorithm', 'crowd_type', 'date_type', 'score', 'utility',
                     'visited_count', 'total_time', 'play_time',
                     'queue_time', 'walk_time']]
    df_all.columns = ['算法', '人群类型', '日期类型', '综合得分', '总效用',
                      '访问项目数', '总耗时(分钟)', '游玩时间(分钟)',
                      '排队时间(分钟)', '步行时间(分钟)']

    csv_all_path = os.path.join(base_output_dir, 'Q1-总汇总结果.csv')
    df_all.to_csv(csv_all_path, index=False, encoding='utf-8-sig')
    print(f"✓ 总汇总表格已保存: {csv_all_path}")

    print(f"\n{'=' * 70}")
    print("汇总统计")
    print(f"{'=' * 70}\n")

    print("各算法平均表现：")
    for algo in algorithms:
        algo_data = df_all[df_all['算法'] == algo]
        print(f"\n{algo}:")
        print(f"  平均得分: {algo_data['综合得分'].mean():.2f}")
        print(f"  平均访问项目数: {algo_data['访问项目数'].mean():.1f}")
        print(f"  平均总耗时: {algo_data['总耗时(分钟)'].mean():.1f}分钟")
        print(f"  平均游玩时间: {algo_data['游玩时间(分钟)'].mean():.1f}分钟")
        print(f"  平均排队时间: {algo_data['排队时间(分钟)'].mean():.1f}分钟")
        print(f"  平均步行时间: {algo_data['步行时间(分钟)'].mean():.1f}分钟")

    print(f"\n{'=' * 70}")
    print("✓ 全部27种情况运行完成！")
    print(f"{'=' * 70}")
    print(f"\n所有结果已保存到 Q1-test 文件夹")


if __name__ == "__main__":
    main()