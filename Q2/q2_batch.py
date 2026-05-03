"""
Q2批量运行脚本（适配重构版）
运行9种情况：3种场景 × 3种人群
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入q2中的函数
from q2 import (
    CONFIG, load_realtime_queue_data, load_projects_data, load_distance_matrix,
    calculate_utility_scores, load_q1_initial_route, dynamic_rolling_replan,
    plot_timeline_q2, plot_queue_comparison_q2
)


def run_single_case(queue_df: pd.DataFrame, projects_df: Dict,
                   distance_matrix: np.ndarray, scenario: str,
                   crowd_type: str, q1_algorithm: str) -> Dict:
    """运行单个案例"""
    print(f"\n{'='*70}")
    print(f"运行: {scenario} - {crowd_type}游客")
    print(f"{'='*70}")
    
    # 计算效用
    import copy
    project_info = copy.deepcopy(projects_df)
    project_info = calculate_utility_scores(project_info, crowd_type)
    
    # 尝试加载Q1初始路径
    initial_route = load_q1_initial_route(scenario, crowd_type, q1_algorithm)
    route_source = f"Q1-{q1_algorithm}"

    if initial_route is None:
        # 使用默认启发式路径
        projects = [(pid, info['utility']) for pid, info in project_info.items()
                   if pid not in [0, 27]]
        projects.sort(key=lambda x: x[1], reverse=True)
        initial_route = [pid for pid, _ in projects[:20]]
        route_source = "启发式回退"
    
    # 运行动态滚动重规划
    state, result = dynamic_rolling_replan(
        initial_route, distance_matrix, project_info,
        queue_df, scenario, crowd_type
    )
    
    print(f"  访问: {result['visited_count']}个项目")
    print(f"  总效用: {result['total_utility']:.2f}")
    print(f"  重规划次数: {result['replan_count']}次")
    
    # 生成可视化
    timeline_path = os.path.join(CONFIG.OUTPUT_DIR, 
                                 f'Q2-timeline-{scenario}-{crowd_type}.png')
    plot_timeline_q2(result, scenario, crowd_type, timeline_path)
    
    queue_path = os.path.join(CONFIG.OUTPUT_DIR,
                             f'Q2-queue-{scenario}-{crowd_type}.png')
    plot_queue_comparison_q2(queue_df, scenario, result['executed_path'],
                            project_info, queue_path)
    
    return {
        'scenario': scenario,
        'crowd_type': crowd_type,
        'route_source': route_source,
        'visited_count': result['visited_count'],
        'total_utility': result['total_utility'],
        'total_time': result['total_time'],
        'play_time': result['total_play'],
        'queue_time': result['total_queue'],
        'wait_time': result['total_wait'],
        'walk_time': result['total_walk'],
        'replan_count': result['replan_count']
    }


def generate_comparison_charts(df_summary: pd.DataFrame):
    """生成对比图表"""
    
    scenarios = df_summary['场景'].unique()
    crowd_types = df_summary['人群类型'].unique()
    
    # 图1：访问项目数对比
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        counts = [data[data['场景'] == s]['访问项目数'].values[0] for s in scenarios]
        ax.bar(x + i*width, counts, width, label=crowd)
    
    ax.set_xlabel('场景', fontsize=12)
    ax.set_ylabel('访问项目数', fontsize=12)
    ax.set_title('Q2-不同场景和人群的访问项目数对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, 'Q2-对比图-访问项目数.png'), 
               dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # 图2：时间分布对比
    fig, ax = plt.subplots(figsize=(14, 6))
    
    combinations = []
    play_times = []
    queue_times = []
    walk_times = []
    
    for scenario in scenarios:
        for crowd in crowd_types:
            data = df_summary[
                (df_summary['场景'] == scenario) & 
                (df_summary['人群类型'] == crowd)
            ]
            combinations.append(f"{scenario}\n{crowd}")
            play_times.append(data['游玩时间(分钟)'].values[0])
            queue_times.append(data['排队时间(分钟)'].values[0])
            walk_times.append(data['步行时间(分钟)'].values[0])
    
    x_pos = np.arange(len(combinations))
    
    ax.bar(x_pos, play_times, label='游玩时间', color='#4ECDC4')
    ax.bar(x_pos, queue_times, bottom=play_times, label='排队时间', color='#FF6B6B')
    ax.bar(x_pos, walk_times, bottom=np.array(play_times)+np.array(queue_times),
          label='步行时间', color='#FFE66D')
    
    ax.set_xlabel('场景 - 人群类型', fontsize=12)
    ax.set_ylabel('时间（分钟）', fontsize=12)
    ax.set_title('Q2-时间分布对比（游玩:排队:步行）', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(combinations, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, 'Q2-对比图-时间分布.png'),
               dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # 图3：总效用对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        utilities = [data[data['场景'] == s]['总效用'].values[0] for s in scenarios]
        ax.bar(x + i*width, utilities, width, label=crowd)
    
    ax.set_xlabel('场景', fontsize=12)
    ax.set_ylabel('总效用', fontsize=12)
    ax.set_title('Q2-不同场景和人群的总效用对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, 'Q2-对比图-总效用.png'),
               dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # 图4：重规划次数对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        replans = [data[data['场景'] == s]['重规划次数'].values[0] for s in scenarios]
        ax.bar(x + i*width, replans, width, label=crowd)
    
    ax.set_xlabel('场景', fontsize=12)
    ax.set_ylabel('重规划次数', fontsize=12)
    ax.set_title('Q2-不同场景和人群的重规划次数对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, 'Q2-对比图-重规划次数.png'),
               dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print("✓ 对比图已生成")


def main():
    """批量运行主函数"""
    print("\n" + "=" * 70)
    print(" " * 10 + "迪士尼乐园路线优化系统 - Q2批量运行（重构版）")
    print("=" * 70)
    print("\n将运行所有9种情况：")
    print("  场景: 工作日、双休日、节假日")
    print("  人群类型: 普通、亲子、情侣")
    print("  Q1初始路径算法: 遗传算法")
    print("  总计: 3 × 3 = 9 种组合\n")
    
    # 创建输出目录
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("=" * 70)
    print("数据加载")
    print("=" * 70)
    
    queue_df = load_realtime_queue_data(
        os.path.join(CONFIG.DATA_DIR, 'new_simulated_queue_10min(1).csv')
    )
    
    project_info, _ = load_projects_data(
        os.path.join(CONFIG.DATA_DIR, 'projects_data.csv')
    )
    
    distance_matrix = load_distance_matrix(
        os.path.join(CONFIG.DATA_DIR, 'poi.csv')
    )
    
    # 定义所有组合
    scenarios = ['工作日', '双休日', '节假日']
    crowd_types = ['普通', '亲子', '情侣']
    q1_algorithm = '遗传算法'
    
    # 存储所有结果
    all_results = []
    
    # 批量运行
    total_cases = len(scenarios) * len(crowd_types)
    current_case = 0
    
    for scenario in scenarios:
        for crowd_type in crowd_types:
            current_case += 1
            print(f"\n{'='*70}")
            print(f"进度: {current_case}/{total_cases}")
            print(f"{'='*70}")
            
            result = run_single_case(queue_df, project_info, distance_matrix,
                                    scenario, crowd_type, q1_algorithm)
            all_results.append(result)
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print("生成汇总报告")
    print(f"{'='*70}\n")
    
    df_summary = pd.DataFrame(all_results)
    df_summary.columns = ['场景', '人群类型', '初始路径来源', '访问项目数', '总效用',
                         '总耗时(分钟)', '游玩时间(分钟)',
                         '排队时间(分钟)', '等待时间(分钟)',
                         '步行时间(分钟)', '重规划次数']
    
    # 打印汇总表格
    print("汇总结果：")
    print(df_summary.to_string(index=False))
    
    # 保存CSV
    csv_path = os.path.join(CONFIG.OUTPUT_DIR, 'Q2-汇总结果.csv')
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 汇总表格已保存: {csv_path}")
    
    # 生成对比图
    generate_comparison_charts(df_summary)
    
    # 打印文件清单
    print(f"\n{'='*70}")
    print("生成的文件清单")
    print(f"{'='*70}\n")
    
    print("【时间线图】（9张）")
    for scenario in scenarios:
        for crowd in crowd_types:
            print(f"  Q2-timeline-{scenario}-{crowd}.png")
    
    print("\n【排队对比图】（9张）")
    for scenario in scenarios:
        for crowd in crowd_types:
            print(f"  Q2-queue-{scenario}-{crowd}.png")
    
    print("\n【汇总文件】")
    print("  Q2-汇总结果.csv")
    print("  Q2-对比图-访问项目数.png")
    print("  Q2-对比图-时间分布.png")
    print("  Q2-对比图-总效用.png")
    print("  Q2-对比图-重规划次数.png")
    
    print(f"\n{'='*70}")
    print("✓ 批量运行完成！所有结果已保存到 Q2-test 文件夹")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
