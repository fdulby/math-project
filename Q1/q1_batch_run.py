"""
批量运行所有9种情况
3种人群类型 × 3种日期类型 = 9种组合
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

# 导入q1_final中的所有函数
import sys
sys.path.insert(0, os.path.dirname(__file__))

from q1_final import (
    CONFIG, load_projects_from_csv, load_distance_matrix,
    calculate_utility_scores, simulated_annealing, evaluate_route,
    plot_crowd_radar, plot_route_map_with_timeline, plot_route_timeline
)


def run_single_case(crowd_type: str, date_type: str, project_info: Dict, 
                   distance_matrix: np.ndarray) -> Dict:
    """运行单个案例"""
    print("\n" + "=" * 70)
    print(f"运行案例：{crowd_type}游客 - {date_type}")
    print("=" * 70)
    
    # 深拷贝project_info（避免修改原数据）
    import copy
    project_info_copy = copy.deepcopy(project_info)
    
    # 计算效用
    project_info_copy = calculate_utility_scores(project_info_copy, crowd_type)
    
    # 调整排队参数
    multiplier = CONFIG.DATE_QUEUE_MULTIPLIER[date_type]
    for proj_id, info in project_info_copy.items():
        if info['type'] == 'normal' and proj_id != 0:
            info['base_q'] *= multiplier
            info['peaks'] = [(A * multiplier, mu, sigma) for A, mu, sigma in info['peaks']]
    
    # 项目列表（排除入口0）
    project_ids = [pid for pid in project_info_copy.keys() if pid != 0]
    
    # 运行优化
    best_route, best_result = simulated_annealing(
        project_ids=project_ids,
        distance_matrix=distance_matrix,
        project_info=project_info_copy,
        start_node=0,
        end_node=0,
        return_to_end=True
    )
    
    # 打印结果
    print(f"\n【结果】")
    print(f"  综合得分: {best_result['final_score']:.2f}")
    print(f"  总效用: {best_result['total_utility']:.2f}")
    print(f"  实际访问: {best_result['visited_count']}/{len(project_ids)} 个项目")
    print(f"  总耗时: {best_result['total_time']:.1f}分钟")
    print(f"  排队时间: {best_result['total_queue']:.1f}分钟")
    print(f"  步行时间: {best_result['total_walk']:.1f}分钟")
    
    # 生成可视化
    output_dir = CONFIG.OUTPUT_DIR
    
    # 雷达图（每种人群只生成一次）
    radar_path = os.path.join(output_dir, f'radar_{crowd_type}.png')
    if not os.path.exists(radar_path):
        plot_crowd_radar(crowd_type, radar_path)
    
    # 时间线图
    timeline_path = os.path.join(output_dir, f'Q1-timeline-{crowd_type}-{date_type}.png')
    plot_route_timeline(best_result, timeline_path)
    
    # 路线图
    route_map_path = os.path.join(output_dir, f'Q1-{crowd_type}-{date_type}.png')
    visited = best_result['visited_projects']
    plot_route_map_with_timeline(visited, project_info_copy, best_result['timeline_log'],
                                 crowd_type, date_type, route_map_path)
    
    return {
        'crowd_type': crowd_type,
        'date_type': date_type,
        'score': best_result['final_score'],
        'utility': best_result['total_utility'],
        'visited_count': best_result['visited_count'],
        'total_time': best_result['total_time'],
        'queue_time': best_result['total_queue'],
        'walk_time': best_result['total_walk'],
        'route': [project_info_copy[pid]['name'] for pid in visited]
    }


def main():
    """批量运行主函数"""
    print("\n" + "=" * 70)
    print(" " * 15 + "迪士尼乐园路线优化系统 - 批量运行")
    print("=" * 70)
    print("\n将运行所有9种情况：")
    print("  人群类型: 普通、亲子、情侣")
    print("  日期类型: 工作日、双休日、节假日")
    print("  总计: 3 × 3 = 9 种组合\n")
    
    # 确保输出目录存在
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    # 读取数据（只读取一次）
    print("=" * 70)
    print("加载数据")
    print("=" * 70)
    
    project_file = os.path.join(CONFIG.DATA_DIR, 'projects_data.csv')
    project_info, df_projects = load_projects_from_csv(project_file)
    print(f"✓ 成功读取 {len(project_info)} 个项目（包含入口）")
    
    poi_file = os.path.join(CONFIG.DATA_DIR, 'poi.csv')
    distance_matrix, poi_mapping = load_distance_matrix(poi_file)
    
    # 定义所有组合
    crowd_types = ['普通', '亲子', '情侣']
    date_types = ['工作日', '双休日', '节假日']
    
    # 存储所有结果
    all_results = []
    
    # 批量运行
    total_cases = len(crowd_types) * len(date_types)
    current_case = 0
    
    for crowd_type in crowd_types:
        for date_type in date_types:
            current_case += 1
            print(f"\n{'='*70}")
            print(f"进度: {current_case}/{total_cases}")
            print(f"{'='*70}")
            
            result = run_single_case(crowd_type, date_type, project_info, distance_matrix)
            all_results.append(result)
    
    # 生成汇总报告
    print("\n" + "=" * 70)
    print("所有案例运行完成！生成汇总报告...")
    print("=" * 70)
    
    # 创建汇总表格
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary[['crowd_type', 'date_type', 'score', 'utility', 
                             'visited_count', 'total_time', 'queue_time', 'walk_time']]
    df_summary.columns = ['人群类型', '日期类型', '综合得分', '总效用', 
                         '访问项目数', '总耗时(分钟)', '排队时间(分钟)', '步行时间(分钟)']
    
    # 打印汇总表格
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    # 保存汇总表格
    summary_path = os.path.join(CONFIG.OUTPUT_DIR, 'Q1-汇总结果.csv')
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 汇总表格已保存: {summary_path}")
    
    # 生成对比图
    generate_comparison_charts(df_summary)
    
    # 打印文件清单
    print("\n" + "=" * 70)
    print("生成的文件清单")
    print("=" * 70)
    print("\n【路线图】（9张）")
    for crowd in crowd_types:
        for date in date_types:
            print(f"  Q1-{crowd}-{date}.png")
    
    print("\n【时间线图】（9张）")
    for crowd in crowd_types:
        for date in date_types:
            print(f"  Q1-timeline-{crowd}-{date}.png")
    
    print("\n【雷达图】（3张）")
    for crowd in crowd_types:
        print(f"  radar_{crowd}.png")
    
    print("\n【汇总文件】")
    print("  Q1-汇总结果.csv")
    print("  Q1-对比图-得分.png")
    print("  Q1-对比图-访问项目数.png")
    print("  Q1-对比图-时间分布.png")
    
    print("\n" + "=" * 70)
    print("✓ 批量运行完成！所有结果已保存到 Q1-test 文件夹")
    print("=" * 70)


def generate_comparison_charts(df_summary: pd.DataFrame):
    """生成对比图表"""
    
    # 图1：综合得分对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    crowd_types = df_summary['人群类型'].unique()
    date_types = df_summary['日期类型'].unique()
    x = np.arange(len(date_types))
    width = 0.25
    
    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        scores = [data[data['日期类型'] == date]['综合得分'].values[0] for date in date_types]
        ax.bar(x + i*width, scores, width, label=crowd)
    
    ax.set_xlabel('日期类型', fontsize=12)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('不同人群和日期的综合得分对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(date_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    score_path = os.path.join(CONFIG.OUTPUT_DIR, 'Q1-对比图-得分.png')
    plt.savefig(score_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 得分对比图已保存: {score_path}")
    plt.close()
    
    # 图2：访问项目数对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, crowd in enumerate(crowd_types):
        data = df_summary[df_summary['人群类型'] == crowd]
        counts = [data[data['日期类型'] == date]['访问项目数'].values[0] for date in date_types]
        ax.bar(x + i*width, counts, width, label=crowd)
    
    ax.set_xlabel('日期类型', fontsize=12)
    ax.set_ylabel('访问项目数', fontsize=12)
    ax.set_title('不同人群和日期的访问项目数对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(date_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    count_path = os.path.join(CONFIG.OUTPUT_DIR, 'Q1-对比图-访问项目数.png')
    plt.savefig(count_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 访问项目数对比图已保存: {count_path}")
    plt.close()
    
    # 图3：时间分布对比（堆叠柱状图）
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 为每种组合创建一个柱子
    combinations = []
    queue_times = []
    walk_times = []
    
    for crowd in crowd_types:
        for date in date_types:
            data = df_summary[(df_summary['人群类型'] == crowd) & (df_summary['日期类型'] == date)]
            combinations.append(f"{crowd}\n{date}")
            queue_times.append(data['排队时间(分钟)'].values[0])
            walk_times.append(data['步行时间(分钟)'].values[0])
    
    x_pos = np.arange(len(combinations))
    
    ax.bar(x_pos, queue_times, label='排队时间', color='#FF6B6B')
    ax.bar(x_pos, walk_times, bottom=queue_times, label='步行时间', color='#4ECDC4')
    
    ax.set_xlabel('人群类型 - 日期类型', fontsize=12)
    ax.set_ylabel('时间（分钟）', fontsize=12)
    ax.set_title('不同情况下的时间分布对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(combinations, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    time_path = os.path.join(CONFIG.OUTPUT_DIR, 'Q1-对比图-时间分布.png')
    plt.savefig(time_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 时间分布对比图已保存: {time_path}")
    plt.close()


if __name__ == "__main__":
    main()
