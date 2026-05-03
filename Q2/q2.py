"""
迪士尼乐园路线优化系统 - 第二问（完全修正版）
基于问题一最优路径的动态滚动修正模型

修正要点：
1. 真正读取Q1最优路径作为初始路径
2. 实现阈值触发（不是每次都重规划）
3. 候选集优先基于剩余后缀
4. 贪心+SA从有序初始解开始
5. 完整的演出项目统计
6. 正确的路径配置
7. 入口0/出口27严格统一
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================
# 配置模块
# ==========================================

class CONFIG:
    """全局配置类"""
    
    # 人群偏好权重
    CROWD_WEIGHTS = {
        '普通': [0.25, 0.25, 0.25, 0.25],
        '亲子': [0.10, 0.20, 0.40, 0.30],
        '情侣': [0.40, 0.40, 0.10, 0.10]
    }
    
    # 时间参数（Q2使用实际时间：9:00-21:00）
    PARK_OPEN_TIME = 540   # 9:00 (分钟)
    PARK_CLOSE_TIME = 1260  # 21:00 (分钟)
    START_TIME = 540
    
    # 节点配置（严格统一）
    START_NODE = 0   # 入口（米奇大街）
    END_NODE = 27    # 出口（米奇大街）
    
    # 评估权重（与Q1保持一致）
    WEIGHT_QUEUE_TIME = 0.1
    WEIGHT_WAIT_TIME = 0.1
    WEIGHT_WALK_TIME = 0.1
    WEIGHT_OVERTIME = 5.0
    WEIGHT_MISSED_SHOW = 1000.0
    WEIGHT_DIVERSITY = 1.0
    
    # Q2特有参数
    REPLAN_THRESHOLD = 0.3  # 排队偏差阈值（30%）
    CANDIDATE_SIZE = 6      # 候选集大小K
    SUFFIX_PRIORITY = 0.7   # 剩余后缀优先权重
    
    # 模拟退火参数（用于局部优化）
    SA_LOCAL_TEMP = 100
    SA_LOCAL_COOLING = 0.95
    SA_LOCAL_ITERATIONS = 300
    
    # 路径配置（基于真实项目结构）
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Q2-test')
    Q1_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Q1-test')
    Q1_ROUTES_DIR = os.path.join(Q1_OUTPUT_DIR, 'routes')
    
    FIGURE_DPI = 100


# ==========================================
# 数据加载模块
# ==========================================

def load_realtime_queue_data(csv_file: str) -> pd.DataFrame:
    """加载实时排队数据"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"✓ 成功读取实时排队数据: {len(df)} 条记录")
    print(f"  场景: {df['scenario'].unique()}")
    print(f"  项目数: {df['project_id'].nunique()}")
    print(f"  时间范围: {df['time_min'].min()}-{df['time_min'].max()}分钟")
    return df


def load_projects_data(csv_file: str) -> Tuple[Dict, pd.DataFrame]:
    """加载项目基础数据（与Q1格式一致）"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
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
        
        # 时间窗
        if pd.notna(row['时间窗开始']) and pd.notna(row['时间窗结束']):
            info['time_window'] = (float(row['时间窗开始']), float(row['时间窗结束']))
        else:
            info['time_window'] = (CONFIG.PARK_OPEN_TIME, CONFIG.PARK_CLOSE_TIME)
        
        # GMM排队参数（用于回退）
        if info['type'] == 'normal' and proj_id not in [0, 27]:
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
    
    print(f"✓ 成功读取项目数据: {len(project_info)} 个项目")
    return project_info, df


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


def load_q1_initial_route(scenario: str, crowd_type: str, 
                          algorithm: str = '模拟退火') -> Optional[List[int]]:
    """
    读取问题一的最优路径作为初始路径R^(0)
    
    修正：读取Q1真实保存的路径文件
    位置：Q1-test/routes/route_{algorithm}_{scenario}_{crowd_type}.json
    """
    filename = f"route_{algorithm}_{scenario}_{crowd_type}.json"
    filepath = os.path.join(CONFIG.Q1_ROUTES_DIR, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                route_data = json.load(f)
            route = route_data['route']
            print(f"✓ 成功读取Q1初始路径: {len(route)} 个项目")
            print(f"  算法: {route_data['algorithm']}")
            print(f"  场景: {route_data['scenario']}")
            print(f"  人群: {route_data['crowd_type']}")
            return route
        except Exception as e:
            print(f"⚠ 读取Q1路径文件失败: {e}")
            return None
    else:
        print(f"⚠ 未找到Q1初始路径文件: {filepath}")
        print(f"  将使用默认启发式路径")
        return None


# ==========================================
# 实时排队接口模块
# ==========================================

def get_realtime_queue(queue_df: pd.DataFrame, scenario: str, 
                      project_id: str, current_time: float) -> Optional[float]:
    """
    获取实时排队时间（10分钟粒度，线性插值）
    若缺失数据，返回None（调用方需回退到GMM预测）
    """
    project_data = queue_df[
        (queue_df['scenario'] == scenario) & 
        (queue_df['project_id'] == project_id)
    ]
    
    if len(project_data) == 0:
        return None
    
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


def get_gmm_queue(current_time: float, base_q: float, peaks: List[Tuple]) -> float:
    """GMM动态排队时间（回退方案）"""
    q_time = float(base_q)
    for A, mu, sigma in peaks:
        q_time += A * np.exp(-((current_time - mu) ** 2) / (2 * sigma ** 2))
    return max(0.0, q_time)


def get_queue_time(queue_df: pd.DataFrame, scenario: str, project_info: Dict,
                  proj_id: int, current_time: float) -> Tuple[float, str]:
    """
    统一排队时间接口
    优先使用实时数据，缺失时回退到GMM预测
    
    返回: (排队时间, 数据来源)
    """
    if proj_id not in project_info:
        return 0.0, 'none'
    
    p_info = project_info[proj_id]
    
    # 演出项目不排队
    if p_info['type'] == 'show':
        return 0.0, 'show'
    
    # 入口/出口不排队
    if proj_id in [0, 27]:
        return 0.0, 'entry_exit'
    
    # 尝试获取实时排队
    proj_code = f"P{proj_id:02d}"
    realtime_queue = get_realtime_queue(queue_df, scenario, proj_code, current_time)
    
    if realtime_queue is not None:
        return realtime_queue, 'realtime'
    
    # 回退到GMM预测
    if 'base_q' in p_info and 'peaks' in p_info:
        gmm_queue = get_gmm_queue(current_time, p_info['base_q'], p_info['peaks'])
        return gmm_queue, 'gmm_fallback'
    
    return 0.0, 'no_data'


# ==========================================
# 效用计算模块
# ==========================================

def calculate_utility_scores(project_info: Dict, crowd_type: str) -> Dict:
    """计算效用得分（与Q1一致）"""
    w = np.array(CONFIG.CROWD_WEIGHTS[crowd_type])
    for proj_id, info in project_info.items():
        if proj_id in [0, 27]:  # 入口/出口不计算效用
            info['utility'] = 0.0
            continue
        features = np.array(info['features'])
        utility = np.dot(features, w) * 10
        info['utility'] = round(utility, 2)
    return project_info


# ==========================================
# 路线评估模块（对齐Q1逻辑）
# ==========================================

def evaluate_route_q2(route: List[int], distance_matrix: np.ndarray, 
                     project_info: Dict, queue_df: pd.DataFrame, scenario: str,
                     start_time: float, start_node: int, 
                     end_node: int, return_to_end: bool = True) -> Dict:
    """
    Q2版本的路线评估函数
    
    与Q1的区别：
    1. 普通项目排队时间使用实时数据（get_queue_time）
    2. 演出项目处理逻辑与Q1完全一致
    3. 评分机制与Q1保持统一
    """
    current_time = float(start_time)
    current_node = start_node
    
    total_utility = 0.0
    total_queue_time = 0.0
    total_wait_time = 0.0
    total_walk_time = 0.0
    total_play_time = 0.0
    missed_shows = 0
    closed_projects = 0
    overtime = 0.0
    timeline_log = []
    feasible = True
    visited_projects = []
    
    if len(route) == 0:
        # 空路线：直接返回终点
        if return_to_end and end_node is not None:
            walk_time = distance_matrix[start_node, end_node]
            if current_time + walk_time <= CONFIG.PARK_CLOSE_TIME:
                current_time += walk_time
                total_walk_time += walk_time
        return {
            'final_score': 0.0, 'total_utility': 0.0, 
            'total_time': current_time - start_time,
            'total_queue': 0.0, 'total_wait': 0.0, 'total_walk': 0.0,
            'total_play': 0.0, 'missed_shows': 0, 'closed_projects': 0,
            'overtime': 0.0, 'feasible': True, 'timeline_log': [],
            'route': route, 'visited_projects': [], 'visited_count': 0
        }
    
    # 遍历路线中的项目
    for next_node in route:
        if next_node not in project_info or next_node in [0, 27]:
            continue
        
        p_info = project_info[next_node]
        walk_time = distance_matrix[current_node, next_node]
        arrive_time = current_time + walk_time
        
        # 提前终止：在路上就超过闭园时间
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
            # 演出项目处理（与Q1一致）
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
                
                gained_utility = utility_score * watched_ratio
                leave_time = time_window_end
                
                if actual_start > time_window_start:
                    status = 'partial_show'
                else:
                    status = 'full_show'
        else:
            # 普通项目：使用实时排队数据
            if arrive_time > time_window_end:
                closed_projects += 1
                status = 'closed'
                leave_time = arrive_time
                gained_utility = 0.0
            else:
                # 关键：使用实时排队数据
                queue_time, data_source = get_queue_time(queue_df, scenario, project_info, 
                                                        next_node, arrive_time)
                total_queue_time += queue_time
                actual_start = arrive_time + queue_time
                leave_time = actual_start + play_duration
                gained_utility = utility_score
        
        # 检查是否超时
        if leave_time > CONFIG.PARK_CLOSE_TIME:
            overtime = leave_time - CONFIG.PARK_CLOSE_TIME
            feasible = False
            total_utility += gained_utility
            total_play_time += play_duration
            timeline_log.append({
                '项目': name, '到达': round(arrive_time, 1), 
                '排队': round(queue_time, 1), '等待': round(wait_time, 1),
                '离开': round(leave_time, 1), '效用': round(gained_utility, 2),
                '状态': status + '_overtime'
            })
            visited_projects.append(next_node)
            current_time = leave_time
            current_node = next_node
            break
        
        total_utility += gained_utility
        total_play_time += play_duration
        timeline_log.append({
            '项目': name, '到达': round(arrive_time, 1),
            '排队': round(queue_time, 1), '等待': round(wait_time, 1),
            '离开': round(leave_time, 1), '效用': round(gained_utility, 2),
            '状态': status
        })
        
        visited_projects.append(next_node)
        current_time = leave_time
        current_node = next_node
    
    # 返回终点
    if return_to_end and end_node is not None and current_node != end_node:
        walk_time = distance_matrix[current_node, end_node]
        arrive_exit_time = current_time + walk_time
        
        if arrive_exit_time <= CONFIG.PARK_CLOSE_TIME:
            current_time = arrive_exit_time
            total_walk_time += walk_time
    
    # 计算综合得分（与Q1一致）
    time_cost = (CONFIG.WEIGHT_QUEUE_TIME * total_queue_time +
                CONFIG.WEIGHT_WAIT_TIME * total_wait_time +
                CONFIG.WEIGHT_WALK_TIME * total_walk_time)
    overtime_penalty = CONFIG.WEIGHT_OVERTIME * (overtime ** 2) if overtime > 0 else 0
    missed_penalty = CONFIG.WEIGHT_MISSED_SHOW * (missed_shows + closed_projects)
    
    visited_types = [project_info[node]['type'] for node in visited_projects 
                    if node in project_info]
    diversity_bonus = CONFIG.WEIGHT_DIVERSITY * len(set(visited_types)) if visited_types else 0
    
    final_score = total_utility - time_cost - overtime_penalty - missed_penalty + diversity_bonus
    
    return {
        'final_score': round(final_score, 2),
        'total_utility': round(total_utility, 2),
        'total_time': round(current_time - start_time, 1),
        'total_queue': round(total_queue_time, 1),
        'total_wait': round(total_wait_time, 1),
        'total_walk': round(total_walk_time, 1),
        'total_play': round(total_play_time, 1),
        'missed_shows': missed_shows,
        'closed_projects': closed_projects,
        'overtime': round(overtime, 1),
        'feasible': feasible,
        'timeline_log': timeline_log,
        'route': route,
        'visited_projects': visited_projects,
        'visited_count': len(visited_projects)
    }


# ==========================================
# 动态状态管理模块（修正版）
# ==========================================

class DynamicState:
    """
    动态规划状态管理（修正版）
    
    修正要点：
    1. 明确维护initial_route和remaining_suffix
    2. 区分executed_prefix和remaining_suffix
    3. 支持基于剩余后缀的候选集筛选
    """
    
    def __init__(self, initial_route: List[int], project_info: Dict):
        self.initial_route = initial_route.copy()  # 原始Q1路径（不变）
        self.remaining_suffix = initial_route.copy()  # 剩余后缀（动态更新）
        self.executed_prefix = []                  # 已执行前缀
        
        self.current_node = CONFIG.START_NODE      # 当前位置
        self.current_time = CONFIG.START_TIME      # 当前时刻
        self.visited = set([CONFIG.START_NODE])    # 已访问集合
        self.timeline = []                         # 时间线记录
        self.replan_count = 0                      # 重规划次数
        self.replan_log = []                       # 重规划日志
        
        # 统计信息
        self.total_utility = 0.0
        self.total_queue = 0.0
        self.total_wait = 0.0
        self.total_walk = 0.0
        self.total_play = 0.0
        self.missed_shows = 0
        self.closed_projects = 0
        
        self.project_info = project_info
    
    def get_remaining_suffix(self) -> List[int]:
        """获取剩余后缀（未访问的部分）"""
        return [p for p in self.remaining_suffix if p not in self.visited]
    
    def get_unvisited_projects(self) -> List[int]:
        """获取所有未访问项目"""
        all_projects = [p for p in self.project_info.keys() 
                       if p not in [0, 27] and p not in self.visited]
        return all_projects
    
    def update_after_visit(self, proj_id: int, arrive_time: float, 
                          leave_time: float, queue_time: float, 
                          wait_time: float, walk_time: float,
                          play_time: float, utility: float, status: str):
        """访问项目后更新状态"""
        self.visited.add(proj_id)
        self.executed_prefix.append(proj_id)
        self.current_node = proj_id
        self.current_time = leave_time
        
        # 从剩余后缀中移除
        if proj_id in self.remaining_suffix:
            self.remaining_suffix.remove(proj_id)
        
        self.total_utility += utility
        self.total_queue += queue_time
        self.total_wait += wait_time
        self.total_walk += walk_time
        self.total_play += play_time
        
        # 记录错过/关闭
        if 'missed' in status:
            self.missed_shows += 1
        if 'closed' in status:
            self.closed_projects += 1
        
        self.timeline.append({
            '项目': self.project_info[proj_id]['name'],
            '到达': round(arrive_time, 1),
            '排队': round(queue_time, 1),
            '等待': round(wait_time, 1),
            '离开': round(leave_time, 1),
            '效用': round(utility, 2),
            '状态': status
        })
    
    def trigger_replan(self, new_suffix: List[int], reason: str):
        """触发重规划，更新剩余后缀"""
        self.remaining_suffix = new_suffix
        self.replan_count += 1
        self.replan_log.append({
            'step': len(self.executed_prefix),
            'time': self.current_time,
            'reason': reason,
            'new_suffix_length': len(new_suffix)
        })


# ==========================================
# 触发判断模块（修正版）
# ==========================================

def check_replan_trigger(state: DynamicState, queue_df: pd.DataFrame, 
                        scenario: str, project_info: Dict) -> Tuple[bool, str, Dict]:
    """
    检查是否触发重规划（修正版）
    
    修正要点：
    1. 不是每次都触发
    2. 阈值触发：排队偏差>30%才触发
    3. 返回详细的触发信息
    
    返回: (是否触发, 触发原因, 详细信息)
    """
    remaining = state.get_remaining_suffix()
    
    if len(remaining) == 0:
        return False, "no_remaining", {}
    
    next_project = remaining[0]
    
    # 检查下一个项目是否有效
    if next_project not in project_info:
        return True, "invalid_next_project", {}
    
    p_info = project_info[next_project]
    
    # 只对普通项目检查排队偏差
    if p_info['type'] == 'normal' and next_project not in [0, 27]:
        # 获取实时排队
        proj_code = f"P{next_project:02d}"
        realtime_queue, _ = get_queue_time(queue_df, scenario, project_info,
                                          next_project, state.current_time)
        
        # 获取GMM预测排队
        predicted_queue = 0.0
        if 'base_q' in p_info and 'peaks' in p_info:
            predicted_queue = get_gmm_queue(state.current_time, 
                                           p_info['base_q'], p_info['peaks'])
        
        if predicted_queue > 0:
            deviation = abs(realtime_queue - predicted_queue) / predicted_queue
            
            trigger_info = {
                'project': p_info['name'],
                'realtime_queue': round(realtime_queue, 1),
                'predicted_queue': round(predicted_queue, 1),
                'deviation': round(deviation, 3)
            }
            
            # 阈值触发
            if deviation > CONFIG.REPLAN_THRESHOLD:
                return True, f"threshold_trigger", trigger_info
            else:
                # 偏差不大，不触发
                return False, "deviation_acceptable", trigger_info
    
    # 演出项目或其他情况：不触发
    return False, "no_trigger_needed", {}


# ==========================================
# 候选集筛选模块（修正版）
# ==========================================

def calculate_instant_attractiveness(proj_id: int, state: DynamicState,
                                    distance_matrix: np.ndarray,
                                    queue_df: pd.DataFrame, scenario: str,
                                    project_info: Dict,
                                    is_in_suffix: bool = False) -> float:
    """
    计算即时吸引力 G_i(t_k)（修正版）
    
    修正要点：
    1. 如果项目在剩余后缀中，给予优先权重
    2. 考虑时间窗约束
    3. 考虑剩余可观看比例
    """
    if proj_id not in project_info:
        return -float('inf')
    
    p_info = project_info[proj_id]
    current_time = state.current_time
    current_node = state.current_node
    
    # 步行时间
    walk_time = distance_matrix[current_node, proj_id]
    arrive_time = current_time + walk_time
    
    # 超过闭园时间
    if arrive_time >= CONFIG.PARK_CLOSE_TIME:
        return -float('inf')
    
    utility = p_info['utility']
    time_window_start, time_window_end = p_info['time_window']
    
    if p_info['type'] == 'show':
        # 演出项目
        if arrive_time >= time_window_end:
            return -float('inf')  # 错过演出
        
        # 计算可观看比例
        actual_start = max(arrive_time, time_window_start)
        full_duration = max(time_window_end - time_window_start, 1e-6)
        watched_duration = max(0.0, time_window_end - actual_start)
        watched_ratio = min(1.0, watched_duration / full_duration)
        
        effective_utility = utility * watched_ratio
        
        # 等待时间
        wait_time = max(0.0, time_window_start - arrive_time)
        
        # 总时间成本
        total_time = walk_time + wait_time + (time_window_end - actual_start)
        
    else:
        # 普通项目
        if arrive_time > time_window_end:
            return -float('inf')  # 项目关闭
        
        # 获取排队时间
        queue_time, _ = get_queue_time(queue_df, scenario, project_info, 
                                      proj_id, arrive_time)
        
        play_time = p_info['duration']
        total_time = walk_time + queue_time + play_time
        effective_utility = utility
    
    # 即时吸引力 = 效用 / 时间成本
    if total_time > 0:
        attractiveness = effective_utility / total_time
    else:
        attractiveness = effective_utility
    
    # 如果在剩余后缀中，给予优先权重
    if is_in_suffix:
        attractiveness *= (1 + CONFIG.SUFFIX_PRIORITY)
    
    return attractiveness


def select_candidate_set(state: DynamicState, distance_matrix: np.ndarray,
                        queue_df: pd.DataFrame, scenario: str,
                        project_info: Dict, K: int = None) -> List[int]:
    """
    贪心筛选候选集 C_k（修正版）
    
    修正要点：
    1. 优先考虑剩余后缀中的项目
    2. 允许引入少量后缀外项目作为替代
    3. 返回按吸引力排序的候选集
    """
    if K is None:
        K = CONFIG.CANDIDATE_SIZE
    
    remaining_suffix = state.get_remaining_suffix()
    all_unvisited = state.get_unvisited_projects()
    
    if len(all_unvisited) == 0:
        return []
    
    # 计算所有未访问项目的即时吸引力
    attractiveness_scores = []
    
    for proj_id in all_unvisited:
        is_in_suffix = proj_id in remaining_suffix
        score = calculate_instant_attractiveness(proj_id, state, distance_matrix,
                                                queue_df, scenario, project_info,
                                                is_in_suffix)
        if score > -float('inf'):
            attractiveness_scores.append((proj_id, score, is_in_suffix))
    
    # 按吸引力排序
    attractiveness_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 选择前K个
    candidate_set = [proj_id for proj_id, _, _ in attractiveness_scores[:K]]
    
    return candidate_set


# ==========================================
# 模拟退火局部修正模块（修正版）
# ==========================================

def local_simulated_annealing(candidate_set: List[int], state: DynamicState,
                             distance_matrix: np.ndarray, queue_df: pd.DataFrame,
                             scenario: str, project_info: Dict) -> List[int]:
    """
    在候选集上使用模拟退火优化后续顺序（修正版）
    
    修正要点：
    1. 初始解使用贪心排序（而不是随机打乱）
    2. 只做局部修正
    3. 返回局部最优后缀路径
    """
    if len(candidate_set) == 0:
        return []
    
    if len(candidate_set) == 1:
        return candidate_set
    
    # 初始解：使用贪心排序（已经按吸引力排序）
    current_route = candidate_set.copy()
    
    # 评估初始解
    current_result = evaluate_route_q2(
        current_route, distance_matrix, project_info, queue_df, scenario,
        state.current_time, state.current_node, CONFIG.END_NODE, True
    )
    current_score = current_result['final_score']
    
    best_route = current_route.copy()
    best_score = current_score
    
    temperature = CONFIG.SA_LOCAL_TEMP
    iteration = 0
    
    while temperature > 0.1 and iteration < CONFIG.SA_LOCAL_ITERATIONS:
        # 生成邻域解：交换两个项目
        new_route = current_route.copy()
        if len(new_route) >= 2:
            i, j = random.sample(range(len(new_route)), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
        
        # 评估新解
        new_result = evaluate_route_q2(
            new_route, distance_matrix, project_info, queue_df, scenario,
            state.current_time, state.current_node, CONFIG.END_NODE, True
        )
        new_score = new_result['final_score']
        
        delta = new_score - current_score
        
        # 接受准则
        if delta > 0:
            current_route = new_route
            current_score = new_score
            if new_score > best_score:
                best_route = new_route.copy()
                best_score = new_score
        else:
            accept_prob = np.exp(delta / temperature)
            if random.random() < accept_prob:
                current_route = new_route
                current_score = new_score
        
        temperature *= CONFIG.SA_LOCAL_COOLING
        iteration += 1
    
    return best_route


# ==========================================
# 主流程模块（修正版）
# ==========================================

def dynamic_rolling_replan(initial_route: List[int], distance_matrix: np.ndarray,
                          project_info: Dict, queue_df: pd.DataFrame,
                          scenario: str, crowd_type: str) -> Tuple[DynamicState, Dict]:
    """
    动态滚动重规划主流程（修正版）
    
    修正要点：
    1. 真正基于Q1初始路径
    2. 阈值触发（不是每次都重规划）
    3. 候选集优先基于剩余后缀
    4. 完整的演出项目统计
    """
    # 初始化状态
    state = DynamicState(initial_route, project_info)
    
    print(f"\n开始动态滚动重规划")
    print(f"  初始路径长度: {len(initial_route)}")
    print(f"  场景: {scenario}, 人群: {crowd_type}")
    print(f"  触发阈值: {CONFIG.REPLAN_THRESHOLD*100:.0f}%")
    
    step = 0
    
    while state.current_time < CONFIG.PARK_CLOSE_TIME:
        step += 1
        remaining = state.get_remaining_suffix()
        
        if len(remaining) == 0:
            print(f"\n步骤{step}: 所有项目已访问完毕")
            break
        
        # 检查触发条件
        should_replan, trigger_reason, trigger_info = check_replan_trigger(
            state, queue_df, scenario, project_info
        )
        
        if should_replan:
            print(f"\n步骤{step}: 触发重规划 ({trigger_reason})")
            if trigger_info:
                if 'deviation' in trigger_info:
                    print(f"  项目: {trigger_info['project']}")
                    print(f"  实时排队: {trigger_info['realtime_queue']}分钟")
                    print(f"  预测排队: {trigger_info['predicted_queue']}分钟")
                    print(f"  偏差: {trigger_info['deviation']*100:.1f}%")
            
            # 贪心筛选候选集
            candidate_set = select_candidate_set(
                state, distance_matrix, queue_df, scenario, project_info
            )
            
            if len(candidate_set) == 0:
                print(f"  无可行候选项目，结束规划")
                break
            
            print(f"  候选集大小: {len(candidate_set)}")
            
            # 模拟退火局部优化
            optimized_suffix = local_simulated_annealing(
                candidate_set, state, distance_matrix, queue_df, 
                scenario, project_info
            )
            
            # 更新剩余后缀
            state.trigger_replan(optimized_suffix, trigger_reason)
            print(f"  优化后后缀长度: {len(optimized_suffix)}")
        else:
            # 不触发重规划，继续按原路径执行
            if trigger_reason == "deviation_acceptable" and trigger_info:
                print(f"\n步骤{step}: 偏差可接受，继续原路径")
                print(f"  下一项目: {trigger_info['project']}")
                print(f"  偏差: {trigger_info['deviation']*100:.1f}%")
        
        # 执行下一个项目
        next_project = state.remaining_suffix[0] if len(state.remaining_suffix) > 0 else None
        
        if next_project is None or next_project in state.visited:
            break
        
        if next_project not in project_info:
            state.remaining_suffix.pop(0)
            continue
        
        p_info = project_info[next_project]
        
        # 计算到达时间
        walk_time = distance_matrix[state.current_node, next_project]
        arrive_time = state.current_time + walk_time
        
        if arrive_time >= CONFIG.PARK_CLOSE_TIME:
            print(f"\n步骤{step}: 到达{p_info['name']}会超过闭园时间，结束规划")
            break
        
        # 执行项目
        time_window_start, time_window_end = p_info['time_window']
        wait_time = 0.0
        queue_time = 0.0
        gained_utility = 0.0
        status = 'completed'
        
        # 等待开放
        if arrive_time < time_window_start:
            wait_time = time_window_start - arrive_time
            arrive_time = time_window_start
            status = 'waited_for_open'
        
        if p_info['type'] == 'show':
            # 演出项目
            if arrive_time >= time_window_end:
                print(f"  错过演出: {p_info['name']}")
                state.missed_shows += 1
                state.remaining_suffix.remove(next_project)
                state.visited.add(next_project)
                continue
            
            actual_start = max(arrive_time, time_window_start)
            full_duration = max(time_window_end - time_window_start, 1e-6)
            watched_duration = max(0.0, time_window_end - actual_start)
            watched_ratio = min(1.0, watched_duration / full_duration)
            
            gained_utility = p_info['utility'] * watched_ratio
            leave_time = time_window_end
            play_time = watched_duration
            
            if actual_start > time_window_start:
                status = 'partial_show'
        else:
            # 普通项目
            if arrive_time > time_window_end:
                print(f"  项目已关闭: {p_info['name']}")
                state.closed_projects += 1
                state.remaining_suffix.remove(next_project)
                state.visited.add(next_project)
                continue
            
            queue_time, data_source = get_queue_time(queue_df, scenario, project_info, 
                                                    next_project, arrive_time)
            play_time = p_info['duration']
            leave_time = arrive_time + queue_time + play_time
            gained_utility = p_info['utility']
        
        # 检查是否超时
        if leave_time > CONFIG.PARK_CLOSE_TIME:
            print(f"\n步骤{step}: 完成{p_info['name']}会超过闭园时间，结束规划")
            break
        
        # 更新状态
        state.update_after_visit(
            next_project, arrive_time, leave_time, queue_time, 
            wait_time, walk_time, play_time, gained_utility, status
        )
        
        print(f"  访问: {p_info['name']}, 效用: {gained_utility:.2f}, "
              f"离开时间: {leave_time:.1f}")
    
    # 返回终点
    if state.current_node != CONFIG.END_NODE:
        walk_time = distance_matrix[state.current_node, CONFIG.END_NODE]
        arrive_exit = state.current_time + walk_time
        
        if arrive_exit <= CONFIG.PARK_CLOSE_TIME:
            state.current_time = arrive_exit
            state.total_walk += walk_time
            print(f"\n返回出口，总耗时: {state.current_time - CONFIG.START_TIME:.1f}分钟")
    
    # 计算最终得分（统一评估）
    time_cost = (CONFIG.WEIGHT_QUEUE_TIME * state.total_queue +
                CONFIG.WEIGHT_WAIT_TIME * state.total_wait +
                CONFIG.WEIGHT_WALK_TIME * state.total_walk)
    overtime = max(0, state.current_time - CONFIG.PARK_CLOSE_TIME)
    overtime_penalty = CONFIG.WEIGHT_OVERTIME * (overtime ** 2) if overtime > 0 else 0
    missed_penalty = CONFIG.WEIGHT_MISSED_SHOW * (state.missed_shows + state.closed_projects)
    
    visited_types = [project_info[node]['type'] for node in state.executed_prefix 
                    if node in project_info]
    diversity_bonus = CONFIG.WEIGHT_DIVERSITY * len(set(visited_types)) if visited_types else 0
    
    final_score = state.total_utility - time_cost - overtime_penalty - missed_penalty + diversity_bonus
    
    # 汇总结果
    result = {
        'executed_path': state.executed_prefix,
        'timeline': state.timeline,
        'final_score': round(final_score, 2),
        'total_utility': state.total_utility,
        'total_queue': state.total_queue,
        'total_wait': state.total_wait,
        'total_walk': state.total_walk,
        'total_play': state.total_play,
        'total_time': state.current_time - CONFIG.START_TIME,
        'visited_count': len(state.executed_prefix),
        'missed_shows': state.missed_shows,
        'closed_projects': state.closed_projects,
        'overtime': round(overtime, 1),
        'replan_count': state.replan_count,
        'replan_log': state.replan_log
    }
    
    return state, result


# ==========================================
# 可视化模块
# ==========================================

def plot_timeline_q2(result: Dict, scenario: str, crowd_type: str, save_path: str):
    """绘制时间线甘特图"""
    df = pd.DataFrame(result['timeline'])
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'completed': '#4ECDC4',
        'waited_for_open': '#FFE66D',
        'partial_show': '#FF8C42',
        'full_show': '#95E1D3',
        'missed_show': '#FF6B6B',
        'closed': '#95A5A6'
    }
    
    for i, row in df.iterrows():
        arrive = row['到达']
        leave = row['离开']
        duration = leave - arrive
        
        color = colors.get(row['状态'], '#4ECDC4')
        
        ax.barh(i, duration, left=arrive, height=0.6, 
               color=color, edgecolor='black', linewidth=1.5)
        ax.text(arrive + duration/2, i, row['项目'], 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{i+1}" for i in range(len(df))])
    ax.set_xlabel('时间（分钟）', fontsize=12)
    ax.set_ylabel('项目序号', fontsize=12)
    ax.set_title(f'Q2-动态滚动规划时间线\n{scenario} - {crowd_type}游客', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['completed'], label='正常完成'),
        Patch(facecolor=colors['waited_for_open'], label='等待开放'),
        Patch(facecolor=colors['full_show'], label='完整观看'),
        Patch(facecolor=colors['partial_show'], label='部分观看'),
        Patch(facecolor=colors['missed_show'], label='错过演出'),
        Patch(facecolor=colors['closed'], label='已关闭')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG.FIGURE_DPI, bbox_inches='tight')
    print(f"✓ 时间线图已保存: {save_path}")
    plt.close()


def plot_queue_comparison_q2(queue_df: pd.DataFrame, scenario: str,
                            executed_path: List[int], project_info: Dict,
                            save_path: str):
    """绘制实时排队时间变化"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for proj_id in executed_path[:10]:  # 只显示前10个
        if proj_id not in project_info:
            continue
        
        p_info = project_info[proj_id]
        if p_info['type'] == 'show':  # 演出不显示排队
            continue
        
        proj_code = f"P{proj_id:02d}"
        data = queue_df[
            (queue_df['scenario'] == scenario) & 
            (queue_df['project_id'] == proj_code)
        ]
        
        if len(data) > 0:
            ax.plot(data['time_min'], data['realtime_wait_min'],
                   label=p_info['name'], marker='o', markersize=3)
    
    ax.set_xlabel('时间（分钟）', fontsize=12)
    ax.set_ylabel('排队时间（分钟）', fontsize=12)
    ax.set_title(f'Q2-实时排队时间变化\n{scenario}', fontsize=14, fontweight='bold')
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
    print(" " * 10 + "迪士尼乐园路线优化系统 - 第二问（完全修正版）")
    print(" " * 5 + "基于问题一最优路径的动态滚动修正模型")
    print("=" * 70 + "\n")
    
    # 加载数据
    print("=" * 70)
    print("数据加载")
    print("=" * 70 + "\n")
    
    queue_df = load_realtime_queue_data(
        os.path.join(CONFIG.DATA_DIR, 'new_simulated_queue_10min(1).csv')
    )
    
    project_info, _ = load_projects_data(
        os.path.join(CONFIG.DATA_DIR, 'projects_data.csv')
    )
    
    distance_matrix = load_distance_matrix(
        os.path.join(CONFIG.DATA_DIR, 'poi.csv')
    )
    
    # 选择场景和人群
    scenario = '工作日'
    crowd_type = '普通'
    algorithm = '模拟退火'
    
    print(f"\n{'='*70}")
    print(f"运行场景: {scenario} - {crowd_type}游客")
    print(f"{'='*70}\n")
    
    # 计算效用
    project_info = calculate_utility_scores(project_info, crowd_type)
    
    # 读取Q1初始路径
    initial_route = load_q1_initial_route(scenario, crowd_type, algorithm)
    
    if initial_route is None:
        # 使用默认启发式路径（按效用排序）
        print("使用默认启发式初始路径")
        projects = [(pid, info['utility']) for pid, info in project_info.items()
                   if pid not in [0, 27]]
        projects.sort(key=lambda x: x[1], reverse=True)
        initial_route = [pid for pid, _ in projects[:20]]  # 取前20个
    
    print(f"初始路径: {len(initial_route)} 个项目")
    
    # 运行动态滚动重规划
    print(f"\n{'='*70}")
    print("开始动态滚动重规划")
    print(f"{'='*70}")
    
    state, result = dynamic_rolling_replan(
        initial_route, distance_matrix, project_info, 
        queue_df, scenario, crowd_type
    )
    
    # 打印结果
    print(f"\n{'='*70}")
    print("优化结果")
    print(f"{'='*70}\n")
    
    print(f"【综合得分】 {result['final_score']:.2f}")
    print(f"【总效用】 {result['total_utility']:.2f}")
    print(f"【访问项目数】 {result['visited_count']} 个")
    print(f"【总耗时】 {result['total_time']:.1f}分钟")
    print(f"【游玩时间】 {result['total_play']:.1f}分钟")
    print(f"【排队时间】 {result['total_queue']:.1f}分钟")
    print(f"【等待时间】 {result['total_wait']:.1f}分钟")
    print(f"【步行时间】 {result['total_walk']:.1f}分钟")
    print(f"【错过演出】 {result['missed_shows']} 场")
    print(f"【关闭项目】 {result['closed_projects']} 个")
    print(f"【超时情况】 {result['overtime']:.1f}分钟")
    print(f"【重规划次数】 {result['replan_count']} 次\n")
    
    print("【实际执行路径】")
    for i, proj_id in enumerate(result['executed_path'], 1):
        print(f"  {i}. {project_info[proj_id]['name']}")
    
    print(f"\n【详细时间线】")
    df_timeline = pd.DataFrame(result['timeline'])
    print(df_timeline.to_string(index=False))
    
    if result['replan_count'] > 0:
        print(f"\n【重规划日志】")
        for log in result['replan_log']:
            print(f"  步骤{log['step']}: {log['reason']}, "
                  f"时间{log['time']:.1f}, 新后缀长度{log['new_suffix_length']}")
    
    # 生成可视化
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    timeline_path = os.path.join(CONFIG.OUTPUT_DIR, 
                                 f'Q2-timeline-{scenario}-{crowd_type}.png')
    plot_timeline_q2(result, scenario, crowd_type, timeline_path)
    
    queue_path = os.path.join(CONFIG.OUTPUT_DIR,
                             f'Q2-queue-{scenario}-{crowd_type}.png')
    plot_queue_comparison_q2(queue_df, scenario, result['executed_path'],
                            project_info, queue_path)
    
    print(f"\n{'='*70}")
    print("✓ 优化完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
