"""
一键运行所有27种情况
3种人群 × 3种日期 × 3种算法 = 27种组合
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    batch_script = os.path.join(project_root, 'Q1', 'q1_batch_run_all.py')
    
    if not os.path.exists(batch_script):
        print(f"错误：找不到 {batch_script}")
        sys.exit(1)
    
    print("=" * 70)
    print(" " * 10 + "开始批量运行所有27种情况")
    print("=" * 70)
    print("\n这将运行：")
    print("  - 3种人群类型（普通、亲子、情侣）")
    print("  - 3种日期类型（工作日、双休日、节假日）")
    print("  - 3种优化算法（模拟退火、遗传算法、蚁群算法）")
    print("  - 总计：3 × 3 × 3 = 27 种组合")
    print("\n预计耗时：15-30分钟")
    print("\n请耐心等待...\n")
    
    os.chdir(os.path.join(project_root, 'Q1'))
    result = subprocess.run([sys.executable, 'q1_batch_run_all.py'])
    sys.exit(result.returncode)
