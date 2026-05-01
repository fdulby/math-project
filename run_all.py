"""
批量运行所有9种情况的入口文件
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    batch_script = os.path.join(project_root, 'Q1', 'q1_batch_run.py')
    
    # 检查文件是否存在
    if not os.path.exists(batch_script):
        print(f"错误：找不到 {batch_script}")
        sys.exit(1)
    
    print("=" * 70)
    print(" " * 15 + "开始批量运行所有9种情况")
    print("=" * 70)
    print("\n这将需要一些时间，请耐心等待...\n")
    
    # 切换到Q1目录运行
    os.chdir(os.path.join(project_root, 'Q1'))
    
    # 运行脚本
    result = subprocess.run([sys.executable, 'q1_batch_run.py'])
    sys.exit(result.returncode)
