"""
迪士尼乐园路线优化系统 - 主入口
直接运行 Q1/q1_final.py
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    q1_script = os.path.join(project_root, 'Q1', 'q1_final.py')
    
    # 检查文件是否存在
    if not os.path.exists(q1_script):
        print(f"错误：找不到 {q1_script}")
        sys.exit(1)
    
    # 切换到Q1目录运行
    os.chdir(os.path.join(project_root, 'Q1'))
    
    # 运行脚本
    result = subprocess.run([sys.executable, 'q1_final.py'])
    sys.exit(result.returncode)
