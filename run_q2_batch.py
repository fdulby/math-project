"""
Q2批量运行入口 - 运行所有9种情况
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    batch_script = os.path.join(project_root, 'Q2', 'q2_batch.py')
    
    if not os.path.exists(batch_script):
        print(f"错误：找不到 {batch_script}")
        sys.exit(1)
    
    print("=" * 70)
    print(" " * 15 + "运行Q2 - 批量运行所有9种情况")
    print("=" * 70)
    print("\n将运行：")
    print("  - 3种场景（工作日、双休日、节假日）")
    print("  - 3种人群（普通、亲子、情侣）")
    print("  - 总计：3 × 3 = 9种组合")
    print("\n预计耗时：2-5分钟\n")
    
    os.chdir(os.path.join(project_root, 'Q2'))
    result = subprocess.run([sys.executable, 'q2_batch.py'])
    sys.exit(result.returncode)
