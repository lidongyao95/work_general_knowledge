#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键构建脚本：合并 content/ 到根目录 MD，然后生成 CSV 文件
"""

import sys
import os
from pathlib import Path

# 添加 scripts 目录到路径，以便导入其他脚本
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# 导入 merge_to_root 和 md_to_anki
from merge_to_root import main as merge_main
from md_to_anki import MarkdownToAnkiConverter


def build_all():
    """执行完整的构建流程：合并 + 生成 CSV"""
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("开始构建：合并 content/ → 根目录 MD → CSV")
    print("=" * 80)
    
    # 步骤 1: 合并 content/ 到根目录 MD
    print("\n步骤 1: 合并 content/ 到根目录 MD")
    print("-" * 80)
    try:
        merge_main()
    except Exception as e:
        print(f"✗ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 步骤 2: 生成 CSV 文件
    print("\n步骤 2: 生成 CSV 文件")
    print("-" * 80)
    
    md_files = {
        'work': 'work_general_knowledge.md',
        'leetcode': 'leetcode_hot100.md',
        'llm': 'llm_knowledge.md',
        'robotics': 'robotics_knowledge.md'
    }
    
    converter = MarkdownToAnkiConverter()
    
    for domain, md_filename in md_files.items():
        md_path = project_root / md_filename
        csv_path = project_root / f"{md_filename.replace('.md', '_anki.csv')}"
        
        if not md_path.exists():
            print(f"警告: MD 文件不存在，跳过: {md_path}")
            continue
        
        print(f"\n处理 {domain}: {md_filename}")
        try:
            # 解析 MD 文件
            cards = converter.parse_markdown_file(str(md_path))
            print(f"  找到 {len(cards)} 张卡片")
            
            # 生成 CSV
            converter.generate_csv(cards, str(csv_path))
            print(f"  ✓ 已生成: {csv_path}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("构建完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    for domain, md_filename in md_files.items():
        md_path = project_root / md_filename
        csv_path = project_root / f"{md_filename.replace('.md', '_anki.csv')}"
        if md_path.exists():
            print(f"  - {md_path.name}")
        if csv_path.exists():
            print(f"  - {csv_path.name}")


if __name__ == '__main__':
    build_all()
