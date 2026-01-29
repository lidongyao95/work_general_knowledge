#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并脚本：将 content/ 下的章节文件合并回根目录的完整 MD 文件
"""

import os
import re
import yaml
import sys
from pathlib import Path
from typing import List


def count_cards(content: str) -> int:
    """统计卡片数量（通过 **问题**： 或 **问题**: 的出现次数）"""
    pattern = r'\*\*问题\*\*[：:]'
    matches = re.findall(pattern, content)
    return len(matches)


def merge_domain(domain: str, content_dir: Path, output_dir: Path) -> bool:
    """
    合并一个域的所有章节文件
    
    Args:
        domain: 域名称（work 或 leetcode）
        content_dir: content/ 目录路径
        output_dir: 输出目录（项目根目录）
    
    Returns:
        bool: 是否成功
    """
    domain_dir = content_dir / domain
    manifest_path = domain_dir / 'manifest.yaml'
    
    if not manifest_path.exists():
        print(f"错误: manifest.yaml 不存在: {manifest_path}")
        return False
    
    # 读取 manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)
    
    title = manifest.get('title', '')
    files = manifest.get('files', [])
    
    if not title or not files:
        print(f"错误: manifest.yaml 格式不正确: {manifest_path}")
        return False
    
    # 读取并拼接所有章节文件
    merged_content_parts = []
    
    for filename in files:
        file_path = domain_dir / filename
        if not file_path.exists():
            print(f"警告: 章节文件不存在，跳过: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            merged_content_parts.append(content)
    
    # 拼接所有内容
    merged_content = '\n'.join(merged_content_parts)
    
    # 统计卡片数量
    card_count = count_cards(merged_content)
    
    # 构建完整的 MD 内容
    header = f"# {title}\n\n共 {card_count} 张卡片\n\n---\n\n"
    full_content = header + merged_content
    
    # 确定输出文件名
    output_filename_map = {
        'work': 'work_general_knowledge.md',
        'leetcode': 'leetcode_hot100.md'
    }
    
    output_filename = output_filename_map.get(domain)
    if not output_filename:
        print(f"错误: 未知的域: {domain}")
        return False
    
    output_path = output_dir / output_filename
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"✓ {domain}: {output_filename} ({card_count} 张卡片)")
    return True


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    content_dir = project_root / 'content'
    
    domains = ['work', 'leetcode']
    
    print("=" * 80)
    print("开始合并 content/ 到根目录 MD 文件")
    print("=" * 80)
    
    success_count = 0
    for domain in domains:
        print(f"\n处理 {domain}...")
        if merge_domain(domain, content_dir, project_root):
            success_count += 1
    
    print("\n" + "=" * 80)
    if success_count == len(domains):
        print("合并完成！")
    else:
        print(f"合并完成，但有 {len(domains) - success_count} 个域失败")
        sys.exit(1)
    print("=" * 80)


if __name__ == '__main__':
    main()
