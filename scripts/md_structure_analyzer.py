#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD文件结构分析脚本
分析Markdown文件的标题结构和章节行号对应关系
"""

import re
import json
import sys
import os
from typing import List, Dict, Optional


class Section:
    """章节类"""
    def __init__(self, title: str, level: int, start_line: int):
        self.title = title
        self.level = level
        self.start_line = start_line
        self.end_line = None
        self.path = []
        self.children = []
        self.cards = []  # 卡片列表，每个卡片包含start_line和end_line
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            'title': self.title,
            'level': self.level,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'path': self.path.copy()
        }
        
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        
        if self.cards:
            result['cards'] = self.cards.copy()
        
        return result


class MarkdownStructureAnalyzer:
    def __init__(self):
        self.sections = []
        self.section_stack = []  # 用于维护章节层级关系
        
    def analyze(self, file_path: str) -> Dict:
        """分析Markdown文件结构"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 重置状态
        self.sections = []
        self.section_stack = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检查是否是标题行
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                # 处理标题
                self._process_heading(title, level, i + 1)
            
            # 检查是否是卡片标题
            elif re.match(r'^####\s+卡片\s+\d+', line):
                card_num_match = re.match(r'^####\s+卡片\s+(\d+)', line)
                if card_num_match:
                    card_num = int(card_num_match.group(1))
                    card_end = self._find_card_end(lines, i)
                    self._add_card(card_num, i + 1, card_end)
            
            i += 1
        
        # 设置所有章节的结束行号
        self._set_end_lines(len(lines))
        
        # 构建结果
        result = {
            'file_path': file_path,
            'total_lines': len(lines),
            'sections': [section.to_dict() for section in self.sections]
        }
        
        return result
    
    def _process_heading(self, title: str, level: int, line_num: int):
        """处理标题"""
        # 弹出栈中层级大于等于当前层级的章节
        while self.section_stack and self.section_stack[-1].level >= level:
            self.section_stack.pop()
        
        # 创建新章节
        section = Section(title, level, line_num)
        
        # 设置路径
        if self.section_stack:
            parent = self.section_stack[-1]
            section.path = parent.path + [title]
            parent.children.append(section)
        else:
            section.path = [title]
            self.sections.append(section)
        
        # 将新章节压入栈
        self.section_stack.append(section)
    
    def _find_card_end(self, lines: List[str], start_idx: int) -> int:
        """查找卡片结束位置"""
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 检查是否是下一个卡片或分隔符
            if line.startswith('#### 卡片') or line == '---':
                return i
            
            i += 1
        
        return len(lines)
    
    def _add_card(self, card_num: int, start_line: int, end_line: int):
        """添加卡片到当前章节"""
        if self.section_stack:
            current_section = self.section_stack[-1]
            current_section.cards.append({
                'card_num': card_num,
                'start_line': start_line,
                'end_line': end_line
            })
    
    def _set_end_lines(self, total_lines: int):
        """设置所有章节的结束行号"""
        def set_end_recursive(sections: List[Section], next_line: int) -> int:
            for i in range(len(sections) - 1, -1, -1):
                section = sections[i]
                
                # 如果有子章节，先处理子章节
                if section.children:
                    section.end_line = set_end_recursive(section.children, section.start_line)
                else:
                    # 没有子章节，结束行是下一个同级或更高级标题之前
                    if i < len(sections) - 1:
                        section.end_line = sections[i + 1].start_line - 1
                    else:
                        # 最后一个章节，查找下一个同级或更高级标题
                        section.end_line = self._find_next_sibling_or_parent(section, total_lines)
                
                # 如果有卡片，确保结束行不小于最后一个卡片的结束行
                if section.cards:
                    last_card_end = max(card['end_line'] for card in section.cards)
                    if section.end_line is None or section.end_line < last_card_end:
                        section.end_line = last_card_end
            
            # 返回第一个章节的开始行号
            return sections[0].start_line - 1 if sections else next_line
        
        set_end_recursive(self.sections, total_lines)
    
    def _find_next_sibling_or_parent(self, section: Section, total_lines: int) -> int:
        """查找下一个同级或更高级标题的行号"""
        # 简化实现：返回文件末尾
        # 实际应该查找下一个同级或更高级标题
        return total_lines
    
    def save_json(self, data: Dict, output_path: str):
        """保存为JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def print_summary(self, data: Dict):
        """打印结构摘要"""
        print(f"文件: {data['file_path']}")
        print(f"总行数: {data['total_lines']}")
        print(f"\n章节结构:")
        self._print_sections(data['sections'], indent=0)
    
    def _print_sections(self, sections: List[Dict], indent: int):
        """递归打印章节"""
        for section in sections:
            prefix = '  ' * indent
            print(f"{prefix}{'#' * section['level']} {section['title']} (行 {section['start_line']}-{section['end_line']})")
            
            if section.get('cards'):
                for card in section['cards']:
                    print(f"{prefix}  - 卡片 {card['card_num']} (行 {card['start_line']}-{card['end_line']})")
            
            if section.get('children'):
                self._print_sections(section['children'], indent + 1)


def main():
    if len(sys.argv) < 2:
        print("用法: python md_structure_analyzer.py <markdown_file> [output_json_file]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # 默认输出文件名
        base_name = os.path.splitext(os.path.basename(md_file))[0]
        output_file = f"output/structure_{base_name}.json"
    
    if not os.path.exists(md_file):
        print(f"错误: 文件不存在: {md_file}")
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"正在分析Markdown文件: {md_file}")
    analyzer = MarkdownStructureAnalyzer()
    structure = analyzer.analyze(md_file)
    
    # 打印摘要
    analyzer.print_summary(structure)
    
    # 保存JSON文件
    print(f"\n正在保存结构信息到: {output_file}")
    analyzer.save_json(structure, output_file)
    
    print("完成！")


if __name__ == '__main__':
    main()

