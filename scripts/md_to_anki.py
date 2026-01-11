#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD到Anki CSV转换脚本
从Markdown文件提取卡片内容，生成Anki格式的CSV文件
"""

import re
import csv
import html
import sys
import os
from typing import List, Dict, Tuple, Optional


class MarkdownToAnkiConverter:
    def __init__(self):
        self.cards = []
        self.current_section_path = []  # 追踪当前章节路径，用于生成标签
        
    def parse_markdown_file(self, file_path: str) -> List[Dict]:
        """解析Markdown文件，提取所有卡片"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cards = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 检查是否是标题行
            if line.startswith('#'):
                self._update_section_path(line)
            
            # 检查是否是卡片标题
            card_match = re.match(r'^####\s+卡片\s+(\d+)', line)
            if card_match:
                card_num = int(card_match.group(1))
                card = self._extract_card(lines, i, card_num)
                if card:
                    cards.append(card)
                    # 跳过已处理的行
                    i = card['end_line']
                    continue
            
            i += 1
        
        return cards
    
    def _update_section_path(self, line: str):
        """更新当前章节路径"""
        # 计算标题层级（#的数量）
        level = len(line) - len(line.lstrip('#'))
        title = line.lstrip('#').strip()
        
        # 只处理二级和三级标题（用于生成标签）
        if level == 2:
            self.current_section_path = [title]
        elif level == 3:
            if len(self.current_section_path) > 0:
                self.current_section_path = [self.current_section_path[0], title]
            else:
                self.current_section_path = [title]
    
    def _extract_card(self, lines: List[str], start_idx: int, card_num: int) -> Optional[Dict]:
        """提取单个卡片的内容"""
        i = start_idx + 1  # 跳过卡片标题行
        question = None
        answer_lines = []
        
        # 查找问题
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('**问题**：') or line.startswith('**问题**:'):
                question = line.replace('**问题**：', '').replace('**问题**:', '').strip()
                i += 1
                break
            i += 1
        
        if not question:
            return None
        
        # 查找答案
        answer_started = False
        while i < len(lines):
            line = lines[i]
            
            # 检查是否是下一个卡片或分隔符
            if line.strip().startswith('#### 卡片') or line.strip() == '---':
                break
            
            # 检查是否是答案开始
            if line.strip().startswith('**答案**：') or line.strip().startswith('**答案**:'):
                answer_started = True
                i += 1
                continue
            
            if answer_started:
                answer_lines.append(line.rstrip('\n'))
            
            i += 1
        
        if not answer_lines:
            return None
        
        # 合并答案行
        answer = '\n'.join(answer_lines).strip()
        
        # 生成标签
        tag = self._generate_tag()
        
        return {
            'card_num': card_num,
            'question': question,
            'answer': answer,
            'tag': tag,
            'start_line': start_idx + 1,
            'end_line': i
        }
    
    def _generate_tag(self) -> str:
        """根据当前章节路径生成标签"""
        if len(self.current_section_path) == 0:
            return "通用"
        elif len(self.current_section_path) == 1:
            return self.current_section_path[0]
        else:
            return '-'.join(self.current_section_path)
    
    def markdown_to_html(self, text: str) -> str:
        """将Markdown文本转换为HTML格式"""
        if not text:
            return ""
        
        # 先提取代码块
        code_blocks = []
        code_block_counter = 0
        
        def extract_code_block(match):
            nonlocal code_block_counter
            language = match.group(1) or ''
            code = match.group(2)
            # 转义HTML特殊字符
            code_escaped = html.escape(code)
            code_html = f'<pre><code style="white-space: pre; font-family: monospace; text-align: left; display: block; background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">{code_escaped}</code></pre>'
            placeholder = f'__CODE_BLOCK_{code_block_counter}__'
            code_blocks.append((placeholder, code_html))
            code_block_counter += 1
            return placeholder
        
        # 提取代码块
        pattern = r'```(\w+)?\n(.*?)```'
        text = re.sub(pattern, extract_code_block, text, flags=re.DOTALL)
        
        # HTML转义（代码块占位符不会被转义，因为它们是特殊格式）
        text = html.escape(text)
        
        # 恢复代码块
        for placeholder, code_html in code_blocks:
            text = text.replace(html.escape(placeholder), code_html)
        
        # 处理粗体 **text** -> <strong>text</strong>
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        
        # 将换行符转换为<br>
        # 注意：保持原有的列表格式，不使用<ul><li>标签
        text = text.replace('\n', '<br>')
        
        return text
    
    def format_for_anki(self, text: str) -> str:
        """格式化为Anki HTML格式"""
        html_text = self.markdown_to_html(text)
        return f'<div style="text-align: left;">{html_text}</div>'
    
    def generate_csv(self, cards: List[Dict], output_path: str):
        """生成Anki CSV文件"""
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            for card in cards:
                question_html = self.format_for_anki(card['question'])
                answer_html = self.format_for_anki(card['answer'])
                tag = card['tag']
                
                writer.writerow([question_html, answer_html, tag])


def main():
    if len(sys.argv) < 2:
        print("用法: python md_to_anki.py <markdown_file> [output_csv_file]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # 默认输出文件名
        base_name = os.path.splitext(os.path.basename(md_file))[0]
        output_file = f"{base_name}_anki.csv"
    
    if not os.path.exists(md_file):
        print(f"错误: 文件不存在: {md_file}")
        sys.exit(1)
    
    print(f"正在解析Markdown文件: {md_file}")
    converter = MarkdownToAnkiConverter()
    cards = converter.parse_markdown_file(md_file)
    
    print(f"找到 {len(cards)} 张卡片")
    
    print(f"正在生成CSV文件: {output_file}")
    converter.generate_csv(cards, output_file)
    
    print("完成！")


if __name__ == '__main__':
    main()

