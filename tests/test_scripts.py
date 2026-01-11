#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本
对比生成的CSV和现有的CSV文件，评估脚本的正确性
"""

import csv
import sys
import os
import difflib
from typing import List, Dict, Tuple

# 添加scripts目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from md_to_anki import MarkdownToAnkiConverter


class CSVComparator:
    def __init__(self):
        self.report_lines = []
    
    def read_csv(self, csv_path: str) -> List[Dict]:
        """读取CSV文件"""
        cards = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    cards.append({
                        'question': row[0],
                        'answer': row[1],
                        'tag': row[2]
                    })
        return cards
    
    def compare_csvs(self, generated_csv: str, existing_csv: str) -> Dict:
        """对比两个CSV文件"""
        print(f"正在读取生成的CSV文件: {generated_csv}")
        generated_cards = self.read_csv(generated_csv)
        
        print(f"正在读取现有的CSV文件: {existing_csv}")
        existing_cards = self.read_csv(existing_csv)
        
        print(f"\n生成的CSV卡片数量: {len(generated_cards)}")
        print(f"现有的CSV卡片数量: {len(existing_cards)}")
        
        # 对比结果
        result = {
            'generated_count': len(generated_cards),
            'existing_count': len(existing_cards),
            'count_match': len(generated_cards) == len(existing_cards),
            'card_matches': [],
            'card_differences': [],
            'tag_differences': []
        }
        
        # 对比每张卡片
        min_count = min(len(generated_cards), len(existing_cards))
        max_count = max(len(generated_cards), len(existing_cards))
        
        for i in range(min_count):
            gen_card = generated_cards[i]
            exist_card = existing_cards[i]
            
            match_result = {
                'index': i + 1,
                'question_match': gen_card['question'] == exist_card['question'],
                'answer_match': gen_card['answer'] == exist_card['answer'],
                'tag_match': gen_card['tag'] == exist_card['tag']
            }
            
            if match_result['question_match'] and match_result['answer_match'] and match_result['tag_match']:
                result['card_matches'].append(i + 1)
            else:
                result['card_differences'].append({
                    'index': i + 1,
                    'generated': gen_card,
                    'existing': exist_card,
                    'differences': match_result
                })
                
                if not match_result['tag_match']:
                    result['tag_differences'].append({
                        'index': i + 1,
                        'generated_tag': gen_card['tag'],
                        'existing_tag': exist_card['tag']
                    })
        
        # 如果有卡片数量不匹配，记录额外的卡片
        if len(generated_cards) > len(existing_cards):
            result['extra_cards'] = generated_cards[len(existing_cards):]
        elif len(existing_cards) > len(generated_cards):
            result['missing_cards'] = existing_cards[len(generated_cards):]
        
        return result
    
    def generate_report(self, comparison_result: Dict, output_file: str = None):
        """生成测试报告"""
        report = []
        report.append("=" * 80)
        report.append("Anki CSV生成脚本测试报告")
        report.append("=" * 80)
        report.append("")
        
        # 卡片数量对比
        report.append("1. 卡片数量对比")
        report.append("-" * 80)
        report.append(f"生成的CSV卡片数量: {comparison_result['generated_count']}")
        report.append(f"现有的CSV卡片数量: {comparison_result['existing_count']}")
        report.append(f"数量匹配: {'✓' if comparison_result['count_match'] else '✗'}")
        report.append("")
        
        # 卡片匹配情况
        report.append("2. 卡片匹配情况")
        report.append("-" * 80)
        match_count = len(comparison_result['card_matches'])
        total_count = min(comparison_result['generated_count'], comparison_result['existing_count'])
        match_rate = (match_count / total_count * 100) if total_count > 0 else 0
        report.append(f"完全匹配的卡片: {match_count} / {total_count} ({match_rate:.1f}%)")
        report.append("")
        
        # 差异详情
        if comparison_result['card_differences']:
            report.append("3. 卡片差异详情")
            report.append("-" * 80)
            for diff in comparison_result['card_differences'][:10]:  # 只显示前10个差异
                report.append(f"\n卡片 #{diff['index']}:")
                if not diff['differences']['question_match']:
                    report.append("  [问题不匹配]")
                    report.append(f"  生成: {diff['generated']['question'][:100]}...")
                    report.append(f"  现有: {diff['existing']['question'][:100]}...")
                if not diff['differences']['answer_match']:
                    report.append("  [答案不匹配]")
                    # 答案可能很长，只显示前200个字符
                    gen_answer = diff['generated']['answer'][:200]
                    exist_answer = diff['existing']['answer'][:200]
                    report.append(f"  生成: {gen_answer}...")
                    report.append(f"  现有: {exist_answer}...")
                if not diff['differences']['tag_match']:
                    report.append(f"  [标签不匹配] 生成: {diff['generated']['tag']}, 现有: {diff['existing']['tag']}")
            
            if len(comparison_result['card_differences']) > 10:
                report.append(f"\n... 还有 {len(comparison_result['card_differences']) - 10} 个差异未显示")
            report.append("")
        
        # 标签差异
        if comparison_result['tag_differences']:
            report.append("4. 标签差异")
            report.append("-" * 80)
            for tag_diff in comparison_result['tag_differences'][:10]:  # 只显示前10个
                report.append(f"卡片 #{tag_diff['index']}: 生成={tag_diff['generated_tag']}, 现有={tag_diff['existing_tag']}")
            if len(comparison_result['tag_differences']) > 10:
                report.append(f"... 还有 {len(comparison_result['tag_differences']) - 10} 个标签差异未显示")
            report.append("")
        
        # 额外或缺失的卡片
        if 'extra_cards' in comparison_result:
            report.append("5. 额外生成的卡片")
            report.append("-" * 80)
            report.append(f"数量: {len(comparison_result['extra_cards'])}")
            for i, card in enumerate(comparison_result['extra_cards'][:5]):  # 只显示前5个
                report.append(f"  卡片 {i+1}: {card['question'][:80]}...")
            if len(comparison_result['extra_cards']) > 5:
                report.append(f"  ... 还有 {len(comparison_result['extra_cards']) - 5} 个额外卡片")
            report.append("")
        
        if 'missing_cards' in comparison_result:
            report.append("6. 缺失的卡片")
            report.append("-" * 80)
            report.append(f"数量: {len(comparison_result['missing_cards'])}")
            for i, card in enumerate(comparison_result['missing_cards'][:5]):  # 只显示前5个
                report.append(f"  卡片 {i+1}: {card['question'][:80]}...")
            if len(comparison_result['missing_cards']) > 5:
                report.append(f"  ... 还有 {len(comparison_result['missing_cards']) - 5} 个缺失卡片")
            report.append("")
        
        # 总结
        report.append("=" * 80)
        report.append("测试总结")
        report.append("=" * 80)
        
        all_match = (
            comparison_result['count_match'] and
            len(comparison_result['card_differences']) == 0 and
            'extra_cards' not in comparison_result and
            'missing_cards' not in comparison_result
        )
        
        if all_match:
            report.append("✓ 所有卡片完全匹配！脚本工作正常。")
        else:
            report.append("✗ 发现差异，请检查脚本实现。")
            if not comparison_result['count_match']:
                report.append(f"  - 卡片数量不匹配: {comparison_result['generated_count']} vs {comparison_result['existing_count']}")
            if comparison_result['card_differences']:
                report.append(f"  - 有 {len(comparison_result['card_differences'])} 张卡片存在差异")
            if 'extra_cards' in comparison_result:
                report.append(f"  - 有 {len(comparison_result['extra_cards'])} 张额外卡片")
            if 'missing_cards' in comparison_result:
                report.append(f"  - 有 {len(comparison_result['missing_cards'])} 张缺失卡片")
        
        report.append("")
        
        # 输出报告
        report_text = '\n'.join(report)
        print(report_text)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n测试报告已保存到: {output_file}")
        
        return report_text


def main():
    if len(sys.argv) < 3:
        print("用法: python test_scripts.py <markdown_file> <existing_csv_file> [output_report_file]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    existing_csv = sys.argv[2]
    output_report = sys.argv[3] if len(sys.argv) >= 4 else None
    
    if not os.path.exists(md_file):
        print(f"错误: Markdown文件不存在: {md_file}")
        sys.exit(1)
    
    if not os.path.exists(existing_csv):
        print(f"错误: 现有CSV文件不存在: {existing_csv}")
        sys.exit(1)
    
    # 生成新的CSV文件
    print("=" * 80)
    print("步骤1: 使用脚本生成新的CSV文件")
    print("=" * 80)
    
    converter = MarkdownToAnkiConverter()
    cards = converter.parse_markdown_file(md_file)
    
    # 创建临时CSV文件
    import tempfile
    temp_csv = os.path.join(tempfile.gettempdir(), 'generated_anki_temp.csv')
    converter.generate_csv(cards, temp_csv)
    
    print(f"已生成临时CSV文件: {temp_csv}")
    print(f"生成卡片数量: {len(cards)}")
    
    # 对比CSV文件
    print("\n" + "=" * 80)
    print("步骤2: 对比生成的CSV和现有的CSV")
    print("=" * 80)
    
    comparator = CSVComparator()
    comparison_result = comparator.compare_csvs(temp_csv, existing_csv)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("步骤3: 生成测试报告")
    print("=" * 80)
    
    comparator.generate_report(comparison_result, output_report)
    
    # 清理临时文件
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
        print(f"\n已清理临时文件: {temp_csv}")


if __name__ == '__main__':
    main()

