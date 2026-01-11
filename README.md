# Anki记忆卡片生成项目

## 项目概述

本项目用于从Markdown文件生成Anki记忆卡片，支持自动化的工作流程，特别适合在Cursor中使用AI Agent协助管理知识卡片。

## 项目结构

```
interview/
├── README.md                          # 本文件，项目总说明
├── work_general_knowledge.md          # 工作通识知识记忆卡片（源文件）
├── work_general_knowledge_anki.csv   # 工作通识知识CSV（正式文件，导入Anki）
├── leetcode_hot100.md                # LeetCode记忆卡片（源文件）
├── leetcode_hot100_anki.csv         # LeetCode CSV（正式文件，导入Anki）
│
├── scripts/                          # 脚本目录
│   ├── md_to_anki.py                # MD到CSV转换脚本
│   ├── md_structure_analyzer.py      # MD结构分析脚本
│   ├── ai_workflow.md               # AI工作流详细说明（重要！）
│   └── SCRIPT_EVALUATION.md         # 脚本评估报告
│
├── tests/                            # 测试脚本目录
│   └── test_scripts.py              # 测试脚本
│
├── output/                           # 临时输出目录（git忽略）
│   ├── structure_*.json             # 结构分析结果
│   ├── *_generated.csv              # 测试生成的CSV
│   └── test_report.txt              # 测试报告
│
└── references/                       # 参考资料目录
    ├── README.md                    # 参考资料说明
    └── [各种原始参考资料文件]
```

## 核心文件说明

### 源文件（.md）
- **`work_general_knowledge.md`** - 工作通识知识卡片源文件，所有卡片内容在这里编辑
- **`leetcode_hot100.md`** - LeetCode卡片源文件

### 正式文件（.csv）
- **`work_general_knowledge_anki.csv`** - 正式CSV文件，可直接导入Anki使用
- **`leetcode_hot100_anki.csv`** - 正式CSV文件，可直接导入Anki使用

### 临时文件（output/）
- `output/` 目录存放脚本运行时的临时输出文件
- 已添加到 `.gitignore`，不会被git跟踪
- 可以随时清理，不影响正式文件

## 在Cursor中使用AI Agent

### 场景：需要添加新的知识卡片

当发现某个知识点缺失时，可以按照以下方式让Cursor AI Agent协助工作：

#### 方式1：直接告诉Agent你的需求

在Cursor中，你可以这样与Agent对话：

```
我需要添加一个新的知识卡片到work_general_knowledge.md中。

主题：StatsD中的某个概念
位置：应该在"项目经历" -> "StatsD"章节下

请按照scripts/ai_workflow.md中规定的方法：
1. 先运行md_structure_analyzer.py分析文件结构
2. 定位到正确的位置
3. 添加新卡片
4. 重新生成CSV文件
```

#### 方式2：引用工作流文档

```
请参考scripts/ai_workflow.md中的工作流程，帮我添加一个新的知识卡片。

具体需求：
- 主题：[你的主题]
- 章节：[目标章节]
- 问题：[问题内容]
- 答案：[答案内容]
```

#### 方式3：让Agent自动定位

```
我发现work_general_knowledge.md中缺少关于[某个主题]的卡片。

请按照scripts/ai_workflow.md的流程：
1. 分析文件结构，找到最合适的位置
2. 查看该章节的最后一个卡片编号
3. 在合适的位置添加新卡片
4. 重新生成CSV并验证
```

### AI Agent工作流程（参考 scripts/ai_workflow.md）

Agent应该按照以下步骤工作：

#### 步骤1：定位缺失知识的位置

```bash
python scripts/md_structure_analyzer.py work_general_knowledge.md output/structure.json
```

这会生成JSON文件，包含：
- 所有章节的标题、层级、起始行号、结束行号
- 每个章节下的卡片位置
- 章节路径（用于生成标签）

Agent应该：
- 读取生成的JSON文件
- 找到目标章节
- 确定新卡片应该插入的位置（最后一个卡片的end_line之后）

#### 步骤2：在MD文件中添加新内容

Agent应该：
- 在确定的位置添加新卡片
- 确保卡片格式正确：
  ```markdown
  #### 卡片 N
  
  **问题**：你的问题内容
  
  **答案**：
  
  你的答案内容...
  
  ---
  ```
- 确保卡片编号连续
- 确保卡片在正确的章节下

#### 步骤3：重新生成CSV文件

```bash
python scripts/md_to_anki.py work_general_knowledge.md work_general_knowledge_anki.csv
```

这会更新根目录下的正式CSV文件。

#### 步骤4：验证正确性（可选）

```bash
python tests/test_scripts.py work_general_knowledge.md work_general_knowledge_anki.csv
```

查看测试报告，确保生成正确。

### 给AI Agent的提示词模板

你可以直接复制以下提示词给Cursor Agent：

```
请按照scripts/ai_workflow.md中规定的方法，帮我添加一个新的知识卡片到work_general_knowledge.md中。

需求：
- 主题：[主题名称]
- 章节路径：[例如：项目经历 -> StatsD]
- 问题：[问题内容]
- 答案：[答案内容]

请执行以下步骤：
1. 运行md_structure_analyzer.py分析文件结构
2. 定位到目标章节和位置
3. 在合适的位置添加新卡片（注意卡片编号要连续）
4. 运行md_to_anki.py重新生成CSV文件
5. （可选）运行test_scripts.py验证结果

请严格按照scripts/ai_workflow.md中的格式要求添加卡片。
```

## 快速开始

### 手动添加新卡片

1. **编辑源文件**：编辑 `work_general_knowledge.md`
2. **生成CSV**：运行 `python scripts/md_to_anki.py work_general_knowledge.md work_general_knowledge_anki.csv`
3. **导入Anki**：将生成的CSV文件导入Anki

### 使用AI Agent添加新卡片

1. 在Cursor中打开项目
2. 告诉Agent你的需求（参考上面的提示词模板）
3. Agent会自动完成所有步骤

## 脚本说明

### md_to_anki.py
将Markdown文件转换为Anki CSV格式。

```bash
python scripts/md_to_anki.py <markdown_file> [output_csv_file]
```

### md_structure_analyzer.py
分析Markdown文件的结构，输出JSON格式的结构信息。

```bash
python scripts/md_structure_analyzer.py <markdown_file> [output_json_file]
```

### test_scripts.py
测试脚本的正确性，对比生成的CSV和现有的CSV。

```bash
python tests/test_scripts.py <markdown_file> <existing_csv_file>
```

详细说明请参考 `scripts/ai_workflow.md`。

## 文件关系

```
源文件（.md）
    ↓ [运行 md_to_anki.py]
正式CSV（.csv，根目录） ← 导入Anki使用
    ↑
临时文件（output/） ← 仅用于测试，git忽略
```

- **源文件**：`.md` 文件是源文件，所有内容编辑都在这里
- **正式文件**：根目录下的 `.csv` 文件是正式文件，用于导入Anki
- **临时文件**：`output/` 目录的文件是临时文件，用于测试，可以随时清理

## 注意事项

1. **正式文件**：根目录下的 `.csv` 文件是正式文件，用于导入Anki
2. **临时文件**：`output/` 目录下的文件是临时文件，已添加到 `.gitignore`
3. **源文件**：`.md` 文件是源文件，所有内容编辑都在这里
4. **备份**：建议定期备份 `.md` 和 `.csv` 文件
5. **格式**：添加卡片时请严格按照 `scripts/ai_workflow.md` 中的格式要求

## 参考资料

- **详细工作流程**：`scripts/ai_workflow.md` - AI工作流详细说明（**重要！**）
- **脚本评估**：`scripts/SCRIPT_EVALUATION.md` - 脚本功能评估报告
- **参考资料**：`references/` - 原始参考资料文件

## 清理建议

如果 `output/` 目录文件过多，可以清理：

```bash
# 清理output目录（保留目录结构）
rm -rf output/*
```

这不会影响正式文件的使用。

