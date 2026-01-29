# Anki记忆卡片生成项目

## 项目概述

本项目用于从Markdown文件生成Anki记忆卡片，支持自动化的工作流程，特别适合在Cursor中使用AI Agent协助管理知识卡片。

## 项目结构

```
interview/
├── README.md                          # 本文件，项目总说明
│
├── content/                           # 源真相：章节文件（按知识域分目录）
│   ├── work/                          # 工作通识知识域
│   │   ├── manifest.yaml              # 章节顺序清单
│   │   ├── 项目经历.md                # 章节文件（按 ## 拆分）
│   │   ├── SoC功耗.md
│   │   └── [其他章节文件...]
│   └── leetcode/                      # LeetCode域
│       ├── manifest.yaml              # 章节顺序清单
│       ├── 数组.md                    # 章节文件（按 ## 拆分）
│       ├── 链表.md
│       └── [其他章节文件...]
│
├── work_general_knowledge.md          # 构建产物：完整MD（由 scripts/build.py 生成）
├── work_general_knowledge_anki.csv   # 构建产物：CSV（由 scripts/build.py 生成）
├── leetcode_hot100.md                # 构建产物：完整MD（由 scripts/build.py 生成）
├── leetcode_hot100_anki.csv         # 构建产物：CSV（由 scripts/build.py 生成）
│
├── llm_knowledge.md                  # 大模型相关知识（仅骨架，不参与重构）
├── llm_knowledge_anki.csv            # 大模型相关知识CSV
│
├── scripts/                          # 脚本目录
│   ├── merge_to_root.py              # 合并脚本：content/ → 根目录MD
│   ├── build.py                      # 一键构建：合并 + 生成CSV
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

### 源真相（content/）
- **`content/work/`** - 工作通识知识域，按章节拆分成多个 `.md` 文件
  - `manifest.yaml` - 记录文档主标题和章节文件顺序
  - `*.md` - 各章节文件（如 `项目经历.md`、`SoC功耗.md` 等）
- **`content/leetcode/`** - LeetCode域，按章节拆分成多个 `.md` 文件
  - `manifest.yaml` - 记录文档主标题和章节文件顺序
  - `*.md` - 各章节文件（如 `数组.md`、`链表.md` 等）

**编辑卡片时，请在 `content/` 下对应的章节文件中编辑。**

### 构建产物（根目录）
- **`work_general_knowledge.md`** - 由 `scripts/build.py` 自动生成，合并 content/work/ 的所有章节
- **`work_general_knowledge_anki.csv`** - 由 `scripts/build.py` 自动生成，可直接导入Anki使用
- **`leetcode_hot100.md`** - 由 `scripts/build.py` 自动生成，合并 content/leetcode/ 的所有章节
- **`leetcode_hot100_anki.csv`** - 由 `scripts/build.py` 自动生成，可直接导入Anki使用
- **`llm_knowledge.md`** - 大模型相关知识（仅骨架，不参与重构，保持现状）
- **`llm_knowledge_anki.csv`** - 大模型相关知识CSV

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
我需要添加一个新的知识卡片。

主题：StatsD中的某个概念
位置：应该在 content/work/项目经历.md 章节下

请按照 scripts/ai_workflow.md 中规定的方法：
1. 在 content/work/项目经历.md 中找到合适的位置
2. 添加新卡片
3. 运行 python scripts/build.py 重新生成根目录MD和CSV
```

#### 方式2：引用工作流文档

```
请参考 scripts/ai_workflow.md 中的工作流程，帮我添加一个新的知识卡片。

具体需求：
- 主题：[你的主题]
- 域：[work 或 leetcode]
- 章节：[目标章节文件名，如 项目经历.md]
- 问题：[问题内容]
- 答案：[答案内容]
```

#### 方式3：让Agent自动定位

```
我发现 content/work/ 中缺少关于[某个主题]的卡片。

请按照 scripts/ai_workflow.md 的流程：
1. 查看 content/work/manifest.yaml 找到合适的章节
2. 在对应的章节文件中添加新卡片
3. 运行 python scripts/build.py 重新生成
```

### AI Agent工作流程（参考 scripts/ai_workflow.md）

Agent应该按照以下步骤工作：

#### 步骤1：定位要编辑的章节文件

- 查看 `content/work/manifest.yaml` 或 `content/leetcode/manifest.yaml` 找到章节列表
- 或直接查看 `content/<domain>/` 目录下的 `.md` 文件名
- 确定要编辑的章节文件（如 `content/work/项目经历.md`）

#### 步骤2：在章节文件中添加新卡片

Agent应该：
- 在确定的章节文件中添加新卡片
- 确保卡片格式正确：
  ```markdown
  **问题**：你的问题内容
  
  **答案**：
  
  你的答案内容...
  
  ---
  ```
- 确保卡片在正确的章节下

#### 步骤3：运行构建脚本

```bash
python scripts/build.py
```

这会：
1. 合并 `content/` 下的所有章节文件到根目录的完整 MD
2. 对根目录的 MD 运行 `md_to_anki` 生成 CSV 文件

#### 步骤4：验证正确性（可选）

```bash
python tests/test_scripts.py work_general_knowledge.md work_general_knowledge_anki.csv
python tests/test_scripts.py leetcode_hot100.md leetcode_hot100_anki.csv
```

查看测试报告，确保生成正确。

### 给AI Agent的提示词模板

你可以直接复制以下提示词给Cursor Agent：

```
请按照 scripts/ai_workflow.md 中规定的方法，帮我添加一个新的知识卡片。

需求：
- 主题：[主题名称]
- 域：[work 或 leetcode]
- 章节：[章节文件名，如 项目经历.md]
- 问题：[问题内容]
- 答案：[答案内容]

请执行以下步骤：
1. 在 content/<domain>/<章节文件>.md 中找到合适的位置
2. 添加新卡片（格式参考 scripts/ai_workflow.md）
3. 运行 python scripts/build.py 重新生成根目录MD和CSV
4. （可选）运行 test_scripts.py 验证结果

请严格按照 scripts/ai_workflow.md 中的格式要求添加卡片。
```

## 快速开始

### 手动添加新卡片

1. **编辑章节文件**：在 `content/work/` 或 `content/leetcode/` 下找到对应的章节 `.md` 文件并编辑
2. **运行构建**：执行 `python scripts/build.py`，这会自动：
   - 合并所有章节文件到根目录的完整 MD
   - 生成对应的 CSV 文件
3. **导入Anki**：将生成的 CSV 文件导入Anki

### 使用AI Agent添加新卡片

1. 在Cursor中打开项目
2. 告诉Agent你的需求（参考上面的提示词模板）
3. Agent会自动完成所有步骤（编辑 content/ 下的章节文件 → 运行 build.py）

## 脚本说明

### build.py（推荐使用）
一键构建脚本：合并 content/ 到根目录 MD，然后生成 CSV 文件。

```bash
python scripts/build.py
```

**这是日常使用的脚本**：每次在 `content/` 下编辑卡片后，运行此脚本即可更新根目录的 MD 和 CSV。

### merge_to_root.py
合并脚本：将 `content/` 下的章节文件合并回根目录的完整 MD 文件。

```bash
python scripts/merge_to_root.py
```

通常不需要单独运行，`build.py` 会自动调用它。

### md_to_anki.py
将Markdown文件转换为Anki CSV格式。

```bash
python scripts/md_to_anki.py <markdown_file> [output_csv_file]
```

通常不需要单独运行，`build.py` 会自动调用它。

### md_structure_analyzer.py
分析Markdown文件的结构，输出JSON格式的结构信息。

```bash
python scripts/md_structure_analyzer.py <markdown_file> [output_json_file]
```

可用于查看根目录合并后的完整 MD 的结构。

### test_scripts.py
测试脚本的正确性，对比生成的CSV和现有的CSV。

```bash
python tests/test_scripts.py <markdown_file> <existing_csv_file>
```

详细说明请参考 `scripts/ai_workflow.md`。

## 文件关系

```
content/（源真相：章节文件）
    ↓ [运行 build.py]
根目录 MD（构建产物：完整MD）
    ↓ [build.py 自动调用 md_to_anki.py]
根目录 CSV（构建产物：CSV） ← 导入Anki使用
```

- **源真相**：`content/` 目录下的章节 `.md` 文件是源文件，所有内容编辑都在这里
- **构建产物**：根目录下的 `.md` 和 `.csv` 文件由 `build.py` 自动生成，用于阅读和导入Anki
- **临时文件**：`output/` 目录的文件是临时文件，用于测试，可以随时清理

**工作流程**：编辑 `content/<domain>/<章节>.md` → 运行 `python scripts/build.py` → 根目录 MD 和 CSV 自动更新

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

