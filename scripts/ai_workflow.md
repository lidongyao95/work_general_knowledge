# AI工作流说明文档

## 项目架构

本项目用于从Markdown文件生成Anki记忆卡片，包含以下核心组件：

1. **md_to_anki.py** - MD到CSV转换脚本
   - 功能：解析Markdown文件，提取卡片内容，生成Anki格式的CSV文件
   - 输入：Markdown文件（如 `work_general_knowledge.md`）
   - 输出：Anki CSV文件（如 `work_general_knowledge_anki.csv`）

2. **md_structure_analyzer.py** - MD结构分析脚本
   - 功能：分析Markdown文件的标题结构和章节行号对应关系
   - 输入：Markdown文件
   - 输出：JSON格式的结构信息（包含章节、卡片的位置信息）

3. **test_scripts.py** - 测试脚本
   - 功能：测试脚本的正确性，对比生成的CSV和现有的CSV文件
   - 输出：测试报告

## 工作流程

### 场景：需要添加新的知识卡片

当发现某个知识点缺失时，按照以下步骤操作：

#### 步骤1：定位缺失知识的位置

使用 `md_structure_analyzer.py` 分析MD文件结构，找到应该添加新内容的位置：

```bash
python scripts/md_structure_analyzer.py work_general_knowledge.md output/structure.json
```

这会生成一个JSON文件，包含：
- 所有章节的标题、层级、起始行号、结束行号
- 每个章节下的卡片位置
- 章节路径（用于生成标签）

**示例输出**：
```json
{
  "sections": [
    {
      "title": "项目经历",
      "level": 2,
      "start_line": 7,
      "end_line": 400,
      "path": ["项目经历"],
      "children": [
        {
          "title": "StatsD",
          "level": 3,
          "start_line": 9,
          "end_line": 399,
          "path": ["项目经历", "StatsD"],
          "cards": [
            {"card_num": 1, "start_line": 11, "end_line": 43},
            {"card_num": 2, "start_line": 45, "end_line": 70}
          ]
        }
      ]
    }
  ]
}
```

**使用方式**：
1. 查看JSON文件，找到目标章节
2. 确定应该在哪个章节下添加新卡片
3. 查看该章节的最后一个卡片的 `end_line`，新卡片应该添加在这个位置之后

#### 步骤2：在MD文件中添加新内容

在确定的位置添加新卡片，格式如下：

```markdown
#### 卡片 N

**问题**：你的问题内容

**答案**：

你的答案内容，可以包含：
- 列表项
- **粗体文本**
- 代码块：

```cpp
// 代码示例
```

---

```

**注意事项**：
- 卡片编号应该连续（如果最后一个卡片是"卡片 10"，新卡片应该是"卡片 11"）
- 问题必须以 `**问题**：` 或 `**问题**:` 开头
- 答案必须以 `**答案**：` 或 `**答案**:` 开头
- 卡片之间用 `---` 分隔
- 确保卡片在正确的章节下（根据标题层级）

#### 步骤3：重新生成CSV文件

使用 `md_to_anki.py` 重新生成CSV文件：

```bash
python scripts/md_to_anki.py work_general_knowledge.md work_general_knowledge_anki.csv
```

这会：
1. 解析MD文件中的所有卡片
2. 根据章节路径生成标签（如：`项目经历-StatsD`）
3. 将Markdown转换为HTML格式
4. 生成Anki CSV文件

#### 步骤4：验证正确性

使用 `test_scripts.py` 验证生成的CSV文件：

```bash
python tests/test_scripts.py work_general_knowledge.md work_general_knowledge_anki.csv
```

测试脚本会：
1. 重新生成CSV文件
2. 对比新生成的CSV和现有的CSV
3. 检查卡片数量、问题文本、答案文本、标签是否一致
4. 生成测试报告

## 脚本使用说明

### md_to_anki.py

**基本用法**：
```bash
python scripts/md_to_anki.py <markdown_file> [output_csv_file]
```

**参数**：
- `markdown_file`: 输入的Markdown文件路径（必需）
- `output_csv_file`: 输出的CSV文件路径（可选，默认为 `{markdown_file}_anki.csv`）

**示例**：
```bash
# 使用默认输出文件名
python scripts/md_to_anki.py work_general_knowledge.md

# 指定输出文件名
python scripts/md_to_anki.py work_general_knowledge.md output/work_general_knowledge_anki.csv
```

**功能说明**：
- 自动识别卡片格式：`#### 卡片 N` + `**问题**：` + `**答案**：`
- 根据标题层级自动生成标签（二级标题 + 三级标题）
- 支持Markdown语法转换：粗体、列表、代码块等
- 输出Anki格式的CSV（三列：问题、答案、标签）

### md_structure_analyzer.py

**基本用法**：
```bash
python scripts/md_structure_analyzer.py <markdown_file> [output_json_file]
```

**参数**：
- `markdown_file`: 输入的Markdown文件路径（必需）
- `output_json_file`: 输出的JSON文件路径（可选，默认为 `output/structure_{markdown_file}.json`）

**示例**：
```bash
# 使用默认输出文件名
python scripts/md_structure_analyzer.py work_general_knowledge.md

# 指定输出文件名
python scripts/md_structure_analyzer.py work_general_knowledge.md output/structure.json
```

**输出格式**：
- JSON格式，包含完整的章节结构和卡片位置信息
- 可以在控制台查看结构摘要

### test_scripts.py

**基本用法**：
```bash
python tests/test_scripts.py <markdown_file> <existing_csv_file>
```

**参数**：
- `markdown_file`: 输入的Markdown文件路径（必需）
- `existing_csv_file`: 现有的CSV文件路径（用于对比）（必需）

**示例**：
```bash
python tests/test_scripts.py work_general_knowledge.md work_general_knowledge_anki.csv
```

**输出**：
- 控制台输出测试结果
- 生成测试报告文件

## 标签生成规则

标签根据Markdown文件的标题层级自动生成：

- **二级标题** (`## 标题`)：标签为 `标题`
- **三级标题** (`### 标题`)：标签为 `二级标题-三级标题`
- **示例**：
  - `## 项目经历` → 标签：`项目经历`
  - `## 项目经历` + `### StatsD` → 标签：`项目经历-StatsD`

## 常见问题

### Q1: 如何确定新卡片应该添加在哪个位置？

A: 使用 `md_structure_analyzer.py` 分析文件结构，查看目标章节的最后一个卡片的 `end_line`，新卡片应该添加在这个位置之后。

### Q2: 卡片编号不连续怎么办？

A: 卡片编号应该连续。如果发现编号不连续，需要手动修正。建议在添加新卡片前，先查看最后一个卡片的编号。

### Q3: 生成的CSV格式不正确？

A: 检查MD文件格式是否正确：
- 卡片标题格式：`#### 卡片 N`
- 问题格式：`**问题**：` 或 `**问题**:`
- 答案格式：`**答案**：` 或 `**答案**:`
- 卡片之间用 `---` 分隔

### Q4: 标签生成不正确？

A: 确保标题层级正确：
- 二级标题使用 `##`
- 三级标题使用 `###`
- 卡片应该在正确的章节下（根据缩进和标题层级）

### Q5: 代码块没有正确转换？

A: 代码块格式应该是：
```markdown
```language
代码内容
```
```

确保代码块前后都有三个反引号，且语言标识符（可选）在同一行。

## 最佳实践

1. **添加新卡片前**：
   - 先运行 `md_structure_analyzer.py` 了解文件结构
   - 确定目标章节和位置
   - 检查最后一个卡片的编号

2. **添加新卡片后**：
   - 运行 `md_to_anki.py` 重新生成CSV
   - 运行 `test_scripts.py` 验证正确性
   - 检查生成的CSV文件是否符合预期

3. **维护MD文件**：
   - 保持卡片编号连续
   - 保持标题层级一致
   - 使用统一的格式（问题、答案格式）

4. **版本控制**：
   - 在修改MD文件前，先备份
   - 提交前运行测试脚本确保正确性

