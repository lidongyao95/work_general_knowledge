# AI工作流说明文档

## 项目架构

本项目采用**方案四（混合结构）**：按知识域分目录（work / leetcode），域内按章节拆成小文件，用 manifest 控制合并顺序。

### 核心组件

1. **build.py** - 一键构建脚本（**推荐使用**）
   - 功能：合并 `content/` 下的章节文件到根目录完整 MD，然后生成 CSV 文件
   - 输入：`content/work/`、`content/leetcode/` 下的章节文件和 manifest.yaml
   - 输出：根目录的完整 MD 和 CSV 文件
   - 用法：`python scripts/build.py`

2. **merge_to_root.py** - 合并脚本
   - 功能：将 `content/` 下的章节文件合并回根目录的完整 MD
   - 输入：`content/<domain>/manifest.yaml` 及章节文件
   - 输出：根目录的完整 MD（如 `work_general_knowledge.md`）
   - 通常不需要单独运行，`build.py` 会自动调用

3. **md_to_anki.py** - MD到CSV转换脚本
   - 功能：解析Markdown文件，提取卡片内容，生成Anki格式的CSV文件
   - 输入：Markdown文件（如 `work_general_knowledge.md`）
   - 输出：Anki CSV文件（如 `work_general_knowledge_anki.csv`）
   - 通常不需要单独运行，`build.py` 会自动调用

4. **md_structure_analyzer.py** - MD结构分析脚本
   - 功能：分析Markdown文件的标题结构和章节行号对应关系
   - 输入：Markdown文件（通常是合并后的根目录 MD）
   - 输出：JSON格式的结构信息（包含章节、卡片的位置信息）

5. **test_scripts.py** - 测试脚本
   - 功能：测试脚本的正确性，对比生成的CSV和现有的CSV文件
   - 输出：测试报告

## 工作流程

### 场景：检查知识点是否已存在

在添加新知识点之前，应该先检查该知识点是否已经存在于内容中。如果已存在，直接给出位置；如果不存在，再按照后续步骤添加。

#### 检查方法

1. **搜索 content/ 下的章节文件**（推荐）：
   - 在 `content/work/` 或 `content/leetcode/` 下搜索关键词
   - 使用grep命令递归搜索
   - 示例：`grep -r "关键词" content/work/`

2. **搜索合并后的根目录 MD**：
   - 在根目录的完整 MD 文件中搜索关键词
   - 示例：`grep -n "关键词" work_general_knowledge.md`
   - 注意：根目录 MD 由 `build.py` 生成，如果未运行 build，可能需要先运行

3. **使用结构分析脚本**（查看合并后的完整结构）：
   ```bash
   python scripts/md_structure_analyzer.py work_general_knowledge.md output/structure.json
   ```
   - 先运行 `python scripts/build.py` 确保根目录 MD 是最新的
   - 查看生成的JSON文件，搜索相关章节
   - 检查该章节下的卡片，看是否已有相关内容

4. **搜索CSV文件**：
   - 直接搜索CSV文件中的问题文本
   - CSV文件格式更简单，便于快速查找
   - 示例：`grep -i "关键词" work_general_knowledge_anki.csv`
   - 注意：CSV 由 `build.py` 生成，如果未运行 build，可能需要先运行

#### 如果知识点已存在

如果找到相关知识点，应该：
1. **给出位置信息**：
   - 章节文件路径（如：`content/work/项目经历.md`）
   - 章节路径（如：`项目经历 -> StatsD`，如果查看合并后的 MD）
   - 问题文本（完整或部分）

2. **提供访问方式**：
   - 可以直接编辑对应的章节文件（如 `content/work/项目经历.md`）
   - 如果内容需要更新，在章节文件中修改

**示例输出格式**：
```
找到相关知识点：
- 章节文件：content/work/项目经历.md
- 章节路径：项目经历 -> StatsD
- 问题：StatsD中Metric分为哪两类？各有什么特点？
- 标签：项目经历-StatsD
```

#### 如果知识点不存在

如果确认知识点不存在，则按照以下步骤添加新卡片。

### 场景：需要添加新的知识卡片

当确认某个知识点缺失时，按照以下步骤操作：

#### 步骤1：定位要编辑的章节文件

有两种方式定位：

**方式1：查看 manifest.yaml**（推荐）
- 查看 `content/work/manifest.yaml` 或 `content/leetcode/manifest.yaml`
- manifest 中列出了所有章节文件的顺序
- 根据章节名称找到对应的 `.md` 文件

**方式2：使用结构分析脚本**（查看合并后的完整结构）
```bash
# 先确保根目录 MD 是最新的
python scripts/build.py

# 分析结构
python scripts/md_structure_analyzer.py work_general_knowledge.md output/structure.json
```

这会生成一个JSON文件，包含：
- 所有章节的标题、层级、起始行号、结束行号
- 每个章节下的卡片位置
- 章节路径（用于生成标签）

**确定章节文件**：
- 根据章节名称，找到 `content/<domain>/<章节名>.md` 文件
- 例如：章节"项目经历"对应 `content/work/项目经历.md`

#### 步骤2：在章节文件中添加新内容

在确定的章节文件中添加新卡片，格式如下：

```markdown
#### 问题：你的问题内容

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
- **不需要**卡片编号（如 `#### 卡片 N`），直接添加问题即可
- 问题必须以 `#### 问题：` 或 `##### 问题：` 开头（作为 Markdown 标题）
- 答案必须以 `**答案**：` 或 `**答案**:` 开头
- 卡片之间用 `---` 分隔
- 确保卡片在正确的章节文件下

#### 步骤3：运行构建脚本

使用 `build.py` 一键完成合并和生成CSV：

```bash
python scripts/build.py
```

这会：
1. 合并 `content/` 下的所有章节文件到根目录的完整 MD
2. 自动统计卡片数量并写入「共 N 张卡片」
3. 对根目录的完整 MD 运行 `md_to_anki.py` 生成 CSV 文件

**输出**：
- 根目录的完整 MD（如 `work_general_knowledge.md`）
- 根目录的 CSV（如 `work_general_knowledge_anki.csv`）

#### 步骤4：验证正确性（可选）

使用 `test_scripts.py` 验证生成的CSV文件：

```bash
python tests/test_scripts.py work_general_knowledge.md work_general_knowledge_anki.csv
python tests/test_scripts.py leetcode_hot100.md leetcode_hot100_anki.csv
```

测试脚本会：
1. 重新生成CSV文件
2. 对比新生成的CSV和现有的CSV
3. 检查卡片数量、问题文本、答案文本、标签是否一致
4. 生成测试报告

## 脚本使用说明

### build.py（推荐使用）

**基本用法**：
```bash
python scripts/build.py
```

**功能**：
- 合并 `content/work/` 和 `content/leetcode/` 下的章节文件到根目录完整 MD
- 自动统计卡片数量并写入「共 N 张卡片」
- 对根目录的完整 MD 运行 `md_to_anki.py` 生成 CSV 文件

**输出**：
- `work_general_knowledge.md`（根目录）
- `work_general_knowledge_anki.csv`（根目录）
- `leetcode_hot100.md`（根目录）
- `leetcode_hot100_anki.csv`（根目录）

**使用场景**：
- **每次在 `content/` 下编辑卡片后，运行此脚本**即可更新根目录的 MD 和 CSV

### merge_to_root.py

**基本用法**：
```bash
python scripts/merge_to_root.py
```

**功能**：
- 读取 `content/work/` 和 `content/leetcode/` 的 manifest.yaml
- 按 manifest 中的 `files` 顺序拼接章节文件
- 生成根目录的完整 MD 文件

**输出**：
- `work_general_knowledge.md`（根目录）
- `leetcode_hot100.md`（根目录）

**使用场景**：
- 通常不需要单独运行，`build.py` 会自动调用
- 如果只需要合并 MD 而不生成 CSV，可以单独运行

### md_to_anki.py

**基本用法**：
```bash
python scripts/md_to_anki.py <markdown_file> [output_csv_file]
```

**参数**：
- `markdown_file`: 输入的Markdown文件路径（必需）
- `output_csv_file`: 输出的CSV文件路径（可选，默认为 `{markdown_file}_anki.csv`）

**功能说明**：
- 自动识别卡片格式：`#### 问题：` 或 `##### 问题：` + `**答案**：`（兼容旧格式 `**问题**：`）
- 根据标题层级自动生成标签（二级标题 + 三级标题）
- 支持Markdown语法转换：粗体、列表、代码块等
- 输出Anki格式的CSV（三列：问题、答案、标签）

**使用场景**：
- 通常不需要单独运行，`build.py` 会自动调用
- 如果只需要对某个 MD 文件生成 CSV，可以单独运行

### md_structure_analyzer.py

**基本用法**：
```bash
python scripts/md_structure_analyzer.py <markdown_file> [output_json_file]
```

**参数**：
- `markdown_file`: 输入的Markdown文件路径（必需，通常是合并后的根目录 MD）
- `output_json_file`: 输出的JSON文件路径（可选，默认为 `output/structure_{markdown_file}.json`）

**示例**：
```bash
# 先运行 build 确保根目录 MD 是最新的
python scripts/build.py

# 分析结构
python scripts/md_structure_analyzer.py work_general_knowledge.md
```

**输出格式**：
- JSON格式，包含完整的章节结构和卡片位置信息
- 可以在控制台查看结构摘要

**使用场景**：
- 查看合并后的完整 MD 的结构
- 定位章节和卡片位置（但编辑在 `content/` 下的章节文件中进行）

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
python tests/test_scripts.py leetcode_hot100.md leetcode_hot100_anki.csv
```

**输出**：
- 控制台输出测试结果
- 生成测试报告文件

**使用场景**：
- 验证 `build.py` 生成的 CSV 是否正确
- 对比新生成的 CSV 和现有的 CSV

## 标签生成规则

标签根据Markdown文件的标题层级自动生成：

- **二级标题** (`## 标题`)：标签为 `标题`
- **三级标题** (`### 标题`)：标签为 `二级标题-三级标题`
- **示例**：
  - `## 项目经历` → 标签：`项目经历`
  - `## 项目经历` + `### StatsD` → 标签：`项目经历-StatsD`

## 常见问题

### Q1: 如何检查知识点是否已存在？

A: 在添加新知识点之前，应该先检查：
1. 在 `content/` 下的章节文件中搜索关键词（推荐）：`grep -r "关键词" content/work/`
2. 搜索合并后的根目录 MD：`grep "关键词" work_general_knowledge.md`（需要先运行 `build.py`）
3. 使用 `md_structure_analyzer.py` 分析合并后的完整 MD 结构
4. 搜索CSV文件中的问题文本：`grep "关键词" work_general_knowledge_anki.csv`（需要先运行 `build.py`）

如果找到相关内容，给出位置信息（章节文件路径、章节路径等）；如果确认不存在，再添加新卡片。

### Q2: 如何确定新卡片应该添加在哪个章节文件？

A: 有两种方式：
1. **查看 manifest.yaml**：查看 `content/work/manifest.yaml` 或 `content/leetcode/manifest.yaml`，找到对应的章节文件名
2. **使用结构分析脚本**：先运行 `build.py`，然后运行 `md_structure_analyzer.py` 分析合并后的完整 MD，找到章节名称，对应到 `content/<domain>/<章节名>.md` 文件

### Q3: 卡片格式有什么要求？

A: 卡片格式：
- **不需要**卡片编号（如 `#### 卡片 N`），直接添加问题即可
- 问题格式：`#### 问题：` 或 `##### 问题：`（作为 Markdown 标题，便于生成文档大纲）
- 答案格式：`**答案**：` 或 `**答案**:`
- 卡片之间用 `---` 分隔
- 确保卡片在正确的章节文件中

### Q4: 编辑后如何更新根目录的 MD 和 CSV？

A: 运行 `python scripts/build.py` 即可：
- 会自动合并 `content/` 下的章节文件到根目录完整 MD
- 会自动生成 CSV 文件

### Q5: 标签生成不正确？

A: 标签根据合并后的完整 MD 的标题层级自动生成：
- 二级标题使用 `##` → 标签为 `标题`
- 三级标题使用 `###` → 标签为 `二级标题-三级标题`
- 确保章节文件中的标题层级正确

### Q6: 代码块没有正确转换？

A: 代码块格式应该是：
```markdown
```language
代码内容
```
```

确保代码块前后都有三个反引号，且语言标识符（可选）在同一行。

## 最佳实践

1. **添加新卡片前**：
   - **首先检查知识点是否已存在**：在 `content/` 下搜索或查看合并后的完整 MD
   - 如果已存在，给出位置信息（章节文件路径），不要重复添加
   - 如果不存在，查看 `manifest.yaml` 或运行 `md_structure_analyzer.py` 确定目标章节文件
   - 在对应的章节文件中添加新卡片

2. **添加新卡片后**：
   - 运行 `python scripts/build.py` 更新根目录 MD 和 CSV
   - （可选）运行 `test_scripts.py` 验证正确性
   - 检查生成的CSV文件是否符合预期

3. **维护章节文件**：
   - 保持标题层级一致（`##` 为章节标题）
   - 使用统一的格式（问题、答案格式）
   - 卡片之间用 `---` 分隔

4. **版本控制**：
   - 源真相在 `content/` 目录下，根目录的 MD 和 CSV 是构建产物
   - 建议将根目录的 MD 和 CSV 也提交到 git（方便阅读和导入Anki）
   - 提交前运行 `build.py` 确保构建产物是最新的

