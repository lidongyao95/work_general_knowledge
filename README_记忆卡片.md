# 记忆卡片使用说明

## 文件说明

本目录包含以下文件：

### 工作通识知识记忆卡片

1. **work_general_knowledge_generator.py** - 工作通识知识记忆卡片生成器程序
2. **work_general_knowledge_anki.csv** - Anki格式的CSV文件（可直接导入Anki）
3. **work_general_knowledge.md** - Markdown格式的卡片文件（方便阅读）

### LeetCode Hot 100记忆卡片

1. **leetcode_hot100_generator.py** - LeetCode Hot 100记忆卡片生成器程序
2. **leetcode_hot100_anki.csv** - Anki格式的CSV文件（可直接导入Anki）
3. **leetcode_hot100.md** - Markdown格式的卡片文件（方便阅读）

## 卡片统计

**总计：100张卡片**

### 分类统计

- **项目经历**（20张）：
  - 算法移植：5张
  - 功耗大数据：6张
  - StatsD：5张
  - 日志开关：2张
  - 性能看板：1张
  - 基线管理：1张

- **SoC功耗**（12张）：
  - CPU状态：3张
  - 调度策略：2张
  - DVFS、Idle状态、热管理、调频策略等

- **SoC架构**（9张）：
  - 缓存层级、DMA、缓存一致性、内存屏障、TLB、预取等

- **Trace工具**（10张）：
  - systrace、ftrace、perfetto的使用和分析方法

- **PMIC/Gauge**（9张）：
  - 电池管理、电源管理、库仑计、DVS等

- **Power Supply**（7张）：
  - Power Supply子系统、属性、驱动实现、uevent机制、充电状态、功耗监控等

- **守护进程/服务**（7张）：
  - 守护进程和服务区别、创建方法、Android Native层daemon、systemd服务、通信方式等

- **会话/进程组**（9张）：
  - 进程组、会话、前后台进程组、作业控制、信号传递、守护进程创建等

- **Kernel调度**（12张）：
  - CFS调度器、进程状态、负载均衡、NUMA、CPU亲和性等

- **面试问答**（5张）：
  - 通用问题、项目问题、业务理解等

## 使用方法

### 方法1：导入Anki（推荐）

1. 安装Anki：从 https://apps.ankiweb.net/ 下载并安装Anki
2. 打开Anki，点击"导入"按钮
3. 选择 `work_general_knowledge_anki.csv` 或 `leetcode_hot100_anki.csv` 文件
4. 设置导入选项：
   - 字段分隔符：逗号
   - 允许HTML：是
   - 标签列：第3列（Tags）
5. 点击"导入"，卡片将自动创建

### 方法2：使用Markdown文件

直接打开 `work_general_knowledge.md` 或 `leetcode_hot100.md` 文件，按分类阅读和复习。

### 方法3：自定义生成

如果需要修改或添加卡片，可以编辑对应的生成器文件，然后运行：

```bash
# 生成工作通识知识卡片
python3 work_general_knowledge_generator.py

# 生成LeetCode Hot 100卡片
python3 leetcode_hot100_generator.py
```

## 卡片内容说明

### 项目经历卡片

涵盖以下项目的关键技术和问题：
- DSP算法移植及优化
- 端侧功耗大数据daemon
- 性能功耗热端侧大数据（StatsD）
- Android日志开关动态下发
- 性能看板开发
- 基线自动管理

### 技术盲区补充卡片

针对高通SOC功耗组的业务需求，补充了以下技术盲区：

1. **SoC功耗优化**：
   - CPU低功耗模式（Active、WFI、Sleep、Deep Sleep）
   - DVFS（动态电压频率调节）
   - CPU Governor（调频策略）
   - 热节流（Thermal Throttling）
   - CPU Idle状态（C-state）

2. **SoC架构**：
   - ARM架构缓存层级（L1/L2/L3）
   - DMA和缓存一致性
   - 内存屏障
   - TLB（页表缓存）
   - 预取机制

3. **调度相关知识**：
   - Linux内核调度器（CFS）
   - 进程状态和优先级
   - 上下文切换
   - 负载均衡
   - CPU亲和性
   - NUMA架构

4. **Trace工具**：
   - systrace/htrace的使用
   - ftrace内核跟踪
   - perfetto分析和优化
   - 锁竞争分析
   - 启动时间分析

5. **PMIC/Gauge**：
   - Battery Gauge工作原理
   - PMIC电源管理
   - 库仑计
   - 电池健康度评估
   - DVS（动态电压调节）

6. **Power Supply子系统**：
   - Power Supply子系统架构
   - sysfs属性接口
   - 驱动实现方法
   - uevent事件机制
   - 充电状态管理
   - 功耗监控方法

7. **守护进程和服务**：
   - 守护进程和服务区别
   - 守护进程创建方法（双重fork）
   - Android Native层daemon
   - systemd服务管理
   - 进程间通信方式

8. **会话和进程组**：
   - 进程组（Process Group）概念
   - 会话（Session）概念
   - 前后台进程组
   - 作业控制（Job Control）
   - 信号传递机制
   - 守护进程创建原理

## 复习建议

1. **分类复习**：按分类逐个复习，确保每个技术点都掌握
2. **重点强化**：对于技术盲区（SoC功耗、架构、调度等），重点复习
3. **项目回顾**：结合项目经历，理解技术在实际项目中的应用
4. **定期复习**：使用Anki的间隔重复功能，定期复习已学内容

## 扩展建议

如果需要添加更多卡片，可以在对应的生成器文件中添加新的问题。建议：

1. 问题要细粒度，避免问题太大
2. 答案要简洁明了，便于记忆
3. 结合实际项目经验
4. 覆盖技术盲区

## 注意事项

- CSV文件使用UTF-8编码，确保Anki正确显示中文
- 如果导入Anki后格式有问题，可以检查字段分隔符设置
- Markdown文件可以直接在编辑器中打开，方便阅读和编辑

