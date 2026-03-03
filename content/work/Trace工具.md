## Trace工具

### ftrace


#### 问题：ftrace可以跟踪哪些内核事件？

**答案**：

1. 函数调用：跟踪内核函数的调用和返回
2. 中断处理：跟踪中断处理流程
3. 调度事件：跟踪进程调度、上下文切换等
4. 定时器事件：跟踪定时器相关事件
5. 内存事件：跟踪内存分配、释放等事件
6. 配合perfetto使用：提供更详细的系统行为分析

---


#### 问题：为什么ftrace使用前一定要挂载debugfs（或tracefs）？

**答案**：

ftrace是Linux内核的跟踪框架，需要通过文件系统接口提供控制和输出功能。ftrace实际上使用tracefs文件系统，但为了向后兼容，也可以挂载在debugfs下。如果不挂载相应的文件系统，就无法访问ftrace的控制和输出文件，无法使用ftrace功能。

1. ftrace的文件系统依赖：
   - tracefs文件系统：
     * ftrace实际上使用tracefs文件系统（不是debugfs）
     * tracefs是专门为内核跟踪设计的文件系统
     * 原理：ftrace通过tracefs文件系统提供用户空间接口
     * 比喻：就像ftrace需要一个"控制面板"（tracefs）来操作
   - debugfs的向后兼容：
     * 为了向后兼容，tracefs可以挂载在debugfs下
     * 路径：/sys/kernel/debug/tracing（旧路径）
     * 原理：旧工具可能使用debugfs路径，为了兼容性支持
     * 比喻：就像为了兼容旧工具，也支持在debugfs下挂载
   - 推荐路径：
     * 现在推荐直接挂载tracefs到/sys/kernel/tracing
     * 路径：/sys/kernel/tracing（新路径，推荐）
     * 原理：直接挂载tracefs更清晰，不依赖debugfs
     * 比喻：就像直接使用"专用控制面板"，更清晰

2. 为什么必须挂载：
   - 文件系统接口：
     * ftrace的所有功能都通过文件系统接口提供
     * 控制和配置文件：current_tracer、available_tracers、tracing_on等
     * 输出文件：trace、trace_pipe等
     * 原理：ftrace通过文件系统接口提供用户空间访问，必须挂载文件系统才能访问
     * 比喻：就像必须挂载"控制面板"才能操作ftrace
   - 无法访问文件：
     * 如果不挂载tracefs（或debugfs），无法访问ftrace的控制和输出文件
     * 无法配置跟踪器、启用/禁用跟踪、读取跟踪数据等
     * 原理：文件系统未挂载，文件路径不存在，无法访问
     * 比喻：就像"控制面板"未安装，无法操作
   - 功能依赖：
     * ftrace的所有功能都依赖文件系统接口
     * 包括：选择跟踪器、设置过滤条件、读取跟踪数据等
     * 原理：ftrace的设计就是通过文件系统接口提供功能，必须挂载
     * 比喻：就像所有功能都通过"控制面板"提供，必须安装

3. 挂载方式：
   - 挂载tracefs（推荐）：
     * mount -t tracefs nodev /sys/kernel/tracing
     * 直接挂载tracefs到/sys/kernel/tracing
     * 原理：直接挂载tracefs，不依赖debugfs，更清晰
     * 比喻：就像直接安装"专用控制面板"
   - 挂载debugfs（向后兼容）：
     * mount -t debugfs nodev /sys/kernel/debug
     * tracefs会自动挂载在/sys/kernel/debug/tracing下
     * 注意：这种方式已废弃，计划在2027年1月移除
     * 原理：为了向后兼容，支持在debugfs下挂载tracefs
     * 比喻：就像为了兼容旧工具，也支持在"通用面板"下安装
   - 自动挂载：
     * 某些系统可能自动挂载tracefs
     * 可以通过/etc/fstab配置自动挂载
     * 原理：系统可以配置自动挂载，方便使用
     * 比喻：就像系统可以自动安装"控制面板"

4. 文件系统的作用：
   - 控制文件：
     * current_tracer：当前使用的跟踪器
     * available_tracers：可用的跟踪器列表
     * tracing_on：启用/禁用跟踪（1启用，0禁用）
     * set_ftrace_filter：设置函数过滤
     * 原理：通过控制文件配置ftrace的行为
     * 比喻：就像通过"控制面板"的按钮配置功能
   - 输出文件：
     * trace：跟踪数据输出（读取后清空）
     * trace_pipe：跟踪数据流（持续输出，不清空）
     * trace_marker：用户空间标记点
     * 原理：通过输出文件获取跟踪数据
     * 比喻：就像通过"控制面板"的显示器查看数据
   - 统计文件：
     * tracing_stats：跟踪统计信息
     * buffer_size_kb：缓冲区大小
     * 原理：通过统计文件了解跟踪状态
     * 比喻：就像通过"控制面板"的统计信息了解状态

5. 不挂载的后果：
   - 无法使用ftrace：
     * 无法访问/sys/kernel/tracing目录
     * 无法配置跟踪器、启用跟踪、读取数据等
     * 原理：文件系统未挂载，文件路径不存在，无法使用
     * 比喻：就像"控制面板"未安装，无法操作
   - 工具失败：
     * 使用ftrace的工具（如perf、perfetto）可能失败
     * 错误信息："No such file or directory"或"mount point does not exist"
     * 原理：工具尝试访问ftrace文件，但文件系统未挂载，访问失败
     * 比喻：就像工具尝试使用"控制面板"，但未安装，操作失败

6. 实际使用示例：
   - 检查是否挂载：
     * ls /sys/kernel/tracing：检查tracefs是否挂载
     * mount | grep tracefs：检查tracefs挂载情况
     * 原理：通过检查文件路径或挂载信息确认是否挂载
     * 比喻：就像检查"控制面板"是否安装
   - 手动挂载：
     * mount -t tracefs nodev /sys/kernel/tracing
     * 挂载后可以使用ftrace功能
     * 原理：手动挂载tracefs，然后可以使用ftrace
     * 比喻：就像手动安装"控制面板"，然后可以使用
   - 使用ftrace：
     * echo function > /sys/kernel/tracing/current_tracer：选择跟踪器
     * echo 1 > /sys/kernel/tracing/tracing_on：启用跟踪
     * cat /sys/kernel/tracing/trace：读取跟踪数据
     * 原理：挂载后可以通过文件系统接口使用ftrace
     * 比喻：就像安装后可以通过"控制面板"操作

7. tracefs vs debugfs：
   - tracefs：
     * 专门为内核跟踪设计的文件系统
     * 更清晰、更专业
     * 推荐使用
     * 原理：tracefs是专门为跟踪设计的，更合适
     * 比喻：就像"专用控制面板"，更专业
   - debugfs：
     * 通用的调试文件系统
     * 可以挂载多种调试工具
     * 向后兼容，但已废弃
     * 原理：debugfs是通用的，tracefs可以挂载在其下，但已废弃
     * 比喻：就像"通用控制面板"，可以安装多种工具，但已过时
   - 迁移建议：
     * 从debugfs路径迁移到tracefs路径
     * 旧路径：/sys/kernel/debug/tracing
     * 新路径：/sys/kernel/tracing
     * 原理：tracefs路径更清晰，推荐使用
     * 比喻：就像从"通用面板"迁移到"专用面板"

8. 系统配置：
   - 自动挂载：
     * 某些Linux发行版可能自动挂载tracefs
     * 可以通过/etc/fstab配置自动挂载
     * 原理：系统可以配置自动挂载，方便使用
     * 比喻：就像系统可以自动安装"控制面板"
   - 权限要求：
     * 挂载tracefs通常需要root权限
     * 某些系统可能允许非root用户访问已挂载的tracefs
     * 原理：挂载需要root权限，但访问可能不需要
     * 比喻：就像安装需要管理员权限，但使用可能不需要

9. 总结：
   - ftrace的文件系统依赖：
     * ftrace使用tracefs文件系统（不是debugfs）
     * 为了向后兼容，也可以挂载在debugfs下（已废弃）
     * 推荐路径：/sys/kernel/tracing
   - 为什么必须挂载：
     * ftrace的所有功能都通过文件系统接口提供
     * 如果不挂载，无法访问控制和输出文件
     * 无法使用ftrace功能
   - 挂载方式：
     * mount -t tracefs nodev /sys/kernel/tracing（推荐）
     * mount -t debugfs nodev /sys/kernel/debug（向后兼容，已废弃）
   - 文件系统作用：
     * 控制文件：配置跟踪器、启用/禁用跟踪等
     * 输出文件：读取跟踪数据
     * 统计文件：了解跟踪状态
   - 不挂载的后果：
     * 无法使用ftrace功能
     * 使用ftrace的工具可能失败
   - 原理：ftrace通过tracefs文件系统提供用户空间接口，所有功能都依赖文件系统接口。如果不挂载tracefs（或debugfs），无法访问ftrace的控制和输出文件，无法使用ftrace功能。tracefs是专门为内核跟踪设计的文件系统，推荐直接挂载到/sys/kernel/tracing，而不是通过debugfs挂载（已废弃）
   - 比喻：就像ftrace需要一个"控制面板"（tracefs）来操作，如果不安装"控制面板"，就无法使用ftrace功能。推荐使用"专用控制面板"（tracefs），而不是"通用控制面板"（debugfs）

---

### perf


#### 问题：perf工具的来源和用法是什么？

**答案**：

perf是Linux内核提供的性能分析工具，是Linux性能分析的标准工具。

1. 来源和历史：
   - 来源：perf工具来源于Linux内核的Performance Events子系统（也称为perf_events）
   - 历史：
     * 2009年引入Linux内核（Linux 2.6.31）
     * 由Ingo Molnar等人开发，整合了之前的多个性能分析工具（oprofile、ptrace等）
     * 成为Linux内核官方推荐的性能分析工具
   - 原理：perf基于Linux内核的Performance Events子系统，使用硬件性能计数器（PMC）和软件事件进行性能分析
   - 比喻：就像Linux内核提供的"性能分析工具箱"，统一了之前的多个工具

2. 核心功能：
   - 性能分析：分析CPU、内存、I/O等系统性能
   - 事件统计：统计硬件和软件事件（如CPU周期、缓存未命中、缺页异常等）
   - 函数级分析：分析函数调用关系和耗时
   - 调用栈分析：分析函数调用栈，定位性能瓶颈
   - 原理：perf使用硬件性能计数器和软件事件，提供全面的性能分析能力
   - 比喻：就像性能分析的多功能工具，可以分析多个方面

3. 主要命令和用法：
   - perf stat：统计性能事件
     * 用法：perf stat [选项] <命令>
     * 示例：
       - perf stat ./program：统计程序运行时的性能事件
       * perf stat -e page-faults ./program：统计缺页异常
       * perf stat -e cache-misses ./program：统计缓存未命中
     * 功能：统计指定事件的发生次数和比率
     * 原理：使用硬件性能计数器统计事件，提供性能统计信息
     * 比喻：就像性能计数器，统计各种事件
   - perf record：记录性能事件
     * 用法：perf record [选项] <命令>
     * 示例：
       * perf record ./program：记录程序运行时的性能事件
       * perf record -g ./program：记录调用栈信息（-g表示记录调用图）
       * perf record -e page-faults ./program：记录缺页异常事件
     * 功能：记录性能事件到perf.data文件，可以后续分析
     * 原理：记录性能事件到文件，支持事后分析
     * 比喻：就像录制性能数据，可以回放分析
   - perf report：查看性能报告
     * 用法：perf report [选项]
     * 示例：
       * perf report：查看perf.data文件的性能报告
       * perf report -g：查看调用图
       * perf report --stdio：以文本格式输出报告
     * 功能：分析perf record记录的数据，生成性能报告
     * 原理：分析记录的性能数据，生成可读的性能报告
     * 比喻：就像分析录制的数据，生成报告
   - perf top：实时性能分析
     * 用法：perf top [选项]
     * 示例：
       * perf top：实时显示占用CPU最多的函数
       * perf top -e page-faults：实时显示缺页异常最多的函数
     * 功能：实时显示系统性能热点，类似top命令
     * 原理：实时采样性能事件，显示热点函数
     * 比喻：就像实时性能监控器，显示热点
   - perf list：列出可用事件
     * 用法：perf list [选项]
     * 示例：
       * perf list：列出所有可用事件
       * perf list | grep cache：列出缓存相关事件
     * 功能：列出系统支持的性能事件
     * 原理：查询系统支持的事件，帮助用户选择合适的事件
     * 比喻：就像列出可用的性能指标

4. 常用事件类型：
   - CPU事件：
     * cpu-cycles：CPU周期数
     * instructions：指令数
     * branches：分支指令数
     * branch-misses：分支预测失败数
     * 原理：CPU硬件性能计数器提供的事件
     * 比喻：就像CPU的性能指标
   - 缓存事件：
     * cache-references：缓存引用
     * cache-misses：缓存未命中
     * L1-dcache-loads：L1数据缓存加载
     * L1-dcache-load-misses：L1数据缓存加载未命中
     * 原理：缓存硬件性能计数器提供的事件
     * 比喻：就像缓存的性能指标
   - 内存事件：
     * page-faults：缺页异常
     * minor-faults：轻微缺页异常
     * major-faults：严重缺页异常
     * 原理：内存管理相关的事件
     * 比喻：就像内存的性能指标
   - TLB事件：
     * dTLB-loads：数据TLB加载
     * dTLB-load-misses：数据TLB加载未命中
     * iTLB-loads：指令TLB加载
     * iTLB-load-misses：指令TLB加载未命中
     * 原理：TLB硬件性能计数器提供的事件
     * 比喻：就像TLB的性能指标
   - 软件事件：
     * cpu-clock：CPU时钟
     * task-clock：任务时钟
     * context-switches：上下文切换
     * page-faults：缺页异常（软件事件）
     * 原理：内核软件提供的事件，不依赖硬件计数器
     * 比喻：就像软件层面的性能指标

5. 实际应用示例：
   - 分析CPU性能：
     * perf stat -e cpu-cycles,instructions,cycles ./program：统计CPU周期和指令数
     * perf record -g ./program：记录调用栈
     * perf report：查看性能报告，找出热点函数
     * 原理：通过统计和记录分析CPU性能
     * 比喻：就像分析CPU的工作效率
   - 分析缓存性能：
     * perf stat -e cache-references,cache-misses ./program：统计缓存引用和未命中
     * perf record -e cache-misses -g ./program：记录缓存未命中的调用栈
     * perf report：查看缓存未命中的热点
     * 原理：通过统计缓存事件分析缓存性能
     * 比喻：就像分析缓存的效率
   - 分析内存性能：
     * perf stat -e page-faults ./program：统计缺页异常
     * perf record -e page-faults -g ./program：记录缺页异常的调用栈
     * perf report：查看缺页异常的热点
     * 原理：通过统计内存事件分析内存性能
     * 比喻：就像分析内存的使用效率
   - 分析TLB性能：
     * perf stat -e dTLB-loads,dTLB-load-misses ./program：统计TLB加载和未命中
     * perf record -e dTLB-load-misses -g ./program：记录TLB未命中的调用栈
     * perf report：查看TLB未命中的热点
     * 原理：通过统计TLB事件分析TLB性能
     * 比喻：就像分析TLB的效率

6. 高级用法：
   - 调用栈分析：
     * perf record -g ./program：记录调用栈（-g表示记录调用图）
     * perf report -g：查看调用栈报告
     * 原理：记录函数调用关系，帮助定位性能瓶颈
     * 比喻：就像记录函数调用链，找出瓶颈
   - 多进程分析：
     * perf record -a ./program：记录所有进程的事件（-a表示all）
     * perf report：查看所有进程的性能报告
     * 原理：可以分析多个进程的性能
     * 比喻：就像分析多个进程的性能
   - 时间范围分析：
     * perf record -e page-faults -- sleep 10：记录10秒内的缺页异常
     * perf report：查看指定时间范围的性能报告
     * 原理：可以分析指定时间范围的性能
     * 比喻：就像分析指定时间段的性能
   - 事件过滤：
     * perf record -e page-faults --filter "addr > 0x1000" ./program：过滤特定地址的缺页异常
     * 原理：可以过滤特定条件的事件
     * 比喻：就像过滤特定条件的事件

7. 与其他工具的区别：
   - perf vs gprof：
     * perf：不需要重新编译程序，使用硬件计数器，开销小
     * gprof：需要重新编译程序，使用代码插桩，开销大
     * 原理：perf使用硬件计数器，不需要修改程序
     * 比喻：就像perf是"非侵入式"工具，gprof是"侵入式"工具
   - perf vs oprofile：
     * perf：Linux内核官方工具，功能更强大，维护更好
     * oprofile：旧工具，已被perf取代
     * 原理：perf整合了oprofile的功能，并提供了更多功能
     * 比喻：就像perf是"升级版"工具
   - perf vs strace：
     * perf：分析性能事件，定位性能瓶颈
     * strace：跟踪系统调用，分析程序行为
     * 原理：perf关注性能，strace关注行为
     * 比喻：就像perf关注"效率"，strace关注"行为"

8. 权限要求：
   - 普通用户：可以分析自己的进程，但功能受限
   - root用户：可以分析所有进程和系统事件，功能完整
   - 原理：perf需要访问硬件性能计数器和内核数据，需要相应权限
   - 比喻：就像需要权限才能访问性能数据

9. 输出格式：
   - 文本格式：perf report --stdio：以文本格式输出
   - 交互式格式：perf report：以交互式格式输出（默认）
   - 原理：支持多种输出格式，方便不同场景使用
   - 比喻：就像支持多种显示方式

10. 性能开销：
    - 开销：perf使用硬件性能计数器，开销很小（通常<1%）
    - 原理：硬件性能计数器是CPU内置的，不需要额外开销
    - 比喻：就像使用内置的性能计数器，开销很小

总结：
- 来源：perf来源于Linux内核的Performance Events子系统，2009年引入
- 功能：性能分析、事件统计、函数级分析、调用栈分析
- 主要命令：perf stat（统计）、perf record（记录）、perf report（报告）、perf top（实时）
- 事件类型：CPU事件、缓存事件、内存事件、TLB事件、软件事件
- 优势：不需要重新编译程序，使用硬件计数器，开销小
- 原理：perf基于Linux内核的Performance Events子系统，使用硬件性能计数器和软件事件进行性能分析
- 比喻：就像Linux内核提供的"性能分析工具箱"，功能强大且易用

---

### perfetto


#### 问题：如何通过perfetto UI分析trace文件？

**答案**：

1. CPU调度情况：查看进程和线程的执行时间线
2. 进程和线程：查看执行时间线、调度延迟等
3. 系统调用和函数调用栈：定位性能瓶颈
4. 锁竞争和等待时间：识别锁竞争问题
5. 内存分配和释放：分析内存使用情况
6. 结合系统日志：定位CPU高负载的根本原因

---


#### 问题：perfetto trace文件包含哪些数据源？

**答案**：

1. ftrace：内核函数调用、调度事件、中断等
2. atrace：Android系统层trace（如SurfaceFlinger、ActivityManager）
3. systrace：系统调用、进程调度等
4. heap profiler：内存分配和释放
5. CPU profiler：CPU使用情况
6. GPU profiler：GPU使用情况
7. 自定义trace：应用层添加的trace点
8. 日志：系统日志和应用日志

---

### systrace


#### 问题：htrace/systrace可以显示哪些内容？

**答案**：

1. CPU调度信息：进程和线程的调度时间线、CPU频率变化、CPU负载分布
2. 系统调用：系统调用的时间点和耗时、系统调用的调用栈
3. 中断和异常：中断发生的时间点、中断处理耗时
4. 锁竞争：锁的获取和释放、锁等待时间
5. 内存操作：内存分配和释放、内存拷贝操作
6. I/O操作：文件读写操作、网络I/O操作

---

### 分析


#### 问题：trace分析中如何定位CPU高负载问题？

**答案**：

1. 查看CPU调度情况：识别占用CPU时间过长的进程或线程
2. 分析系统调用：查找频繁的系统调用或中断
3. 检查锁竞争：识别锁竞争导致的线程阻塞
4. 分析内存分配：查找内存分配导致的GC压力
5. 结合系统日志：综合分析定位根本原因

---

### 启动分析


#### 问题：如何通过trace分析应用启动时间？

**答案**：

1. 查看Activity启动流程：从启动Intent到Activity显示
2. 分析关键阶段：
   - 进程创建时间
   - Application.onCreate()时间
   - Activity.onCreate()时间
   - 布局inflate时间
   - 首帧渲染时间
3. 识别瓶颈：找出耗时最长的阶段
4. 优化建议：
   - 延迟初始化
   - 异步加载
   - 优化布局
   - 减少主线程阻塞

---

### 抓取


#### 问题：如何触发抓取trace？

**答案**：

1. 通过OLC修改系统property
2. 系统中管理perfetto的服务先拼接config
3. 使用system函数执行perfetto的cmd命令
4. 可以配置预警条件，在特定条件下自动触发（如CPU负载达到97%时）
5. 借助StatsD的预警和OLC接口，实现任意条件触发

---

### 指标


#### 问题：systrace中的关键指标有哪些？

**答案**：

1. Frame时间：每帧渲染时间，应该小于16.67ms（60fps）
2. VSync：垂直同步信号，用于同步渲染
3. SurfaceFlinger：合成和显示帧的时间
4. CPU频率：CPU频率变化情况
5. 调度延迟：进程等待调度的时间
6. 中断频率：中断发生的频率
7. 锁等待时间：等待锁的时间
8. I/O等待时间：等待I/O操作的时间

---

### 解析


#### 问题：解析Trace的看板通常是如何工作的？

**答案**：

1. 离线解析：先抓取trace文件，然后通过脚本解析
2. 数据入库：将解析后的数据存储到数据库中
3. 看板展示：通过Web界面展示分析结果
4. 不是实时显示：需要先抓取trace文件，然后解析和展示
5. 常见入门项目：帮助新员工熟悉trace文件格式、性能分析方法和数据处理

---

### 锁分析


#### 问题：如何在perfetto中分析锁竞争问题？

**答案**：

1. 查看锁事件：识别锁的获取和释放时间点
2. 分析等待时间：查看线程等待锁的时间
3. 识别热点锁：找出被频繁竞争的锁
4. 查看调用栈：分析为什么需要这个锁
5. 优化建议：
   - 减少锁的持有时间
   - 使用更细粒度的锁
   - 使用无锁数据结构
   - 避免锁嵌套

---


