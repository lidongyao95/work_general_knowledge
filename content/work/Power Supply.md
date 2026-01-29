## Power Supply

### Gauge/库仑计


**问题**：Gauge和库仑计的关系是什么？

**答案**：

电量计和库仑计，严格来说不是同一个东西，但它们经常被放在一起说，甚至有些场合会混用：
1. 电量计是一个更宽泛的概念，它指的是所有能用来测量电池电量的装置或芯片
2. 库仑计是电量计里面最常用、也最准确的一种，它的原理就是通过精确测量电池充放电时的电流，再乘以时间，也就是"库仑"这个电量单位的定义，来直接计算出到底用了多少电或者充进去多少电
3. 它们能获取的数据：主要就是电池的剩余电量、已用电量、充电速度、放电速度这些
4. 相比其他类型的电量计，比如通过测电池电压来估算电量的那种，库仑计因为是直接计量电流，所以结果会准确得多，尤其是在电池使用时间比较长、老化之后，电压法会越来越不准，库仑计的优势就更明显了

---

### PMIC/FuelGauge


**问题**：PMIC和Fuel Gauge在实际应用中的协作关系是什么？

**答案**：

PMIC和Fuel Gauge不是竞争关系，而是"协同工作"的关系：

充电阶段：
1. PMIC负责与充电器协商快充协议，实时采样电池电压/电流，调整充电档位
2. Fuel Gauge实时计算SoC，当SoC达到100%时，通知PMIC停止快充，切换到涓流充电
3. 若PMIC检测到过压/过流，立即切断充电，同时Fuel Gauge记录此次异常事件

放电阶段：
1. PMIC负责给系统供电，实时监测放电电流，防止过流/欠压
2. Fuel Gauge通过积分计算剩余容量，结合当前放电电流预测续航时间，显示在手机状态栏
3. 当电池电压低于3.0V时，PMIC触发关机，Fuel Gauge记录关机前的SoC和电压

---

### PMU/PMIC


**问题**：PMU和PMIC的区别是什么？

**答案**：

PMU（Power Management Unit，电源管理单元）：
- 定位：通常集成在SOC内部，主要负责管理SOC内部各模块的电源，比如CPU核心、GPU、DSP等
- 能获取的数据：主要是SOC内部的电源状态，比如各核心的电压、电流、工作频率，以及SOC整体的功耗统计

PMIC（Power Management IC，电源管理芯片）：
- 定位：独立于SOC的外部芯片，负责给整个设备供电，包括给SOC、电池充电、还有屏幕、摄像头等外设供电
- 能获取的数据：范围更广，比如电池的电压、电流、充电状态、以及各外设的供电情况

---

### uevent


**问题**：Power Supply子系统中的uevent机制是什么？Power Supply发送uevent，是不是userspace有监听才能收到，或者说注册回调？给出完整过程。

**答案**：

是的，Power Supply发送uevent后，userspace必须主动监听netlink socket才能收到事件。这不是注册回调机制，而是基于netlink socket的事件监听机制。完整过程包括内核发送uevent、通过netlink传输、userspace监听接收三个步骤。

1. 核心概念：
   - uevent机制：
     * uevent是Linux内核向用户空间发送设备事件通知的机制
     * 用于通知设备添加、删除、状态变化等事件
     * 原理：内核通过netlink socket向userspace发送事件消息，userspace需要主动监听才能接收
     * 比喻：就像内核通过广播发送消息，userspace需要调频到正确的频道才能听到
   - 与回调机制的区别：
     * 回调机制：内核主动调用userspace注册的回调函数（如中断处理、信号处理）
     * uevent机制：内核发送事件消息，userspace主动监听接收（如netlink socket、文件监听）
     * 原理：uevent是异步消息机制，不是同步回调机制，userspace需要主动监听
     * 比喻：就像回调是"电话通知"（主动呼叫），uevent是"广播消息"（需要调频收听）

2. 内核发送uevent的完整过程：
   - 步骤1：驱动调用power_supply_changed()：
     * 当Power Supply状态发生变化时（如开始充电、电量变化、温度变化等）
     * 驱动调用power_supply_changed(psy)函数通知Power Supply子系统
     * 原理：驱动检测到状态变化，主动通知Power Supply子系统
     * 比喻：就像传感器检测到变化，通知管理中心
   - 步骤2：Power Supply子系统处理：
     * power_supply_changed()函数内部调用kobject_uevent(&psy->dev.kobj, KOBJ_CHANGE)
     * 构建uevent消息，包含设备信息、属性变化等
     * 原理：Power Supply子系统将状态变化转换为uevent消息
     * 比喻：就像管理中心将变化转换为标准格式的消息
   - 步骤3：内核发送uevent到netlink：
     * kobject_uevent()函数通过netlink socket发送uevent消息
     * 使用NETLINK_KOBJECT_UEVENT协议族
     * 消息格式：ACTION=change\nPOWER_SUPPLY_NAME=battery\nPOWER_SUPPLY_STATUS=Charging\n...
     * 原理：内核通过netlink socket将uevent消息发送到userspace，所有监听的进程都能收到
     * 比喻：就像通过广播频道发送消息，所有调频到这个频道的接收器都能收到
   - 步骤4：内核不等待userspace响应：
     * 内核发送uevent后立即返回，不等待userspace处理
     * uevent是异步的、单向的（内核 -> userspace）
     * 原理：uevent是事件通知机制，不是请求-响应机制，内核不关心userspace是否收到
     * 比喻：就像广播消息，发送后不关心是否有人听到

3. uevent消息内容：
   - 基本字段：
     * ACTION：事件动作（add、remove、change等）
     * DEVPATH：设备路径（如/class/power_supply/battery）
     * SUBSYSTEM：子系统名称（power_supply）
     * SEQNUM：序列号（用于去重）
     * 原理：这些字段标识了事件的基本信息
     * 比喻：就像消息的标题和来源
   - Power Supply特定字段：
     * POWER_SUPPLY_NAME：电源设备名称（如battery）
     * POWER_SUPPLY_STATUS：电源状态（Charging、Discharging、Full等）
     * POWER_SUPPLY_CAPACITY：电量百分比
     * POWER_SUPPLY_VOLTAGE_NOW：当前电压
     * POWER_SUPPLY_CURRENT_NOW：当前电流
     * 原理：这些字段包含了Power Supply状态变化的详细信息
     * 比喻：就像消息的详细内容
   - 消息格式示例：
     * ACTION=change\nDEVPATH=/class/power_supply/battery\nSUBSYSTEM=power_supply\nPOWER_SUPPLY_NAME=battery\nPOWER_SUPPLY_STATUS=Charging\nPOWER_SUPPLY_CAPACITY=85\nSEQNUM=1234
     * 原理：uevent消息是文本格式，以换行符分隔的键值对
     * 比喻：就像标准格式的文本消息

4. userspace接收uevent的完整过程：
   - 方式1：通过udev监听（推荐）：
     * udev是Linux系统的设备管理守护进程，默认监听所有uevent
     * udev创建netlink socket，绑定到NETLINK_KOBJECT_UEVENT协议
     * 收到uevent后，udev根据规则处理（如更新设备节点、触发脚本等）
     * 原理：udev是系统级的uevent监听器，统一处理所有设备事件
     * 比喻：就像系统级的消息接收中心，统一处理所有消息
   - 方式2：应用程序直接监听netlink socket：
     * 应用程序创建netlink socket：socket(AF_NETLINK, SOCK_DGRAM, NETLINK_KOBJECT_UEVENT)
     * 绑定socket到NETLINK_KOBJECT_UEVENT协议
     * 使用select/poll/epoll监听socket，等待接收uevent消息
     * 收到消息后解析并处理
     * 原理：应用程序可以直接监听netlink socket，接收uevent消息
     * 比喻：就像应用程序自己调频到广播频道，直接接收消息
   - 方式3：通过libudev API监听（推荐用于应用程序）：
     * 使用libudev库提供的API，简化netlink socket操作
     * 创建udev_monitor，过滤特定子系统（如power_supply）
     * 使用poll/epoll监听monitor的文件描述符
     * 收到事件后通过API解析设备信息
     * 原理：libudev封装了netlink socket操作，提供更高级的API
     * 比喻：就像使用高级API简化消息接收和处理
   - 关键点：userspace必须主动监听：
     * 内核发送uevent后，不会主动通知userspace进程
     * userspace进程必须创建netlink socket并监听，才能收到uevent
     * 如果没有进程监听，uevent消息会丢失（内核不缓存）
     * 原理：uevent是基于netlink socket的异步消息机制，需要主动监听
     * 比喻：就像广播消息，如果不调频收听，就收不到消息

5. 完整数据流示例（Power Supply状态变化）：
   - 步骤1：硬件状态变化：
     * 电池开始充电，Gauge芯片检测到状态变化
     * 原理：硬件检测到状态变化，触发中断或轮询检测
     * 比喻：就像传感器检测到变化
   - 步骤2：驱动处理：
     * Gauge驱动读取状态寄存器，发现status从Discharging变为Charging
     * 驱动调用power_supply_changed(battery_psy)
     * 原理：驱动检测到状态变化，通知Power Supply子系统
     * 比喻：就像传感器通知管理中心
   - 步骤3：内核发送uevent：
     * power_supply_changed() -> kobject_uevent() -> netlink_send()
     * 通过NETLINK_KOBJECT_UEVENT socket发送uevent消息
     * 消息内容：ACTION=change\nPOWER_SUPPLY_STATUS=Charging\n...
     * 原理：内核通过netlink socket发送uevent消息
     * 比喻：就像通过广播频道发送消息
   - 步骤4：userspace监听接收：
     * udev或应用程序的netlink socket收到uevent消息
     * 解析消息，提取POWER_SUPPLY_STATUS=Charging等信息
     * 原理：监听的进程收到uevent消息并解析
     * 比喻：就像调频收听的接收器收到消息
   - 步骤5：userspace处理：
     * Android系统：BatteryService收到uevent，更新电池状态，通知应用
     * 应用程序：根据状态变化执行相应操作（如更新UI、调整功耗策略）
     * 原理：userspace根据uevent消息执行相应处理
     * 比喻：就像根据收到的消息执行相应操作

6. 是否需要注册回调：
   - 不是注册回调机制：
     * uevent不是回调机制，内核不会主动调用userspace的函数
     * userspace不需要向内核注册回调函数
     * 原理：uevent是异步消息机制，不是同步回调机制
     * 比喻：就像不是"电话通知"（回调），而是"广播消息"（监听）
   - 需要主动监听：
     * userspace必须创建netlink socket并监听
     * 使用select/poll/epoll等待socket可读事件
     * 收到消息后主动读取并处理
     * 原理：userspace需要主动监听netlink socket，才能收到uevent
     * 比喻：就像需要主动调频到广播频道，才能听到消息
   - 监听是持续的过程：
     * 监听不是一次性的，需要持续监听socket
     * 通常使用循环+select/poll/epoll实现持续监听
     * 原理：uevent是持续的事件流，需要持续监听
     * 比喻：就像需要持续调频收听，才能持续收到消息

7. Android系统中的实现：
   - BatteryService监听uevent：
     * Android的BatteryService通过netlink socket监听power_supply的uevent
     * 收到uevent后，更新BatteryManager的状态
     * 通过Binder通知应用层电池状态变化
     * 原理：Android系统通过BatteryService统一监听和处理电池uevent
     * 比喻：就像Android系统的电池管理服务统一监听电池消息
   - 应用层接收：
     * 应用通过BroadcastReceiver监听BatteryManager.ACTION_BATTERY_CHANGED广播
     * 或通过BatteryManager API直接查询电池状态
     * 原理：应用层通过Android框架接收电池状态变化，不直接监听uevent
     * 比喻：就像应用通过Android框架接收电池消息，不直接调频收听

8. 代码示例（userspace监听uevent）：
   - 使用libudev监听（推荐）：
     * struct udev *udev = udev_new();
     * struct udev_monitor *mon = udev_monitor_new_from_netlink(udev, "udev");
     * udev_monitor_filter_add_match_subsystem_devtype(mon, "power_supply", NULL);
     * udev_monitor_enable_receiving(mon);
     * int fd = udev_monitor_get_fd(mon);
     * poll(&fds, 1, -1); // 等待事件
     * struct udev_device *dev = udev_monitor_receive_device(mon);
     * 原理：使用libudev API简化netlink socket操作
     * 比喻：就像使用高级API简化消息接收
   - 直接使用netlink socket监听：
     * int sock = socket(AF_NETLINK, SOCK_DGRAM, NETLINK_KOBJECT_UEVENT);
     * bind(sock, ...);
     * recv(sock, buf, sizeof(buf), 0); // 接收uevent消息
     * 解析buf中的文本消息（键值对格式）
     * 原理：直接使用netlink socket接收uevent消息
     * 比喻：就像直接调频到广播频道接收消息

9. 总结：
   - 核心答案：
     * 是的，Power Supply发送uevent后，userspace必须主动监听netlink socket才能收到
     * 这不是注册回调机制，而是基于netlink socket的事件监听机制
     * 完整过程：内核发送uevent -> netlink传输 -> userspace监听接收
     * 原理：uevent是异步消息机制，需要userspace主动监听
   - 内核发送过程：
     * 驱动调用power_supply_changed() -> kobject_uevent() -> netlink_send()
     * 通过NETLINK_KOBJECT_UEVENT socket发送uevent消息
     * 内核不等待userspace响应，发送后立即返回
     * 原理：内核通过netlink socket异步发送uevent消息
   - userspace接收过程：
     * 创建netlink socket，绑定到NETLINK_KOBJECT_UEVENT协议
     * 使用select/poll/epoll监听socket，等待接收uevent消息
     * 收到消息后解析并处理（通过udev或自定义程序）
     * 原理：userspace需要主动监听netlink socket，才能收到uevent
   - 是否需要注册回调：
     * 不是注册回调机制，内核不会主动调用userspace函数
     * 需要主动监听netlink socket，持续等待接收uevent消息
     * 原理：uevent是异步消息机制，不是同步回调机制
   - 实际应用：
     * udev默认监听所有uevent，统一处理设备事件
     * 应用程序可以通过libudev API或直接监听netlink socket接收uevent
     * Android系统通过BatteryService监听电池uevent，通知应用层
     * 原理：不同系统和服务可以通过不同方式监听uevent
   - 原理：Power Supply发送uevent后，userspace必须主动监听netlink socket才能收到事件。这不是注册回调机制，而是基于netlink socket的事件监听机制。完整过程包括：内核通过power_supply_changed()触发uevent，通过netlink socket发送到userspace；userspace必须创建netlink socket并监听（通过udev、libudev API或直接监听），才能收到uevent消息；收到消息后解析并处理。如果没有进程监听，uevent消息会丢失，因为内核不缓存消息。这是异步消息机制，不是同步回调机制
   - 比喻：就像内核通过广播频道发送消息（uevent），userspace需要调频到正确的频道（监听netlink socket）才能听到消息。如果不调频收听，就收不到消息。这不是"电话通知"（回调），而是"广播消息"（监听），需要主动调频收听

---

### 充电状态


**问题**：Power Supply子系统中的充电状态有哪些？

**答案**：

充电状态（status属性）包括：
1. Unknown：未知状态
2. Charging：正在充电
3. Discharging：正在放电
4. Not charging：未充电（可能已满或温度异常）
5. Full：已充满
状态转换：
- 插入充电器：Discharging -> Charging
- 充满：Charging -> Full
- 拔出充电器：Charging/Full -> Discharging
- 温度异常：可能变为Not charging
应用：系统根据充电状态调整功耗策略，如充电时可以提高性能，放电时降低功耗。

---

### 功耗监控


**问题**：如何通过Power Supply子系统监控功耗？

**答案**：

1. 读取电流：
   - /sys/class/power_supply/battery/current_now（微安）
   - 正数表示充电，负数表示放电
2. 读取电压：
   - /sys/class/power_supply/battery/voltage_now（微伏）
3. 计算功耗：
   - 功耗 = 电流 × 电压
   - 单位：微安 × 微伏 = 皮瓦（需要转换为毫瓦）
4. 采样频率：
   - 根据需求设置采样周期（如1秒）
   - 避免过度频繁读取，增加系统开销
5. 数据聚合：
   - 按时间聚合（如每小时、每天）
   - 按事件聚合（如亮屏、应用启动等）
6. 误差控制：
   - 与功耗板对比，进行校准
   - 确保误差在可接受范围内

---

### 基础


**问题**：什么是Power Supply子系统？

**答案**：

Power Supply子系统是Linux内核中用于管理电源供应的框架。
功能：
1. 统一管理各种电源设备（电池、USB、AC适配器等）
2. 提供统一的用户空间接口（/sys/class/power_supply/）
3. 监控电源状态（电量、电压、电流、温度等）
4. 支持多种电源类型：Battery、USB、AC、Wireless等
5. 事件通知：电源状态变化时通知用户空间
应用：Android系统通过Power Supply子系统获取电池信息，进行功耗管理和优化。

---

### 属性


**问题**：Power Supply子系统中的属性有哪些？

**答案**：

Power Supply子系统通过sysfs暴露属性，常见属性包括：
1. 状态属性：
   - status：电源状态（Charging、Discharging、Full、Not charging等）
   - present：电源是否存在
   - type：电源类型（Battery、USB、AC等）
2. 电量属性：
   - capacity：电量百分比（0-100）
   - capacity_level：电量等级（Normal、High、Low、Critical等）
3. 电压电流：
   - voltage_now：当前电压（微伏）
   - current_now：当前电流（微安）
4. 温度：
   - temp：温度（0.1度为单位）
5. 健康状态：
   - health：电池健康状态（Good、Overheat、Dead等）

---

### 数据获取


**问题**：如何通过Power Supply子系统获取电池信息？

**答案**：

1. 读取sysfs节点：
   - /sys/class/power_supply/battery/capacity（电量）
   - /sys/class/power_supply/battery/voltage_now（电压）
   - /sys/class/power_supply/battery/current_now（电流）
2. 在Android Native层：
   - 直接读取节点文件
   - 或通过BatteryManager API获取
3. 在驱动层：
   - 实现power_supply驱动
   - 通过power_supply_register注册设备
   - 通过power_supply_changed通知状态变化
4. 事件监听：
   - 通过uevent机制监听电源状态变化
   - 用户空间可以监听这些事件

---

### 电池消耗


**问题**：如何获取电池消耗量？三种方式有什么区别？

**答案**：

获取一段时间内的电池消耗量，通常有几种方式：
1. 电池内部的电量计（Fuel Gauge/Gauge）：会直接累计电池的充放电电流，通过积分计算出电量消耗，这是最直接准确的方式。电量计是一个更宽泛的概念，它指的是所有能用来测量电池电量的装置或芯片。库仑计是电量计里面最常用、也最准确的一种，它的原理就是通过精确测量电池充放电时的电流，再乘以时间，来直接计算出到底用了多少电或者充进去多少电
2. PMIC监测：PMIC也会监测电池的输出电流和电压，通过这些数据可以估算电量消耗。但PMIC主要关注SOC内部的功耗，虽然可以通过SOC的功耗间接反映一部分电池消耗，但对于整个设备的电池消耗量，还是PMIC和电池电量计的数据更全面准确
3. PMU数据：PMU主要关注SOC本身的功耗，虽然可以通过SOC的功耗间接反映一部分电池消耗，但对于整个设备的电池消耗量，还是PMIC和电池电量计的数据更全面准确

总结：如果想获取整个设备的电池消耗量，优先从电池电量计或PMIC获取；如果只关注SOC本身的功耗，那PMU的数据就足够了。

---

### 通信模块


**问题**：Modem、WiFi、蓝牙、基带的关系是什么？

**答案**：

Modem（调制解调器）：
- 作用：主要负责通过蜂窝网络联网，也就是我们说的移动数据，像4G、5G这些，让手机能在没有Wi-Fi的地方上网、打电话、发短信
- 位置：现在很多中高端手机，Modem会和CPU、GPU这些核心一起，直接集成在SOC芯片里面。但有些低端手机或者早期的手机，为了降低成本，会采用分离式设计，这时候Modem就是一个独立的芯片

WiFi和蓝牙：
- WiFi作用：负责连接无线路由器，在有Wi-Fi覆盖的地方，比如家里、公司，用它上网速度更快，也更省电量，不会消耗手机的流量
- 蓝牙作用：主要用于短距离通信，比如连接耳机、手环、车载系统这些，传输的数据量比较小，距离也近，一般在10米左右
- 硬件集成：现在很多手机会把Wi-Fi和蓝牙集成到一个芯片里，叫Combo芯片，而Modem可能是集成在SOC里，也可能是独立的
- 协调工作：它们之间会通过SOC内部的总线或者外部接口来协调工作，比如手机会优先选择Wi-Fi上网，这时Modem就会进入低功耗状态，只负责接收电话和短信，这样能节省电量

基带（Baseband）：
- 作用：手机里的基带，你可以理解成"通信大脑"，它主要负责处理所有和蜂窝网络相关的通信任务。比如说，你用移动数据上网、打电话、发短信，这些信号的编码解码、调制解调，还有和基站之间的信号交互，都是基带在管
- 与Modem的关系：Modem是基带的核心硬件部分。基带是一个更完整的系统，它除了包含Modem芯片，还包括负责信号处理的数字信号处理器，以及存储通信协议和算法的固件。所以可以说，Modem是基带的"心脏"，而基带则是整个手机通信功能的基础
- 与WiFi/蓝牙的关系：基带主要负责的就是蜂窝网络，像4G、5G这些。蓝牙和WiFi有专门的芯片来处理，它们仨就像是三个不同的通信通道，各自管一块，这样分工明确，效率也更高

---

### 驱动实现


**问题**：Power Supply驱动如何实现？

**答案**：

1. 定义power_supply_desc结构：
   - name：设备名称
   - type：电源类型（POWER_SUPPLY_TYPE_BATTERY等）
   - properties：属性数组
   - get_property：获取属性值的回调函数
2. 注册设备：
   - power_supply_register()注册设备
   - 返回power_supply指针
3. 更新属性：
   - power_supply_changed()通知属性变化
   - 触发uevent，通知用户空间
4. 注销设备：
   - power_supply_unregister()注销设备
5. 属性实现：
   - 在get_property回调中实现属性读取
   - 从硬件寄存器或Gauge芯片读取数据

---


