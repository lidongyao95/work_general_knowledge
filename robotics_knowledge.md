# 机器人学相关知识记忆卡片

共 5 张卡片

---

## 机器人学

### 机器人学导论


#### 问题：坐标平移和旋转是如何使用矩阵表示的？

**答案**：

1. 平移变换（Translation）：
   - 原理：在齐次坐标系中，平移可以通过4×4矩阵表示。对于3D空间中的平移向量$(t_x, t_y, t_z)$，平移矩阵$T$为：
     $$
     T = \begin{bmatrix}
     1 & 0 & 0 & t_x \\
     0 & 1 & 0 & t_y \\
     0 & 0 & 1 & t_z \\
     0 & 0 & 0 & 1
     \end{bmatrix}
     $$
   - 比喻：就像给坐标系的每个点都加上一个固定的偏移量，就像把整个坐标系"搬"到一个新位置

2. 旋转变换（Rotation）：
   - 原理：旋转可以通过旋转矩阵表示。对于绕Z轴旋转角度$\theta$，旋转矩阵$R_z$为：
     $$
     R_z(\theta) = \begin{bmatrix}
     \cos(\theta) & -\sin(\theta) & 0 & 0 \\
     \sin(\theta) &  \cos(\theta) & 0 & 0 \\
     0 & 0 & 1 & 0 \\
     0 & 0 & 0 & 1
     \end{bmatrix}
     $$
   - 绕X轴和Y轴的旋转矩阵类似，只是旋转轴不同
   - 比喻：就像把坐标系绕某个轴"转"一个角度，就像旋转一个立方体

3. 齐次变换矩阵（Homogeneous Transformation Matrix）：
   - 原理：将平移和旋转组合在一起，使用4×4齐次变换矩阵$H$表示：
     $$
     H = \begin{bmatrix}
     R & \mathbf{t} \\
     \mathbf{0} & 1
     \end{bmatrix} = \begin{bmatrix}
     \text{旋转矩阵} & \text{平移向量} \\
     \mathbf{0} & 1
     \end{bmatrix}
     $$
   - 其中$R$是3×3旋转矩阵，$\mathbf{t}$是3×1平移向量
   - 比喻：就像先旋转坐标系，再平移，两步操作合并成一个矩阵，就像"先转后移"的组合动作

4. 变换的复合：
   - 原理：多个变换可以通过矩阵乘法组合。如果先应用变换$H_1$，再应用$H_2$，则组合变换为$H_2 \times H_1$（注意顺序）
   - 比喻：就像连续执行多个动作，每个动作对应一个矩阵，最终效果是所有矩阵相乘的结果

5. 应用：
   - 机器人学中用于描述机器人各关节之间的坐标变换
   - 用于描述机器人末端执行器相对于基座的位姿
   - 用于坐标系的转换和变换链的计算

---

#### 问题：坐标系间映射的平移、旋转和一般变换算子是什么？它们如何工作？

**答案**：

1. 坐标系映射的基本概念：
   - 原理：坐标系映射描述如何将一个坐标系中的点或向量转换到另一个坐标系中。给定两个坐标系A和B，需要找到从A到B的变换算子
   - 比喻：就像在地图上标注位置，需要知道如何从一个坐标系（如经纬度）转换到另一个坐标系（如像素坐标）

2. 平移算子（Translation Operator）：
   - 原理：只改变坐标系的原点位置，不改变坐标轴方向。平移算子$T(\mathbf{t})$将点$p$映射为$p' = p + \mathbf{t}$，其中$\mathbf{t}$是平移向量
   - 矩阵表示：在齐次坐标中，平移矩阵为：
     $$
     T(\mathbf{t}) = \begin{bmatrix}
     I & \mathbf{t} \\
     \mathbf{0} & 1
     \end{bmatrix} = \begin{bmatrix}
     1 & 0 & 0 & t_x \\
     0 & 1 & 0 & t_y \\
     0 & 0 & 1 & t_z \\
     0 & 0 & 0 & 1
     \end{bmatrix}
     $$
   - 比喻：就像把整个坐标系"搬"到一个新位置，但方向不变

3. 旋转算子（Rotation Operator）：
   - 原理：只改变坐标轴的方向，不改变原点位置。旋转算子$R(\theta)$将点$p$映射为$p' = R \times p$，其中$R$是旋转矩阵
   - 矩阵表示：旋转矩阵$R$是正交矩阵（$R^T \times R = I$），行列式为1。绕Z轴旋转$\theta$角的旋转矩阵为：
     $$
     R_z(\theta) = \begin{bmatrix}
     \cos(\theta) & -\sin(\theta) & 0 \\
     \sin(\theta) &  \cos(\theta) & 0 \\
     0 & 0 & 1
     \end{bmatrix}
     $$
   - 性质：旋转矩阵的逆等于其转置（$R^{-1} = R^T$），表示反向旋转
   - 比喻：就像把坐标系绕某个轴"转"一个角度，原点位置不变

4. 一般变换算子（General Transformation Operator）：
   - 原理：同时包含旋转和平移的复合变换。一般变换算子$T(R, \mathbf{t})$将点$p$映射为$p' = R \times p + \mathbf{t}$
   - 齐次变换矩阵：使用4×4齐次变换矩阵$H$表示：
     $$
     H = \begin{bmatrix}
     R & \mathbf{t} \\
     \mathbf{0} & 1
     \end{bmatrix} = \begin{bmatrix}
     r_{11} & r_{12} & r_{13} & t_x \\
     r_{21} & r_{22} & r_{23} & t_y \\
     r_{31} & r_{32} & r_{33} & t_z \\
     0 & 0 & 0 & 1
     \end{bmatrix}
     $$
   - 其中$R$是3×3旋转矩阵，$\mathbf{t}$是3×1平移向量
   - 比喻：就像先旋转坐标系，再平移，两步操作合并成一个矩阵

5. 变换算子的性质：
   - 复合性：多个变换可以通过矩阵乘法组合，$T_{AB} = T_{BC} \times T_{AC}$（注意顺序）
   - 可逆性：变换的逆变换存在，$H^{-1} = \begin{bmatrix} R^T & -R^T \times \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}$
   - 传递性：如果知道$T_{AB}$和$T_{BC}$，可以计算$T_{AC} = T_{BC} \times T_{AB}$
   - 比喻：就像链条，知道每两个相邻环节的关系，就能推导出任意两个环节的关系

6. 坐标系变换链：
   - 原理：在机器人学中，通常需要计算从基座坐标系到末端执行器坐标系的变换，这需要经过多个关节坐标系
   - 计算方法：$T_{\text{base}}^{\text{end}} = T_{\text{base}}^{\text{joint1}} \times T_{\text{joint1}}^{\text{joint2}} \times \cdots \times T_{\text{jointN}}^{\text{end}}$
   - 比喻：就像接力赛，每个关节负责一段变换，最终得到完整的变换链

7. 应用场景：
   - 机器人正运动学：计算末端执行器相对于基座的位姿
   - 机器人逆运动学：根据末端位姿反推各关节角度
   - 传感器数据融合：将不同传感器坐标系的数据转换到统一坐标系
   - 路径规划：在统一坐标系中规划机器人运动轨迹

---

#### 问题：复合变换和逆变换是什么？它们如何计算和应用？

**答案**：

1. 复合变换（Composition of Transformations）：
   - 原理：复合变换是将多个变换按顺序组合成一个变换。如果先应用变换$H_1$，再应用变换$H_2$，则复合变换为$H_{\text{composite}} = H_2 \times H_1$（注意矩阵乘法的顺序是从右到左）
   - 数学表示：对于点$p$，先应用$H_1$得到$p_1 = H_1 \times p$，再应用$H_2$得到$p_2 = H_2 \times p_1 = H_2 \times H_1 \times p$
   - 比喻：就像先穿衣服再穿外套，顺序很重要，不能颠倒

2. 复合变换的顺序重要性：
   - 原理：矩阵乘法不满足交换律，即$H_2 \times H_1 \neq H_1 \times H_2$。变换的顺序决定了最终结果
   - 示例：先平移后旋转与先旋转后平移的结果不同
     - 先平移$(t_x, t_y)$再旋转$\theta$：$R(\theta) \times T(t_x, t_y)$
     - 先旋转$\theta$再平移$(t_x, t_y)$：$T(t_x, t_y) \times R(\theta)$
   - 比喻：就像先向左走再向前走，与先向前走再向左走，到达的位置不同

3. 坐标系变换链的复合：
   - 原理：在机器人学中，从坐标系A到坐标系C的变换可以通过中间坐标系B计算：$T_{AC} = T_{BC} \times T_{AB}$
   - 计算方法：如果知道$T_{AB}$（从A到B）和$T_{BC}$（从B到C），则$T_{AC} = T_{BC} \times T_{AB}$
   - 推广：对于多个中间坐标系，$T_{A \to E} = T_{D \to E} \times T_{C \to D} \times T_{B \to C} \times T_{A \to B}$
   - 比喻：就像导航路线，知道每段路的方向和距离，就能计算出总路线

4. 逆变换（Inverse Transformation）：
   - 原理：逆变换$H^{-1}$是将变换$H$的效果完全撤销的变换。如果$H$将点$p$映射到$p'$，则$H^{-1}$将$p'$映射回$p$，即$H^{-1} \times H = I$（单位矩阵）
   - 数学表示：$p = H^{-1} \times p' = H^{-1} \times H \times p$
   - 比喻：就像"撤销"操作，能把变换的效果完全还原

5. 齐次变换矩阵的逆变换计算：
   - 原理：对于齐次变换矩阵$H = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}$，其逆变换为：
     $$
     H^{-1} = \begin{bmatrix}
     R^T & -R^T \times \mathbf{t} \\
     \mathbf{0} & 1
     \end{bmatrix}
     $$
   - 推导过程：
     - 旋转部分的逆：$R^{-1} = R^T$（旋转矩阵的逆等于转置）
     - 平移部分的逆：需要先旋转回原方向，再反向平移，即$-R^T \times \mathbf{t}$
   - 比喻：就像先反向旋转，再反向平移，两步操作都要反过来

6. 逆变换的应用：
   - 坐标系反向变换：如果知道从坐标系A到B的变换$T_{AB}$，则从B到A的变换为$T_{BA} = T_{AB}^{-1}$
   - 逆运动学：已知末端执行器位姿，反推各关节角度时需要使用逆变换
   - 传感器标定：将传感器坐标系的数据转换到机器人基座坐标系时，需要计算逆变换
   - 比喻：就像知道从家到学校的路线，就能反推出从学校到家的路线

7. 复合变换与逆变换的组合应用：
   - 原理：在复杂变换链中，可能需要计算部分逆变换。例如，已知$T_{AB}$和$T_{AC}$，求$T_{BC}$：
     $$
     T_{BC} = T_{AC} \times T_{AB}^{-1}
     $$
   - 推导：因为$T_{AC} = T_{BC} \times T_{AB}$，所以$T_{BC} = T_{AC} \times T_{AB}^{-1}$
   - 比喻：就像知道总路程和第一段路程，就能计算出第二段路程

8. 计算效率考虑：
   - 原理：旋转矩阵的转置计算比求逆快得多，因为旋转矩阵是正交矩阵（$R^T = R^{-1}$）
   - 优化：在计算逆变换时，直接使用转置而不是通用的矩阵求逆算法
   - 比喻：就像走捷径，知道旋转矩阵的特殊性质，就能更快地计算逆变换

9. 实际应用示例：
   - 机器人正运动学：$T_{\text{base}}^{\text{end}} = T_{\text{base}}^{\text{joint1}} \times T_{\text{joint1}}^{\text{joint2}} \times \cdots \times T_{\text{jointN}}^{\text{end}}$
   - 机器人逆运动学：已知$T_{\text{base}}^{\text{end}}$，反推各关节角度
   - 工具坐标系变换：将工具坐标系中的点转换到基座坐标系：$p_{\text{base}} = T_{\text{base}}^{\text{tool}} \times p_{\text{tool}}$
   - 反向变换：$p_{\text{tool}} = (T_{\text{base}}^{\text{tool}})^{-1} \times p_{\text{base}}$

---

#### 问题：表示姿态的三种常用方法：XYZ固定角、ZYX欧拉角和ZYZ欧拉角是什么？它们有什么区别？

**答案**：

1. 姿态表示的基本概念：
   - 原理：姿态（Orientation）描述物体在空间中的方向，可以用旋转矩阵表示，但旋转矩阵有9个元素，冗余且不直观。因此常用更紧凑的表示方法：固定角（Fixed Angles）和欧拉角（Euler Angles）
   - 区别：固定角是相对于固定坐标系的旋转，欧拉角是相对于旋转后坐标系的旋转
   - 比喻：固定角就像站在地面上看物体旋转，欧拉角就像坐在旋转的物体上看它继续旋转

2. XYZ固定角（Fixed Angles）：
   - 原理：按照固定坐标系的X、Y、Z轴顺序依次旋转。旋转角度为$(\alpha, \beta, \gamma)$，分别表示绕固定X、Y、Z轴的旋转角度
   - 旋转矩阵：$R_{XYZ}(\alpha, \beta, \gamma) = R_Z(\gamma) \times R_Y(\beta) \times R_X(\alpha)$
   - 展开形式：
     $$
     R_{XYZ} = \begin{bmatrix}
     c_\gamma c_\beta & c_\gamma s_\beta s_\alpha - s_\gamma c_\alpha & c_\gamma s_\beta c_\alpha + s_\gamma s_\alpha \\
     s_\gamma c_\beta & s_\gamma s_\beta s_\alpha + c_\gamma c_\alpha & s_\gamma s_\beta c_\alpha - c_\gamma s_\alpha \\
     -s_\beta & c_\beta s_\alpha & c_\beta c_\alpha
     \end{bmatrix}
     $$
     其中$c_\alpha = \cos(\alpha)$，$s_\alpha = \sin(\alpha)$，其他类似
   - 特点：每次旋转都相对于原始固定坐标系
   - 比喻：就像站在地面上，先让物体绕X轴转，再绕Y轴转，最后绕Z轴转，每次都是相对于地面坐标系

3. ZYX欧拉角（Euler Angles，也叫Roll-Pitch-Yaw）：
   - 原理：按照旋转后坐标系的Z、Y、X轴顺序依次旋转。旋转角度为$(\psi, \theta, \phi)$，分别表示绕当前Z、Y、X轴的旋转角度
   - 旋转矩阵：$R_{ZYX}(\psi, \theta, \phi) = R_X(\phi) \times R_Y(\theta) \times R_Z(\psi)$
   - 展开形式：
     $$
     R_{ZYX} = \begin{bmatrix}
     c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi \\
     s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi \\
     -s_\theta & c_\theta s_\phi & c_\theta c_\phi
     \end{bmatrix}
     $$
   - 特点：每次旋转都相对于旋转后的坐标系（当前坐标系）
   - 应用：在机器人学和航空航天中广泛使用，$\phi$称为Roll（横滚），$\theta$称为Pitch（俯仰），$\psi$称为Yaw（偏航）
   - 比喻：就像坐在旋转的物体上，先绕自己的Z轴转，再绕新的Y轴转，最后绕新的X轴转

4. ZYZ欧拉角（Euler Angles）：
   - 原理：按照旋转后坐标系的Z、Y、Z轴顺序依次旋转。旋转角度为$(\alpha, \beta, \gamma)$，分别表示绕当前Z、Y、Z轴的旋转角度
   - 旋转矩阵：$R_{ZYZ}(\alpha, \beta, \gamma) = R_Z(\gamma) \times R_Y(\beta) \times R_Z(\alpha)$
   - 展开形式：
     $$
     R_{ZYZ} = \begin{bmatrix}
     c_\alpha c_\beta c_\gamma - s_\alpha s_\gamma & -c_\alpha c_\beta s_\gamma - s_\alpha c_\gamma & c_\alpha s_\beta \\
     s_\alpha c_\beta c_\gamma + c_\alpha s_\gamma & -s_\alpha c_\beta s_\gamma + c_\alpha c_\gamma & s_\alpha s_\beta \\
     -s_\beta c_\gamma & s_\beta s_\gamma & c_\beta
     \end{bmatrix}
     $$
   - 特点：两次绕Z轴旋转，中间一次绕Y轴旋转，适合描述对称物体的旋转
   - 应用：在经典力学和机器人学中常用，特别适合描述球关节的旋转
   - 比喻：就像先绕自己的Z轴转，再绕新的Y轴转，最后再绕新的Z轴转

5. 三种方法的对比：
   - **旋转顺序**：
     - XYZ固定角：固定坐标系X → Y → Z
     - ZYX欧拉角：当前坐标系Z → Y → X
     - ZYZ欧拉角：当前坐标系Z → Y → Z
   - **旋转矩阵的关系**：
     - XYZ固定角：$R_{XYZ} = R_Z(\gamma) \times R_Y(\beta) \times R_X(\alpha)$（从右到左）
     - ZYX欧拉角：$R_{ZYX} = R_X(\phi) \times R_Y(\theta) \times R_Z(\psi)$（从右到左）
     - ZYZ欧拉角：$R_{ZYZ} = R_Z(\gamma) \times R_Y(\beta) \times R_Z(\alpha)$（从右到左）
   - **应用场景**：
     - XYZ固定角：适合描述相对于固定参考系的旋转
     - ZYX欧拉角：适合描述飞行器姿态（Roll-Pitch-Yaw）
     - ZYZ欧拉角：适合描述对称物体的旋转，避免万向锁问题

6. 万向锁（Gimbal Lock）问题：
   - 原理：当中间旋转角度为$\pm 90°$时，第一个和第三个旋转轴重合，导致失去一个自由度
   - 影响：ZYX欧拉角在$\theta = \pm 90°$时会出现万向锁，ZYZ欧拉角在$\beta = 0°$或$\pm 180°$时会出现万向锁
   - 解决方案：使用四元数（Quaternion）或旋转矩阵直接表示姿态
   - 比喻：就像两个旋转轴重合了，无法区分是绕哪个轴旋转

7. 从旋转矩阵提取角度：
   - XYZ固定角：$\alpha = \operatorname{atan2}(r_{32}, r_{33})$，$\beta = \arcsin(-r_{31})$，$\gamma = \operatorname{atan2}(r_{21}, r_{11})$
   - ZYX欧拉角：$\phi = \operatorname{atan2}(r_{32}, r_{33})$，$\theta = \arcsin(-r_{31})$，$\psi = \operatorname{atan2}(r_{21}, r_{11})$
   - ZYZ欧拉角：$\alpha = \operatorname{atan2}(r_{23}, r_{13})$，$\beta = \arccos(r_{33})$，$\gamma = \operatorname{atan2}(r_{32}, -r_{31})$
   - 注意：提取角度时需要注意象限和奇异性问题

8. 实际应用：
   - 机器人末端执行器姿态：常用ZYX欧拉角表示
   - 飞行器姿态控制：使用Roll-Pitch-Yaw（ZYX欧拉角）
   - 球关节机器人：使用ZYZ欧拉角更自然
   - 传感器融合：需要将不同表示方法转换为统一格式

---

#### 问题：等效角度轴线表示法（Axis-Angle Representation）是什么？它如何表示姿态？

**答案**：

1. 等效角度轴线表示法的基本概念：
   - 原理：根据欧拉旋转定理，任何旋转都可以表示为绕某个单位轴$\mathbf{k}$旋转角度$\theta$。这种表示方法称为等效角度轴线表示法或轴角表示法
   - 数学表示：用四元组$(\mathbf{k}, \theta)$表示，其中$\mathbf{k} = [k_x, k_y, k_z]^T$是单位旋转轴（$\|\mathbf{k}\| = 1$），$\theta$是旋转角度
   - 比喻：就像用螺丝刀拧螺丝，需要知道拧的方向（轴线）和拧的角度

2. 旋转向量（Rotation Vector）表示：
   - 原理：将轴角表示压缩为三维向量$\mathbf{r} = \theta \mathbf{k}$，其中$\mathbf{r} = [r_x, r_y, r_z]^T$，$\theta = \|\mathbf{r}\|$，$\mathbf{k} = \mathbf{r} / \|\mathbf{r}\|$
   - 特点：旋转向量同时编码了旋转轴和旋转角度，方向表示旋转轴，模长表示旋转角度
   - 优点：只需要3个参数，比旋转矩阵（9个参数）更紧凑，比欧拉角更直观
   - 比喻：就像用一根箭，箭的方向表示旋转轴，箭的长度表示旋转角度

3. 从轴角到旋转矩阵的转换（Rodrigues公式）：
   - 原理：给定旋转轴$\mathbf{k}$和旋转角度$\theta$，旋转矩阵$R$可以通过Rodrigues公式计算：
     $$
     R(\mathbf{k}, \theta) = I + \sin(\theta) K + (1 - \cos(\theta)) K^2
     $$
     其中$K$是$\mathbf{k}$的反对称矩阵：
     $$
     K = \begin{bmatrix}
     0 & -k_z & k_y \\
     k_z & 0 & -k_x \\
     -k_y & k_x & 0
     \end{bmatrix}
     $$
   - 展开形式：
     $$
     R = \begin{bmatrix}
     k_x^2 v_\theta + c_\theta & k_x k_y v_\theta - k_z s_\theta & k_x k_z v_\theta + k_y s_\theta \\
     k_x k_y v_\theta + k_z s_\theta & k_y^2 v_\theta + c_\theta & k_y k_z v_\theta - k_x s_\theta \\
     k_x k_z v_\theta - k_y s_\theta & k_y k_z v_\theta + k_x s_\theta & k_z^2 v_\theta + c_\theta
     \end{bmatrix}
     $$
     其中$c_\theta = \cos(\theta)$，$s_\theta = \sin(\theta)$，$v_\theta = 1 - \cos(\theta)$
   - 比喻：就像用公式把"拧螺丝"的动作转换成旋转矩阵

4. 从旋转矩阵到轴角的转换：
   - 原理：给定旋转矩阵$R$，可以提取旋转轴和旋转角度
   - 旋转角度：$\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right)$，其中$\text{tr}(R)$是矩阵的迹
   - 旋转轴：$\mathbf{k} = \frac{1}{2\sin(\theta)} \begin{bmatrix} r_{32} - r_{23} \\ r_{13} - r_{31} \\ r_{21} - r_{12} \end{bmatrix}$（当$\sin(\theta) \neq 0$时）
   - 特殊情况：当$\theta = 0$或$\pi$时，需要特殊处理（旋转轴不唯一）
   - 比喻：就像从旋转矩阵中"提取"出旋转轴和角度信息

5. 旋转向量的指数映射和对数映射：
   - 指数映射（从旋转向量到旋转矩阵）：$R = \exp([\mathbf{r}]_\times)$，其中$[\mathbf{r}]_\times$是旋转向量$\mathbf{r}$的反对称矩阵
   - 对数映射（从旋转矩阵到旋转向量）：$\mathbf{r} = \log(R)$
   - 数学关系：$\exp([\mathbf{r}]_\times) = I + \frac{\sin(\|\mathbf{r}\|)}{\|\mathbf{r}\|} [\mathbf{r}]_\times + \frac{1 - \cos(\|\mathbf{r}\|)}{\|\mathbf{r}\|^2} [\mathbf{r}]_\times^2$
   - 比喻：就像在旋转向量和旋转矩阵之间建立"桥梁"

6. 等效角度轴线表示法的优点：
   - 紧凑性：只需要3个参数（旋转向量）或4个参数（轴角），比旋转矩阵（9个参数）更紧凑
   - 直观性：旋转轴和旋转角度有明确的物理意义，比欧拉角更直观
   - 无奇异性：不存在万向锁问题，任意姿态都可以唯一表示（除了$\theta = 0$的情况）
   - 插值友好：旋转向量之间的插值比欧拉角插值更平滑
   - 比喻：就像用"方向+角度"描述旋转，比用"三个角度"更直观

7. 等效角度轴线表示法的缺点：
   - 不唯一性：旋转角度$\theta$和$-\theta$绕相反轴旋转得到相同结果，即$(\mathbf{k}, \theta)$和$(-\mathbf{k}, -\theta)$表示相同旋转
   - 周期性：旋转角度$\theta$和$\theta + 2\pi$表示相同旋转
   - 计算复杂度：与旋转矩阵的转换需要三角函数计算，比欧拉角稍复杂
   - 比喻：就像同一个旋转可以用不同的"拧螺丝"方式表示

8. 与其他表示方法的转换：
   - 与旋转矩阵：通过Rodrigues公式或指数/对数映射
   - 与欧拉角：先转换为旋转矩阵，再提取欧拉角
   - 与四元数：旋转向量$\mathbf{r}$可以转换为四元数$q = [\cos(\theta/2), \sin(\theta/2) \mathbf{k}]$
   - 比喻：就像不同语言之间的翻译，可以互相转换

9. 实际应用：
   - 机器人运动规划：旋转向量便于插值和优化
   - 姿态估计：在SLAM和视觉里程计中常用旋转向量表示姿态
   - 优化问题：旋转向量是3维无约束优化变量，比旋转矩阵（6个约束）更适合优化
   - 数值计算：旋转向量的导数计算比旋转矩阵更简单
   - 比喻：就像用旋转向量作为"中间语言"，方便在不同表示方法之间转换和计算

10. 旋转向量的组合：
    - 原理：两个旋转的组合不能直接相加旋转向量，需要先转换为旋转矩阵，相乘后再转换回旋转向量
    - 近似：对于小角度旋转，旋转向量可以近似相加：$\mathbf{r}_{12} \approx \mathbf{r}_1 + \mathbf{r}_2$（当角度很小时）
    - 精确计算：$\mathbf{r}_{12} = \log(\exp([\mathbf{r}_1]_\times) \times \exp([\mathbf{r}_2]_\times))$
    - 比喻：就像两个"拧螺丝"动作不能简单相加，需要先转换成旋转矩阵再组合

---

### 人形机器人近期成果


---

### Apollo


---

### ROS/ROS2


---
