---
title: Optimal Transport-3.Kantorovich-OT Dual Problem
tags:
    - Optimal Transport
    - Math
---


## toc

&emsp;&emsp;在前面的小节，我们介绍了 Kantorovich OT问题的基本形式，其基本形式如下：

$$
L_{\boldsymbol{C}}(\boldsymbol{a}, \boldsymbol{b}) \quad \overset{\text{def.}}{=} \quad \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{C},\boldsymbol{P} \right>_{F}=\sum_{i,j}p_{ij}c_{ij}
$$  

其中，$\boldsymbol{P} \in \mathbb{R}_{+}^{n \times m}$ 为运输矩阵，$\boldsymbol{C} \in \mathbb{R}_{+}^{n \times m}$ 为成本矩阵，且约束条件为：  

$$
\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b}) = \{ \boldsymbol{P} \in \mathbb{R}^{n \times m}_{+}: \boldsymbol{P} \mathbf{1}_{m}=\boldsymbol{a} \quad and \quad \boldsymbol{P^{T}} \mathbf{1}_{n}=\boldsymbol{b} \}
$$  

&emsp;&emsp;从今天的视角来看，Kantorovich OT问题本质上是一种线性规划问题，然而在 Kantorovich 给出 OT 问题的上述形式时，严格的线性规划理论还没有被建立。作为线性规划理论的主要建立人，Kantorovich 对于 OT 问题的研究，在其建立线性规划理论上起到了重要的推动作用，线性规划中很多与经济资源分配的实际案例，便来源于 Kantorovich 对于 OT 问题的研究。Kantorovich 在线性规划理论中，提出了著名的**对偶关系**，这成为了最优化学科的核心理论之一。  
&emsp;&emsp;在本节，我们将首先介绍一些凸优化与对偶关系的基础知识，并给出 Kantorovich OT 问题的标准线性规划形式，最后我们将讨论其对偶问题的经济内涵。  

## Convex Optimization Foundations  

### Basic Concept  

**仿射集**  
&emsp;&emsp;设$C$为某一集合，若 $\forall x_1,x_2 \in C, \theta \in \mathbb{R}, \theta x_1+(1-\theta)x_2 \in C$，则称集合$C$为仿射集。  

**仿射函数**  
&emsp;&emsp;设有映射 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$，若 $f(x)=Ax+b, A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^{m}$，则称映射$f$为仿射函数。  

**凸集**  
&emsp;&emsp;设$C$为某一集合，若 $\forall x_1,x_2 \in C, \theta \in [0,1], \theta x_1+(1-\theta)x_2 \in C$，则称集合$C$为凸集。  

**凸函数**  
&emsp;&emsp;一个函数$f(x)$被称为凸函数，如果它的定义域 $dom f$ 为凸集，并且对 $\forall x_1,x_2 \in dom f, \alpha \in [0,1]$，有以下不等式成立：  

$$
f[\alpha x_1+(1-\alpha) x_2] \leq \alpha f(x_1)+(1-\alpha)f(x_2)
$$  

&emsp;&emsp;**凸函数的一阶条件**   
 &emsp;&emsp;设函数$f(x)$**一阶可微**，$x \in \mathbb{R}^{n}$，则$f(x)$为凸函数的**充要条件**为：$dom \space f$为凸集，且对$\forall x,y \in dom \space f$，有以下不等式成立：  

$$
f(y) \ge f(x)+\nabla f^{T}(x)(y-x)
$$  

&emsp;&emsp;**凸函数的二阶条件**   
&emsp;&emsp;设函数$f(x)$**二阶可微**，$x \in \mathbb{R}^{n}$，则$f(x)$为凸函数的**充要条件**为：$dom \space f$为凸集，且对$\forall x \in dom \space f$，有$\nabla^{2}f(x) \succeq 0$，即$Hessain$矩阵半正定。  

**最优化问题**  
&emsp;&emsp;最优化问题的基本形式(1)：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \quad & f(x) \\
        s.t. \quad & m_{i}(x) \leq 0,\quad i=1,2,\dots,M \\
        & n_{j}(x) = 0,\quad j=1,2,\dots,N
    \end{split}
\end{equation}
$$

- $x=\begin{bmatrix}
    x_1,x_2,\dots,x_n
\end{bmatrix}^{T},x \in \mathbb{R}^{n}$ 称为**优化变量(Optimization Variable)**.  
- $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$，称为**目标函数(Objective Function)**.  
- $m_i: \mathbb{R}^{n} \rightarrow \mathbb{R}$，称为**不等式约束(Inequality Constraint)**.  
- $n_j: \mathbb{R}^{n} \rightarrow \mathbb{R}$，称为**等式约束(Equality Constraint)**.  
- $D = \left( dom \space f \right) \bigcap \{x \vert m_{i}(x) \leq 0,i=1,2,\dots,M\} \bigcap \{x \vert h_{j}(x) = 0,j=1,2,\dots,N\}$，称为最优化问题的**域(Domain)**，$x \in D$ 称为**可行解(Feasible Solution)**.
- $p^{*} = \inf \{ f(x) \vert x \in D \}$，称为最优化问题的**最优值(Optimum)**.  
- $f(x^{*})=p^{*}$，称$x^{*}$为最优化问题的**最优解(Optimal Solution)**.  
- $X_{opt} = \{ x \vert x \in D,f(x)=p^{*} \}$，称为最优化问题的**最优解集(Optimal Set)**.  

**凸优化问题**  
&emsp;&emsp;若在优化问题(7)中，目标函数$f(x)$为凸函数，不等式约束$m_i(x)$为凸函数，等式约束$n_j(x)$为仿射函数，则称该优化问题为凸优化问题。  

### Dual Problem  

**拉格朗日函数**  
&emsp;&emsp;原问题(1)的拉格朗日函数为：  

$$
L(x,\lambda,\eta)=f(x)+\sum_{i=1}^{M}\lambda_i m_i(x)+\sum_{j=1}^{N}\eta_j n_j(x)
$$  

$$
\lambda_i \ge 0, \quad i=1,2,\dots,M
$$  

**原问题的无约束形式**  
&emsp;&emsp;原问题(1)的无约束形式(2)为：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \max_{\lambda,\eta} \quad & L(x,\lambda,\eta)  \\
        s.t. \quad & \lambda_i \ge 0, \quad i=1,2,\dots,M \\
    \end{split}
\end{equation}
$$

&emsp;&emsp;原问题(1)与其无约束形式(2)是等价的。  

**对偶问题**  
&emsp;&emsp;原问题(1)的拉格朗日对偶问题(3)为：  

$$
\begin{equation}
    \begin{split}
        \max_{\lambda,\eta} \min_{x} \quad & L(x,\lambda,\eta) \\
        s.t. \quad & \lambda_i \ge 0, \quad i = 1,2,\dots,M
    \end{split}
\end{equation}
$$

**弱对偶关系**  
&emsp;&emsp;原问题(1)与其对偶问题(3)满足弱对偶关系：  

$$
\max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) \leq \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

**强对偶关系**  
&emsp;&emsp;若原问题(1)与其对偶问题(3)满足：  

$$
\max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) = \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

&emsp;&emsp;则称原问题(1)与其对偶问题(3)满足强对偶关系。  

**$Slater$条件**  
&emsp;&emsp;若原问题(1)是凸问题，同时 $\exists x \in relint(D)$，使得约束满足：  

$$
\begin{split}
    & m_{i}(x) < 0, \quad i=1,2,\dots,M \\
    & n_{j}(x) = 0, \quad j=1,2,\dots,N \\
\end{split}
$$

&emsp;&emsp;则原问题与对偶问题满足强对偶关系。  

- 注：$relint(D)$ 表示原始凸问题的域的相对内部，即域内除了边界点以外的所有点。

&emsp;&emsp;$Slater$条件是一个**充分不必要条件**，若满足$Slater$条件，则强对偶一定成立，不满足$Slater$条件，强对偶也可能成立。大多数凸优化问题均满足$Slater$条件，即有强对偶性。  

### Linear Programming  

&emsp;&emsp;线性规划问题的一般形式为(4)：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \quad & c^{T}x + d \\
        s.t. \quad &  Gx \leq h \\
        & Ax=b  \\
    \end{split}
\end{equation}
$$

其中，$x,c,d \in \mathbb{R}^{n}$；$G \in \mathbb{R}^{M \times n}, h \in \mathbb{R}^{M}; A \in \mathbb{R}^{N \times n}, b \in \mathbb{R}^{N}$.  
&emsp;&emsp;一般的线性规划问题都可以通过变形，转化成线性规划的标准形式(5)：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \quad & c^{T}x  \\
        s.t. \quad &  Ax \leq b \\
        & x \leq 0
    \end{split}
\end{equation}
$$

&emsp;&emsp;有一个重要的结论是：**线性规划问题是凸问题且满足强对偶关系。**  

## Linear Programming Form of Kantorovich-OT Problem

&emsp;&emsp;Kantorovich-OT Problem 本质上是一个线性规划问题，我们可以通过变形，将其转化为线性规划形式，下面我们尝试将Kantorovich-OT Problem 转化为与其等价的线性规划问题。  

### Primal Linear Programming

&emsp;&emsp;原始的Kantorovich-OT Problem的形式如下：

$$
L_{\boldsymbol{C}}(\boldsymbol{a}, \boldsymbol{b}) \quad \overset{\text{def.}}{=} \quad \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{C},\boldsymbol{P} \right>_{F}=\sum_{i,j}p_{ij}c_{ij}
$$  

&emsp;&emsp;运输矩阵 $\boldsymbol{P}$ 与 成本矩阵 $\boldsymbol{C}$ 可以写成列向量形式：  

$$
\boldsymbol{P}=\begin{bmatrix}
    \boldsymbol{p_1}, \boldsymbol{p_2}, \cdots,\boldsymbol{p_m}
\end{bmatrix},\quad \boldsymbol{C} = \begin{bmatrix}
    \boldsymbol{c_1},\boldsymbol{c_2}, \cdots,\boldsymbol{c_m}
\end{bmatrix}
$$  

&emsp;&emsp;构造向量 $\boldsymbol{p},\boldsymbol{c} \in \mathbb{R}^{nm}$，矩阵 $A \in \mathbb{R}^{(n+m) \times nm}$:  

$$
\boldsymbol{p} = \begin{bmatrix}
    \boldsymbol{p_{1}^{T}}, \boldsymbol{p_{2}^{T}}, \cdots, \boldsymbol{p_{m}^{T}}
\end{bmatrix}^{T}, \quad \boldsymbol{c} = \begin{bmatrix}
    \boldsymbol{c_{1}^{T}}, \boldsymbol{c_{2}^{T}}, \cdots, \boldsymbol{c_{m}^{T}}
\end{bmatrix}^{T}
$$  

$$
\boldsymbol{A} = \begin{bmatrix}
    \boldsymbol{1_{n}^{T}} \otimes \boldsymbol{I_{m}} \\
    \boldsymbol{I_{n}} \otimes \boldsymbol{1_{m}^{T}} \\
\end{bmatrix}
$$  

其中，符号"$\otimes$"表示矩阵的 kronecker's product。  

$$
\boldsymbol{c^{T}}\boldsymbol{p}=\sum_{i=1}^{m}\boldsymbol{c_{i}^{T}}\boldsymbol{p_{i}}=\sum_{i,j}p_{ij}c_{ij}
$$  

$$
\boldsymbol{A}\boldsymbol{p} = \begin{bmatrix}
    \boldsymbol{1_{n}^{T}} \otimes \boldsymbol{I_{m}} \\
    \boldsymbol{I_{n}} \otimes \boldsymbol{1_{m}^{T}} \\
\end{bmatrix}\boldsymbol{p}=\begin{bmatrix}
    (\boldsymbol{1_{n}^{T}} \otimes \boldsymbol{I_{m}})\boldsymbol{p} \\
    (\boldsymbol{I_{n}} \otimes \boldsymbol{1_{m}^{T}})\boldsymbol{p} \\
\end{bmatrix} \in \mathbb{R}^{n+m}
$$  

其中，  

$$
\begin{split}
    (\boldsymbol{1_{n}^{T}} \otimes \boldsymbol{I_{m}})\boldsymbol{p} &= \begin{bmatrix}
        \boldsymbol{I_{m}}, \boldsymbol{I_{m}}, \cdots,\boldsymbol{I_{m}}
    \end{bmatrix}_{m \times nm} \cdot \boldsymbol{p} \\
    &= \begin{bmatrix}
        \sum_{j}p_{1 \cdot j} \\
        \sum_{j}p_{2 \cdot j} \\
        \vdots \\
        \sum_{j}p_{n \cdot j}
    \end{bmatrix}_{m \times 1} = \boldsymbol{P1_{m}} = \boldsymbol{a}
\end{split}
$$  

$$
\begin{split}
    (\boldsymbol{I_{n}} \otimes \boldsymbol{1_{m}^{T}})\boldsymbol{p} &= \begin{bmatrix}
        \boldsymbol{1_{m}^{T}} & & & \\
        & \boldsymbol{1_{m}^{T}} & & \\
        & & \ddots & \\
        & & & \boldsymbol{1_{m}^{T}}
    \end{bmatrix}_{n \times nm} \cdot \boldsymbol{p} \\
    &= \begin{bmatrix}
        \sum_{j}p_{j \cdot 1} \\
        \sum_{j}p_{j \cdot 2} \\
        \vdots \\
        \sum_{j}p_{j \cdot m} \\
    \end{bmatrix}_{n \times 1} = \boldsymbol{P^{T}1_{n}}=\boldsymbol{b}
\end{split}
$$  

故有：  

$$
\boldsymbol{Ap}=\begin{bmatrix}
    (\boldsymbol{1_{n}^{T}} \otimes \boldsymbol{I_{m}})\boldsymbol{p} \\
    (\boldsymbol{I_{n}} \otimes \boldsymbol{1_{m}^{T}})\boldsymbol{p} \\
\end{bmatrix}=\begin{bmatrix}
    \boldsymbol{a} \\
    \boldsymbol{b} \\
\end{bmatrix} \overset{\text{def.}}{=} \boldsymbol{d} \in \mathbb{R^{n+m}}
$$  

故 Kantorovich-OT Problem 可以转化为等价形式(6):  

$$
\begin{equation}
    \begin{split}
        \min_{\boldsymbol{p} \in \mathbb{R^{nm}}} \quad &\boldsymbol{c^{T}p} \\
        s.t. \quad & \boldsymbol{Ap} = \boldsymbol{d} \\
        & \boldsymbol{p} \ge \boldsymbol{0} \\
    \end{split}
\end{equation}
$$  

以上问题(6)即为 Kantorovich-OT Problem 的一般线性规划形式，其中等式约束也可以改为不等式约束。  

### Dual Linear Programming  

&emsp;&emsp;接下来我们来讨论一下问题(6)的对偶问题，首先写出问题(6)的拉格朗日函数：  

$$
\mathbf{L}(\boldsymbol{p},\boldsymbol{h},\boldsymbol{u}) = \boldsymbol{c^{T}p} + \boldsymbol{h^{T}} \left[ \boldsymbol{Ap} - \boldsymbol{d} \right] - \boldsymbol{u^{T}p}
$$  

原问题的无约束形式为：  

$$
\begin{split}
    \min_{\boldsymbol{p}}\max_{\boldsymbol{h,u}} \quad & \mathbf{L}(\boldsymbol{p},\boldsymbol{h},\boldsymbol{u}) \\
    s.t. \quad & \boldsymbol{u} \ge \boldsymbol{0}
\end{split}
$$  

则原问题(6)的对偶问题(7)为：  

$$
\begin{equation}
    \max_{\boldsymbol{h,u}}\min_{\boldsymbol{p}} \quad \mathbf{L}(\boldsymbol{p},\boldsymbol{h},\boldsymbol{u})
\end{equation}
$$  

我们首先来考察一下内部极小化问题：  

$$
\begin{split}
    \min_{p} \mathbf{L}(\boldsymbol{p},\boldsymbol{h},\boldsymbol{u}) &= \boldsymbol{c^{T}p} + \boldsymbol{h^{T}}\left( \boldsymbol{Ap} - \boldsymbol{d} \right) - \boldsymbol{u^{T}p} \\
    &= \left(  \boldsymbol{c^{T}} + \boldsymbol{h^{T}A} - \boldsymbol{u^{T}} \right)\boldsymbol{p} - \boldsymbol{h^{T}d} \\
    &= \left( \boldsymbol{c} + \boldsymbol{A^{T}h - \boldsymbol{u}} \right)^{T}\boldsymbol{p} - \boldsymbol{h^{T}d}
\end{split}
$$  

令 $\boldsymbol{h'}=-\boldsymbol{h}$，则有：  

$$
\min_{p} \mathbf{L}(\boldsymbol{p},\boldsymbol{h},\boldsymbol{u}) = \left( \boldsymbol{c} - \boldsymbol{u} - \boldsymbol{A^{T}h'} \right)^{T}\boldsymbol{p} + \boldsymbol{(h')^{T}d}
$$  

$$
\min_{p} \mathbf{L}(\boldsymbol{p},\boldsymbol{h'},\boldsymbol{u}) = \left \{
\begin{array}{rcl}
\boldsymbol{(h')^{T}d}, & {\boldsymbol{A^{T}h'} \leq \boldsymbol{c} - \boldsymbol{u}}\\
-\infty,& {\boldsymbol{A^{T}h'} \nleq \boldsymbol{c} - \boldsymbol{u}}\\
\end{array} \right.
$$  

故问题(7)等价于:  

$$
\begin{split}
    \max_{\boldsymbol{h',u}} \quad & \boldsymbol{(h')^{T}d} \\
    s.t. \quad & \boldsymbol{A^{T}h'} \leq \boldsymbol{c} - \boldsymbol{u} \\
    & \boldsymbol{u} \ge \boldsymbol{0}
\end{split}
$$  

合并两个等式约束得到原问题(6)的对偶问题(8):  

$$
\begin{equation}
    \begin{split}
    \max_{\boldsymbol{h'} \in \mathbb{R}^{n+m}} \quad & \boldsymbol{d^{T}h'} \\
    s.t. \quad & \boldsymbol{A^{T}h'} \leq \boldsymbol{c} \\
    & \boldsymbol{h'} \ge \boldsymbol{0}
\end{split}
\end{equation}
$$  

## Economic interpretation of Kantorovich-OT Dual Problem  

&emsp;&emsp;Kantorovich-OT Problem 的对偶问题具有现实的经济解释，下面我们首先导出其对偶问题的形式。其原问题为：  

$$
\min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{C},\boldsymbol{P} \right>_{F}=\sum_{i,j}p_{ij}c_{ij}
$$  

$$
\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b}) = \{ \boldsymbol{P} \in \mathbb{R}^{n \times m}_{+}: \boldsymbol{P} \mathbf{1}_{m}=\boldsymbol{a} \quad and \quad \boldsymbol{P^{T}} \mathbf{1}_{n}=\boldsymbol{b} \}
$$  

原问题的拉格朗日函数：  

$$
\mathbf{L}(\boldsymbol{P},\boldsymbol{f},\boldsymbol{g}) = \left< \boldsymbol{C},\boldsymbol{P} \right> + \left< \boldsymbol{a} - \boldsymbol{P1_{m}},\boldsymbol{f} \right>+\left< \boldsymbol{b} - \boldsymbol{P^{T}1_{n}},\boldsymbol{g} \right>
$$  

其中，$\boldsymbol{f} \in \mathbb{R}^{n}, \boldsymbol{g} \in \mathbb{R}^{m}$。则原问题的无约束形式为：  

$$
\min_{\boldsymbol{P} \ge 0}\max_{\boldsymbol{f},\boldsymbol{g}}\quad \left< \boldsymbol{C},\boldsymbol{P} \right> + \left< \boldsymbol{a} - \boldsymbol{P1_{m}},\boldsymbol{f} \right>+\left< \boldsymbol{b} - \boldsymbol{P^{T}1_{n}},\boldsymbol{g} \right>
$$  

上式等价于  

$$
\max_{\boldsymbol{f},\boldsymbol{g}} \left< \boldsymbol{a},\boldsymbol{f} \right> + \left< \boldsymbol{b},\boldsymbol{g} \right> + \min_{\boldsymbol{P} \ge \boldsymbol{0}} \left< \boldsymbol{C - \boldsymbol{f1_{m}^{T}}-\boldsymbol{1_{n}g^{T}}}, \boldsymbol{P} \right>
$$  

令 $\boldsymbol{Q}=\boldsymbol{C - \boldsymbol{f1_{m}^{T}}-\boldsymbol{1_{n}g^{T}}} = \boldsymbol{C} - \boldsymbol{f}\oplus\boldsymbol{g}$，有：  

$$
\min_{\boldsymbol{P} \ge \boldsymbol{0}} \left< \boldsymbol{Q},\boldsymbol{P} \right> = \left \{
\begin{array}{rcl}
0, & {\boldsymbol{Q} \ge \boldsymbol{0}}\\
-\infty,& {otherwise}\\
\end{array} \right.
$$  

故原问题的对偶问题可以写成(9):  

$$
\begin{equation}
    L_{\boldsymbol{C}}(\boldsymbol{a},\boldsymbol{b}) = \max_{(\boldsymbol{f},\boldsymbol{g}) \in \boldsymbol{R}(\boldsymbol{C})} \left< \boldsymbol{f},\boldsymbol{a} \right> + \left< \boldsymbol{g,\boldsymbol{b}} \right>
\end{equation}
$$  

$$
\boldsymbol{R}(\boldsymbol{C}) \overset{def.}{=} \{ (\boldsymbol{f},\boldsymbol{g}) \in \mathbb{R}^{n} \times \mathbb{R}^{m}: \boldsymbol{f}\oplus\boldsymbol{g} \leq \boldsymbol{C} \}
$$  

&emsp;&emsp;借助原问题的经济含义，我们来思考一下对偶问题的经济解释。在原问题中，我们需要将 n 个木材工厂生产的木材运输到 m 个城市，各个木材工厂的产量为 $\boldsymbol{a} \in \mathbb{R}_{+}^{n}$，各个城市的需求量为 $\boldsymbol{b} \in \mathbb{R}_{+}^{m}$。在原问题中，我们所需要求解的运输矩阵 $\boldsymbol{P}$ 包含了从工厂 i 到城市 j 的具体运输计划，即需要将 工厂 i 生产多少木材运输到城市 j。成本矩阵 $\boldsymbol{C}$ 则表示从工厂 i 到城市 j 每单位木材运输所消耗的成本。约束条件 $\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})$ 表示运输计划需要符合各个工厂的产量以及各个城市的需求量。  
&emsp;&emsp;**中间商运输**  
&emsp;&emsp;考虑目标函数。木材供应商面对复杂的运输问题，他选择将运输的工作外包给某个中间商。该中间商会首先从各个木材工厂统一将木材进行回收，然后再分发给不同的城市。假设木材的回收价格为 $\boldsymbol{f} \in \mathbb{R}^{n}$，$f_{i}$ 表示从木材工厂 i 回收单位木材的价格；木材的分发价格为 $\boldsymbol{g} \in \mathbb{R}^{m}$，$g_{i}$ 表示将单位木材分发给城市 i 的价格。中间商的目的自然是最大化它的利润，故对于中间商来说，这个运输的优化问题为：  

$$\max_{\boldsymbol{f},\boldsymbol{g}} \quad \left< \boldsymbol{f},\boldsymbol{a} \right> + \left< \boldsymbol{g},\boldsymbol{b} \right>$$  

&emsp;&emsp;考虑约束条件。对于木材供应商来说，他认为如果中间商的定价满足 $\boldsymbol{f} \oplus \boldsymbol{g} \leq \boldsymbol{C}$，则意味着自己一定不会吃亏，因为这意味着 $\forall i,j, f_i+g_j \leq \boldsymbol{C}_{ij}$，即对于将单位木材从工厂 i 运输到城市 j，中间商收取的费用一定是小于等于自己运输的成本。因此供应商将这条定价约束写进了招商合同。这样对于中间商来说，它所面对的最优化问题即为：  

$$
\begin{equation}
    L_{\boldsymbol{C}}(\boldsymbol{a},\boldsymbol{b}) = \max_{(\boldsymbol{f},\boldsymbol{g}) \in \boldsymbol{R}(\boldsymbol{C})} \left< \boldsymbol{f},\boldsymbol{a} \right> + \left< \boldsymbol{g,\boldsymbol{b}} \right>
\end{equation}
$$  

$$
\boldsymbol{R}(\boldsymbol{C}) \overset{def.}{=} \{ (\boldsymbol{f},\boldsymbol{g}) \in \mathbb{R}^{n} \times \mathbb{R}^{m}: \boldsymbol{f}\oplus\boldsymbol{g} \leq \boldsymbol{C} \}
$$  

&emsp;&emsp;考虑经济收益。由于供应商面对的原问题本质上是一个线性规划问题，故其满足强对偶关系，这意味着中间商可以通过合理地设置收费价格 $\boldsymbol{f},\boldsymbol{g}$ 使得：  

$$
\min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{P},\boldsymbol{C} \right> = \max_{(\boldsymbol{f},\boldsymbol{g}) \in \boldsymbol{R}(\boldsymbol{C})} \left< \boldsymbol{f},\boldsymbol{a} \right> + \left< \boldsymbol{g},\boldsymbol{b} \right>
$$  

&emsp;&emsp;值得注意的是，中间商所设置的价格 $\boldsymbol{f},\boldsymbol{g}$ 可以为负数，这也很好理解，中间商可以补贴的形式，从供应商那里收取木材，即将 $\boldsymbol{f}$ 的某些值设置为负数；而对于需要木材的城市，收取高额的分发费用，即增加 $\boldsymbol{g}$ 的某些分量的值，这样对于中间商来说，它仍然可以保证利润不变，但却给供应商营造了一种获利的错觉。  

## Reference  

- **[1] Book: Peyré G, Cuturi M. Computational optimal transport[J]. Center for Research in Economics and Statistics Working Papers, 2017 (2017-86).**  
- **[2] Lecture: 李向东. 最优传输理论及其应用. BIMSA**  

