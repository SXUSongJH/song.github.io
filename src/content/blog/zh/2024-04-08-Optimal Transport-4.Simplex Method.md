---
title: Optimal Transport-4.Simplex Method
tags:
    - Optimal Transport
    - Math
---

# Simplex Method  

&emsp;&emsp;单纯形方法(Simplex Method)是解决线性规划问题的一种重要算法。它由 George Dantzig 在1947年发明。线性规划问题涉及到在一系列线性等式或不等式约束条件下，找到一个线性函数的最大值或最小值。**单纯形方法的基本思想是将问题的解空间视为一个几何形状（通常是多维的），这个几何形状的顶点代表可能的解。算法从这个形状的一个顶点开始，沿着边缘移动到相邻的顶点，以此方式逐步改进解，直到找到最优解为止。每一步都保证目标函数的值不会变差。**  
&emsp;&emsp;单纯形方法在实际应用中非常广泛，包括运筹学、工程优化、金融和经济学等领域。尽管存在某些局限性，它仍然是解决线性规划问题的一个强大工具。  

## Linear Programming Form of OT Problem  

&emsp;&emsp;在之前的章节中，我们已经介绍了 Kantorovich OT Problem 的经济含义与数学形式，我们也说明了其能够转化为线性规划的标准形式，这里我们再回顾一下之前的内容。  
**数学形式**  
&emsp;&emsp;Kantorovich OT Problem 的数学形式如下(1)：  

$$\begin{equation}
    L_{\boldsymbol{C}}(\boldsymbol{a}, \boldsymbol{b}) \quad \overset{\text{def.}}{=} \quad \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{C},\boldsymbol{P} \right>_{F}=\sum_{i,j}p_{ij}c_{ij}
\end{equation}$$  

其中 $\boldsymbol{a} \in \mathbb{R}^{n},\boldsymbol{b} \in \mathbb{R}^{m}, \boldsymbol{P},\boldsymbol{C} \in \mathbb{R}^{n \times m}_{+}$，约束条件 $\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})$ 为：  

$$\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b}) = \{ \boldsymbol{P} \in \mathbb{R}^{n \times m}_{+}: \boldsymbol{P} \mathbf{1}_{m}=\boldsymbol{a} \quad and \quad \boldsymbol{P^{T}} \mathbf{1}_{n}=\boldsymbol{b} \}$$  

**经济含义**  
&emsp;&emsp;考虑一个实际问题，有 n 个生产木材的工厂，生产的木材需要供应给 m 个城市，每个工厂每个月的产能是固定的，记为 $\boldsymbol{a}, \boldsymbol{a} \in \mathbb{R}^{n}$；每个城市每个月木材的需求量也是固定的，记为 $\boldsymbol{b}, \boldsymbol{b} \in \mathbb{R}^{m}$。记 $\boldsymbol{P} = [p_{ij}]_{n \times m} \in \mathbb{R}^{n \times m}_{+}$ 表示运输方案，其中 $p_{ij}$ 表示第 i 个工厂运输到第 j 个城市的木材量。记 $\boldsymbol{C} = [c_{ij}]_{n \times m} \in \mathbb{R}^{n \times m}_{+}$ 表示运输成本，其中 $c_{ij}$ 表示将单位木材从第i个工厂运输到第j个城市的成本。  
&emsp;&emsp;则 Kantorovich OT Problem 实际上是在寻找满足供应与需求条件下，使得总成本最低的运输方案。  

**线性规划形式**  

&emsp;&emsp;Kantorovich OT Problem 实际上是一个线性规划问题，通过一定的变换，我们能够将原始的 Kantorovich OT Problem 转化为线性规划形式。  
&emsp;&emsp;将矩阵 $\boldsymbol{P},\boldsymbol{C}$ 写成列向量形式：  

$$\boldsymbol{P}=\begin{bmatrix}
    \boldsymbol{p_1}, \boldsymbol{p_2}, \cdots,\boldsymbol{p_m}
\end{bmatrix},\quad \boldsymbol{C} = \begin{bmatrix}
    \boldsymbol{c_1},\boldsymbol{c_2}, \cdots,\boldsymbol{c_m}
\end{bmatrix}$$  

&emsp;&emsp;构造向量 $\boldsymbol{p},\boldsymbol{c} \in \mathbb{R}^{nm}, \boldsymbol{d} \in \mathbb{R}^{n+m}$，矩阵 $A \in \mathbb{R}^{(n+m) \times nm}$:  

$$\boldsymbol{p} = \begin{bmatrix}
    \boldsymbol{p_{1}^{T}}, \boldsymbol{p_{2}^{T}}, \cdots, \boldsymbol{p_{m}^{T}}
\end{bmatrix}^{T}, \quad \boldsymbol{c} = \begin{bmatrix}
    \boldsymbol{c_{1}^{T}}, \boldsymbol{c_{2}^{T}}, \cdots, \boldsymbol{c_{m}^{T}}
\end{bmatrix}^{T}$$  

$$\boldsymbol{A} = \begin{bmatrix}
    \boldsymbol{1_{n}^{T}} \otimes \boldsymbol{I_{m}} \\
    \boldsymbol{I_{n}} \otimes \boldsymbol{1_{m}^{T}} \\
\end{bmatrix}, \quad \boldsymbol{d}=\begin{bmatrix}
    \boldsymbol{a} \\
    \boldsymbol{b} \\
\end{bmatrix}$$  

其中，符号"$\otimes$"表示矩阵的 kronecker's product。则(1)式可以被转化为：  

$$\begin{equation}
    \begin{split}
        \min_{\boldsymbol{p} \in \mathbb{R^{nm}}} \quad &\boldsymbol{c^{T}p} \\
        s.t. \quad & \boldsymbol{Ap} = \boldsymbol{d} \\
        & \boldsymbol{p} \ge \boldsymbol{0} \\
    \end{split}
\end{equation}$$  

&emsp;&emsp;优化问题(2)即为 Kantorovich OT Problem 的线性规划形式。  

## Theoretical Foundations of Simplex Method  

&emsp;&emsp;解决线性规划的单纯形算法的理论基础是**凸多面体分解定理**，在正式介绍单纯形算法之前，我们首先来介绍凸多面体分解定理。要想理解凸多面体分解定理，我们需要从凸多面体、极点、极方向等开始讲起。  

### Polyhedron

**凸包(Convex Hall)**  
&emsp;&emsp;凸包是点集中元素的凸组合所构成的集合。数学形似为：  

$$C \subseteq R^{n},Conv C= \{ \sum_{i=1}^{n}\theta_{i}x_{i} | \forall x_i \in C, \forall \theta_{i} \in [0,1], \sum_{i=1}^{n}\theta_i = 1, i = 1,...,n \}$$  

&emsp;&emsp;凸包是欧式空间中一组点集的最小凸集。  

<center>
    <img src="https://s2.loli.net/2024/04/08/DfV5GLTQ8ncANRi.jpg" width=60% height=60%>
    <div align="center">Image1: 凸包</div>
</center>  
<br/>  

**凸多面体(Convex Polyhedron)**  
&emsp;&emsp;凸多面体是有限个线性等式和不等式的解集。其数学形式为：  

$$P = \{ x | a_{i}^{T}x \leq b_i, i=1,...,m; c_{j}^{T}x = d_{j}, j=1,...,p \}$$  

&emsp;&emsp;从凸多面体的数学形式上可以多面体是有限个半空间和超平面的交集，因此凸多面体也是凸集。  

<center>
    <img src="https://s2.loli.net/2024/04/08/5yoEIzh3YAZRdX1.jpg" width=60% height=60%>
    <div align="center">Image2: 凸多面体</div>
</center>  
<br/>  

&emsp;&emsp;设线性规划问题(2)的可行解集为 $S = \{ \boldsymbol{x}| \boldsymbol{x} \in \mathbb{R}^{nm},\boldsymbol{Ax}=\boldsymbol{d},\boldsymbol{x} \ge 0 \}$。由凸多面体的定义可知**可行解集 $S$ 为凸多面体**。  

## Properties of Convex Polyhedron  

**极点(Extreme Point)**  
&emsp;&emsp;对于凸集$C$，若 $x \in C$ 不能表示成 $C$ 中另外两点的凸组合，则称 $x$ 为凸集 $C$ 的极点。极点的数学形式为：  

$$Ep(C) = \{x | x \in C, \forall x_1,x_2 \in C, \forall \theta \in [0,1], x \ne \theta x_1 + (1-\theta)x_2 \}$$  

&emsp;&emsp;通过极点的定义可以得知：**极点不能落在凸集中另外两点组成的线段上。**  

**方向(Recession Direction)**  
&emsp;&emsp;对于凸集$C$，若存在非零向量 $d$，满足：对于任意 $x \in C$，均有 $x + \lambda d \in C, \forall \lambda > 0$，则称 $d$ 为凸集 $C$ 的一个方向。方向的数学形式为：  

$$d: d \ne 0, \forall x \in C, \forall \lambda >0, x + \lambda d \in C$$  

&emsp;&emsp;凸集$C$中所有方向构成了一个方向锥：  

$$RDcone(C) = \{d | d \ne 0, \forall x \in C, \forall \lambda >0, x + \lambda d \in C \}$$  

**极方向(Extreme Direction)**  
&emsp;&emsp;对于凸集$C$，若方向 $d$ 不能表示成另外两个方向的正线性组合，则称 $d$ 为凸集 $C$ 的极方向。极方向的数学形式为：  

$$Ed(C) = \{ d | d \in RDcone(C),\forall d_1,d_2 \in RDcone(C), \forall \lambda_1,\lambda_2 > 0,d \ne \lambda_1 d_1 + \lambda_2 d_2 \}$$  

&emsp;&emsp;**凸集$C$中任意一个方向$d$可以表示成其极方向的线性组合。**

<center>
    <img src="https://s2.loli.net/2024/04/08/aTVmyZB4udYf7gA.jpg" width=60% height=60%>
    <div align="center">Image3: 极点、方向、极方向</div>
</center>  
<br/>  

**极点与极方向的数学表示**  
&emsp;&emsp;设 $\boldsymbol{x} \in \mathbb{R}^{n}, \boldsymbol{A} \in \mathbb{R}^{m \times n}, \boldsymbol{b} \in \mathbb{R}^{m}, r(\boldsymbol{A})=m$，则有凸多面体：  

$$S = \{\boldsymbol{x} | \boldsymbol{Ax}=\boldsymbol{b},\boldsymbol{x} \ge0 \}$$  

**极点定理:** $\boldsymbol{x} \in S$ 是凸多面体 $S$ 的极点当且仅当存在 $\boldsymbol{A}$ 的分块: $\boldsymbol{A}=\begin{bmatrix}
    \boldsymbol{B},\boldsymbol{N}
\end{bmatrix}$，使得 $\boldsymbol{x}$ 可以表示为：  

$$\boldsymbol{x} = \begin{bmatrix}
    \boldsymbol{B^{-1}b} \\
    \boldsymbol{0} \\
\end{bmatrix}, \quad \boldsymbol{A}=\begin{bmatrix}
    \boldsymbol{B},\boldsymbol{N}
\end{bmatrix},\boldsymbol{B} \in \mathbb{R}^{m \times m}$$  

其中，$\boldsymbol{B}$ 可逆，且 $\boldsymbol{B^{-1}b} \ge \boldsymbol{0}$。  
&emsp;&emsp;由极点定理可知，若 $\boldsymbol{x}$ 为凸多面体 $S$ 的一个极点，则 $\boldsymbol{x}$ 正分量对应的矩阵 $\boldsymbol{A}$ 的列必然线性无关。同时凸多面体的极点必然是有限个的。  

**极方向定理:** $\boldsymbol{d} \in \mathbb{R}^{n}$ 是凸多面体 $S$ 的极方向当且仅当存在 $\boldsymbol{A}$ 的分块: $\boldsymbol{A}=\begin{bmatrix}
    \boldsymbol{B},\boldsymbol{N}
\end{bmatrix}$， 使得 $\boldsymbol{d}$ 可以表示为:  

$$\boldsymbol{d} = t\begin{bmatrix}
    -\boldsymbol{B^{-1}n_{j}} \\
    \boldsymbol{e_{j}}
\end{bmatrix},\quad \boldsymbol{A} = \begin{bmatrix}
    \boldsymbol{B},\boldsymbol{N}
\end{bmatrix},\boldsymbol{B} \in \mathbb{R}^{m \times m}$$  

其中，$\boldsymbol{B}$ 可逆，$t > 0, \boldsymbol{B^{-1}n_{j}} \leq 0$，$\boldsymbol{n_{j}}$ 为矩阵 $\boldsymbol{N}$ 的第j列，$\boldsymbol{e_{j}} \in \mathbb{R}^{n-m}$ 的第j个分量为1，其余为0.  

## Decomposition of Convex Polyhedron  

&emsp;&emsp;设凸多面体 $S$ 的极点为 $\boldsymbol{x_1,x_2,\dots,x_{k}}$，极方向为 $\boldsymbol{d_1,d_2,\dots,d_{I}}$，则 $\boldsymbol{x} \in S$ 当且仅当 $\boldsymbol{x}$ 具有如下形式：  

$$\boldsymbol{x} = \sum_{i=1}^{k}\lambda_{i}\boldsymbol{x_{i}} + \sum_{j=1}^{I}\mu_{j}\boldsymbol{d_{j}}$$  

其中，$\sum_{i=1}^{k}\lambda_{i}=1,\lambda_{i} \ge 0, i=1,\dots,k; \mu_{j} \ge 0, j = 1,\dots,I$  
&emsp;&emsp;通过几何分析我们可以，该定理表示的含义为，**以凸多面体的顶点构成的线段上的点为起点，以凸多面体的方向为方向所形成的所有射线的集合即为凸多么体中的点集。**  

<center>
    <img src="https://s2.loli.net/2024/04/08/vYnHhyJeDd2PcTq.jpg" width=60% height=60%>
    <div align="center">Image4: 多面体的分解</div>
</center>  
<br/>  

## Simplex Method of Linear Programming  

&emsp;&emsp;在介绍了单纯形方法的理论——凸多面体分解定理后，我们现在可以推导出单纯形算法，我们首先基于分解定理，对原始的线性规划问题做一些分析。

### Analysis for LP Based on Decomposition Theroem  

&emsp;&emsp;线性规划问题的形式为：  

$$\begin{equation}
    \begin{split}
        \min_{\boldsymbol{c} \in \mathbb{R^{n}}} \quad &\boldsymbol{c^{T}x} \\
        s.t. \quad & \boldsymbol{Ax} = \boldsymbol{b} \\
        & \boldsymbol{x} \ge \boldsymbol{0} \\
    \end{split}
\end{equation}$$  

其中，$\boldsymbol{x} \in \mathbb{R}^{n}, \boldsymbol{A} \in \mathbb{R}^{m \times n}, \boldsymbol{b} \in \mathbb{R}^{m}, r(\boldsymbol{A})=m$.  
&emsp;&emsp;定义可行解集 $S = \{ \boldsymbol{x} | \boldsymbol{Ax}=\boldsymbol{b} \}$，则线性规划问题(3)也可以写成如下形式：  

$$\begin{equation}
    \begin{split}
        \min_{\boldsymbol{c} \in \mathbb{R^{n}}} \quad &\boldsymbol{c^{T}x} \\
        s.t. \quad & \boldsymbol{x} \in S \\
    \end{split}
\end{equation}$$  

&emsp;&emsp;由于可行解集 $S$ 为一个凸多面体，设凸多面体 $S$ 的极点为 $\boldsymbol{x_1,x_2,\dots,x_{k}}$，极方向为 $\boldsymbol{d_1,d_2,\dots,d_{I}}$，则对 $\forall x \in S, \exists \lambda_{i},\mu_{j}$，使得:  

$$\boldsymbol{x} = \sum_{i=1}^{k}\lambda_{i}\boldsymbol{x_{i}} + \sum_{j=1}^{I}\mu_{j}\boldsymbol{d_{j}}$$  

其中，$\sum_{i=1}^{k}\lambda_{i}=1,\lambda_{i} > 0, i=1,\dots,k; \mu_{j} \ge 0, j = 1,\dots,I$. 则优化问题(4)可以转化为：  

$$\begin{split}
    \min_{\boldsymbol{c} \in \mathbb{R^{n}}} \quad &\boldsymbol{c^{T}} \left( \sum_{i=1}^{k}\lambda_{i}\boldsymbol{x_{i}} + \sum_{j=1}^{I}\mu_{j}\boldsymbol{d_{j}} \right) \\
    s.t. \quad & \sum_{i=1}^{k}\lambda_{i}=1,\lambda_{i} \ge 0 \\
    \quad & \mu_{j} \ge 0
\end{split}$$  

即：  

$$\begin{equation}
    \begin{split}
        \min_{\boldsymbol{c} \in \mathbb{R^{n}}} \quad &\sum_{i=1}^{k}\lambda_{i}\boldsymbol{c^{T}x_{i}} + \sum_{j=1}^{I}\mu_{j}\boldsymbol{c^{T}d_{j}} \\
        s.t. \quad & \sum_{i=1}^{k}\lambda_{i}=1,\lambda_{i} \ge 0 \\
        \quad & \mu_{j} \ge 0
    \end{split}
\end{equation}$$  

&emsp;&emsp;对于优化问题(5)：  

- 若 $\exists j, \boldsymbol{c^{T}d_{j}} < 0$，令 $\mu_{j} \rightarrow +\infty$，则该问题的最小值为负无穷，无意义。  
- 若 $\forall j, \boldsymbol{c^{T}d_{j}} \ge 0$，则有 $\mu_{j} = 0, \forall j$，该问题存在最优解。

&emsp;&emsp;通过以上的分析，优化问题(5)可被化为：  

$$\begin{equation}
    \begin{split}
        \min_{\boldsymbol{c} \in \mathbb{R^{n}}} \quad &\sum_{i=1}^{k}\lambda_{i}\boldsymbol{c^{T}x_{i}} \\
        s.t. \quad & \sum_{i=1}^{k}\lambda_{i}=1,\lambda_{i} \ge 0 \\
    \end{split}
\end{equation}$$  

&emsp;&emsp;对于优化问题(6)，我们实际上只需要找到 i 使得 $\boldsymbol{c^{T}x_{i}}$ 最小，此时令对应的 $\lambda_{i}=1$，目标函数即为最小值。故优化问题(6)又可被转化为：  

$$\begin{equation}
    \begin{split}
        \min_{i} \quad \boldsymbol{c^{T}x_{i}} \\
    \end{split}
\end{equation}$$  

&emsp;&emsp;故实际上，**线性规划问题的最优值一定在其可行解集的某个极点上取到，其最优解也是某一个极点。**  

**结论**  
&emsp;&emsp;通过以上的分析，我们来总结一下得到的结论。考虑如下线性规划的形式：  

$$\begin{split}
    \min_{\boldsymbol{c} \in \mathbb{R^{n}}} \quad &\boldsymbol{c^{T}x} \\
    s.t. \quad & \boldsymbol{Ax} = \boldsymbol{b} \\
    & \boldsymbol{x} \ge \boldsymbol{0} \\
\end{split}$$  

&emsp;&emsp;其中，$\boldsymbol{A} \in \mathbb{R}^{m \times n},r(\boldsymbol{A})=m$，记其可行解集为 $S = \{ \boldsymbol{x} | \boldsymbol{Ax}=\boldsymbol{b},\boldsymbol{x} \ge \boldsymbol{0} \}$，设 $S$ 的极点为 $\boldsymbol{x_1,x_2,\dots,x_{k}}$，极方向为 $\boldsymbol{d_1,d_2,\dots,d_{I}}$，则有：  

- **线性规划有最优解当且仅当 $\forall j, \boldsymbol{c^{T}d_{j}} \ge 0$.**
- **若线性规划有最优值，则必然可在某个极点上达到。**  

## Mathematical Analysis of the Simplex Method  

&emsp;&emsp;有了前文中关于线性规划问题解的结论后，我们便能够比较容易地理解单纯形法的基本思想了。单纯形法的基本思想是**首先找到可行解集的某一个极点，判断其是否为最优解，若是则算法终止，否则寻找一个更有的极点。**  
&emsp;&emsp;这里有两个核心问题，**如何判断某一个极点是否是最优解**？以及**如何找到一个更优的极点**? 如果这两个问题解决了，我们便能够依据单纯形的基本思想完成整个算法的设计。接下来，我们便来分析这两个问题。  
**Analysis**  
&emsp;&emsp;设矩阵 $\boldsymbol{A}$ 的某种分块为 $\boldsymbol{A} = \begin{bmatrix}
    \boldsymbol{B},\boldsymbol{N}
\end{bmatrix}$，$\boldsymbol{B} \in \mathbb{R}^{m}$，使得 $\boldsymbol{B}$ 可逆，且 $\boldsymbol{B^{-1}b} \ge \boldsymbol{0}$，令：  

$$\boldsymbol{\bar{x}} = \begin{bmatrix}
    \boldsymbol{B^{-1}b} \\
    \boldsymbol{0} \\
\end{bmatrix}$$  

由极点定理可知 $\boldsymbol{\bar{x}}$ 是可行解集 $S$ 的某一个极点。**接下来我们要判断 $\boldsymbol{\bar{x}}$ 是否是最优解。**  
&emsp;&emsp;取 $\boldsymbol{x} \in S$，根据矩阵 $\boldsymbol{A}$ 的分块结果，相应地将向量 $\boldsymbol{x}$ 与 $\boldsymbol{c}$ 进行分块：  

$$\boldsymbol{x} = \begin{bmatrix}
    \boldsymbol{x_{B}} \\
    \boldsymbol{x_{N}} \\
\end{bmatrix}, \quad \boldsymbol{c} = \begin{bmatrix}
    \boldsymbol{c_{B}} \\
    \boldsymbol{c_{N}} \\
\end{bmatrix}$$  

$$\boldsymbol{Ax}=\boldsymbol{b} \quad \Rightarrow \quad \begin{bmatrix}
    \boldsymbol{B,N}
\end{bmatrix}\begin{bmatrix}
    \boldsymbol{x_{B}} \\
    \boldsymbol{x_{N}} \\
\end{bmatrix}=\boldsymbol{Bx_{B}}+\boldsymbol{Nx_{N}}=\boldsymbol{b} \quad \Rightarrow \quad \boldsymbol{x_{B}} = \boldsymbol{B^{-1}b}-\boldsymbol{B^{-1}Nx_{N}}$$  

&emsp;&emsp;若对 $\forall x \in S$，均有 $\boldsymbol{c^{T}x}-\boldsymbol{c^{T}\bar{x}} \ge 0$，则 $\boldsymbol{\bar{x}}$ 为最优解。对判断条件 $\boldsymbol{c^{T}x}-\boldsymbol{c^{T}\bar{x}} \ge 0$ 做一些化简：  

$$\begin{split}
    \boldsymbol{c^{T}x}-\boldsymbol{c^{T}\bar{x}} =& \begin{bmatrix}
        \boldsymbol{c_{B}^{T},c_{N}^{T}} \\
    \end{bmatrix}\begin{bmatrix}
        \boldsymbol{x_{B}} \\
        \boldsymbol{x_{N}} \\
    \end{bmatrix} - \begin{bmatrix}
        \boldsymbol{c_{B}^{T},c_{N}^{T}} \\
    \end{bmatrix}\begin{bmatrix}
        \boldsymbol{B^{-1}b} \\
        \boldsymbol{0} \\
    \end{bmatrix} = \boldsymbol{c_{B}^{T}x_{B}}+\boldsymbol{c_{N}^{T}x_{N}} - \boldsymbol{c_{B}^{T}B^{-1}b}  \\
    &= \boldsymbol{c_{B}^{T}B^{-1}b} - \boldsymbol{c_{B}^{T}B^{-1}Nx_{N}}+\boldsymbol{c_{N}^{T}x_{N}} - \boldsymbol{c_{B}^{T}B^{-1}b} = \boldsymbol{c_{N}^{T}x_{N}} - \boldsymbol{c_{B}^{T}B^{-1}Nx_{N}} \\
    &= \left( \boldsymbol{c_{N}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}N} \right)\boldsymbol{x_{N}}
\end{split}$$  

&emsp;&emsp;**(1)** 若 $\boldsymbol{c_{N}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}N}  \ge \boldsymbol{0}$，则有 $\boldsymbol{c^{T}x}-\boldsymbol{c^{T}\bar{x}} \ge 0$，$\boldsymbol{\bar{x}}$ 即为最优解，此时我们已经找到了最优解，算法可以终止了。  
&emsp;&emsp;**(2)** 若 $\boldsymbol{c_{N}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}N}  \ngeq \boldsymbol{0}$，此时 $\boldsymbol{\bar{x}}$ 不是最优解，我们需要**寻找一个更优的极点。**  
&emsp;&emsp;不妨设 $\boldsymbol{c_{N}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}N}$ 中的第j个分量小于零，即 $\boldsymbol{c_{Nj}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}n_{j}} < 0$，记：  

$$\boldsymbol{d_{j}} = \begin{bmatrix}
    -\boldsymbol{B^{-1}n_{j}} \\
    \boldsymbol{e_{j}}
\end{bmatrix}$$  

则：  

$$\boldsymbol{c^{T}d_{j}} = \begin{bmatrix}
        \boldsymbol{c_{B}^{T},c_{N}^{T}} \\
    \end{bmatrix}\begin{bmatrix}
    -\boldsymbol{B^{-1}n_{j}} \\
    \boldsymbol{e_{j}}
\end{bmatrix} = \boldsymbol{c_{Nj}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}n_{j}} < 0$$  

记 $\boldsymbol{r_{j}} = \boldsymbol{B^{-1}n_{j}}$  
&emsp;&emsp;**1.** 若 $\boldsymbol{r_{j}} \leq \boldsymbol{0}$，则 $\boldsymbol{d_{j}} \ge 0$，由极方向定理可知: $\boldsymbol{d_{j}}$ 为 $S$ 的一个极方向，则对 $\forall \lambda \ge 0, \boldsymbol{\bar{x}}+\lambda \boldsymbol{d_{j}} \in S$，故有：  

$$\boldsymbol{c^{T}}\left( \boldsymbol{\bar{x}} + \lambda \boldsymbol{d_{j}} \right) = \boldsymbol{c^{T}\bar{x}}+\lambda \boldsymbol{c^{T}d_{j}} \rightarrow -\infty, \quad \lambda \rightarrow +\infty$$  

&emsp;&emsp;**2.** 若 $\boldsymbol{r_{j}} \nleq \boldsymbol{0}$，则 $\boldsymbol{d_{j}} \ngeq \boldsymbol{0}$，故 $\boldsymbol{d_{j}}$ 不是极方向，此时：  

$$\boldsymbol{c^{T}}\left( \boldsymbol{\bar{x}} + \lambda \boldsymbol{d_{j}} \right) = \boldsymbol{c^{T}\bar{x}}+\lambda \boldsymbol{c^{T}d_{j}} < \boldsymbol{c^{T}\bar{x}}$$  

&emsp;&emsp;由上式可知沿着 $\boldsymbol{d_{j}}$ 方向，能够使得目标函数下降，此时我们需要**选取一个合适的步长 $\lambda$，** 使得下降速度尽可能大，但又不会使得新得到的解 $\boldsymbol{\bar{x}} + \lambda \boldsymbol{d_{j}}$ 违反约束条件。  
&emsp;&emsp;首先来检验等式约束，对于 $\forall \lambda \ge 0$：  

$$\begin{split}
    \boldsymbol{A}\left( \boldsymbol{\bar{x}} + \lambda \boldsymbol{d_{j}} \right) =& \boldsymbol{A\bar{x}} + \lambda \boldsymbol{Ad_{j}} \\
    =& \boldsymbol{b} + \lambda \begin{bmatrix}
        \boldsymbol{B,N} \\
    \end{bmatrix} \begin{bmatrix}
        -\boldsymbol{B^{-1}n_{j}} \\
        \boldsymbol{e_{j}} \\
    \end{bmatrix} \\
    =& \boldsymbol{b} + \lambda \left( -\boldsymbol{n_{j}} + \boldsymbol{n_{j}} \right) \\
    =& \boldsymbol{b}  
\end{split}$$  

&emsp;&emsp;再来检验不等式约束：  

$$\boldsymbol{\bar{x}} + \lambda \boldsymbol{d_{j}} = \begin{bmatrix}
    \boldsymbol{B^{-1}b} \\
    \boldsymbol{0} \\
\end{bmatrix} + \lambda \begin{bmatrix}
    -\boldsymbol{B^{-1}n_{j}} \\
    \boldsymbol{e_{j}}
\end{bmatrix} \ge \boldsymbol{0}$$  

$$\Rightarrow \lambda \boldsymbol{e_{j}} \ge 0,\quad \boldsymbol{B^{-1}b}-\lambda \boldsymbol{B^{-1}n_{j}}=\boldsymbol{B^{-1}b} - \lambda \boldsymbol{r_{j}} \ge \boldsymbol{0}$$  

记 $\boldsymbol{h} = \boldsymbol{B^{-1}b} \ge \boldsymbol{0}$，则：  

$$\boldsymbol{h}-\lambda \boldsymbol{r_{j}} \ge \boldsymbol{0} \quad \Leftrightarrow \quad \begin{bmatrix}
    \boldsymbol{h_{1}} - \lambda \boldsymbol{r_{j1}} \\
    \vdots \\
    \boldsymbol{h_{m}} - \lambda \boldsymbol{r_{jm}} \\
\end{bmatrix} \ge \boldsymbol{0} \quad \Leftrightarrow \quad \forall i, \boldsymbol{h_{i}} - \lambda \boldsymbol{r_{ji}} \ge 0$$  

$$\Rightarrow \quad \lambda \leq \frac{\boldsymbol{h_{i}}}{\boldsymbol{r_{ji}}}, \forall i,且 \boldsymbol{r_{ji}} > 0 \quad \Rightarrow \quad \lambda^{*} = \min_{i} \{ \frac{\boldsymbol{h_{i}}}{\boldsymbol{r_{ji}}} | \boldsymbol{r_{ji}} > 0 \}$$  

&emsp;&emsp;通过以上分析，我们找到了一个最佳的步长 $\lambda^{*}$，有一个非常重要的结论：**利用 $\lambda^{*}$ 构造的新解 $\boldsymbol{\hat{x}}=\boldsymbol{\bar{x}}+\lambda \boldsymbol{d_{j}}$ 为可行解集 $S$ 的一个新的极点。**  
&emsp;&emsp;此时我们只需要重复以上步骤判断 $\boldsymbol{\hat{x}}$ 是否是最优解，若是则算法终止，否则再构造下一个极点。  

## Algorithm Steps for Simplex Method  

&emsp;&emsp;输入: $\boldsymbol{c} \in \mathbb{R}^{n}, \boldsymbol{A} \in \mathbb{R}^{m \times n}, \boldsymbol{b} \in \mathbb{R}^{m}$  
&emsp;&emsp;输出: $\boldsymbol{x} \in \mathbb{R}^{n}$  
&emsp;&emsp;算法步骤：  
&emsp;&emsp;(1) 确定某种 $\boldsymbol{A}$ 的分块: $\boldsymbol{A} = \begin{bmatrix}
    \boldsymbol{B,N}
\end{bmatrix}$，以及对应的初始极点：  

$$\boldsymbol{x} = \begin{bmatrix}
    \boldsymbol{B^{-1}b} \\
    \boldsymbol{0} \\
\end{bmatrix}$$  

&emsp;&emsp;(2) 判断 $\boldsymbol{c_{N}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}N} \ge \boldsymbol{0}?$ 若是，则算法终止，输出 $\boldsymbol{x}$，否则转入下一步(3);  
&emsp;&emsp;(3) 找到 $\boldsymbol{c_{N}^{T}} - \boldsymbol{c_{B}^{T}B^{-1}N}$ 的负值分量$j$，检验是否 $\exists j$，使得 $\boldsymbol{r_{j}}=\boldsymbol{Bn_{j}} \leq \boldsymbol{0}$，若存在，则该问题是无界的，无最小值，算法终止，否则转入(4);  
&emsp;&emsp;(4) 挑选一个$j$，构造一个新的极点$\boldsymbol{\bar{x}}$:  

$$\boldsymbol{\bar{x}} = \boldsymbol{x} + \lambda^{*} \boldsymbol{d_{j}},\quad \boldsymbol{d_{j}} = \begin{bmatrix}
    -\boldsymbol{B^{-1}n_{j}} \\
    \boldsymbol{e_{j}}
\end{bmatrix},\quad \lambda^{*} = \min_{i} \{ \frac{\boldsymbol{h_{i}}}{\boldsymbol{r_{ji}}} | \boldsymbol{r_{ji}} > 0 \}$$  

&emsp;&emsp;(5) 根据 $\boldsymbol{\bar{x}}$，对矩阵$\boldsymbol{A}$进行重新分块 $\boldsymbol{A} = \begin{bmatrix}
    \boldsymbol{\bar{B},\bar{N}}
\end{bmatrix}$，重复(2)~(5)步。  

## Python Impletation of Simplex Method  

&emsp;&emsp;给定算法的输入：  

$$\boldsymbol{c}=\begin{bmatrix}
    1 \\
    2 \\
    3 \\
    4 \\
    5 \\
\end{bmatrix},\quad \boldsymbol{A}= \begin{bmatrix}
    1 & 1 & 1 & 1 & 1 \\
    2 & 2 & 3 & 4 & 5 \\
\end{bmatrix},\quad \boldsymbol{b}=\begin{bmatrix}
    10 \\
    20 \\
\end{bmatrix}$$  

&emsp;&emsp;我们可以调用 Python 的 `scipy.optimize` 类中的 `linprog`方法来执行单纯形算法，代码如下：  

```python
from scipy.optimize import linprog

c = [1, 2, 3, 4, 5]

A_eq = [
    [1, 1, 1, 1, 1],
    [2, 2, 3, 4, 5] 
]
b_eq = [
    10, 
    20  
]

result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='simplex')

print(result['x'])
```  
&emsp;&emsp;使用单纯形算法解得最优解为：  

$$\boldsymbol{x^{*}} = \begin{bmatrix}
    10 & 0 & 0 & 0 & 0
\end{bmatrix}^{T}$$  

## Reference  

**[1] Video: 最优化理论与方法-第十一讲-线性规划, superfatseven, B站.**  
**[2] Book: Boyd S P, Vandenberghe L. Convex optimization[M]. Cambridge university press, 2004.**



