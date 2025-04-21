---
title: Optimal Transport-6.Sinkhorn Algorithm
tags:
    - Optimal Transport
    - Math
---  

## toc

&emsp;&emsp;在上一节中我们介绍了带熵正则项的最优传输问题，通过在原始的最优传输问题中引入熵正则项，可以原问题转化为严格的凸问题，从而可以使用更高效的算法来更快地求解最优传输问题，计算效率的提高在最优传输的应用领域至关重要。本节我们将会介绍求解带熵正则项的最优传输问题的经典算法——**Sinkhorn算法**。我们将会从算法推导、收敛性证明、代码实现几个方面来讨论。  
&emsp;&emsp;Sinkhorn算法是一种用于解决凸优化问题的迭代算法，主要用于解决带有非负矩阵约束的最优传输问题。其主要思想是通过迭代地调整两个非负矩阵的行和列，使它们近似满足指定的边际约束条件。  
&emsp;&emsp;Sinkhorn算法最早由Richard Sinkhorn在1964年提出，用于解决数学物理中的问题。后来，该算法被应用于最优传输问题，并在机器学习、图像处理等领域得到了广泛应用。随着计算机算力的提高和研究的深入，Sinkhorn算法及其变体在处理大规模数据和优化问题上发挥了重要作用。  

## Algorithm iteration  

&emsp;&emsp;首先，我们来回顾一下上一节熵正则的内容。在引入熵正则项后，原始的最优传输问题变成了如下(1)形式：  

$$
\begin{equation}
    L_{\boldsymbol{C}}^{\varepsilon}(\boldsymbol{a},\boldsymbol{b}) := \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{P},\boldsymbol{C} \right> - \varepsilon H(\boldsymbol{P})
\end{equation}
$$  

$$
U(\boldsymbol{\alpha, \beta}):=\{ \boldsymbol{P}: \boldsymbol{P1_{m}}=\boldsymbol{a}, \boldsymbol{P^{T}1_{n}}=\boldsymbol{b} \}, \quad H(\boldsymbol{P})=-\sum_{i,j}p_{ij}\log{p_{ij}}
$$  

&emsp;&emsp;为了求导的便利性，我们将熵的表达式做一些小的修改：  

$$
H(\boldsymbol{P})=-\sum_{i,j}p_{ij}\log{p_{ij}}+1=-\sum_{i,j}p_{ij}(\log{p_{ij}}-1)
$$  

&emsp;&emsp;由于只是在原始的优化问题的目标函数上加了一个常数，故此修改并不会对问题(1)的最优解。为了推导Sinkhorn算法的迭代公式，我们首先来写出优化问题(1)的拉格朗日函数：  

$$
\mathcal{L}(\boldsymbol{P,f,g})=\left< \boldsymbol{P,C} \right>-\varepsilon H(\boldsymbol{P})-\left< \boldsymbol{f,P1_{m}-a} \right>-\left< \boldsymbol{g,P^{T}1_{n}-b} \right>
$$  

&emsp;&emsp;由费马定理可知，对于优化问题(1)的最优解，有下式成立：  

$$
\frac{\partial \mathcal{L}(\boldsymbol{P,f,g})}{\partial p_{ij}}=c_{ij}+\varepsilon\log{p_{ij}}-f_{i}-g_{j}=0
$$  

$$
\Rightarrow p_{ij}=e^{\frac{f_i+g_j-c_{ij}}{\varepsilon}}=e^{\frac{f_i}{\varepsilon}}e^{\frac{-c_{ij}}{\varepsilon}}e^{\frac{g_j}{\varepsilon}}
$$  

&emsp;&emsp;由以上的结果可知，优化问题(1)的最优解的元素可以分解成三部分之积，进一步，我们可以将以上结果写成矩阵形式，设：  

$$
\boldsymbol{u}=\begin{bmatrix}
    e^{\frac{f_1}{\varepsilon}} \\
    e^{\frac{f_2}{\varepsilon}} \\
    \vdots \\
    e^{\frac{f_n}{\varepsilon}} \\
\end{bmatrix},\quad \boldsymbol{v}=\begin{bmatrix}
    e^{\frac{g_1}{\varepsilon}} \\
    e^{\frac{g_2}{\varepsilon}} \\
    \vdots \\
    e^{\frac{g_m}{\varepsilon}} \\
\end{bmatrix},\quad \boldsymbol{K}=\begin{bmatrix}
    e^{\frac{-c_{11}}{\varepsilon}} & e^{\frac{-c_{12}}{\varepsilon}} & \dotsb & e^{\frac{-c_{1m}}{\varepsilon}} \\
    e^{\frac{-c_{21}}{\varepsilon}} & e^{\frac{-c_{22}}{\varepsilon}} & \dotsb & e^{\frac{-c_{2m}}{\varepsilon}} \\  
    \vdots & \vdots &  & \vdots \\
    e^{\frac{-c_{n1}}{\varepsilon}} & e^{\frac{-c_{n2}}{\varepsilon}} & \dotsb & e^{\frac{-c_{nm}}{\varepsilon}} \\
\end{bmatrix}
$$  

&emsp;&emsp;则优化问题(1)的最优解可以写成：  

$$
\boldsymbol{P}=diag(\boldsymbol{u})\boldsymbol{K}diag(\boldsymbol{v})
$$  

&emsp;&emsp;考虑边际约束条件：  

$$
\left \{
\begin{array}{l}
 \boldsymbol{P1_{m}} =diag(\boldsymbol{u})\boldsymbol{K}diag(\boldsymbol{v})\boldsymbol{1_{m}}=\boldsymbol{a} \\
 \boldsymbol{P^{T}1_{n}} =  diag(\boldsymbol{v})\boldsymbol{K^{T}}diag(\boldsymbol{u})\boldsymbol{1_{n}}=\boldsymbol{b}
\end{array} \right.
$$  

$$
\because diag(\boldsymbol{u})\boldsymbol{1_{n}}=\boldsymbol{u}, \quad diag(\boldsymbol{v})\boldsymbol{1_{m}}=\boldsymbol{v}
$$  

$$
\Rightarrow \left \{
\begin{array}{l}
 \boldsymbol{P1_{m}} =diag(\boldsymbol{u})(\boldsymbol{K}\boldsymbol{v})=\boldsymbol{u} \odot (\boldsymbol{Kv})=\boldsymbol{a} \\
 \boldsymbol{P^{T}1_{n}} = diag(\boldsymbol{v})(\boldsymbol{K^{T}}\boldsymbol{u})=\boldsymbol{v} \odot (\boldsymbol{K^{T}u})=\boldsymbol{b} \\
\end{array} \right.
$$  

&emsp;&emsp;在实际求解优化问题(1)时，由于成本矩阵 $\boldsymbol{C}$ 与惩罚系数 $\varepsilon$ 是已知的，故矩阵 $\boldsymbol{K}$ 是已知的，因此我们需要求解向量 $(\boldsymbol{u,v})$ 使得其满足边际条件，则问题(1)的最优解即为：$\boldsymbol{P}=diag(\boldsymbol{u})\boldsymbol{K}diag(\boldsymbol{v})$. 该问题在数值分析领域被称为矩阵缩放问题，一个直观的解决方法是通过迭代求解。**首先固定住$\boldsymbol{v}$，更新$\boldsymbol{u}$，使其满足边界条件；然后固定更新后的$\boldsymbol{u}$，更新$\boldsymbol{v}$，使其同样满足边界条件。** 这便是Sinkhorn算法的基本思想，写成迭代公式如下：  

$$
\Rightarrow \left \{
\begin{array}{l}
\boldsymbol{u}^{(l+1)} = \frac{\boldsymbol{a}}{\boldsymbol{K}\boldsymbol{v}^{(l)}} \\
\\
\boldsymbol{v}^{(l+1)} = \frac{\boldsymbol{b}}{\boldsymbol{K}\boldsymbol{u}^{(l+1)}} \\
\end{array} \right. \quad \boldsymbol{v}^{(0)}=\boldsymbol{1_{m}}
$$  

&emsp;&emsp;其中，除法运算表示向量之间各个元素对应相除。$\boldsymbol{v}^{(0)}$可以初始化为任意的正值向量。通过不断的迭代，我们可以得到近似解$(\boldsymbol{u^{*},v^{*}})$，此时，最优运输矩阵及Sinkhorn Distance则为：  

$$
\boldsymbol{P^{*}} = diag(\boldsymbol{u^{*}})\boldsymbol{K}diag(\boldsymbol{v^{*}}), \quad Sinkhorn \ Distance = \left< \boldsymbol{P^{*}, C} \right>
$$  

&emsp;&emsp;矩阵$\boldsymbol{K}$被称为 Gibbs distributions，实际上，对于优化问题(1)的最优解$\boldsymbol{P^{*}}$，其实际上是矩阵$\boldsymbol{K}$在可行解集$U(\boldsymbol{a,b})$上的投影，即(2)：  

$$
\begin{equation}
    \boldsymbol{P^{*}}=Proj_{U(\boldsymbol{a,b})}^{KL}(\boldsymbol{K}) := \argmin_{\boldsymbol{P \in U(\boldsymbol{a,b})}} KL(\boldsymbol{P | K})
\end{equation}
$$  

&emsp;&emsp;问题(2)被称为**静态薛定谔问题**。进一步地，Sinkhorn算法的迭代过程实际上也是一个迭代投影过程，设：  

$$
\mathcal{C}_{\boldsymbol{a}}^{1}:=\{ \boldsymbol{P}: \boldsymbol{P1_{m}=a} \} \quad and \quad \mathcal{C}_{\boldsymbol{b}}^{2} := \{ \boldsymbol{P}: \boldsymbol{P^{T}1_{n}=b} \}
$$  

&emsp;&emsp;则有 $U(\boldsymbol{a,b})=\mathcal{C}_{\boldsymbol{a}}^{1} \cap \mathcal{C}_{\boldsymbol{b}}^{2}$，**Bregman迭代投影**的过程如下：  

$$
\boldsymbol{P}^{(l+1)} := Proj_{\mathcal{C}_{\boldsymbol{a}}^{1}}^{KL}(\boldsymbol{P}^{(l)}) \quad and \quad \boldsymbol{P}^{(l+2)} := Proj_{\mathcal{C}_{\boldsymbol{b}}^{2}}^{KL}(\boldsymbol{P}^{(l+1)})
$$  

&emsp;&emsp;其与Sinkhorn迭代过程的对应关系为：  

$$
\boldsymbol{P}^{(2l)} := diag(\boldsymbol{u}^{(l)})\boldsymbol{K}diag(\boldsymbol{v}^{(l)})
$$  

$$
\boldsymbol{P}^{(2l+1)} := diag(\boldsymbol{u}^{(l+1)})\boldsymbol{K}diag(\boldsymbol{v}^{(l)})
$$

$$
\boldsymbol{P}^{(2l+2)} := diag(\boldsymbol{u}^{(l+1)})\boldsymbol{K}diag(\boldsymbol{v}^{(l+1)})
$$

## Algorithm Covergence

## Algorithm Implementation  

&emsp;&emsp;接下来，我们来通过代码实现一下Sinkhorn算法，首先来给定原始分布、目标分布、成本矩阵以及惩罚系数：  

$$
\boldsymbol{a} = \begin{bmatrix}
    0.2 \\
    0.5 \\
    0.3 \\
\end{bmatrix}, \boldsymbol{b}=\begin{bmatrix}
    0.3 \\
    0.4 \\
    0.3 \\
\end{bmatrix}, \quad \boldsymbol{C}=\begin{bmatrix}
    0.1 & 0.2 & 0.3 \\
    0.4 & 0.5 & 0.6 \\
    0.7 & 0.8 & 0.9 \\
\end{bmatrix},\quad \varepsilon=0.01
$$  

&emsp;&emsp;相应的Python代码为：  

```python
a = np.array([0.2, 0.5, 0.3])
b = np.array([0.3, 0.4, 0.3])
C = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])
reg_param = 0.01
```

&emsp;&emsp;按照前文推导出的Sinkhorn算法的迭代公式，我们可以写出其实现代码：  

```python
import numpy as np

def sinkhorn(a, b, C, reg_param, max_iters=100):

    n = len(a)
    m = len(b)
    u = np.ones(n)
    v = np.ones(m)
    K = np.exp(-C / reg_param)

    for _ in range(max_iters):
        u = a / (K.dot(v))
        v = b / (K.T.dot(u))

    transport_matrix = np.diag(u).dot(K).dot(np.diag(v))
    sinkhorn_distance = np.sum(transport_matrix * C)

    return transport_matrix, sinkhorn_distance

transport_matrix, sinkhorn_distance = sinkhorn(a, b, C, reg_param)
```  
&emsp;&emsp;我们同样可以使用Python中专门用于解决最优传输领域问题的库POT来进行求解，其代码如下：  

```python
import ot

transport_matrix = ot.sinkhorn(a, b, C, reg_param)
sinkhorn_distance = ot.sinkhorn2(a, b, C, reg_param)
```
&emsp;&emsp;这两种求解方法得到最优运输矩阵及Sinkhorn距离相同，均为：  

$$
\boldsymbol{P^{*}}=\begin{bmatrix}
    0.06 & 0.08 & 0.06 \\
    0.15 & 0.2 & 0.15 \\
    0.09 & 0.12 & 0.09 \\
\end{bmatrix}, \quad Sinkhorn \ Distance = 0.53
$$  

## Reference  

- **[1] Book: Peyré G, Cuturi M. Computational optimal transport[J]. Center for Research in Economics and Statistics Working Papers, 2017 (2017-86).**  