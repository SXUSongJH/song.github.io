---
title: Optimal Transport-2.Wasserstein Distance
tags:
  - Optimal Transport
  - Math
---  

# Wasserstein Distance  

&emsp;&emsp;在上一节 Monge-Kantorovich Problem 中，我们介绍了最优传输问题。最优传输一个重要的应用是它可以用来衡量分布之间的距离，从而将距离的概念由点与点之间拓展到分布与分布之间。本节我们将介绍分布之间的距离定义，即 Wasserstein Distance，以及为什么其能够用于表示分布之间的距离。  

## Metric Properties on Probility Space  

&emsp;&emsp;在上一节中，我们从概率视角描述了最优传输问题。设 $X,Y$ 是服从分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 的两个随机变量，运输矩阵为 $\boldsymbol{P}$，成本矩阵为 $\boldsymbol{C}$，则分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的最优传输问题可以被定义为：  

$$
L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right> = \min_{(X,Y)} \{ \mathbb{E}_{(X,Y)}(c(X,Y)): X \sim \boldsymbol{\alpha}, Y \sim \boldsymbol{\beta} \}
$$  

&emsp;&emsp;$L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 的含义是将分布 $\boldsymbol{\alpha}$ 传输到分布 $\boldsymbol{\beta}$ 所花费的最小成本，我们很自然地就会想到 $L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 也许能够表示分布 $\boldsymbol{\alpha}$ 和 $\boldsymbol{\beta}$ 之间的距离或相似度。当然，要说明这个问题，我们需要证明函数 $L_{\boldsymbol{C}}$ 满足概率空间中距离函数的性质。  
&emsp;&emsp;设分布 $\boldsymbol{\alpha},\boldsymbol{\beta} $ 取自概率空间 $\mathcal{X}$，$W(\boldsymbol{\alpha},\boldsymbol{\beta})$ 是分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的距离函数，如果函数 $W$ 满足:  

- **非负性(Non-negativity):** 对 $\forall \boldsymbol{\alpha},\boldsymbol{\beta} \in \mathcal{X}, W(\boldsymbol{\alpha},\boldsymbol{\beta}) \ge 0.$  
- **同一性(Identity of Indiscernibles):** $W(\boldsymbol{\alpha},\boldsymbol{\beta})=0$ 当且仅当 $\boldsymbol{\alpha} = \boldsymbol{\beta}.$  
- **对称性(Symmetry):** $\forall \boldsymbol{\alpha},\boldsymbol{\beta} \in \mathcal{X}, W(\boldsymbol{\alpha},\boldsymbol{\beta}) = W(\boldsymbol{\beta},\boldsymbol{\alpha}).$  
- **三角不等式(Triangle Inequality):** $\forall \boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma} \in \mathcal{X}, W(\boldsymbol{\alpha},\boldsymbol{\gamma}) \leq W(\boldsymbol{\alpha},\boldsymbol{\beta})+W(\boldsymbol{\beta},\boldsymbol{\gamma}).$  

&emsp;&emsp;学者们通过研究发现，当对成本矩阵 $\boldsymbol{C}$ 设置一些条件后，可以使得概率空间中最优传输问题的解 $L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 满足距离函数的性质，从而使得其可以用于衡量分布之间的距离。  

## Wasserstein Distance  

### Definition

&emsp;&emsp;我们首先来定义离散分布下的 Wasserstein Distance。设 $\boldsymbol{\alpha},\boldsymbol{\beta} \in \sum_{n}:=\{ \boldsymbol{x} \in \mathbb{R}^{n}_{+}: \boldsymbol{x^{T}}\mathbf{1}_{n}=1 \}$，设矩阵 $\boldsymbol{D} \in \mathbb{R}^{n \times n}$ 是一个度量矩阵，即矩阵 $\boldsymbol{D}$ 满足:  
**(1)** $\boldsymbol{D} \in \mathbb{R}^{n \times n}_{+};$  
**(2)** $\boldsymbol{D}_{i,j}=0$，当且仅当 $i=j$;  
**(3)** $\boldsymbol{D}$ 是对称矩阵;  
**(4)** $\forall i,j,k \in \{ 1,\dotsb,n\}, \boldsymbol{D}_{i,k} \leq \boldsymbol{D}_{i,j}+\boldsymbol{D}_{j,k}$.  
令成本矩阵 $\boldsymbol{C} = \boldsymbol{D}^{p}= \left[  \boldsymbol{D}_{i,j}^{p} \right]_{n \times n} \in \mathbb{R}^{n \times n}_{+}(p \ge 1)$，定义：  

$$
W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta}) := L_{\boldsymbol{D}^{p}}(\boldsymbol{\alpha,\boldsymbol{\beta}})^{1/p}
$$  

则称 $W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 为概率分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的 **p-Wasserstein 距离**。  
&emsp;&emsp;现在来证明$W_{p}$可以作为概率空间$\sum_{n}$上的距离函数。  

### Proof  

$$
W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta}) = L_{\boldsymbol{D}^{p}}(\boldsymbol{\alpha,\boldsymbol{\beta}})^{1/p} = \left( \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{D}^{p} \right> \right)^{\frac{1}{p}}
$$  

其中 $\boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) = \{ \boldsymbol{P} \in \mathbb{R}^{n \times n}_{+} : \boldsymbol{P}\mathbf{1}_{n}=\boldsymbol{\alpha} \quad and \quad \boldsymbol{P^{T}}\mathbf{1}_n=\boldsymbol{\beta} \}$.  
&emsp;&emsp;要证明 $W_{p}$ 可以作为概率空间$\sum_{n}$上的距离函数，则需要证明 $W_{p}$ 满足概率空间中距离函数的性质，即非负性、同一性、对称性、三角不等式。  
&emsp;&emsp;**(1) 非负性证明**  
&emsp;&emsp; $\boldsymbol{P},\boldsymbol{D}^{p} \in \mathbb{R}^{n \times n}_{+} \Rightarrow \left<  \boldsymbol{P},\boldsymbol{D}^{p} \right>=\sum_{ij}\boldsymbol{P}_{ij}\boldsymbol{D}^{p}_{ij} \ge 0 \Rightarrow W_{p}(\boldsymbol{\alpha}, \boldsymbol{\beta}) \ge 0.$  
&emsp;&emsp;**(2) 同一性证明**  
&emsp;&emsp;由度量矩阵的性质可知:  $\boldsymbol{D}_{i,i}=0, \forall i \in \{ 1,\dotsb,n \}$，则有 $\boldsymbol{D}_{i,i}^{p}=0$，即成本矩阵 $\boldsymbol{D}^{p}$ 的对角线元素均为零。  
&emsp;&emsp;当 $\boldsymbol{\alpha}=\boldsymbol{\beta}$ 时，可行域 $\boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\alpha}) = \{ \boldsymbol{P} \in \mathbb{R}^{n \times n}_{+} : \boldsymbol{P}\mathbf{1}_{n}=\boldsymbol{P^{T}}\mathbf{1}_n=\boldsymbol{\alpha} \}$，则 $\boldsymbol{P}^{*}=diag(\boldsymbol{\alpha}) \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\alpha})$，此时：  

$$
\left<  \boldsymbol{P}^{*}, \boldsymbol{D}^{p} \right>=\sum_{i}\boldsymbol{\alpha}_{i}\boldsymbol{D}_{i,i}^{p}=0 \Rightarrow W_{p}(\boldsymbol{\alpha},\boldsymbol{\alpha})=0
$$  

&emsp;&emsp;当 $W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})=0$ 时，由于成本矩阵 $\boldsymbol{D}^{p}$ 的非对角线元素均大于零，故运输矩阵 $\boldsymbol{P}$ 的非对角线元素均为零，即运输矩阵 $\boldsymbol{P}$ 为对角矩阵，$\boldsymbol{P}=\boldsymbol{P}^{T}$. 此时有 $\boldsymbol{P}\mathbf{1}_{n}=\boldsymbol{P^{T}}\mathbf{1}_n$，即 $\boldsymbol{\alpha}=\boldsymbol{\beta}$.  
&emsp;&emsp;**(3) 对称性证明**  
&emsp;&emsp;设 $\boldsymbol{P}^{*}$ 为$W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})$所对应的最优运输矩阵，则有:  

$$
W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})=\left<  \boldsymbol{P}^{*},\boldsymbol{D}^{p} \right>^{\frac{1}{p}}
$$  

&emsp;&emsp;由于成本矩阵 $\boldsymbol{D}^{p}$ 是对称矩阵，故有：  

$$
W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})=\left<  \boldsymbol{P}^{*},\boldsymbol{D}^{p} \right>^{\frac{1}{p}}=\left<  \boldsymbol{(P^{*})^{T}},\boldsymbol{D}^{p} \right>^{\frac{1}{p}}
$$  

&emsp;&emsp;$\boldsymbol{(P^{*})^{T}}\mathbf{1}_{n}=\boldsymbol{\beta}, \boldsymbol{P}^{*}\mathbf{1}_{n}=\boldsymbol{\alpha} \Rightarrow \boldsymbol{(P^{*})^{T}} \in \boldsymbol{U}(\boldsymbol{\beta},\boldsymbol{\alpha})$. 由于 $\boldsymbol{U}(\boldsymbol{\beta},\boldsymbol{\alpha})$ 与 $\boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 中的运输矩阵是对应转置的关系，故有：  

$$
W_{p}(\boldsymbol{\beta},\boldsymbol{\alpha})=\left( \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\beta},\boldsymbol{\alpha})} \left< \boldsymbol{P},\boldsymbol{D}^{p} \right> \right)^{\frac{1}{p}}=\left<  \boldsymbol{(P^{*})^{T}},\boldsymbol{D}^{p} \right>^{\frac{1}{p}}
$$  

$$\Rightarrow W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta}) = W_{p}(\boldsymbol{\beta},\boldsymbol{\alpha})$$  
&emsp;&emsp;**(4) 三角不等式性质证明**  
&emsp;&emsp;设 $\boldsymbol{\gamma} \in \sum_{n}$, 现证明：$W_{p}(\boldsymbol{\alpha},\boldsymbol{\gamma}) \leq W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})+W_{p}(\boldsymbol{\beta},\boldsymbol{\gamma})$.  
&emsp;&emsp;设 $\boldsymbol{P}$ 是 $W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 所对应的最优运输矩阵，$\boldsymbol{Q}$ 是 $W_{p}(\boldsymbol{\beta},\boldsymbol{\gamma})$ 所对应的最优运输矩阵，则有  

$$
\begin{split}
    W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta}) &= \left<  \boldsymbol{P},\boldsymbol{D}^{p} \right>^{\frac{1}{p}} = \left(\sum_{ij}\boldsymbol{P}_{ij}\boldsymbol{D}^{p}_{ij}\right)^{\frac{1}{p}}  \\
    W_{p}(\boldsymbol{\beta},\boldsymbol{\gamma}) &= \left<  \boldsymbol{Q},\boldsymbol{D}^{p} \right>^{\frac{1}{p}} = \left(\sum_{ij}\boldsymbol{Q}_{ij}\boldsymbol{D}^{p}_{ij}\right)^{\frac{1}{p}}  \\
\end{split}
$$  

&emsp;&emsp;定义：  

$$
\tilde{\boldsymbol{\beta}} = [\tilde{\boldsymbol{\beta}}_{j}],\quad \tilde{\boldsymbol{\beta}}_{j} = \left \{ \begin{array}{lr}
    \boldsymbol{\beta}_{j}, \quad\boldsymbol{\beta}_{j} > 0 \\
    1, \quad\boldsymbol{\beta}_{j} = 0
\end{array} \right.
$$  

$$
\boldsymbol{S} := \boldsymbol{P}diag(1/\tilde{\boldsymbol{\beta}})\boldsymbol{Q} \in \mathbb{R}^{n \times n}_{+}
$$  

&emsp;&emsp;则有：  

$$
\begin{split}
    \boldsymbol{S}\mathbf{1}_{n} &= \boldsymbol{P}diag(1/\tilde{\boldsymbol{\beta}})\boldsymbol{Q}\mathbf{1}_{n}=\boldsymbol{P}diag(1/\tilde{\boldsymbol{\beta}})\boldsymbol{\beta}  \\
    &= \boldsymbol{P}\boldsymbol{[\boldsymbol{\beta}_{j}/\tilde{\boldsymbol{\beta}}_{j}]_{n}} = \boldsymbol{P}\mathbf{1}_{Supp(\boldsymbol{\beta})} = \boldsymbol{P}\mathbf{1}_{n} \\
    &= \boldsymbol{\alpha}
\end{split}
$$  

&emsp;&emsp;同理可得：$\boldsymbol{S}^{T}\mathbf{1}_{n}=\boldsymbol{\gamma}$，则可以得到:  $\boldsymbol{S} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\gamma})$.  

$$
\begin{split}
    W_{p}(\boldsymbol{\alpha}, \boldsymbol{\gamma}) &= \left( \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\gamma})} \left< \boldsymbol{P},\boldsymbol{D}^{p} \right> \right)^{\frac{1}{p}} \leq \left< \boldsymbol{S},\boldsymbol{D}^{p} \right>^{\frac{1}{p}}  \\
    &= \left(  \sum_{ik}\boldsymbol{D}_{ik}^{p}\boldsymbol{S}_{ik} \right)^{\frac{1}{p}} = \left(  \sum_{ik}\boldsymbol{D}_{ik}^{p}\sum_{j}\frac{\boldsymbol{P}_{ij}\boldsymbol{Q}_{jk}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}} = \left(  \sum_{ijk}\boldsymbol{D}_{ik}^{p}\frac{\boldsymbol{P}_{ij}\boldsymbol{Q}_{jk}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}}  \\
    & \leq \left(  \sum_{ijk}(\boldsymbol{D}_{ij}+\boldsymbol{D}_{jk})^{p}\frac{\boldsymbol{P}_{ij}\boldsymbol{Q}_{jk}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}} \leq \left(  \sum_{ijk}\boldsymbol{D}_{ij}^{p}\frac{\boldsymbol{P}_{ij}\boldsymbol{Q}_{jk}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}} + \left(  \sum_{ijk}\boldsymbol{D}_{jk}^{p}\frac{\boldsymbol{P}_{ij}\boldsymbol{Q}_{jk}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}}  \\
    &= \left(  \sum_{ij}\boldsymbol{D}_{ij}^{p}\boldsymbol{P}_{ij}\sum_{k}\frac{\boldsymbol{Q}_{jk}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}} + \left(  \sum_{jk}\boldsymbol{D}_{jk}^{p}\boldsymbol{Q}_{jk}\sum_{i}\frac{\boldsymbol{P}_{ij}}{\tilde{\boldsymbol{\beta}}_{j}} \right)^{\frac{1}{p}}  \\
    &= \left(  \sum_{ij}\boldsymbol{D}_{ij}^{p}\boldsymbol{P}_{ij} \right)^{\frac{1}{p}} + \left(  \sum_{jk}\boldsymbol{D}_{jk}^{p}\boldsymbol{Q}_{jk} \right)^{\frac{1}{p}}  \\
    &= W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta}) + W_{p}(\boldsymbol{\beta},\boldsymbol{\gamma})  \\
\end{split}
$$  

&emsp;&emsp;故有：  

$$
W_{p}(\boldsymbol{\alpha},\boldsymbol{\gamma}) \leq W_{p}(\boldsymbol{\alpha},\boldsymbol{\beta})+W_{p}(\boldsymbol{\beta},\boldsymbol{\gamma})
$$  

&emsp;&emsp;综上所述，$W_{p}$可以作为概率空间$\sum_{n}$上的距离函数。  

## Ground Cost  

&emsp;&emsp;证明了$W_{p}$可以作为概率空间$\sum_{n}$上的距离函数。接下来我们就可以考虑如何定义度量矩阵 $\boldsymbol{D}$，从而生成成本矩阵$\boldsymbol{C}$, 得到成本矩阵$\boldsymbol{C}$后，我们便可以来计算分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的 Wasserstein 距离。当我们在欧式空间中考虑最优传输问题时，一种常用的生成成本矩阵$\boldsymbol{C}$的方法是 **Ground Cost**。  
&emsp;&emsp;Ground Cost 使用原始分布与目标分布的取值之差的 $L_2$ 范数来定义度量矩阵 $\boldsymbol{D}$，容易验证矩阵 $\boldsymbol{D}$ 满足度量矩阵的性质，然后使用度量矩阵的平方生成成本矩阵 $\boldsymbol{C}$，即 $\boldsymbol{C}=\boldsymbol{D}^2$，故 Ground Cost 是欧式空间中的一种 2-Wasserstein 距离。  
&emsp;&emsp;仍然考虑离散分布 $\boldsymbol{\alpha}, \boldsymbol{\beta}$，设:  

$$
\boldsymbol{\alpha} = \begin{bmatrix}
    \alpha_1 \\
    \alpha_2 \\
    \vdots \\
    \alpha_n \\
\end{bmatrix}, \quad \boldsymbol{\beta} = \begin{bmatrix}
    \beta_1 \\
    \beta_2 \\
    \vdots \\
    \beta_n \\
\end{bmatrix}
$$  

则分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 的分布列可以写成：  

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<p align="center"><font face="黑体" size=2.></font></p>
<div class="center">

||1|2|$\cdots$|n|  
|:---:|:---:|:---:|:---:|:---:|  
|p|$\alpha_1$|$\alpha_2$|$\cdots$|$\alpha_n$|  

||1|2|$\cdots$|n|  
|:---:|:---:|:---:|:---:|:---:|  
|p|$\beta_1$|$\beta_2$|$\cdots$|$\beta_n$|  
</div>  

定义度量矩阵$\boldsymbol{D}$为离散分布$\boldsymbol{\alpha},\boldsymbol{\beta}$取值之差的$L_2$范数：  

$$
\boldsymbol{D} = [\boldsymbol{D}_{ij}]_{n \times n}=[ ||i-j||_{2} ]_{n \times n} = \begin{bmatrix}
    0 & \cdots & ||1-n||_{2}  \\
    \vdots & & \vdots \\
    ||n-1||_{2} & \cdots & 0 \\
\end{bmatrix}
$$  

定义成本矩阵$\boldsymbol{C}$为度量矩阵$\boldsymbol{D}$的平方：  

$$
\boldsymbol{C} = \boldsymbol{D}^{2} = [\boldsymbol{D}_{ij}^{2}]_{n \times n} = \begin{bmatrix}
    0 & \cdots & ||1-n||_{2}^{2}  \\
    \vdots & & \vdots \\
    ||n-1||_{2}^{2} & \cdots & 0 \\
\end{bmatrix}
$$  

概率分布 $\boldsymbol{\alpha}, \boldsymbol{\beta}$ 之间的 Wasserstein 距离可以被定义为：  

$$
W_{2}(\boldsymbol{\alpha},\boldsymbol{\beta}) = L_{\boldsymbol{C}}(\boldsymbol{\alpha,\boldsymbol{\beta}})^{1/2} = \left( \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right> \right)^{\frac{1}{2}}
$$  

## Example  

&emsp;&emsp;我们用一个实际的例子来展示如何基于 Ground Cost 来计算离散分布之间的 Wasserstein 距离。我们将使用Python中专门用于OT问题的库 **POT** 来完成这个实例的计算。首先导入所需要的包：  

```python
import numpy as np
import matplotlib.pyplot as plt
import ot
```

&emsp;&emsp;假设离散分布 $\boldsymbol{\alpha}, \boldsymbol{\beta} \in \sum_{5}:=\{ \boldsymbol{x} \in \mathbb{R}^{5}_{+}: \boldsymbol{x^{T}}\mathbf{1}_{5}=1 \}$：  

$$
\boldsymbol{\alpha} = \begin{bmatrix}
    0.1 \\
    0.3 \\
    0.2 \\  
    0.1 \\
    0.3 \\
\end{bmatrix}, \quad \boldsymbol{\beta} = \begin{bmatrix}
    0.1 \\
    0.3 \\
    0.2 \\
    0.3 \\
    0.1 \\
\end{bmatrix}
$$  

&emsp;&emsp;画出离散分布 $\boldsymbol{\alpha}, \boldsymbol{\beta}$ 的概率分布直方图：  

```python
def pbar(x,y,color,title):
    plt.bar(x,y, width=1, color=color,alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability')
# 1*2 plot
def multiplot(x,y_1,y_2):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    pbar(x,y_1, color="blue", title='alpha distribution')
    plt.subplot(1,2,2)
    pbar(x,y_2, color="green", title='beta distribution')
    plt.show()

# values of probalility distribution
x = np.array([1,2,3,4,5])
# probability vector
a = np.array([0.1,0.3,0.2,0.1,0.3])
b = np.array([0.1,0.3,0.2,0.3,0.1])
# draw distribution barplot
multiplot(x,a,b)
```  

得到的图像为：  

<center>
    <img src="https://s2.loli.net/2024/03/04/amv6pIF52GMkEds.png" width=60% height=60%>
    <div align="center">Image1: 原始分布与目标分布的直方图</div>
</center>  
<br/>

&emsp;&emsp;基于 Ground Cost 我们可以定义成本矩阵 $\boldsymbol{C}$，相应的代码为：  

```python
# ground cost
def ground_cost(n):
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = i+1
            y = j+1
            C[i][j] = (x-y)**2
    return C

C = ground_cost(5)
```  

&emsp;&emsp;得到的成本矩阵$\boldsymbol{C}$为：  

$$
\boldsymbol{C} = \begin{bmatrix}
    0 & 1 & 4 & 9 & 16 \\
    1 & 0 & 1 & 4 & 9 \\
    4 & 1 & 0 & 1 & 4 \\
    9 & 4 & 1 & 0 & 1 \\
    16 & 9 & 4 & 1 & 0 \\
\end{bmatrix}
$$  

则概率分布 $\boldsymbol{\alpha}, \boldsymbol{\beta}$ 之间的 Wasserstein 距离可以被定义为：

$$
W_{2}(\boldsymbol{\alpha},\boldsymbol{\beta}) = L_{\boldsymbol{C}}(\boldsymbol{\alpha,\boldsymbol{\beta}})^{1/2} = \left( \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right> \right)^{\frac{1}{2}}
$$  

&emsp;&emsp;我们可以使用POT库的API来求解离散分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的最优传输矩阵 $P^{*}$ 以及 Wasserstein 距离 $W_{2}(\boldsymbol{\alpha},\boldsymbol{\beta})$，其代码如下：  

```python
# optimal transport matrix
P = ot.emd(a, b, C)
# wasserstein distence
wasserstein_distence = ot.emd2(a, b, C)

print(P.round(4))
print(round(np.sqrt(wasserstein_distence),4))
```  

求解结果如下：  

$$
\boldsymbol{P}^{*} = \begin{bmatrix}
    0.1 & 0 & 0 & 0 & 0 \\
    0 & 0.3 & 0 & 0 & 0 \\
    0 & 0 & 0.2 & 0 & 0 \\
    0 & 0 & 0 & 0.1 & 0 \\
    0 & 0 & 0 & 0.2 & 0.1 \\
\end{bmatrix},\quad W_{2}(\boldsymbol{\alpha},\boldsymbol{\beta})=0.4472
$$  

## Reference  

- **[1] Book: Peyré G, Cuturi M. Computational optimal transport[J]. Center for Research in Economics and Statistics Working Papers, 2017 (2017-86).**  

