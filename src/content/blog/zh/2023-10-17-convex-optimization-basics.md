---
title: 凸优化基础
tags:
    - Optimization
    - Math
---

## toc 

## 基础概念

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
&emsp;&emsp;最优化问题的基本形式(7)：  

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
- $X_{opt} = \{ x \vert x \in D,f(x)=p^{*} \}$，称为最优化问题的**最优解集(Optima Set)**.  

**凸优化问题**  
&emsp;&emsp;若在优化问题(7)中，目标函数$f(x)$为凸函数，不等式约束$m_i(x)$为凸函数，等式约束$n_j(x)$为仿射函数，则称该优化问题为凸优化问题。  

## 对偶关系  

**拉格朗日函数**  
&emsp;&emsp;原问题(7)的拉格朗日函数为：  

$$
L(x,\lambda,\eta)=f(x)+\sum_{i=1}^{M}\lambda_i m_i(x)+\sum_{j=1}^{N}\eta_j n_j(x)
$$  

$$
\lambda_i \ge 0, \quad i=1,2,\dots,M
$$  

**原问题的无约束形式**  
&emsp;&emsp;原问题(7)的无约束形式(8)为：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \max_{\lambda,\eta} \quad & L(x,\lambda,\eta)  \\
        s.t. \quad & \lambda_i \ge 0, \quad i=1,2,\dots,M \\
    \end{split}
\end{equation}
$$

&emsp;&emsp;原问题(7)与其无约束形式(8)是等价的，现证明这个结论。  
**证明:**  
&emsp;&emsp;令$h(x) = \max_{\lambda,\eta} L(x,\lambda,\eta)$,  
&emsp;&emsp;若优化变量$x$不满足某个不等式约束$m_i(x)$，即 $\exists i, m_i(x) > 0$,则有：  

$$
h(x)=\max_{\lambda,\eta} L(x,\lambda,\eta) \rightarrow +\infty
$$  

&emsp;&emsp;若优化变量$x$不满足某个等式约束$n_j(x)$，即 $\exists j,n_j(x) \ne 0$，则有：  

$$
h(x)=\max_{\lambda,\eta} L(x,\lambda,\eta) \rightarrow +\infty
$$  

&emsp;&emsp;若优化变量$x$满足所有的不等式约束$m_i(x)$与等式约束$n_j(x)$,即 $\forall i,j, m_i(x) \leq 0, n_j(x)=0$，则有:  

$$
h(x) = \max_{\lambda,\eta} L(x,\lambda,\eta) < +\infty
$$  

$$
\lambda_i = 0, \quad i=1,2,\dots,M
$$

&emsp;&emsp;设集合$S_1,S_2$分别为：  

$$
S_1 = \{ x \vert \exists i,m_i(x) > 0 \} \cup \{ x \vert \exists j, n_j(x) \ne 0 \}
$$  

$$
S_2 = \{ x \vert \forall i,j, m_i(x) \leq 0,n_j(x) = 0 \}
$$  

&emsp;&emsp;有:  

$$
S_1 \cup S_2 = \mathbb{R}^{n}, S_1 \cap S_2 = \emptyset
$$  

&emsp;&emsp;则无约束问题(8)可以写为  

$$
\min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta) \Leftrightarrow \min_{x} h(x) = \left \{
\begin{array}{l}
+\infty, & {x \in S_1}\\
c(x)<+\infty,& {x \in S_2}\\
\end{array} \right.
$$  

$$
\Rightarrow \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta) \Leftrightarrow \min_{x \in S_2} h(x)=\min_{x \in S_2} f(x)
$$  

&emsp;&emsp;故原问题(7)与其无约束形式(8)等价。  
&emsp;&emsp;证毕.  

**对偶问题**  
&emsp;&emsp;原问题(7)的拉格朗日对偶问题(9)为：  

$$
\begin{equation}
    \begin{split}
        \max_{\lambda,\eta} \min_{x} \quad & L(x,\lambda,\eta) \\
        s.t. \quad & \lambda_i \ge 0, \quad i = 1,2,\dots,M
    \end{split}
\end{equation}
$$

**弱对偶关系**  
&emsp;&emsp;原问题(7)与其对偶问题(9)满足弱对偶关系：  

$$
\max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) \leq \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

**证明:**  
&emsp;&emsp;令：  

$$
A(\lambda,\eta)=\min_{x} L(x,\lambda,\eta),\quad B(x)=\max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

$$
\because \min_{x}L(x,\lambda,\eta) \leq L(x,\lambda,\eta) \leq \max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

$$
\Rightarrow A(\lambda,\eta) \leq B(x),\quad \forall \lambda,\eta, \forall x
$$  

$$
\Rightarrow \max_{\lambda,\eta} A(\lambda,\eta) \leq \min_{x} B(x)
$$  

&emsp;&emsp;即：  

$$
\max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) \leq \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

&emsp;&emsp;证毕.  

**强对偶关系**  
&emsp;&emsp;若原问题(7)与其对偶问题(9)满足：  

$$
\max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) = \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta)
$$  

&emsp;&emsp;则称原问题(7)与其对偶问题(9)满足强对偶关系。  

**强对偶关系的几何理解**  
&emsp;&emsp;设原问题为仅有不等式约束的优化问题(10)：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \quad & f(x) \\
        s.t. \quad & m_{i}(x) \leq 0, \quad i=1,2,\dots,M \\
    \end{split}
\end{equation}
$$

&emsp;&emsp;原问题(10)的拉格朗日函数为：  

$$
L(x,\lambda)=f(x)+\sum_{i=1}^{M}\lambda_i m_i(x)
$$

&emsp;&emsp;若原问题与其对偶问题满足强对偶关系，则有：  

$$
\min_{x} \max_{\lambda} L(x,\lambda) = \max_{\lambda} \min_{x} L(x,\lambda)
$$  

&emsp;&emsp;**鞍点的定义:** 若 $\exists (\hat{w},\hat{z})$，使得：  

$$
\sup_{z} \inf_{w} f(w,z) =f(\hat{w},\hat{z})= \inf_{w} \sup_{z} f(w,z)
$$  

&emsp;&emsp;则称 $(\hat{w},\hat{z})$ 为 $f(w,z)$ 的鞍点。由于 $f(\hat{w},z)= \inf_{w} f(w,z), f(w,\hat{z})=sup_{z} f(w,z)$，则以上等式也可以写成：  

$$
\sup_{z} f(\hat{w},z)=f(\hat{w},\hat{z})=\inf_{w} f(w,\hat{z})
$$  

<center>
    <img src="https://s2.loli.net/2023/10/29/lH3OtvE7UN9A15m.jpg" width=60% height=60%>
    <div align="center">Image: 鞍点</div>
</center>  

&emsp;&emsp;结合鞍点以及强对偶关系的概率，我们可以得出结论：**原问题的拉格朗日函数存在鞍点是原问题与对偶问题满足强对偶关系的充要条件，且鞍点即为原问题与对偶问题的最优解。**  

## KKT条件  

&emsp;&emsp;下面来介绍如何判断原问题与对偶问题是否满足强对偶关系，以及如何求出相应的最优解。首先来介绍$Slater$条件。  
**$Slater$条件**  
&emsp;&emsp;若原问题(7)是凸问题，同时 $\exists x \in relint(D)$，使得约束满足：  

$$
\begin{split}
    & m_{i}(x) < 0, \quad i=1,2,\dots,M \\
    & n_{j}(x) = 0, \quad j=1,2,\dots,N \\
\end{split}
$$

&emsp;&emsp;则原问题与对偶问题满足强对偶关系。  

- 注：$relint(D)$ 表示原始凸问题的域的相对内部，即域内除了边界点以外的所有点。

&emsp;&emsp;$Slater$条件是一个**充分不必要条件**，若满足$Slater$条件，则强对偶一定成立，不满足$Slater$条件，强对偶也可能成立。大多数凸优化问题均满足$Slater$条件，即有强对偶性。  
&emsp;&emsp;若我们已知原问题与对偶问题满足强对偶关系，如何求解出原问题以及对偶问题的最优解？下面我们来介绍凸优化中一个非常经典的理论——KKT条件。  

**KKT条件**  
&emsp;&emsp;设$x^{*}$为原始问题(7)的最优解，$\lambda^{*},\eta^{*}$为对偶问题(9)的最优解，且原始问题与对偶问题满足强对偶关系，则有以下四组条件成立：  

$$
KKT条件:  \left \{
\begin{array}{l}
m_{i}(x^{*}) \leq 0, h_{j}(x^{*}) = 0 &&(primal \space feasibility)  \\
\\
\lambda^{*} \ge 0 &&(dual \space feasibility)\\
\\
\lambda^{*}m_{i}(x^{*})=0 &&(complementary \space slackness)\\
\\
\frac{\partial L(x,\lambda^{*},\eta^{*})}{\partial x} \vert_{x=x^{*}}=0 &&(stationarity)\\
\end{array} \right.
$$  

&emsp;&emsp;原问题、对偶问题的可行性条件，稳定性条件都很好理解，我们主要来推导一下互补松弛条件是如何得到的。  
&emsp;&emsp;**证明:**  
&emsp;&emsp;记 $p^{*}$ 为原问题(7)的最优值，$d^{*}$ 为对偶问题(9)的最优值，即：  

$$
p^{*}= \min_{x} f(x)=f(x^{*})
$$  

$$
d^{*}=\max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) \triangleq \max_{\lambda,\eta} g(\lambda,\eta)=g(\lambda^{*},\eta^{*})
$$  

&emsp;&emsp;通过分析可得：  

$$
\begin{align*}
    d^{*} &= \max_{\lambda,\eta} \min_{x} L(x,\lambda,\eta) = \min_{x} \max_{\lambda,\eta} L(x,\lambda,\eta) = \min_{x} L(x,\lambda^{*},\eta^{*}) \\
    &\leq L(x^{*},\lambda^{*},\eta^{*}) = f(x^{*})+\sum_{i=1}^{M}\lambda_{i}^{*}m_{i}(x^{*}) + \sum_{j=1}^{N}\eta_{j}^{*}n_{j}(x^{*}) \\
    &\leq f(x^{*}) = p^{*}
\end{align*}
$$  

&emsp;&emsp;由原问题与对偶问题满足强对偶关系可知：$p^{*}=d^{*}$，则以上式子中的小于等于号均取等号，故有：  

$$
\sum_{i=1}^{M}\lambda_{i}^{*}m_{i}(x^{*})=0 \Rightarrow \lambda^{*}m_{i}(x^{*})=0, \forall i
$$  

&emsp;&emsp;互补松弛条件成立，证毕.  

&emsp;&emsp;对于**一般的原问题**，KKT条件是 $x^{*},\lambda^{*},\eta^{*}$ 为最优解的**必要条件**，即只要 $x^{*},\lambda^{*},\eta^{*}$ 为原问题及对偶问题的最优解，则一定满足KKT条件。  
&emsp;&emsp;对于**原问题为凸问题**，KKT条件是 $x^{*},\lambda^{*},\eta^{*}$ 为最优解的**充要条件**，即只要 $x^{*},\lambda^{*},\eta^{*}$ 满足KKT条件，则其一定为原问题及对偶问题的最优解，反过来，只要 $x^{*},\lambda^{*},\eta^{*}$ 为原问题及对偶问题的最优解，则其一定满足KKT条件。

## 凸二次规划  

&emsp;&emsp;凸二次规划问题的基本形式为：  

$$
\begin{equation}
    \begin{split}
        \min_{x} \quad & \frac{1}{2} x^{T}Px+q^{T}x+r \\
        s.t. \quad &  Gx \leq h \\
        & Ax=b  \\
    \end{split}
\end{equation}
$$

&emsp;&emsp;其中，$x \in \mathbb{R}^{n}, P \in S_{+}^{n}$，为对称正定矩阵；$q,r \in \mathbb{R}^{n}; G \in \mathbb{R}^{M \times n}, h \in \mathbb{R}^{M}; A \in \mathbb{R}^{N \times n}, b \in \mathbb{R}^{N}$.  
&emsp;&emsp;结论：**凸二次规划问题满足强对偶关系。**  

## 参考  

**[1] Video: bilibili,欧拉的费米子,凸优化理论-中科大凌青**  
**[2] Blog: 知乎,Lauer,[凸优化笔记6]-拉格朗日对偶、KKT条件**