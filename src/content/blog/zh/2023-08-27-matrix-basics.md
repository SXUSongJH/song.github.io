---
title: 矩阵-基与坐标
tags:
    - Matrix
    - Math
---

## toc

&emsp;&emsp;在前面的章节，我们介绍了线性空间的概念，线性空间是一个抽象的概念，但在实际应用中，出于对向量运算的需求，我们通常更需要标准线性空间中的向量。为了将抽象线性空间中的向量映射到标准线性空间，我们引入了基向量的概念。抽象向量沿着基向量展开后得到坐标向量，我们用坐标向量来表示映射后的抽象向量。  

## 有限维线性空间基坐标  

&emsp;&emsp;设$V$是数域$\mathbb{F}$上的线性空间，若有正整数$n$，及$V$中的向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$，使得:  

1. **线性无关性**: 向量组$\{ \alpha_1,\alpha_2,\dots,\alpha_n \}$为线性无关向量组.  
2. **线性生成性**: $\forall \alpha \in V$，均可由向量组$\{ \alpha_1,\alpha_2,\dots,\alpha_n \}$线性表示.  

$$
\alpha = \alpha_{1} k_1 + \alpha_{2} k_2 + \dots + \alpha_{n} k_n = \begin{bmatrix}
    \alpha_1,\alpha_2,\dots,\alpha_n \\
\end{bmatrix} \begin{bmatrix}
    k_1 \\
    k_2 \\
    \vdots \\
    k_n \\
\end{bmatrix} = Ak
$$  

&emsp;&emsp;则称$V$为$n$维线性空间，向量组$\{ \alpha_1,\alpha_2,\dots,\alpha_n \}$称为$V$中的一个基(坐标系)，$k \in \mathbb{F}^n$称为$\alpha \in V$，沿着该基的坐标向量.  

注：  

$$
\color{green} \begin{bmatrix}
    抽 \\
    象 \\
    向 \\
    量 \\
\end{bmatrix} = \begin{bmatrix}
    基矩阵 \\
\end{bmatrix} \begin{bmatrix}
    坐 \\
    标 \\
    向 \\
    量 \\
\end{bmatrix}
$$  

## 关于基向量组的定理  

**定理1(基向量个数的唯一性)** 设向量组$\{\alpha_1,\alpha_2,\dots,\alpha_m\}$及$\{\beta_1,\beta_2,\dots,\beta_s\}$分别是线性空间$V$的两个基，则有$m=s$.  
**证明:**  
&emsp;&emsp;从线性空间中任意取$p$个向量组成一个向量组$\{v_1,v_2,\dots,v_p\}$，要求$m \leq p, s \leq p$.  
&emsp;&emsp;由基向量组$\{\alpha_1,\alpha_2,\dots,\alpha_m\}$的定义可知：  
&emsp;&emsp;1. 向量组$\{\alpha_1,\alpha_2,\dots,\alpha_m\}$为线性无关向量组.  
&emsp;&emsp;2. 对$\forall v \in \{v_1,v_2,\dots,v_p\}, v$均可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_m\}$线性表示.  
&emsp;&emsp;$\Rightarrow$向量组$\{\alpha_1,\alpha_2,\dots,\alpha_m\}$是向量组$\{v_1,v_2,\dots,v_p\}$的极大线性无关组.  
&emsp;&emsp;同理可知：向量组$\{\beta_1,\beta_2,\dots,\beta_s\}$也为向量组$\{v_1,v_2,\dots,v_p\}$的极大线性无关组.  
&emsp;&emsp;由向量组的极大线性无关组中向量个数的唯一性可知: $m=s$.  

**定理2** $\color{green}{基实现了抽象线性空间到标准线性空间的一一映射.}$  
**证明:**  
&emsp;&emsp;设$V$为$n$维线性空间，向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$是$V$的一个基.  
&emsp;&emsp;设由基向量组实现的映射为:  

$$
\sigma: V \longrightarrow \mathbb{F}^{n}
$$  

$$
v \longmapsto k = \begin{bmatrix}
    k_1 \\
    k_2 \\
    \vdots \\
    k_n \\
\end{bmatrix}
$$  

&emsp;&emsp;$k \in \mathbb{F}^{n}$是$v$在基下的坐标向量.  
&emsp;&emsp;现需要验证映射$\sigma$满足一一映射的两个条件.  
&emsp;&emsp;(1) 验证对$\forall k \in \mathbb{F}^n, \exists v \in V$，使得$v=\alpha_1k_1+\alpha_2k_2+\dots+\alpha_nk_n$.  
&emsp;&emsp;$\{\alpha_1,\alpha_2,\dots,\alpha_n\} \in V$. 由线性空间$V$对加法与数乘封闭的性质可知:  
&emsp;&emsp;$\exists v \in V$使得$v = \alpha_1k_1+\alpha_2k_2+\dots+\alpha_nk_n$.  
&emsp;&emsp;(2) 验证若$\sigma(v)=\sigma(v_0)=k$，则有$v=v_0$.  
&emsp;&emsp;$\sigma(v)=k \Leftrightarrow v=\alpha_1k_1+\alpha_2k_2+\dots+\alpha_nk_n$.  
&emsp;&emsp;$\sigma(v_0)=k \Leftrightarrow v_0=\alpha_1k_1+\alpha_2k_2+\dots+\alpha_nk_n$.  
&emsp;&emsp;$\Rightarrow v_0 = v$.  
&emsp;&emsp;综上所述映射$\sigma$为一一映射.  

**注：一一映射的定义**  
&emsp;&emsp;设映射$\sigma: S_1 \rightarrow S_2$满足:  
&emsp;&emsp;(1)满射：对$\forall s_2 \in S_2, \exists s_1 \in S_1, \sigma(s_1)=s_2 .$  
&emsp;&emsp;(2)单射：若$\sigma(s_1)=\sigma(s^{*}_{1})$,则$s_1=s^{*}_{1}.$  
&emsp;&emsp;则称映射$\sigma$为集合$S_1$与$S_2$之间的一一映射.  

## 标准线性空间$\mathbb{R}^n$的标准基与一般基  

### 标准基

&emsp;&emsp;标准线性空间$\mathbb{R}^n$的标准基：  

$$
e_1=\begin{bmatrix}
    1 \\
    0 \\
    \vdots \\
    0 \\
\end{bmatrix}, e_2 = \begin{bmatrix}
    0 \\
    1 \\
    \vdots \\
    0
\end{bmatrix}, \dots, e_n = \begin{bmatrix}
    0 \\
    0 \\
    \vdots \\
    1
\end{bmatrix}
$$  

**证明:**  
&emsp;&emsp;(1)先证明标准基向量组的线性无关性：  
&emsp;&emsp;令 $I_n = \begin{bmatrix}
    e_1,e_2,\dots,e_n
\end{bmatrix}$，有线性方程组$I_n x = 0$.  
&emsp;&emsp;该方程组仅有$x=0$唯一解，故标准基向量组$\{e_1,e_2,\dots,e_n\}$线性无关.  
&emsp;&emsp;(2)再证标准基向量组的线性生成性：  
&emsp;&emsp;对$\forall v \in \mathbb{R}^n$，判断线性方程组$I_n x = v$是否有解.  
&emsp;&emsp;$I_n x = v \Rightarrow x = v \Rightarrow$方程有解$\Rightarrow v$可由标准基向量组$\{e_1,e_2,\dots,e_n\}$线性表示.  
&emsp;&emsp;综上所述，向量组$\{e_1,e_2,\dots,e_n\}$可作为标准线性空间$\mathbb{R}^n$的基向量组.  

### 一般基  

&emsp;&emsp;$\alpha_1,\alpha_2,\dots,\alpha_n \in \mathbb{R}^n$，向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$构成标准线性空间$\mathbb{R}^n$的一组基的充要条件为：向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$的秩为$n$.  

**证明:**  
&emsp;&emsp;向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$的秩为$n$ $\Rightarrow$ 向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$线性无关  
&emsp;&emsp;令$A=\begin{bmatrix}
    \alpha_1,\alpha_2,\dots,\alpha_n
\end{bmatrix}$，有$rank(A) = n$.  
&emsp;&emsp;对$\forall \beta \in \mathbb{R}^n$，判断矩阵方程$Ax=\beta$是否有解.  
&emsp;&emsp;$\because rank(A)=rank([A,b])=n$，$\therefore$方程$Ax=\beta$有解.  

$\color{green}{注：线性方程组 Ax=\beta 的几何语言：在n维线性空间中，将向量\beta 沿着矩阵A的列向量组所构成的基展开.}$  

## 多项式函数空间作为线性空间的基


&emsp;&emsp;在第一节我们已经说明函数空间$V=\mathcal{F}(I,\mathbb{F}^n)$可以作为线性空间。多项式是函数的一种形式，我们可以定义以多项式为元素的线性空间：  

$$
\mathbb{F}[x]=\{以x为自变量，以数域\mathbb{F}中的数为系数的多项式\}=\{f=a_0+a_1x+a_2x^2+\dots \vert a_i \in \mathbb{F}, i =1,2,\dots\}
$$  

&emsp;&emsp;通过分析我们可以得知这是一个无限维的线性空间，这里我们不讨论无限维线性空间的基，通过对$x$的次数添加限制，我们可以将这个线性空间变为有限维：  

$$
\mathbb{F}_n[x]=\{以x为自变量，以数域\mathbb{F}中的数为系数,次数小于n的多项式\}=\{f=a_0+a_1x++\dots+a_{n-1}x^{n-1} \vert a_i \in \mathbb{F}, i =1,2,\dots,n-1\}
$$  

&emsp;&emsp;$\mathbb{F}_n[x]$是一个$n$维线性空间，取$\mathbb{F}=\mathbb{R}$，接下来我们来讨论$\mathbb{R}_n[x]$的基向量组。  
&emsp;&emsp;在高等代数中，我们知道多项式函数空间中的元素与标准线性空间中的元素一一对应，对$\forall f \in \mathbb{R}_n[x]$，有：

$$
f=a_0+a_1x++\dots+a_{n-1}x^{n-1}=\begin{bmatrix}
    1,x,\dots,x^{n-1} \\
\end{bmatrix}\begin{bmatrix}
    a_0 \\
    a_1 \\
    \vdots \\
    a_{n-1} \\
\end{bmatrix}, f \rightarrow \begin{bmatrix}
    a_0 \\
    a_1 \\
    \vdots \\
    a_{n-1} \\
\end{bmatrix}
$$  

&emsp;&emsp;可以取$\{1,x,\dots,x^{n-1}\}$为$\mathbb{R}_n[x]$的一组基，以下是证明$\{1,x,\dots,x^{n-1}\}$可以作为$\mathbb{R}_n[x]$的基.  
**证明:**  
&emsp;&emsp;(1)线性无关性  
&emsp;&emsp;若$a_0+a_1x+\dots+a_{n-1}x^{n-1}=0$，带入$x=1,2,\dots,n$，得：  

$$
\begin{bmatrix}
    1^{0}&1^{1}&\dots&1^{n-1} \\
    2^{0}&2^{1}&\dots&2^{n-1} \\
    \vdots&\vdots&&\vdots \\
    n^{0}&n^{1}&\dots&n^{n-1} \\
\end{bmatrix}\begin{bmatrix}
    a_0 \\
    a_1 \\
    \vdots \\
    a_{n-1} \\
\end{bmatrix}=\begin{bmatrix}
    0 \\
    0 \\
    \vdots \\
    0
\end{bmatrix}
$$  

&emsp;&emsp;令$A=\begin{bmatrix}
    1^{0}&1^{1}&\dots&1^{n-1} \\
    2^{0}&2^{1}&\dots&2^{n-1} \\
    \vdots&\vdots&&\vdots \\
    n^{0}&n^{1}&\dots&n^{n-1} \\
\end{bmatrix},a=\begin{bmatrix}
    a_0 \\
    a_1 \\
    \vdots \\
    a_{n-1} \\
\end{bmatrix}$，$det(A)$为范德蒙行列式,且$det(A)\neq 0$，故$rank(A)=n$.  
&emsp;&emsp;$rank(A)=n \Rightarrow$ 方程$Aa=0$只有零解,即 $a_i=0, i=1,2,\dots,n-1$.  
&emsp;&emsp;故$\{1,x,\dots,x^{n-1}\}$线性无关.  
&emsp;&emsp;(2)线性生成性  
&emsp;&emsp;由多项式的定义可知，$\mathbb{R}_n[x]$中的元素均可由$\{1,x,\dots,x^{n-1}\}$线性表示.  
&emsp;&emsp;综上所述：$\{1,x,\dots,x^{n-1}\}$可以作为$\mathbb{R}_n[x]$的一组基.  

注：$\{1,x,\dots,x^{n-1}\}$对应于$\mathbb{R}^n$中的标准基  
&emsp;&emsp;设$\sigma$为以$\{1,x,\dots,x^{n-1}\}$为基时，从$\mathbb{R}_n[x]$到$\mathbb{R}^n$的映射，有：  

$$
1=\begin{bmatrix}
    1,x,\dots,x^{n-1} \\
\end{bmatrix}\begin{bmatrix}
    1 \\
    0 \\
    \vdots \\
    0 \\
\end{bmatrix}, \sigma: 1 \rightarrow \begin{bmatrix}
    1 \\
    0 \\
    \vdots \\
    0 \\
\end{bmatrix}
$$  

$$
x=\begin{bmatrix}
    1,x,\dots,x^{n-1} \\
\end{bmatrix}\begin{bmatrix}
    0 \\
    1 \\
    \vdots \\
    0 \\
\end{bmatrix}, \sigma: x \rightarrow \begin{bmatrix}
    0 \\
    1 \\
    \vdots \\
    0 \\
\end{bmatrix}
$$  

$$
\vdots
$$  

$$
x^{n-1}=\begin{bmatrix}
    1,x,\dots,x^{n-1} \\
\end{bmatrix}\begin{bmatrix}
    0 \\
    0 \\
    \vdots \\
    1 \\
\end{bmatrix}, \sigma: x^{n-1} \rightarrow \begin{bmatrix}
    0 \\
    0 \\
    \vdots \\
    1 \\
\end{bmatrix}
$$






