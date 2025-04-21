---
title: 矩阵-线性映射
tags:
    - Matrix
    - Math
---

## toc

## 线性映射的定义  

&emsp;&emsp;设$V_1,V_2$是数域$\mathbb{F}$上的线性空间，有映射 $\sigma: V_1 \rightarrow V_2$，如果$\sigma$满足：  
(1) **加法关系:** 对$\forall e_1,e_2 \in V_1, \sigma(e_1+e_2) = \sigma(e_1)+\sigma(e_2) \in V_2$.  
(2) **数乘关系:** d对$\forall e \in V_1, k \in \mathbb{F}, \sigma(ek)=\sigma(e)k$  
则称映射$\sigma$是$V_1$到$V_2$的**线性映射**。特别地，若有$V_1=V_2=V$，则称映射$\sigma$为$V$上的**线性变换**。  

**注:** 若线性映射$\sigma: V_1 \rightarrow V_2$ 是可逆映射(一一映射)，则称$\sigma$为**线性同构**。  

## 线性映射的实例  

### 例1 线性与非线性映射  

**非线性映射的实例**
&emsp;&emsp;设线性空间$V_1,V_2=\mathbb{R}^2$，有映射：  

$$
\mathcal{A}: \begin{bmatrix}
    x_1 \\
    x_2 \\
\end{bmatrix} \mapsto \begin{bmatrix}
    x_1+x_2 \\
    x_1x_2 \\
\end{bmatrix}
$$  

则映射$\mathcal{A}$为$V_1$到$V_2$的非线性映射。  
**证明:**  
&emsp;&emsp;取$e_1,e_2 \in V_1$，其中:  

$$
e_1=e_2=\begin{bmatrix}
    1 \\
    1 \\
\end{bmatrix},e_1+e_2=\begin{bmatrix}
    2 \\
    2 \\
\end{bmatrix}
$$  

$$
\mathcal{A}(e_1+e_2)=\begin{bmatrix}
    2+2 \\
    2 \times 2 \\
\end{bmatrix}=\begin{bmatrix}
    4 \\
    4 \\
\end{bmatrix}
$$  

$$
\mathcal{A}(e_1)+\mathcal{A}(e_1)=\begin{bmatrix}
    1+1 \\
    1 \times 1 \\
\end{bmatrix}+\begin{bmatrix}
    1+1 \\
    1 \times 1 \\
\end{bmatrix}=\begin{bmatrix}
    4 \\
    2 \\
\end{bmatrix}
$$  

&emsp;&emsp;$\because \mathcal{A}(e_1+e_2) \ne \mathcal{A}(e_1)+\mathcal{A}(e_1)$，故映射$\mathcal{A}$不满足加法关系，其不是$V_1$到$V_2$上的线性映射。  

**线性映射的实例**  
&emsp;&emsp;设线性空间$V_1=\mathbb{R}^3,V_2=\mathbb{R}^2$，有映射：  

$$
\mathcal{B}: \begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3 \\
\end{bmatrix} \mapsto \begin{bmatrix}
    x_1+x_2-x_3 \\
    \frac{1}{2}x_1-3x_2
\end{bmatrix}
$$  

则线性映射$\mathcal{B}$为$V_1$到$V_2$上的线性映射。  
**证明:**  
&emsp;&emsp;(1) 先验证映射$\mathcal{B}$满足线性映射的加法关系:  
&emsp;&emsp;设$\forall \alpha,\beta \in V_1, \alpha = \begin{bmatrix}
    \alpha_1 ,\alpha_2,\alpha_3
\end{bmatrix}^T,\beta = \begin{bmatrix}
    \beta_1,\beta_2,\beta_3
\end{bmatrix}^T$  

$$
\mathcal{B}(\alpha)=\begin{bmatrix}
    \alpha_1+\alpha_2-\alpha_3 \\
    \frac{1}{2}\alpha_1-3\alpha_2 \\
\end{bmatrix},\mathcal{B}(\beta)=\begin{bmatrix}
    \beta_1+\beta_2-\beta_3 \\
    \frac{1}{2}\beta_1-3\beta_2 \\
\end{bmatrix}
$$  

$$
\mathcal{B}(\alpha)+\mathcal{B}(\beta)=\begin{bmatrix}
    \alpha_1+\beta_1+\alpha_2+\beta_2-(\alpha_3+\beta_3) \\
    \frac{1}{2}(\alpha_1+\beta_1)-3(\alpha_2+\beta_2) \\
\end{bmatrix}
$$  

$$
\alpha+\beta=\begin{bmatrix}
    \alpha_1+\beta_1 \\
    \alpha_2+\beta_2 \\
    \alpha_3+\beta_3 \\
\end{bmatrix},\mathcal{B}(\alpha+\beta)=\begin{bmatrix}
    \alpha_1+\beta_1+\alpha_2+\beta_2-(\alpha_3+\beta_3) \\
    \frac{1}{2}(\alpha_1+\beta_1)-3(\alpha_2+\beta_2) \\
\end{bmatrix}
$$  

$$
\Rightarrow \mathcal{B}(\alpha+\beta)=\mathcal{B}(\alpha)+\mathcal{B}(\beta)
$$

&emsp;&emsp;故映射$\mathcal{B}$满足线性映射的加法关系。  
&emsp;&emsp;(2) 再验证映射$\mathcal{B}$满足线性映射的数乘关系:  
&emsp;&emsp;设$\forall \alpha \in V_1,k \in \mathbb{F},\alpha = \begin{bmatrix}
    \alpha_1 ,\alpha_2,\alpha_3 \\
\end{bmatrix}^T$,有:  

$$
\alpha k = \begin{bmatrix}
    \alpha_1 k \\
    \alpha_2 k \\
    \alpha_3 k \\
\end{bmatrix}, \mathcal{B}(\alpha k)=\begin{bmatrix}
    \alpha_1 k+\alpha_2 k-\alpha_3 k \\
    \frac{1}{2}\alpha_1 k-3\alpha_2 k \\
\end{bmatrix}=\begin{bmatrix}
    (\alpha_1+\alpha_2-\alpha_3)k \\
    (\frac{1}{2}\alpha_1-3\alpha_2) k \\
\end{bmatrix}
$$  

$$
\mathcal{B}(\alpha)=\begin{bmatrix}
    \alpha_1+\alpha_2-\alpha_3 \\
    \frac{1}{2}\alpha_1-3\alpha_2 \\
\end{bmatrix},\mathcal{B}(\alpha)k=\begin{bmatrix}
    (\alpha_1+\alpha_2-\alpha_3)k \\
    (\frac{1}{2}\alpha_1-3\alpha_2) k \\
\end{bmatrix}
$$  

$$
\Rightarrow \mathcal{B}(\alpha k)=\mathcal{B}(\alpha)k
$$  

&emsp;&emsp;故映射$\mathcal{B}$满足线性映射的数乘关系。  
&emsp;&emsp;综上所述，映射$\mathcal{B}$为$V_1$到$V_2$上的线性映射。  

### 例2 矩阵与标准线性空间之间的线性映射的等同性  
 
&emsp;&emsp;给定矩阵 $A \in \mathbb{F}^{m \times n}, x \in \mathbb{F}^n$，则矩阵$A$可以作为线性映射$\sigma_{A}$：  

$$
\begin{align*}
    \sigma_{A}: &\mathbb{F}^{n} \rightarrow \mathbb{F}^{m} \\
    &x \mapsto y=Ax
\end{align*}
$$  

&emsp;&emsp;若我们已知有线性映射 $\sigma: \mathbb{F}^{n} \rightarrow \mathbb{F}^{m}$，如例1中的线性映射$\mathcal{B}$，能否求得相应的矩阵$A$?  
**解:**  
&emsp;&emsp;记标准线性空间$\mathbb{F}^n$的标准基为：$e_1,e_2,\dots,e_n$，可以构造矩阵：  

$$
\sigma(\begin{bmatrix}
    e_1,e_2,\dots,e_n \\
\end{bmatrix})=\begin{bmatrix}
    \sigma(e_1),\sigma(e_2),\dots,\sigma(e_n) \\
\end{bmatrix} \triangleq A \in \mathbb{F}^{n \times m}
$$  

&emsp;&emsp;对$\forall x \in \mathbb{F}^{n}$，将$x$沿着标准基展开：  

$$
x = \begin{bmatrix}
    e_1,e_2,\dots,e_n \\
\end{bmatrix}x=e_1x_1+e_2x_2,\dots,e_nx_n
$$  

$$
\begin{align*}
    \sigma(x)&=\sigma(e_1x_1+e_2x_2,\dots,e_nx_n)=\sigma(e_1)x_1+\sigma(e_2)x_2+\dots+\sigma(e_n)x_n \\
    &= \begin{bmatrix}
    \sigma(e_1),\sigma(e_2),\dots,\sigma(e_n) \\
\end{bmatrix}x=Ax
\end{align*}
$$  

&emsp;&emsp;因此矩阵$A$与线性映射$\sigma$具有等同性.  

## 线性映射的矩阵表示  

### 定义

&emsp;&emsp;设有标准线性空间$V=\mathbb{F}^n,W=\mathbb{F}^m$，给定线性映射：  

$$
\begin{align*}
    \sigma: &V \rightarrow W \\
    & v \mapsto w
\end{align*}
$$

&emsp;&emsp;选取$V$的基向量$v_1,v_2,\dots,v_n$，作为**入口基**，$W$的基向量$w_1,w_2,\dots,w_m$作为**出口基**，记$V$中第$j$个入口基向量$v_j$在$W$中的象$\sigma(v_j)$在出口基下的坐标表示为$a_j = \begin{bmatrix}
    a_{1j},a_{2j},\dots,a_{mj} \\
\end{bmatrix}^T$，即：  

$$
\sigma(v_j)=\begin{bmatrix}
    w_1,w_2,\dots,w_n \\
\end{bmatrix}\begin{bmatrix}
    a_{1j} \\
    a_{2j} \\
    \vdots \\
    a_{mj} \\
\end{bmatrix}
$$

&emsp;&emsp;将所有入口基的象在出口基下的坐标拼成矩阵$A$:  

$$
A=\begin{bmatrix}
    \sigma(v_1),\sigma(v_2),\dots,\sigma(v_n) \\
\end{bmatrix}=\begin{bmatrix}
    a_{11} & a_{21} & \dots & a_{1n} \\
    a_{21} & a_{22} & \dots & a_{2n} \\
    \vdots & \vdots &  & \vdots \\
    a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
$$  

&emsp;&emsp;则有：  

$$
\sigma(\begin{bmatrix}
    v_1,v_2,\dots,v_n
\end{bmatrix})=\begin{bmatrix}
    w_1,w_2,\dots,w_m
\end{bmatrix}A
$$

&emsp;&emsp;则称矩阵$A$为线性映射$\sigma$在入口基$v_i,i=1,\dots,n$和出口基$w_j,j=1,\dots,m$下的矩阵表示。  

**注:**  

$$
\color{green} \begin{bmatrix}
    线性 \\
    映射 \\
\end{bmatrix}\begin{bmatrix}
    入口基 \\
    矩阵  \\
\end{bmatrix}=\begin{bmatrix}
    出口基 \\
    矩阵  \\
\end{bmatrix}\begin{bmatrix}
    线性映射 \\
    矩阵表示  \\
\end{bmatrix}
$$  

### 定理(线性映射的坐标计算)  

&emsp;&emsp;设有$n$维线性空间$V$和$m$维线性空间$W$，$v_1,v_2,\dots,v_n$为$V$的一组基向量，$w_1,w_2,\dots,w_m$为$W$的一组基向量，映射$\sigma$为$V$到$W$上的线性映射，矩阵$A$为线性映射$\sigma$在入口基$\{v_i \vert i=1,2,\dots,n\}$与出口基$\{w_i \vert i=1,2,\dots,m\}$下的矩阵表示。  
&emsp;&emsp;给定$\forall v \in V$，其沿着基向量组展开的坐标为$x$，即：  

$$
v=\begin{bmatrix}
    v_1,v_2,\dots,v_n \\
\end{bmatrix}x=\begin{bmatrix}
    v_1,v_2,\dots,v_n \\
\end{bmatrix}\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_m \\
\end{bmatrix}
$$  

则经过线性映射$\sigma$后，$\sigma(v) \in W$在沿着基向量组展开的坐标为$Ax$，即：  

$$
\sigma(v) = \begin{bmatrix}
    w_1,w_2,\dots,w_m \\
\end{bmatrix}Ax
$$  

**证明:**  
&emsp;&emsp;由题意可知：  

$$
v=\begin{bmatrix}
    v_1,v_2,\dots,v_n \\
\end{bmatrix}\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_m \\
\end{bmatrix}=v_1x_1+v_2x_2+\dots+v_nx_n
$$  

&emsp;&emsp;则有  

$$
\begin{align*}
    \sigma(v)&=\sigma(v_1x_1+v_2x_2+\dots+v_nx_n)=\sigma(v_1)x_1+\sigma(v_2)x_2+\dots+\sigma(v_n)x_n \\
    &=\begin{bmatrix}
        \sigma(v_1),\sigma(v_2),\dots,\sigma(v_n) \\
    \end{bmatrix}=(\begin{bmatrix}
        w_1,w_2,\dots,w_m
    \end{bmatrix}A)x \\
    &=\begin{bmatrix}
        w_1,w_2,\dots,w_m
    \end{bmatrix}(Ax)
\end{align*}
$$  

&emsp;&emsp;故$\sigma(v)$沿着基向量组$\{w_i \vert i=1,2,\dots,m\}$展开后的坐标为$Ax$.  

<span style="color: green;">结论：基向量组将抽象的线性空间映射为标准线性空间，在基向量组的表示下，原本抽象线性空间之间的线性映射可以被表示为具体的矩阵。</span>   

<center>
    <img src="https://s2.loli.net/2023/10/05/GTHiFnjpR4v2CQr.jpg
    " width=60% height=60%>
    <div align="center">Image1: 线性映射的矩阵表示</div>
</center>  

## 线性映射矩阵表示的实例  

### 例1 微分算子的矩阵表示  

&emsp;&emsp;设线性空间$V=\mathbb{R}_{4}[x], W=\mathbb{R}_{3}[x]$($\mathbb{R}_{n}[x]$表示$n$维多项式函数空间)，微分算子$\sigma$为$V$到$W$上的线性映射，求$\sigma$在标准基向量组下的矩阵表示.  

**解:**  
&emsp;&emsp;$V$的标准基向量组为：$\{1,x,x^2,x^3\}$(入口基)，$W$的标准基向量组为$\{1,x,x^2\}$，则有：  

$$
\begin{align*}
    \sigma(\begin{bmatrix}
        1,x,x^2,x^3 \\
    \end{bmatrix})&=\begin{bmatrix}
        \sigma(1),\sigma(x),\sigma(x^2),\sigma(x^3) \\
    \end{bmatrix} \\
    &=\begin{bmatrix}
        0,1,2x,3x^2 \\
    \end{bmatrix}=\begin{bmatrix}
        1,x,x^2 \\
    \end{bmatrix}A
\end{align*}
$$  

$$
A=\begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 2 & 0 \\
    0 & 0 & 0 & 3 \\
\end{bmatrix}
$$  

&emsp;&emsp;故矩阵$A$即为微分算子$\sigma$在$V$与$W$的标准基向量组下的矩阵表示.  

**应用:**  

&emsp;&emsp;$v = \frac{1}{2}x^3+5x \in V$，将$v$沿着$V$的标准基向量组$\{1,x,x^2,x^3\}$展开得其坐标:  

$$
v=\frac{1}{2}x^3+5x=\begin{bmatrix}
    1,x,x^2,x^3 \\
\end{bmatrix}\begin{bmatrix}
    0 \\
    5 \\
    0 \\
    \frac{1}{2} \\
\end{bmatrix}
$$  

&emsp;&emsp;则$\sigma(v)$在$W$的标准基向量组$\{1,x,x^2\}$下的坐标表示为:  

$$
A\begin{bmatrix}
    0 \\
    5 \\
    0 \\
    \frac{1}{2} \\
\end{bmatrix}=\begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 2 & 0 \\
    0 & 0 & 0 & 3 \\
\end{bmatrix}\begin{bmatrix}
    0 \\
    5 \\
    0 \\
    \frac{1}{2} \\
\end{bmatrix}=\begin{bmatrix}
    5 \\
    0 \\
    \frac{3}{2} \\
\end{bmatrix}
$$  

&emsp;&emsp;故对$v$求微分的结果为：  

$$
\sigma(v)=\begin{bmatrix}
    1,x,x^2,x^3 \\
\end{bmatrix}\begin{bmatrix}
    5 \\
    0 \\
    \frac{3}{2} \\
\end{bmatrix}=\frac{3}{2}x^2+5
$$  

### 例2 旋转变换的矩阵表示  

&emsp;&emsp;欧几里得空间中的某一物体绕固定轴旋转$\theta$，求该变换的矩阵表示。  
**解:**  
&emsp;&emsp;设$V=W$为欧几里得空间，映射$\sigma$为旋转变换，易知$\sigma$为线性变换.
&emsp;&emsp;设欧几里得空间中的一组标准正交基向量为$e_1,e_2,e_3$，

$$
e_1=\begin{bmatrix}
    1 \\
    0 \\
    0 \\
\end{bmatrix}, e_2=\begin{bmatrix}
    0 \\
    1 \\
    0 \\
\end{bmatrix}, e_3=\begin{bmatrix}
    0 \\
    0 \\
    1 \\
\end{bmatrix}
$$

&emsp;&emsp;其中$e_3$为旋转变换所固定的轴，则$e_1,e_2$为与旋转轴所垂直的平面的一组正交基，可以$e_1,e_2,e_3$的方向为$x,y,z$轴建立坐标系. 向量组$e_1,e_2,e_3$既为入口基也为出口基. 
&emsp;&emsp;旋转变换$\sigma$作用于基向量组$e_1,e_2,e_3$时，$e_3$并不会改变，$e_1,e_2$在其所在的平面上旋转$\theta$，旋转变换可用图2表示。  

<center>
    <img src="https://s2.loli.net/2023/10/05/hU9ixmNgfXRtWze.jpg
    " width=40% height=40%>
    <div align="center">Image2: 旋转变换</div>
</center>  

&emsp;&emsp;由几何知识可得：  

$$
\sigma(e_1)=\begin{bmatrix}
    cos\theta \\
    sin\theta \\
    0 \\
\end{bmatrix},\sigma(e_2)=\begin{bmatrix}
    -sin\theta \\
    cos\theta \\
    0 \\
\end{bmatrix},\sigma(e_3)=\begin{bmatrix}
    0 \\
    0 \\
    1 \\
\end{bmatrix}
$$  

&emsp;&emsp;故有:  

$$
\begin{align*}
    \sigma(\begin{bmatrix}
    e_1,e_2,e_3 \\
\end{bmatrix})&=\begin{bmatrix}
    \sigma(e_1),\sigma(e_2),\sigma(e_3)
\end{bmatrix}=\begin{bmatrix}
    cos\theta & -sin\theta & 0 \\
    sin\theta & cos\theta & 0 \\
    0 & 0 & 1 \\
\end{bmatrix} \\
    &=\begin{bmatrix}
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1 \\
    \end{bmatrix}\begin{bmatrix}
    cos\theta & -sin\theta & 0 \\
    sin\theta & cos\theta & 0 \\
    0 & 0 & 1 \\
\end{bmatrix}=\begin{bmatrix}
    e_1,e_2,e_3 \\
\end{bmatrix}A
\end{align*}
$$  

$$
A=\begin{bmatrix}
    cos\theta & -sin\theta & 0 \\
    sin\theta & cos\theta & 0 \\
    0 & 0 & 1 \\
\end{bmatrix}
$$  

&emsp;&emsp;矩阵$A$即为欧几里得空间中旋转变换$\sigma$在标准正交基下的矩阵表示.  

### 例3 镜面反射的矩阵表示

&emsp;&emsp;欧几里得空间中的某一物体对固定平面进行镜面反射，求该变换的矩阵表示.  
**解:**  
&emsp;&emsp;设$V=M$为欧几里得空间，映射$\sigma$为镜面反射，易知$\sigma$为线性变换.
&emsp;&emsp;设欧几里得空间中的一组标准正交基向量为$e_1,e_2,e_3$，

$$
e_1=\begin{bmatrix}
    1 \\
    0 \\
    0 \\
\end{bmatrix}, e_2=\begin{bmatrix}
    0 \\
    1 \\
    0 \\
\end{bmatrix}, e_3=\begin{bmatrix}
    0 \\
    0 \\
    1 \\
\end{bmatrix}
$$

&emsp;&emsp;其中，$e_1,e_2$所在的平面即为进行镜面反射所依赖的平面，$e_3$为垂直于该平面的一个基向量。可以$e_1,e_2,e_3$的方向为$x,y,z$轴建立坐标系. 向量组$e_1,e_2,e_3$既为入口基也为出口基.  
&emsp;&emsp;镜面反射$\sigma$作用于基向量组$e_1,e_2,e_3$时，$e_1,e_2$并不会改变，$e_3$变为相反方向，镜面反射可用图3表示。  

<center>
    <img src="https://s2.loli.net/2023/10/05/oSeUqtnj5I74Kyf.jpg
    " width=40% height=40%>
    <div align="center">Image3: 镜面反射</div>
</center>  

&emsp;&emsp;由几何知识可知：  

$$
\sigma(e_1)=\begin{bmatrix}
    1 \\
    0 \\
    0 \\
\end{bmatrix},\sigma(e_2)=\begin{bmatrix}
    0 \\
    1 \\
    0 \\
\end{bmatrix},\sigma(e_3)=\begin{bmatrix}
    0 \\
    0 \\
    -1 \\
\end{bmatrix}
$$  

&emsp;&emsp;故有:  

$$
\begin{align*}
    \sigma(\begin{bmatrix}
    e_1,e_2,e_3 \\
\end{bmatrix})&=\begin{bmatrix}
    \sigma(e_1),\sigma(e_2),\sigma(e_3)
\end{bmatrix}=\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & -1 \\
\end{bmatrix} \\
    &=\begin{bmatrix}
        1 & 0 & 0 \\
        0 & 1 & 0 \\
        0 & 0 & 1 \\
    \end{bmatrix}\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & -1 \\
\end{bmatrix}=\begin{bmatrix}
    e_1,e_2,e_3 \\
\end{bmatrix}A
\end{align*}
$$  

$$
A=\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & -1 \\
\end{bmatrix}
$$  

&emsp;&emsp;矩阵$A$即为欧几里得空间中镜面反射$\sigma$在标准正交基下的矩阵表示.  

