---
title: 矩阵-线性空间
tags:
  - Matrix
  - Math
---

## toc

## 加法与数乘  
**加法的定义**  
&emsp;&emsp;给定非空集合V和数域$\mathbb{F}$，若存在映射：  

$$
\sigma: V \times V \rightarrow V
$$

$$
(\alpha,\beta) \mapsto \sigma(\alpha,\beta)
$$

即对V中任意元素$\alpha$，$\beta$，在集合V中都存在唯一元素$\gamma$,使得$\gamma=\alpha+\beta \in V$，则称映射$\sigma$为集合V上的加法。  
  
**数乘的定义**  
&emsp;&emsp;给定非空集合V和数域$\mathbb{F}$，若存在映射：

$$
\sigma: V \times \mathbb{F} \rightarrow V
$$

$$
(\alpha,k) \mapsto \sigma(\alpha,k)
$$

即对集合V中的任意元素$\alpha$和数域$\mathbb{F}$中任意元素k，在集合V中都存在唯一元素$\gamma$，使得$\gamma=\alpha k \in V$，则称映射$\sigma$为集合V上的数乘。  
  
- **注1：关于映射的符号**  
&emsp;&emsp;设映射$\sigma$为正弦函数$sin(*)$  
&emsp;&emsp;“$\rightarrow$”表示将集合映射到集合。例如 $\sigma: R \rightarrow [-1,1]$，表示将实数集R映射到集合[-1,1]。  
&emsp;&emsp;“$\mapsto$”表示将元素映射到元素。例如 $\sigma: \pi \mapsto 0$，表示将元素$\pi$映射到元素0。  
  
- **注2：集合的笛卡尔积**  
&emsp;&emsp;集合$S_1$和$S_2$的笛卡尔积的数学表示为：

$$
S_1 \times S_1 = \{\begin{bmatrix}s_1\\s_2 \\\end{bmatrix} \mid s_1 \in S_1,s_1 \in S_2\}
$$  
  
- **注3：域的定义及常用的域**  
&emsp;&emsp;域的定义：包含加法与乘法的，满足通常运算规则的代数结构称为域。  
&emsp;&emsp;常用的域：$\mathbb{Q}$(有理数域)、$\mathbb{R}$(实数域)、$\mathbb{C}$(复数域)等。  

## 通常的运算规则  
1. **加法交换律**  
&emsp;&emsp;已知非空集合V，对 $\forall v_1,v_2 \in V$，有 $v_1+v_2=v_2+v_1$。  
2. **加法结合律**  
&emsp;&emsp;已知非空集合V，对 $\forall v_1,v_2,v_3 \in V$，有 $(v_1+v_2)+v_3=v_1+(v_2+v_3)$。  
3. **加法零元素**  
&emsp;&emsp;已知非空集合V，对 $\forall v \in V, \exists e \in V$，使得 $e+v=v$。  
4. **加法逆元素**  
&emsp;&emsp;已知非空集合V，对 $\forall v \in V, \exists a \in V$，使得 $v+a=e$，记 $a=-v$。  
5. **数乘分配律**  
&emsp;&emsp;已知非空集合V和数域$\mathbb{F}$，对 $\forall v_1,v_2 \in V, k \in \mathbb{F}$，有 $(v_1+v_2)k=v_1k+v_2k$。  
&emsp;&emsp;已知非空集合V和数域$\mathbb{F}$，对 $\forall v \in V;k_1,k_2 \in \mathbb{F}$，有 $v(k_1+k_2)=vk_1+vk_2$。  
6. **数乘结合律**  
&emsp;&emsp;已知非空集合V和数域$\mathbb{F}$，对 $\forall v \in V;k,l \in \mathbb{F}$，有 $(vk)l=v(kl)$。  
7. **数乘单位元素**  
&emsp;&emsp;已知非空集合V和数域$\mathbb{F}$，对 $\forall v \in V,\exists 1 \in \mathbb{F}$，使得 $v1=v$。  
  
## 线性空间的定义  
&emsp;&emsp;<font color=#008000>若集合V满足<b>加法</b>与<b>数乘</b>两种运算，且这两种运算满足<b>通常的运算规则</b>，则称<b>集合V</b>关于此加法和数乘是<b>数域$\mathbb{F}$</b>上的线性空间。</font>一般也把这种线性空间称为向量空间，集合V中的元素称为向量。  
  
## 线性空间的具体实例  
### 例1：数域$\mathbb{F}$上的标准线性空间$\mathbb{F}^n$  
&emsp;&emsp;已知数域$\mathbb{F}$,设  

$$
V := \mathbb{F}^n=\mathbb{F} \times \mathbb{F} \times \dots \times \mathbb{F}=\{\begin{bmatrix}
  v_1\\
  v_2\\
  \vdots\\
  v_n\\
\end{bmatrix} \mid v_i \in \mathbb{F},i=1, \dots,n\}
$$  

&emsp;&emsp;对 $\forall \alpha,\beta \in V, k \in \mathbb{F}$，  
&emsp;&emsp;定义集合V上的加法运算：  

$$
\alpha + \beta=\begin{bmatrix}
  \alpha_1\\
  \alpha_2\\
  \vdots\\
  \alpha_n\\
\end{bmatrix}+\begin{bmatrix}
  \beta_1\\
  \beta_2\\
  \vdots\\
  \beta_n\\
\end{bmatrix}=\begin{bmatrix}
  \alpha_1+\beta_1\\
  \alpha_2+\beta_2\\
  \vdots\\
  \alpha_n+\beta_n\\
\end{bmatrix} \in V
$$  

&emsp;&emsp;定义集合V上的数乘运算： 

$$
\alpha \cdot k=\begin{bmatrix}
  \alpha_1\\
  \alpha_2\\
  \vdots\\
  \alpha_n\\
\end{bmatrix} \cdot k = \begin{bmatrix}
  \alpha_1 k \\
  \alpha_2 k \\
  \vdots \\
  \alpha_n k \\
\end{bmatrix} \in V
$$  

&emsp;&emsp;易证此加法与数乘满足通常的运算法则，此时称集合V为数域$\mathbb{F}$上的n维标准线性空间，记为$\mathbb{F}^n$。  
&emsp;&emsp;一些常用数域上的n维标准线性空间：$\mathbb{R}^n$(实数域)、$\mathbb{C}^n$(复数域)等。  

### 例2：欧几里得空间作为线性空间  
&emsp;&emsp;令数域$\mathbb{F}=\mathbb{R}$，集合$V=\{欧几里得空间中的全体有向线段\}$。(当两条有向线段经过平移能够重叠时，则把这两条线段算做一条线段)  
&emsp;&emsp;定义集合V上的加法运算：向量运算的平行四边形法则。  
&emsp;&emsp;定义集合V上的数乘运算：向量同向或反向伸缩。  
<center class="half">
    <img src="https://s2.loli.net/2023/07/27/GNvpJQHgdrOLMRA.png" width=50%><img src="https://s2.loli.net/2023/07/27/lFhM6DNEmVkRW7c.png" width=50%>
    <p>Image 1 向量的平行四边形法则&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Image 2 向量的伸缩</p>
</center>
&emsp;&emsp;易证此加法与数乘满足通常的运算法则，则集合V是线性空间，说明欧几里得空间可以作为线性空间。  

### 例3：函数空间作为线性空间  
&emsp;&emsp;已知数域$\mathbb{F}$，集合$V=\mathcal{F}(I,\mathbb{F}^n)$。集合V为函数空间，V中的元素是以数域$\mathbb{F}$中的区间$I$为定义域，具有n个分量的n维向量值函数。例如：  

$$
f=\begin{bmatrix}
  f_1(x) \\
  f_2(x) \\
\end{bmatrix}=\begin{bmatrix}
  sin(x) \\
  \frac{1}{2}x^3 \\
\end{bmatrix}, f \in \mathcal{F}([-1,1],\mathbb{R}^2) 
$$  

&emsp;&emsp;对 $\forall f,g \in \mathcal{F}(I,\mathbb{F}^n),k \in \mathbb{F}$，  
&emsp;&emsp;定义集合V上的加法运算：  

$$
f+g = \begin{bmatrix}
  f_1(x) \\
  f_2(x) \\
  \vdots \\
  f_n(x) \\
\end{bmatrix}+\begin{bmatrix}
  g_1(x) \\
  g_2(x) \\
  \vdots \\
  g_n(x) \\
\end{bmatrix} = \begin{bmatrix}
  f_1(x)+g_1(x) \\
  f_2(x)+g_2(x) \\
  \vdots \\
  f_n(x)+g_n(x) \\
\end{bmatrix} \in \mathcal{F}(I,\mathbb{F}^n)
$$  

&emsp;&emsp;定义集合V上的数乘运算： 

$$
f \cdot k = \begin{bmatrix}
  f_1(x) \\
  f_2(x) \\
  \vdots \\
  f_n(x) \\
\end{bmatrix} \cdot k = \begin{bmatrix}
  kf_1(x) \\
  kf_2(x) \\
  \vdots \\
  kf_n(x) \\
\end{bmatrix} \in \mathcal{F}(I,\mathbb{F}^n)
$$  

&emsp;&emsp;易证此加法与数乘满足通常的运算法则，则集合V是线性空间，说明函数空间可以作为线性空间。

