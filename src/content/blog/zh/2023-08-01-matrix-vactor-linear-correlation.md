---
title: 矩阵-向量组与线性相关性
tags:
    - Matrix
    - Math
---

## toc 

## 向量组  

### 向量组的定义  
&emsp;&emsp;设$V$是数域$\mathbb{F}$上的线性空间，$V$中有限序列$\alpha_1,\alpha_2,\dots,\alpha_n$称为$V$中的一个向量组，记为向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$。  

### 向量组所拼成的抽象矩阵  
&emsp;&emsp;若矩阵$A$中的每一列是由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$所构成的，则称矩阵$A$是由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$所拼成的抽象矩阵，记为：  

$$
A = \begin{bmatrix}
    \alpha_1,\alpha_2,\dots,\alpha_n \\
\end{bmatrix}
$$  

## 向量组的线性相关性  

### 线性相关与线性无关的定义  

&emsp;&emsp;向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$称为线性相关，如果存在不全为零的$P$个数$k_i \in \mathbb{F}, i=1,2,\dots,p$，使得：  

$$
\begin{equation}\
\alpha_1k_1+\alpha_2k_2+\dots+\alpha_pk_p=0
\end{equation}
$$  

&emsp;&emsp;向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$称为线性无关，如果任意不全为零的$P$个数$k_i \in \mathbb{F}, i=1,2,\dots,p$，使得：  

$$
\begin{equation}\
\alpha_1k_1+\alpha_2k_2+\dots+\alpha_pk_p \ne 0
\end{equation}
$$  

- **注1** 线性无关的另外一种表述：  
&emsp;&emsp;向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$称为线性无关，若 $\alpha_1k_1+\alpha_2k_2+\dots+\alpha_pk_p = 0$ 当且仅当 $k_i=0,i=1,\dots,p$ 时成立。  

### 线性相关性的矩阵表述  

&emsp;&emsp;设向量组$\{\alpha_1,\alpha_2,\dots,\alpha_n\}$所拼成的抽象矩阵为$A$，向量$x = \begin{bmatrix}
    x_1,x_2,\dots,x_p
\end{bmatrix}^T$，有齐次线性方程：  

$$
\begin{equation}
    Ax=\begin{bmatrix}
    \alpha_1,\alpha_2,\dots,\alpha_p \\
\end{bmatrix}\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_p
\end{bmatrix}=0
\end{equation}
$$  

<span style="color: green;">向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性相关 $\Leftrightarrow$ 齐次线性方程(3)有非零解。</span>  
<span style="color: green;">向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性无关 $\Leftrightarrow$ 齐次线性方程(3)只有零解。</span>  

## 向量组之间的线性表示  

### 线性表示的定义

&emsp;&emsp;设有向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$，$\{\beta_1,\beta_2,\dots,\beta_q\}$以及向量$\beta$。  
&emsp;&emsp;称向量$\beta$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示，如果$\exists k_1,k_2,\dots,k_p \in \mathbb{F}$，使得：  

$$
\begin{equation}
   \beta = \alpha_1k_1+\alpha_2k_2+\dots+\alpha_pk_p
\end{equation}
$$  

&emsp;&emsp;称向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示，如果每个$\beta_i$均可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示。  

### 线性表示的矩阵表述  

&emsp;&emsp;设向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$所拼成的抽象矩阵为$A$，向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$所拼成的抽象矩阵为$B$,有非齐次线性方程：  

$$
\begin{equation}
    Ax=\begin{bmatrix}
        \alpha_1,\alpha_2,\dots,\alpha_p
    \end{bmatrix}\begin{bmatrix}
        x_1 \\
        x_2 \\
        \vdots \\
        x_p
    \end{bmatrix} = \beta
\end{equation}
$$  

$$
\begin{equation}
    AX= \begin{bmatrix}
        \alpha_1,\alpha_2,\dots,\alpha_p
    \end{bmatrix}\begin{bmatrix}
        x_{11} & x_{12} & \dotsb & x_{1q} \\
        x_{21} & x_{22} & \dotsb & x_{2q} \\
        \vdots & \vdots &        & \vdots \\
        x_{p1} & x_{p2} & \dotsb & x_{pq} \\
    \end{bmatrix} = \begin{bmatrix}
        \beta_1,\beta_2,\dots,\beta_q
    \end{bmatrix}=B
\end{equation}
$$  

<span style="color: green;">向量$\beta$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示 $\Leftrightarrow$ 非齐次线性方程(5)有解。</span>  
<span style="color: green;">向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示 $\Leftrightarrow$ 非齐次线性方程组(6)有解。</span>  

### 线性表示的传递性  

&emsp;&emsp;设有三个向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$、$\{\beta_1,\beta_2,\dots,\beta_q\}$、$\{v_1,v_2,\dots,v_t\}$，其所拼成抽象矩阵分别为$A、B、C$。  
&emsp;&emsp;若向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示，而向量组$\{v_1,v_2,\dots,v_t\}$可由向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$线性表示，则向量组$\{v_1,v_2,\dots,v_t\}$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示。  
  
**证明：**  
向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示 $\Rightarrow$ 矩阵方程 $AX_{p\times q}=B$ 有解  
向量组$\{v_1,v_2,\dots,v_t\}$可由向量组$\{\beta_1,\beta_2,\dots,\beta_q\}$线性表示 $\Rightarrow$ 矩阵方程 $BY_{q\times t}=C$ 有解

$$
\left\{
             \begin{array}{lr}
             AX_{p\times q}=B \\
             BY_{q \times t}=C
             \end{array}
\right. \Rightarrow AXY=C \Rightarrow 矩阵方程AZ_{p \times t} = C 有解，Z_{p \times t}=XY
$$  

故向量组$\{v_1,v_2,\dots,v_t\}$可由向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$线性表示。  

## 向量组的极大线性无关组  

### 极大线性无关组的定义  

&emsp;&emsp;设向量组$\{\beta_1,\beta_2,\dots,\beta_s\}$是向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$的子组，即 $\{\beta_1,\beta_2,\dots,\beta_s\} \subseteq \{\alpha_1,\alpha_2,\dots,\alpha_p\}$。子组$\{\beta_1,\beta_2,\dots,\beta_s\}$称为向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$的极大线性无关组，若其满足：  
1. 子组$\{\beta_1,\beta_2,\dots,\beta_s\}$线性无关。  
2. 若向量组$\{v_1,v_2,\dots,v_t\}$也是向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$的子组，且 $s < t$，则子组$\{v_1,v_2,\dots,v_t\}$线性相关。  

- **注：“极大性”的另一种说法：“生成性”。**
  
&emsp;&emsp;若子组$\{\beta_1,\beta_2,\dots,\beta_s\}$线性无关，且 $\forall \alpha_i \in \{\alpha_1,\alpha_2,\dots,\alpha_p\}$，$\alpha_i$都可由子组$\{\beta_1,\beta_2,\dots,\beta_s\}$线性表示，则子组$\{\beta_1,\beta_2,\dots,\beta_s\}$是向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$的极大线性无关组。  

$$
\color{green} 极大线性无关组 \Leftrightarrow 线性无关生成组
$$  

### 向量个数的唯一性  

**定理：** 向量组的极大线性无关组所含向量的个数是唯一的。  
**证明：**  
&emsp;&emsp;设有向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$，向量组$\{\beta_1,\beta_2,\dots,\beta_s\}$与$\{v_1,v_2,\dots,v_t\}$均是其极大线性无关组，  
&emsp;&emsp;要证明 $ s=t $，可以利用反证法，假设 $s < t$，   
&emsp;&emsp;设 $A = \begin{bmatrix}
    \alpha_1,\alpha_2,\dots,\alpha_p
\end{bmatrix}$，$B=\begin{bmatrix}
    \beta_1,\beta_2,\dots,\beta_s
\end{bmatrix}$，$C = \begin{bmatrix}
    v_1,v_2,\dots,v_t 
\end{bmatrix}$，  
&emsp;&emsp;$\because$ 向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p\}$可由其极大线性无关组$\{\beta_1,\beta_2,\dots,\beta_s\}$、$\{v_1,v_2,\dots,v_t\}$线性表示，  
&emsp;&emsp;$\therefore$ 矩阵方程 $BX=A, AY=C$均有解，  
&emsp;&emsp;$\Rightarrow$ 矩阵方程 $BZ=C$ 有解，其中$Z_{s \times t}=XY$，  
&emsp;&emsp;$ s < t$ $\Rightarrow$ $rank(Z) < t$ $\Rightarrow$ 矩阵方程$ZW= \textbf{0}$有无穷多解 $\Rightarrow$ 矩阵方程$ZW= \textbf{0}$必有非零解，  
&emsp;&emsp;设矩阵方程 $ZW= \textbf{0}$ 的非零解为$W$，将矩阵方程 $BZ=C$ 的两边同时乘以$W$，  
&emsp;&emsp;有 $B(ZW) = CW, \because ZW=\textbf{0}$， $\therefore CW= \textbf{0}$，  
&emsp;&emsp;与向量组$\{v_1,v_2,\dots,v_t\}$线性无关矛盾，  
&emsp;&emsp;$\Rightarrow$ $s \nless t$，同理可证 $s \ngtr t$，故有 $s = t$  

## 向量组的秩  

&emsp;&emsp;<span style="color: green;">向量组的秩等于向量组的极大线性无关组所含向量的个数。</span> 


