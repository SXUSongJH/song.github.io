---
title: 矩阵-线性子空间
tags:
    - Matrix
    - Math
---

## toc

## 线性子空间的定义  

&emsp;&emsp;设$V$是数域$\mathbb{F}$上的线性空间，$W \in V$是非空子集，若$W$有以下两个属性:  
(1) **对加法封闭**：$\forall \alpha,\beta \in W, \alpha+\beta \in W$.  
(2) **对数乘封闭**：$\forall k \in \mathbb{F}, \alpha \in W, \alpha k \in W$.  
则称集合$W$是线性空间$V$的线性子空间.  

注：线性子空间$W$本身按$V$中定义的加法与数乘也构成线性空间.  

## 线性子空间的实例  

### 例1 二维空间的线性子空间  

&emsp;&emsp;$\color{green}{结论: 二维空间的线性子空间是任意经过原点的直线.}$  
&emsp;&emsp;**证明:**  
&emsp;&emsp;设$V = \mathbb{R}^{2}$，$V$表示二维平面.  
&emsp;&emsp;二维平面中经过原点的某一直线：$W = \{x \vert \theta^{T} x=0, x \in V\}, \theta \in \mathbb{R}^{2}$.  
&emsp;&emsp;验证直线$W$是二维平面$V$的线性子空间：  
&emsp;&emsp;(1) 对$\forall x_1,x_2 \in W, \theta^{T}(x_1+x_2)= \theta^{T}x_1+\theta^{T}x_2=0$,且$x_1+x_2 \in V,$  
&emsp;&emsp;$\Rightarrow x_1+x_2 \in W$，即$W$对加法封闭.  
&emsp;&emsp;(2) 对$\forall \beta \in \mathbb{R}, \forall x \in W, \theta^{T}(\beta x)=\beta(\theta^{T}x)=0$，且$\beta x \in V,$  
&emsp;&emsp;$\Rightarrow \beta x \in W$，即$W$对数乘封闭.  
&emsp;&emsp;综上所述：直线$W$是二维空间$V$的线性子空间.  

&emsp;&emsp;**注：二维空间中不经过原点的直线不是其线性子空间。**  
&emsp;&emsp;**证明:**  
&emsp;&emsp;设$V=\mathbb{R}^2$，$V$表示二维平面.  
&emsp;&emsp;二维平面中不经过原点的某一直线：$W=\{x \vert \theta^{T}x+b=0,b \neq 0,x \in V\}, \theta \in \mathbb{R}^{2}.$  
&emsp;&emsp;验证直线$W$不是二维平面$V$的线性子空间：  
&emsp;&emsp;(1) 对$\forall x_1,x_2 \in W, x_1,x_2 \neq 0, \theta^{T}(x_1+x_2)+b = (\theta^{T}x_1+b)+\theta^{T}x_2=\theta^{T}x_2 \neq 0.$  
&emsp;&emsp;又$\because x_1+x_2 \in V \Rightarrow x_1+x_2 \notin W \Rightarrow$ 直线$W$不满足对加法封闭，故其不是二维空间$V$的线性子空间.  

&emsp;&emsp;对于二维空间的线性子空间，我们也可以借助图像来直观理解：  

<center>
    <img src="https://s2.loli.net/2023/09/17/6eWufsxXb3FpVYh.jpg" width=40% height=40%>
    <div align="center">Image1: 二维空间的子空间</div>
</center>  

&emsp;&emsp;从图中我们可以发现，经过原点的直线$W_1$上任意两个向量$x_1,x_2$的和仍位于直线$W_1$上，说明$W_1$对加法封闭，同时$W_1$上的向量伸缩后仍位于直线$W_1$上，说明$W_1$对数乘封闭。故经过原点的直线$W_1$是二维空间$V$的线性子空间.  
&emsp;&emsp;不经过原点的直线$W_2$上的任意两个向量$y_1,y_2$，这两个向量的和$y_3$不位于直线$W_2$上，故直线$W_2$对加法不封闭，其不是二维空间$V$的线性子空间.  

### 例2 向量组生成的线性子空间及线性子空间的生成组  

&emsp;&emsp;设有向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p \}$，定义：  

$$
W = span\{ \alpha_1,\alpha_2,\dots,\alpha_p\} =\{\alpha_{1}c_{1}+\alpha_{2}c_{2}+\dots+\alpha_{p}c_{p} \vert c_{i} \in \mathbb{F},i=1,2,\dots,p\} =\{向量组\alpha_i 的线性组合的全体\}
$$  

&emsp;&emsp;称$W$为向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p \}$所生成的线性子空间,向量组$\{\alpha_1,\alpha_2,\dots,\alpha_p \}$称为线性子空间的生成组.  

### 例3 矩阵的核与象  

&emsp;&emsp;设矩阵$A \in \mathbb{F}^{m\times n}$，定义集合$\{x \vert x \in \mathbb{F}^{n},Ax=0 \}$为矩阵$A$的核，记为$ker A$，集合$\{ y \vert y \in \mathbb{F}^{m}, \exists x \in \mathbb{F}^{n}, y=Ax\}$为矩阵$A$的象，记为$im A$. $ker A$与$im A$分别是$\mathbb{F}^n$与$\mathbb{F}^{m}$的线性子空间.  
  
**证明:**  
&emsp;&emsp;设矩阵$A \in \mathbb{F}^{m \times n}$，记矩阵$A$的核为集合$ker A = \{x \vert x \in \mathbb{F}^{n},Ax=0 \}$，  
&emsp;&emsp;矩阵$A$的象为集合$im A = \{ y \vert y \in \mathbb{F}^{m}, \exists x \in \mathbb{F}^{n}, y=Ax\}$.  
&emsp;&emsp;(1) 验证$ker A$是线性空间$\mathbb{F}^{n}$的线性子空间:  
&emsp;&emsp;对 $\forall x_1,x_2 \in ker A$，有$Ax_1=Ax_2=0 \Rightarrow A(x_1+x_2)=0$，故有：  
&emsp;&emsp;$x_1+x_2 \in ker A$，即$ker A$对加法封闭.  
&emsp;&emsp;对$\forall k \in \mathbb{F}, A(x_1 k) = (Ax_1)k = 0$，故有$x_1 k \in ker A$，即$ker A$对数乘封闭.  
&emsp;&emsp;棕上所述，$ker A$是线性空间$\mathbb{F}^{n}$的线性子空间.  
&emsp;&emsp;(2) 验证$im A$是线性空间$\mathbb{F}^{m}$的线性子空间:  
&emsp;&emsp; 对$\forall y_1,y_2 \in im A, \exists x_1,x_2 \in \mathbb{F}^n$，使得$y_1 = Ax_1, y_2 = Ax_2$，  
&emsp;&emsp;$\Rightarrow y_1+y_2 = Ax_1 + Ax_2 = A(x_1+x_2)$，令$y_3=y_1+y_2,x_3 = x_1+x_2$，  
&emsp;&emsp;$\because y_1,y_2 \in \mathbb{F}^{m}$，且线性空间$\mathbb{F}^{m}$对加法封闭，故有$y_3 \in \mathbb{F}^{m}$，同理可得$x_3 \in \mathbb{F}^{n}$，  
&emsp;&emsp;$y_3 = Ax_3 \Rightarrow y_3 \in im A$，即$im A$对加法封闭.  
&emsp;&emsp;对$\forall k \in \mathbb{F}, y_1 k = (Ax_1)k = A(x_1 k)$，  
&emsp;&emsp;$\because x_1 \in \mathbb{F}^{n}$，且线性空间$\mathbb{F}^{n}$对数乘封闭, 故有$x_1 k \in \mathbb{F}^{n}$,  
&emsp;&emsp;则有$y_1 k \in im A$，即$im A$对数乘封闭.  
&emsp;&emsp;综上所述，$im A$是线性空间$\mathbb{F}^{m}$的线性子空间.  

**注:** $Ax$为矩阵$A$的列向量组$\{\alpha_1, \alpha_2, \dots, \alpha_n \}$以$x$为系数的线性组合. $im A$也就是向量组$\{\alpha_1, \alpha_2, \dots, \alpha_n \}$所张成的线性子空间.  

## 线性子空间的交与和  

&emsp;&emsp;设$U,W$是$V$的线性子空间，则有：  
(1) $U \cap W = \{v \vert v \in U, v \in W\}$，$U \cap W$也是$V$的线性子空间，称为$U,W$的交.  
(2) $U+W = span \{U,W\} = \{u+w \vert u \in U, w \in W\}$，$U+W$也是$V$的线性子空间，称为$U,W$的和.  

**证明:**  
&emsp;&emsp;设$V$是线性空间,$U,W$是$V$的线性子空间,  
&emsp;&emsp;(1) 验证$U \cap W$是$V$的线性子空间  
&emsp;&emsp;对$\forall v_1,v_2 \in U \cap W$，有$v_1,v_2 \in U, v_1,v_2 \in W$  
&emsp;&emsp;$\because U,W$是$V$的线性子空间，故$U,W$对加法封闭 $\Rightarrow v_1+v_2 \in U, v_1+v_2 \in W \Rightarrow v_1+v_2 \in U \cap W$  
&emsp;&emsp;即$U \cap W$对加法封闭.  
&emsp;&emsp;对$\forall k \in \mathbb{F}, v_1 \in U \cap W$，$U,W$对数乘封闭，故有:  
&emsp;&emsp;$v_1k \in U, v_1k \in W \Rightarrow v_1k \in U \cap W$，即$U \cap W$对数乘封闭.  
&emsp;&emsp;(2) 验证$U+W$是$V$的线性子空间  
&emsp;&emsp;对$\forall v_1,v_2 \in U+W, \exists u_1,u_2 \in U, w_1,w_2 \in W$，使得$v_1 = u_1+w_1,v_2 = u_2+w_2$  
&emsp;&emsp;$v_1+v_2=u_1+w_1+u_2+w_2=(u_1+u_2)+(w_1+w_2)$，  
&emsp;&emsp;令$v_3=v_1+v_2, u_3 = u_1+u_2, w_3 = w_1+w_2 \Rightarrow v_3=u_3+w_3$，由$U,W$对加法的封闭性可知：  
&emsp;&emsp;$u_3 \in U, w_3 \in W \Rightarrow v_3 \in U+W$，即$U+W$对加法封闭.  
&emsp;&emsp;对$\forall k \in \mathbb{F}, v_1 \in U+W, v_1k = (u_1+w_1)k=u_1k+w_1k$,  
&emsp;&emsp;$\because U,W$对数乘封闭，故有$u_1k \in U, w_1k \in W \Rightarrow v_1 k \in U+W$，即$U+W$对数乘封闭.  
&emsp;&emsp;综上所述，$U \Cap w, U+W$均为$V$的线性子空间.  
















