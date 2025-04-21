---
title: Optimal Transport-5.Entropy Regularization
tags:
    - Optimal Transport
    - Math
---  

# Entropy Regularization  

&emsp;&emsp;在之前的文章中，我们从概率视角描述了最优传输问题。设 $X,Y$ 是服从分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 的两个随机变量，运输矩阵为 $\boldsymbol{P}$，成本矩阵为 $\boldsymbol{C}$，则分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的最优传输问题可以被定义为(1)式：  

$$\begin{equation}
    L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right> = \min_{(X,Y)} \{ \mathbb{E}_{(X,Y)}(c(X,Y)): X \sim \boldsymbol{\alpha}, Y \sim \boldsymbol{\beta} \}
\end{equation}$$  

&emsp;&emsp;我们在之前讨论过，运输矩阵 $\boldsymbol{P}$ 可以看作随机变量 $X,Y$ 的联合分布。从Monge的前向运输法到Kantorovich的局部前向运输，最优传输问题的解通常是满足边界条件下，"耦合度"最低的运输方式，即最优运输矩阵$\boldsymbol{P}$ 是一个很稀疏的矩阵，其非零元素主要集中在对角线附近。如果我们用概率论与信息论的视角来分析，这意味原始最优传输问题的解是**满足边际分布条件下，熵最小的联合分布。**  
&emsp;&emsp;熵正则的主要思想对联合分布$\boldsymbol{P}$的信息熵进行惩罚，以期望求得的联合分布$\boldsymbol{P^{*}}$的熵更大。熵正则的数学表达式如下(2)式：  

$$\begin{equation}
    L_{\boldsymbol{C}}^{\varepsilon}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right> - \varepsilon H(\boldsymbol{P})
\end{equation}$$  

&emsp;&emsp;其中，常数 $\varepsilon$ 为惩罚系数，用于控制对联合分布的熵的惩罚力度，$\varepsilon$ 越大，则我们解得的联合分布 $\boldsymbol{P^{*}}$ 的熵越大。除了这种形式，有一些文献也将熵正则的定义为如下形式(3)式：  

$$\begin{equation}
    L_{\boldsymbol{C}}^{\gamma}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \min_{\boldsymbol{P} \in \boldsymbol{U_{\gamma}}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right>
\end{equation}$$  

$$\boldsymbol{U_{\gamma}}(\boldsymbol{\alpha},\boldsymbol{\beta}) = \{ \boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) | KL(\boldsymbol{P} || \boldsymbol{\alpha \beta^{T}}) \leq \gamma \} = \{ \boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) | h(\boldsymbol{P}) \ge h(\boldsymbol{\alpha}) + h(\boldsymbol{\beta}) - \gamma  \} \subset \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})$$  

&emsp;&emsp;(2)和(3)两种熵正则的形式实际上是等价的，熵正则在最优传输理论中起着十分重要的作用，**在本节我们将会讨论熵正则的作用；信息论以及几何视角下的熵正则内涵，以及证明其求解结果能够作为分布之间距离的度量。**  

## Why Entropy Regularzation ?  

&emsp;&emsp;对于熵正则的作用，我主要会从两个方面来讨论，**数值计算以及最大熵思想。**

### Lightspeed Computation of Optimal Transport
&emsp;&emsp;在很多对于最优传输问题中熵正则项的作用，大部分文章都会提到引入正则项能够提高最优传输问题的计算效率，这当然是其最突出的优点。  
&emsp;&emsp;原始的最优传输问题是一个线性规划问题，对于线性规划问题，通常的求解方法为单纯形法以及内点法，这些算法的计算复杂度为 $O(d^{3}\log(d))$，$d$ 为随机变量的维度。这些算法能够处理一些简单的最优传输问题，但随着维度的增加，巨大的计算成本使得最优传输很难应用到高维数据中。而在机器学习领域中，很多数据都是高维数据，例如长宽均为256的一张图片，其一个数据样本便是 $\mathbb{R}^{256 \times 256 \times 3}$ 的空间中的一个向量。如果想在这种数据集上应用最优传输理论来做一些工作，例如图像生成、图像匹配、风格变化等，庞大的计算成本便是一个很难解决的问题。  
&emsp;&emsp;**通过在原始的最优传输问题中引入一个熵正则项，可以使得该优化问题变成一个严格的凸问题。** 最优传输问题本身在没有正则化时可能不是严格凸的，特别是在离散设置中，可能存在多个最优解，这使得我们求解的计算成本大大增加，例如在单纯形算法中，如果我们需要找到所有的最优解，我们可能需要遍历多面体所有极点，这在高维问题中意味着庞大的计算开销，因为此时多面体可能存在很多极点。而严格的凸性意味着该优化问题只存在唯一的最优解，此时我们只要找到一个最优解，则这个最优解一定是唯一的，算法便可以终止了。对于严格的凸问题我们已经有很多高效的数值求解方法。  
&emsp;&emsp;**引入熵正则项后的最优传输问题可以通过特定的算法，如Sinkhorn迭代算法来高效求解。** 这些算法通常基于矩阵缩放技术，即通过迭代调整矩阵的行和列以使得传输矩阵逼近目标边缘分布。这种方法简单、高效，并且易于实现。Franklin和Lorenz (1989) 以及 Knight (2008) 的研究表明，Sinkhorn算法具有线性收敛性。这意味着每次迭代后，误差以固定比率减少，确保了算法在实际应用中的快速收敛。对于Sinkhorn算法，我们将在下一篇博客中详细讨论，这里不作过多的展开。  
&emsp;&emsp;**Sinkhorn算法能够依靠矩阵乘法的运算来进行迭代，这使得我们能够使用GPU强大的并行计算能力来快速求解高维空间中的最优传输问题。** 近二十年来，计算机硬件厂商对GPU不断地进行升级迭代，使得电脑GPU的性能不断提升，而GPU的多核结构十分适合用于矩阵运算，计算性能的提示是得以往一些难以取得进展的领域得到了突破，深度学习便是一个很好的例子，而深度学习的兴起又反过来促使硬件厂商不断提升GPU性能，GPU的算力得到了高速发展。Sinkhorn算法可以使用矩阵乘法进行迭代运算，这使得最优传输问题的求解能够享受到GPU性能不断提升的红利，这大大促进了最优传输理论在机器学习领域的应用。  

### Maximum Entropy Principle  

&emsp;&emsp;除开耳熟能详的计算优势外，熵正则所蕴含的最大熵原理的思想是另外一个我想重点讨论的点。熵这个概念最早可以追溯到19世纪 R.J.E.Clausius 对于热力学的研究，他引入熵这个概念来描述系统的无序程度或者信息缺失的状态。热力学第二定律指出，**一个孤立的系统的总熵不会自发减少，即熵总是趋于增加的。** 最大熵思想最初由物理学家 J.W.Gibbs 在《物理学原理》这本著作中提出。Gibbs讨论了如何从统计力学的角度来理解熵，以及如何通过概率分布来计算系统的熵，特别是在平衡状态下。Gibbs的工作强调了**在已知部分信息的情况下，选择熵最大的分布可以带来最大的“不确定性”或“混乱度”，这种分布认为每一种可能状态出现的机会均等，直到有额外信息来提供更多的偏好。**  
&emsp;&emsp;我们用一个例子来说明一下系统的这种熵增现象。Albert Einstein 在1905年发表了一篇划时代的论文《关于运动着的水分子所引起的、需用显微镜观察的悬浮粒子的运动》，在这篇论文中，他发展了数学模型来描述微观粒子在流体中的随机运动，这种运动后来被称为布朗运动，也称随机扩散过程。我们便用粒子的随机扩散过程体会熵增的过程。  
&emsp;&emsp;假设现在有一条“又长又细的水管”，我们可以将这个水管看作实数轴，我们在这个实数轴的原点$O$处滴入一滴粒子数量为$C$的墨水，设 $\tau$ 代表扩散过程的时间间隔，$\rho(y)$ 表示在时间$\tau$内粒子移动距离为$y$的概率分布，其是一个概率密度函数。$f(x,t)$ 表示$t$时刻在实数轴$x$处的粒子的数量，其也是一个概率密度函数，同时是关于时间$t$的一个随机过程。  
  &emsp;&emsp;假设扩散过程是对称的，即$\rho(y)$和$f(x,t)$均为关于原点$O$的对称分布，此时有:  

$$\rho(y) = \rho(-y) \Leftrightarrow \int_{-\infty}^{+\infty}y\rho(y)=0$$  

&emsp;&emsp;由假设我们可以得知这个系统的初始状态 $f(0,0) = C$，即墨水集中在原点处，扩散过程还没有发生，**此时我们可以将这种初始状态看作一个狄拉克$\delta$分布，这种分布是熵最小的分布，其熵实际上为零，系统不具有随机性。** 我们要来推导$f(x,t)$的表达式，由粒子质量守恒定律可以得到方程：  

$$f(x,t+\tau) = \int_{-\infty}^{+\infty}f(x-y,t)\rho(y)dy$$  

&emsp;&emsp;另外，如果我们把$f(x,t)$看作关于$t$的一元函数，我们可以将$f(x,t+\tau)$在$t$处进行泰勒展开：  

$$f(x,t+\tau) = f(x,t)+\frac{\partial f}{\partial t}\tau+O(\tau)$$  

&emsp;&emsp;同理，我们可以把$f(x-y,t)$在$x$处展开：  

$$f(x-y,t)=f(x,t) - \frac{\partial f}{\partial x}y + \frac{1}{2} \frac{\partial^{2}f}{\partial x^{2}}y^{2}+O(y^{2})$$  

&emsp;&emsp;联立以上三个等式得到扩散过程的SDE：  

$$f(x,t)+\frac{\partial f}{\partial t}\tau = \int_{-\infty}^{+\infty} \left[ f(x,t) - \frac{\partial f}{\partial x}y + \frac{1}{2} \frac{\partial^{2}f}{\partial x^{2}}y^{2} \right] \rho(y)dy$$  

&emsp;&emsp;通过化简，我们得到了最终的**扩散方程(4):**  

$$\begin{equation}
    \frac{\partial f}{\partial t} = \frac{D}{2\tau} \frac{\partial^{2}f}{\partial x^{2}}
\end{equation}$$  

&emsp;&emsp;求解这个偏微分方程，我们发现**随机过程$f(x,t)$是一个高斯过程(5):**  

$$\begin{equation}
    f(x,t) = \frac{1}{\sqrt{2\pi Ct}}\exp\left( -\frac{x^{2}}{2Ct} \right)
\end{equation}$$  

&emsp;&emsp;在之前机器学习系列的《交叉熵与KL散度》这篇博客文章中，我们证明了在给定均值与方差的条件下，高斯分布具有最大的熵，具体来说，**若有定义在整个实数轴上的随机变量$X$，给定其均值为$\mu$，方差为$\sigma^{2}$，则当$X \sim N(\mu, \sigma^{2})$时，随机变量$X$的熵$H(X)$最大，其熵为$H(X)=\frac{1}{2}\log(2\pi e \sigma^{2})$。而粒子的扩散过程是一个高斯过程，其在每个时间点$t$ 是都是一个高斯分布，具有最大的信息熵，同时随着时间$t$的增大，高斯分布的方差逐渐增大，其熵会进一步增大，当时间$t$趋于无穷时，高斯分布的熵也趋于无穷。整个粒子的扩散过程从开始熵为零的单点分布，到熵为无穷的高斯分布，系统的扩散总是趋向于熵最大的方式。**  

<center>
    <img src="https://s2.loli.net/2024/04/22/k1cszEIoYygA9MU.jpg" width=60% height=60%>
    <div align="center">Image1: 扩散过程</div>
</center>  
<br/>  

&emsp;&emsp;举上面这个例子，我是想让读者对熵增的过程有一个直观的感受，最大熵原理便是基于熵增定律，选择熵最大的分布是一个理智的选择，因为这符合自然界的规律。言归正传，最大熵原理和最优传输的熵正则理论有什么联系了？实际上我们只要对(2)式做一个小的等价变形，就可以发现其中的联系了，(2)式实际上也等价于(6)：  

$$\begin{equation}
    L_{\boldsymbol{C}}^{\eta}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \max_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})}  H(\boldsymbol{P}) - \eta \left< \boldsymbol{P},\boldsymbol{C} \right>
\end{equation}$$  

&emsp;&emsp;**我们原先对于熵正则的理解是在考虑熵的约束的条件下使得运输成本最小化，(6)式告诉我们，从最大熵原理的角度看，熵正则也可以理解成在考虑运输成本的情况下，使得联合分布$P(X,Y)$的熵最大。**  
&emsp;&emsp;实际上，早在1969年，Alan Wilson 在将最优传输理论应用到交通运输领域时，便考虑到了最大熵原理。交通系统往往具有复杂性于动态性，其包含了多种变量的随机性。例如，交通流量会受到天气、时间、事故等因素的影响。最优运输问题的传统解决方案往往倾向于找出成本最低的几条路线，这在理论上是有效的，但这些路线可能无法准确地反映这些动态变化。Wilson在最优传输理论中引入熵正则项，这种正则化旨在使运输方案更加现实，通过平滑运输路线的分布，从而更好地模仿现实世界网络中观察到的实际交通流。  

## From the Perspective of Information Theory and Geometry  

### Entropic Constraints on Joint Probabilities  

&emsp;&emsp;在(1)式中，运输矩阵$\boldsymbol{P}$的可行解集$U(\boldsymbol{\alpha},\boldsymbol{\beta})$ 可以看作随机变量$X,Y$的联合分布。对于联合分布$P(X,Y)$，有：  

$$\boldsymbol{\hat{P}} = \argmax_{\boldsymbol{P} \in U(\boldsymbol{\alpha,\beta})} H(\boldsymbol{P}) = \boldsymbol{\alpha \beta^{T}}$$  

&emsp;&emsp;即当$P(X,Y)=\boldsymbol{\alpha \beta^{T}}$ 时，联合分布的熵最大。如果我们想要最优传输问题的解具有较大的熵，则我们可以将最优传输问题的可行解集约束到$\boldsymbol{\hat{P}}$附近，这样可行使得求解出的联合分布便可以具有具有较大的信息熵。要实现这种约束，可以对解集$U(\boldsymbol{\alpha,\beta})$中的$\boldsymbol{P}$与$\boldsymbol{\hat{P}}$的KL散度进行约束，这样可以使得解集中的$\boldsymbol{P}$与$\boldsymbol{\hat{P}}$的分布相似，即约束在$\boldsymbol{\hat{P}}$ 附近，这样我们得到的新的可行解集为：  

$$\boldsymbol{U_{\gamma}}(\boldsymbol{\alpha},\boldsymbol{\beta}) = \{ \boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) | KL(\boldsymbol{P} || \boldsymbol{\alpha \beta^{T}}) \leq \gamma \}$$  

&emsp;&emsp;其中，$\gamma$ 是一个常数，用来控制约束程度。由信息论的知识我们可以有以下结论：  

$$\begin{split}
        & \frac{1}{2} \left( H(\boldsymbol{\alpha}) + H(\boldsymbol{\beta}) \right) \leq H(\boldsymbol{P}) \leq H(\boldsymbol{\alpha}) + H(\boldsymbol{\beta}) = H(\boldsymbol{\alpha \beta^{T}}) \\  
        & KL(\boldsymbol{P} || \boldsymbol{\alpha \beta^{T}}) = H(\boldsymbol{\alpha}) + H(\boldsymbol{\beta}) - H(\boldsymbol{P}) = I(X || Y)
\end{split}$$  

&emsp;&emsp;第一个式子确定了联合分布$P(X,Y)$的上下界，第二个式子说明$\boldsymbol{P}$ 与 $\boldsymbol{\alpha \beta^{T}}$ 的KL散度实际上是随机变量$X,Y$的**互信息量**。互信息量是一种衡量两个随机变量之间相互依赖程度的度量。给定两个随机变量X和Y，它们的互信息量 $I(X \parallel Y)$ 可以用联合概率分布来计算：  

$$I(X \parallel Y) = \sum_{x}\sum_{y}P(x,y)\log{\frac{P(x,y)}{P(x)P(y)}} = H(X)-H(X | Y)$$  

&emsp;&emsp;互信息量的含义为给定$Y$后，随机变量$X$所包含的信息的减少量。减少量越大，说明$X$和$Y$的依赖程度越大。也可以理解为联合概率分布$P(X,Y)$所提供的信息与独立分布$P(X),P(Y)$所提供的信息之间的信息差，信息差越大，说明$X$和$Y$的依赖关系越强。  
&emsp;&emsp;因此对$\boldsymbol{P}$与$\boldsymbol{\alpha \beta^{T}}$ 之间的KL散度的约束，也可以理解成对随机变量$X,Y$之间的互信息量的约束，即依赖程度的约束。我们希望找到随机变量$X,Y$之间依赖程度足够小的联合分布:  

$$\boldsymbol{U_{\gamma}}(\boldsymbol{\alpha},\boldsymbol{\beta}) = \{ \boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) | KL(\boldsymbol{P} || \boldsymbol{\alpha \beta^{T}}) \leq \gamma \} = \{ \boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) | h(\boldsymbol{P}) \ge h(\boldsymbol{\alpha}) + h(\boldsymbol{\beta}) - \gamma  \} \subset \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})$$  

&emsp;&emsp;此时引入熵正则的最优传输问题便转化为(3)式的形式，如果我们去写(3)式的拉格朗日函数，可以很容易将优化问题(3)转化为优化问题(2)，这里不再说明。  
&emsp;&emsp;我们来讨论一下极限情况，设$\boldsymbol{P^{*}}$是原始最优传输问题(1)的最优解，$\boldsymbol{P^{\gamma}}$是引入熵正则约束的最优传输问题(3)的最优解，很容易可以得到以下结论：  

- 当 $\gamma \rightarrow \infty$ 时，$\boldsymbol{U_{\gamma}}(\boldsymbol{\alpha},\boldsymbol{\beta}) \rightarrow \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})$，此时 $\boldsymbol{P^{\gamma}} \rightarrow \boldsymbol{P^{*}}, L_{\boldsymbol{C}}^{\gamma}(\boldsymbol{\alpha},\boldsymbol{\beta}) \rightarrow L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta}).$  
- 当 $\gamma \rightarrow 0$ 时，$KL(\boldsymbol{P} || \boldsymbol{\alpha \beta^{T}}) \rightarrow 0$，此时 $\boldsymbol{P^{\gamma}} \rightarrow \boldsymbol{\alpha \beta^{T}}, L_{\boldsymbol{C}}^{\gamma}(\boldsymbol{\alpha},\boldsymbol{\beta}) \rightarrow \boldsymbol{\alpha^{T}C\beta}$  

### Geometric Intuition  

&emsp;&emsp;我们可以从几何的角度来理解熵正则，这里我使用 Marco Cuturi 2013年发表在Nips上的论文《Sinkhorn Distances: Lightspeed Computation of Optimal Transport》中的配图来说明：  

<center>
    <img src="https://s2.loli.net/2024/04/22/fvY9PXokwb4ZHxR.png" width=60% height=60%>
    <div align="center">Image2: 熵正则的几何示例</div>
</center>  
<br/>  

&emsp;&emsp;论文中的部分符号与本博客不完全一致，读者需要自行对照理解，这里的论述以配图中的符号为标准。在之前关于最优传输的博客中，我们说明了原始的最优传输问题(1)实际上是一个线性规划问题，其解$\boldsymbol{P}$被约束在一个$\mathbb{R^{d \times d}}$空间中的一个多面体$U(\boldsymbol{r,c})$中，如图2中的红色多面体所示。在关于线性规划单纯形的讨论中，我们证明了线性规划问题的最优解一定在多面体的极点上，设原始最优传输问题(1)的最优解为$\boldsymbol{P^{*}}$，即图2中多面体上绿色的极点。多面体的内部存在着某一点$\boldsymbol{rc^{T}}$，其表示熵最大的联合分布$\boldsymbol{P}$。在前文的讨论中，我们说明了引入熵正则项的最优传输问题(3)实际上是将可行解集从原来的多面体，约束到熵最大的解$\boldsymbol{rc^{T}}$的附近，并将问题变成了一个严格的凸问题，这个新的严格凸的可行解集即为图中的蓝色区域。设问题(3)的最优解为$\boldsymbol{P_{\alpha}}$，即图2中凸区域上的蓝色点。$\boldsymbol{P_{\alpha}}$关于凸区域的切线是垂直于梯度方向$\boldsymbol{M}$。当 $\alpha \rightarrow 0$ 时，$\boldsymbol{P_{\alpha}} \rightarrow \boldsymbol{rc^{T}}$；当 $\alpha \rightarrow \infty$ 时，$\boldsymbol{P_{\alpha}} \rightarrow \boldsymbol{P^{*}}$，因此随着 $\alpha$ 的增大，问题(3)的最优解$\boldsymbol{P_{\alpha}}$会在多面体的内部形成一条路径，这条路径从起点$\boldsymbol{rc^{T}}$逐渐逼近到$\boldsymbol{P^{*}}$，如图2中红色的曲线所示。由于问题(3)实际上是与问题(2)等价的，给定一个$\alpha$，就会有一个对应的$\lambda$，当 $\lambda \rightarrow 0$ 时，$\boldsymbol{P^{\lambda}} \rightarrow \boldsymbol{rc^{T}}$；当 $\lambda \rightarrow \infty$ 时，$\boldsymbol{P^{\lambda}} \rightarrow \boldsymbol{P^{*}}$。因此这条路径也可看作是问题(2)的最优解$\boldsymbol{P^{\lambda}}$形成的。图2中右端放大的图示的含义是，虽然理论上当 $\lambda \rightarrow \infty$ 时，$\boldsymbol{P^{\lambda}} \rightarrow \boldsymbol{P^{*}}$，但计算机的数值精度是有限的，不可能真正现实趋于无穷大，因此在实际计算中，硬件所能表示的最大的$\lambda_{max}$所对应的解$\boldsymbol{P^{\lambda_{max}}}$只是$\boldsymbol{P^{*}}$的一个近似值。  

## Sinkhorn Distance  

&emsp;&emsp;在引入熵正则项后，我们可以定义一个新的距离：  

$$d_{\boldsymbol{C,\gamma}}(\boldsymbol{\alpha,\beta}) :=  \min_{\boldsymbol{P} \in \boldsymbol{U_{\gamma}}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right>$$  

若成本矩阵$\boldsymbol{C}$为度量矩阵，即：  

$$\boldsymbol{C} \in \{ \boldsymbol{C} \in \mathbb{R_{+}^{d \times d}}: \boldsymbol{C^{T}}=\boldsymbol{C};\forall i,j \leq d, c_{ij}=0 \Leftrightarrow i=j; \forall i,j,k \leq d, c_{ij} \leq c_{ik}+c_{kj} \}$$ 

则 $d_{\boldsymbol{C,\gamma}}(\boldsymbol{\alpha,\beta})$ 被定义为分布 $\boldsymbol{\alpha,\beta}$ 之间的**Sinkhorn Distance**。  
&emsp;&emsp;证明 Sinkhorn Distance 能够作为分布之间的距离类似于之前博客中 Wasserstein Distance 的证明过程，这里由于篇幅原因不再多加赘述。基于熵正则的最优传输问题所导出的 Sinkhorn Distance 具有广泛的应用，以下是几个典型领域：  

- **图像处理与计算机视觉**：在图像生成、图像配准和图像分类等任务中，可以使用Sinkhorn距离来度量两个图像之间的相似性，尤其是在非刚性配准和变形问题中。  
- **自然语言处理**：在文本生成、文本分类和词嵌入等任务中，Sinkhorn距离可以用于比较两个文本之间的相似性，帮助解决文本对齐和翻译等问题。  
- **机器学习与模式识别**：在模式匹配、聚类和异常检测等任务中，Sinkhorn距离可以用作特征之间相似性的度量，从而提高模型的性能和鲁棒性。  
- **经济学与运筹学**：在经济学中，Sinkhorn距离可以用于衡量两个市场之间的相似性，从而帮助分析市场结构和预测市场变化。在运筹学中，它可以用于解决运输和分配等问题。  

## References  

- **[1] Book: Peyré G, Cuturi M. Computational optimal transport[J]. Center for Research in Economics and Statistics Working Papers, 2017 (2017-86).**  
- **[2] Paper: Cuturi M. Sinkhorn distances: Lightspeed computation of optimal transport[J]. Advances in neural information processing systems, 2013, 26.**  
- **[3] Video: B站《随机过程》张灏 UP-我在人间凑数的这几年.**

