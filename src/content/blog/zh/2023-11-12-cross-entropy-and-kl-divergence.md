---
title: 交叉熵与KL散度
tags:
    - Machine Learning
---  

## toc

&emsp;&emsp;在机器学习中，我们经常使用信息熵、交叉熵、KL散度等概率，例如在决策树中，我们使用基于信息熵的信息增益来构造树形结构；而交叉熵常被用于分类问题的损失函数；KL散度则被用于衡量两个分布之间的差异。本文将会介绍这些常用的概率，以便于我们今后学习相应的机器学习模型。

## 信息熵  (entropy)  

### 熵

&emsp;&emsp;在现实中，我们会接触到各种各样的信息，如何对信息进行量化便成为了一个重要的问题。信息论是应用数学的一个分支，由美国数学家香农提出并发展壮大，主要研究的是对一个事件包含信息的多少进行量化。  
&emsp;&emsp;信息论的基本思想是一个小概率事件发生了，要比大概率事件发生，提供的信息更多。在信息论中，我们认为事件的信息量具有以下三条性质：  

- 大概率事件所包含的信息量较小。  
- 小概率事件所包含的信息量较大。
- 独立事件的信息量可以进行累加。

&emsp;&emsp;由以上三条性质，我们定义了某一事件$X=x$的**自信息量**(self-information)为:  

$$
I(x)=-\log{P(x)}
$$  

&emsp;&emsp;其中$X$为随机变量，表示某一事件；$x$为随机变量$X$的取值。当上式中的$\log$以2为底数时，$I(x)$的单位是比特(bit)或者香农(shannons)；当$\log$以2为底数时，$I(x)$单位是奈特(nats)。这两个单位之间可以通过对数换底公式相互转换。  
&emsp;&emsp;自信息量表示单个事件的信息量。若我们已知事件$X$服从某一概率分布$P(X)$，我们可以使用**香农熵**(Shannon entropy)来对整个概率分布所包含的信息总量进行量化：  

$$
H(X)=\mathbb{E}_{X \sim P}[I(x)]
$$  

&emsp;&emsp;若随机变量$X$为离散型随机变量，则熵可以写为求和形式：  

$$
H(X)=\sum_{i=1}^{N}P(x_i)I(x_i)=-\sum_{i=1}^{N}P(x_i)\log{P(x_i)}
$$  

&emsp;&emsp;若随机变量$X$为连续型随机变量，则熵可以写为积分形式：  

$$
H(X)=\int_{\mathcal{X}}P(x)I(x)dx=-\int_{\mathcal{X}}P(x)\log{P(x)}
dx$$  

### 联合熵  

&emsp;&emsp;若有两个随机变量$X,Y$，且服从某个联合分布$P(X,Y)$，我们可以使用联合熵来对联合概率分布所包含的信息量进行量化：  

$$
H(X,Y)=-\sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}}P(x,y)\log{P(x,y)}
$$  

&emsp;&emsp;以上给出的是$X,Y$为离散型随机变量的联合熵，若$X,Y$为连续型随机变量，则依照熵的形式，联合熵也可以写成积分形式：  

$$
H(X,Y)=-\int_{\mathcal{X}}\int_{\mathcal{Y}}P(x,y)\log{P(x,y)} dydx
$$

### 条件熵  

&emsp;&emsp;在数理统计中，我们还学习了条件概率分布，表示在某个事件发生后，另一个事件所发生的概率，在信息论中，用条件熵来表示条件概率分布所包含的信息。若有两个离散随机变量$X,Y$，且已知联合概率分布$P(X,Y)$，条件概率分布$P(Y|X)$，则该条件分布的条件熵为：  

$$
H(Y|X)=-\sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}}P(x,y)\log{P(y|x)}
$$  

&emsp;&emsp;当$X,Y$为连续型随机变量时，上式可以写出积分形式：  

$$
H(Y|X)=-\int_{\mathcal{X}}\int_{\mathcal{Y}}P(x,y)\log{P(y|x)dydx}
$$  

### 最大熵思想  

&emsp;&emsp;既然信息熵可以用来表示信息量的大小，人们自然希望找到包含信息量最大的概率分布。当我们的概率分布中存在未知参数时，可以使用**最大化分布的熵**的思想来估计未知参数，这类方法在机器学习中被称为最大熵学习，这里不展开讨论最大熵学习，有兴趣的读者可查阅相关资料进行了解。  
&emsp;&emsp;通过最大熵思想，我们有一个非常有意思的发现：**若有定义在整个实数轴上的随机变量$X$，其均值为$\mu$，方差为$\sigma^{2}$，当$X \sim N(\mu,\sigma^2)$时，熵 $H(X)$ 最大.** 这个结论的证明放在目录，有兴趣的读者可自行阅读。高斯分布具有最大的信息熵，这也是为什么在现实生活中大量随机事件都服从高斯分布，因为自然界总是偏向于制造最大的不确定性，从而包含最多的信息，即最大的熵。  

## 交叉熵(cross entropy)

&emsp;&emsp;在机器学习中，我们常常需要比较两个分布之间的相似程度，例如当我们用一个估计的分布去近似真实分布时，我们自然希望这两个分布越相似越好。交叉熵便是衡量两个分布之间相似程度的一种度量方法，因此经常被用作机器学习模型的损失函数。  
&emsp;&emsp;为了理解交叉熵为什么能够度量两个分布的相似程度，我们借助贝叶斯统计的思想来进行解释。根据贝叶斯思想，对于某一个事件$X$，我们会有一个先验的认知，即$X$的先验分布$P_0(x)$，在先验认知下，事件$X$带给我们的信息量为：  

$$
I_{0}(x)=-\log{P_{0}(x)}
$$  

&emsp;&emsp;假设事件$X$的真实分布为$P_{1}(x)$，则**我们在主观认知下，通过客观事实所得到的事件$X$的概率分布所包含的信息总量**为：  

$$
H(P_0,P_1)=-\sum_{i=1}^{N}P_{1}(x)\log{P_{0}(x)}dx
$$  

&emsp;&emsp;当$X$为连续型随机变量时，其积分形式为：  

$$
H(P_{0},P_{1})=-\int_{\mathcal{X}}P_{1}(x)\log{P_{0}(x)}dx
$$  

&emsp;&emsp;上式即为分布$P_0,P_1$的**交叉熵**。若交叉熵较大，说明在已有主观先验认知下，事件$X$的实际情况带给我们的信息量较大，说明我们的先验认知与实际情况差别较大，即$P_{0}(x)$与$P_{1}(x)$的相似度较低；若交叉熵较小，说明在已有主观先验认知下，事件$X$的实际情况带给我们的信息量较小，说明我们的先验认知与实际情况差别较小，即$P_{0}(x)$与$P_{1}(x)$的相似度较高。  
&emsp;&emsp;在机器学习中，我们希望学习到的概率分布$P_{m}$与训练数据所估计的真实分布$P_{t}$足够相似，这时我们通常将分布$P_{m}$与$P_{t}$的交叉熵作为损失函数，例如逻辑回归模型，通过最小化交叉熵来调整$P_{m}$，使得$P_{m}$与真实分布$P_{t}$足够相似。  
&emsp;&emsp;另外，有一个非常有意思的结论，**最小化交叉熵实际上是等价于极大似然估计**，这个结论的证明将会放在附录。

## KL散度(KL Divergence)  

&emsp;&emsp;上文介绍了交叉熵，其可以用来衡量两个分布的相似程度。但交叉熵存在一个问题，即若我们主观的先验认知与客观实际完全一致，即$P_{0}=P_{1}$，此时我们实际上并没有得到任何信息，但交叉熵的计算结果为$P_{0}$的信息熵，并不为$0$。因此，我们可以转而考虑信息的增量，KL散度实际上就时在考虑信息熵的增量，我们先给出两个分布$P_0,P_1$的KL散度的计算公式：  

$$
KL(P_1||P_0)=\mathbb{E}_{X \sim P_1}[\log{\frac{P_{1}(x)}{P_{0}(x)}}]
$$  

&emsp;&emsp;通过将上式进行调整，我们可以将$P_{0}$与$P_{1}$的KL散度写成$P_{0}$与$P_{1}$的交叉熵$H(P_{0},P_{1})$与$P_{0}$的信息熵之差：  

$$
KL(P_{1}||P_{0})=H(P_{0},P_{1})-H(P_{0})
$$  

&emsp;&emsp;通过上式我们可以发现，当$P_1$与$P_0$的KL散度较大，说明在已有主观认知下，我们从客观事件获得信息增量较大，说明我们的主观认知与客观现实不一致，即$P_1$与$P_0$的相似度较低；当$P_1$与$P_0$的KL散度较小，说明在已有主观认知下，我们从客观事件获得信息增量较小，说明我们的主观认知与客观现实较为一致，即$P_1$与$P_0$的相似度较大；当$P_1$与$P_0$的KL散度等于0时，说明已有主观认知下，我们从客观事件中没有获得额外的信息，说明我们的主观认知与客观现实完全一致，即$P_0=P_1$.  
&emsp;&emsp;KL散度衡量了一种信息增益，因此也被称为**相对熵**。在机器学习中我们同样可以使用KL散度作为损失函数。

## 附录  

### (一) 高斯分布具有最大的信息熵  

**结论:** 已知定义在整个实数轴上的连续型随机变量$X$的均值为$\mu$，方差为$\sigma^{2}$，当$X \sim N(\mu,\sigma^{2})$，$X$的信息熵$H(X)$最大。  
**证明:**  
&emsp;&emsp;设随机变量$X$的概率密度函数为$p(x)$，由题意可知：  

$$
E(X)=\int_{-\infty}^{+\infty}xp(x)dx=\mu,\quad Var(X)=\int_{-\infty}^{+\infty}(x-\mu)^{2}p(x)dx=\sigma^{2}
$$  

&emsp;&emsp;根据最大熵思想，可以得到如下一个带约束的优化问题：  

$$
\begin{split}
    \max_{p(x)} \quad & H(X)=-\int_{-\infty}^{+\infty}p(x)\ln{p(x)}dx  \\
    s.t. \quad & \int_{-\infty}^{+\infty}p(x)=1 \\
    & \int_{-\infty}^{+\infty}xp(x)dx=\mu  \\
    & \int_{-\infty}^{+\infty}(x-\mu)^{2}p(x)dx=\sigma^{2}
\end{split}
$$  

&emsp;&emsp;上述原问题的拉格朗日函数为：  

$$
\begin{split}
    Q(p(x),\lambda_1,\lambda_2,\lambda_3) = &-\int_{-\infty}^{+\infty}p(x)\ln{p(x)}dx+\lambda_1 \left( \int_{-\infty}^{+\infty}p(x)-1 \right) \\
    &+ \lambda_2 \left( \int_{-\infty}^{+\infty}xp(x)dx-\mu \right)+\lambda_3 \left( \int_{-\infty}^{+\infty}(x-\mu)^{2}p(x)dx-\sigma^{2} \right)
\end{split}
$$  

&emsp;&emsp;令$\lambda=\begin{bmatrix}
    \lambda_1,\lambda_2,\lambda_3
\end{bmatrix}^{T}$，则原问题的无约束形式可以写为  

$$
\begin{split}
     \max_{p(x)}\min_{\lambda} \quad & Q(p(x),\lambda_1,\lambda_2,\lambda_3)  \\
     \space s.t. \quad & \lambda \ge 0
\end{split}
$$  

&emsp;&emsp;则原问题的对偶问题为：  

$$
\begin{split}
     \min_{\lambda}\max_{p(x)} \quad & Q(p(x),\lambda_1,\lambda_2,\lambda_3)  \\
     \space s.t. \quad & \lambda \ge 0
\end{split}
$$  

&emsp;&emsp;设$p^{*}(x)$为原问题最优解，$\lambda^{*}$为对偶问题最优解，由KKT条件得：  

$$
\frac{\partial Q}{\partial p(x)} |_{p(x)=p^{*}(x)}=-[\ln{p^{*}(x)+1}]+\lambda_1^{*}+\lambda_2^{*}x+\lambda_3^{*}(x-\mu)^{2}=0
$$  

$$
\Rightarrow p^{*}(x)=e^{\lambda_1^{*}-1+\lambda_2^{*}x+\lambda_3^{*}(x-\mu)^{2}}
$$  

&emsp;&emsp;之后对$p^{*}(x)$做一些变形：  

$$
\begin{split}
    p^{*}(x) &= e^{\lambda_1^{*}-1+\lambda_2^{*}x+\lambda_3^{*}(x-\mu)^{2}}=e^{\lambda_1^{*}-1}e^{\lambda_3^{*}x^2+(\lambda_2^{*}-2\mu\lambda_3^{*})x+\mu^{2}\lambda_3^{*}} \\
    &= Ce^{\lambda_3^{*}[x^2+(\frac{\lambda_2^{*}}{\lambda_3^{*}}-2\mu)x+\mu^2]} = Ce^{\lambda_3^{*}[x-(\mu-\frac{\lambda_2^{*}}{2\lambda_3^{*}})]^{2}}
\end{split}
$$  

&emsp;&emsp;由密度函数$p^{*}(x)$的非负性以及正则性可知：$C>0,\lambda_3^{*}<0$；指数部分中的二次函数 $[x-(\mu-\frac{\lambda_2^{*}}{2\lambda_3^{*}})]^{2}$ 表明 $p^{*}(x)$的对称轴为 $\mu-\frac{\lambda_2^{*}}{2\lambda_3^{*}}$，由于对称的密度函数，其对称轴一定等于均值可知：$\lambda_2^{*}=0$。  
&emsp;&emsp;由于$\lambda_3^{*}>0$，可设$\lambda_3^{*}=-\beta$，则$p^{*}(x)$可化简为：  

$$
p^{*}(x)=Ce^{-\beta(x-\mu)^{2}}
$$

&emsp;&emsp;将$p^{*}(x)$代入正则化约束得：  

$$
\begin{split}
    1 &= \int_{-\infty}^{+\infty}p^{*}(x)dx=C\int_{-\infty}^{+\infty}e^{-\beta(x-\mu)^{2}}dx \\
    &=C\int_{-\infty}^{+\infty}e^{-\beta y^{2}}dy=\frac{C}{\sqrt{\beta}}\int_{0}^{+\infty} z^{-\frac{1}{2}}e^{-z}dz \\
    &=\frac{C}{\sqrt{\beta}}\Gamma \left( \frac{1}{2} \right) = C\sqrt{\frac{\pi}{\beta}}
\end{split}
$$  

&emsp;&emsp;从而得：$C=\sqrt{\frac{\beta}{\pi}}$，再利用方差约束条件得：  

$$
\begin{split}
    \sigma^{2} &= \int_{-\infty}^{+\infty}(x-\mu)^2p^{*}(x)dx = C\int_{-\infty}^{+\infty}(x-\mu)^{2}e^{-\beta(x-\mu)^{2}}dx  \\
    &=2C\int_{0}^{+\infty}y^{2}e^{-\beta  y^{2}}dx = \frac{C}{\beta \sqrt{\beta}} \int_{0}^{+\infty} z^{\frac{1}{2}}e^{-z}dz \\
    &= \frac{C}{\beta \sqrt{\beta}} \Gamma \left( \frac{3}{2} \right) = \sqrt{\frac{\beta}{\pi}} \cdot \frac{1}{\beta \sqrt{\beta}} \cdot \frac{\sqrt{\pi}}{2}= \frac{1}{2\beta}
\end{split}
$$  

&emsp;&emsp;从而得：$\beta = \frac{1}{2\sigma^{2}}$，联立：  

$$
\left \{
\begin{array}{l}
p^{*}(x)=Ce^{-\beta(x-\mu)^{2}}  \\
C=\sqrt{\frac{\beta}{\pi}}  \\
\beta = \frac{1}{2\sigma^{2}} \\
\end{array} \right.
$$  

&emsp;&emsp;解得：  

$$
p^{*}(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}
$$  

&emsp;&emsp;即当$X \sim N(\mu,\sigma^{2})$时，熵$H(X)$最大，证毕.  

### (二) 最小化交叉熵等价于极大似然估计  

**结论：** 假设我们从训练数据集$T_{train}$中所得到的数据的经验分布为$P_{t}(x)=\frac{1}{N}$(最大熵思想)，$N$为训练数据的样本容量，我们所需要学习的模型为$P_{m}(x;\theta)$，则有：  

$$
\min_{\theta} H(P_{t}(x),P_{m}(x;\theta)) \Leftrightarrow \max_{\theta} \prod_{x \in T} P_{m}(x;\theta)
$$  

**证明:**  

$$
\begin{split}
    \max_{\theta} \prod_{x \in T} P_{m}(x;\theta) & \Leftrightarrow \max_{\theta} \sum_{x \in T} \log{P_{m}(x;\theta)} \Leftrightarrow -\min_{\theta} \sum_{x \in T} \log{P_{m}(x;\theta)} \\
    & \Leftrightarrow -\min_{\theta} \sum_{i=1}^{N} \frac{1}{N}\log{P_{m}(x;\theta)} \Leftrightarrow \min_{\theta} \mathbb{E}_{X \sim P_{t}}[-\log{P_{m}(x;\theta)}]  \\
    & \Leftrightarrow \min_{\theta} H(P_{t}(x),P_{m}(x;\theta))
\end{split}
$$  

&emsp;&emsp;证毕.  

## 参考  

**[1] Book: 董平,机器学习中的统计思维(Python实现)**  
**[2] Blog：知乎,康斯坦丁,一篇文章讲清楚交叉熵和KL散度**

