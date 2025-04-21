---
title: 机器学习概论
tags:
    - Machine Learning
---  

## toc

## 机器学习  
**定义**  
&emsp;&emsp;机器学习(Machine Learning)是关于计算机基于数据构建模型并运用模型对数据进行预测与分析的一门学科。  

**特点**  
&emsp;&emsp;机器学习的主要特点是:  
&emsp;&emsp;(1) 机器学习以计算机及网络为**平台**，是建立在计算机网络上的。  
&emsp;&emsp;(2) 机器学习以数据为**研究对象**，是数据驱动的学科。  
&emsp;&emsp;(3) 机器学习的**目的**是对数据进行预测与分析。  
&emsp;&emsp;(4) 机器学习以**方法**为中心，机器学习方法构建模型并应用模型进行预测与分析。  
&emsp;&emsp;(5) 机器学习是微积分、线性代数、概率论、统计学、信息论、数值计算、最优化及计算机科学等多个领域的交叉**学科**，并且在发展中逐步形成独自的理论体系与方法论。  

**步骤**  
&emsp;&emsp;实现机器学习方法的步骤：  
&emsp;&emsp;**(1) 得到一个有限的训练数据集合；**  
&emsp;&emsp;**(2) 确定包含所有可能的模型的假设空间，即学习模型的集合；**  
&emsp;&emsp;**(3) 确定模型选择的准则，即学习的策略；**  
&emsp;&emsp;**(4) 实现求解最优模型的算法，即学习的算法；**  
&emsp;&emsp;**(5) 通过学习方法选择最优模型；**  
&emsp;&emsp;**(6) 利用学习的最优模型对新数据进行预测与分析。**  

## 机器学习的分类  
&emsp;&emsp;我们可以从多个方面对机器学习方法进行分类。  

&emsp;&emsp;(1) 基本分类  

- **监督学习(Supervised Learning):** 从标注数据中学习预测模型的机器学习问题。标注数据表示输入到输出的对应关系，预测模型对给定的输入产生相应的输出。监督学习的本质是学习输入到输出的映射的统计规律。  
    - **监督学习的实例:** 线性回归、逻辑回归、决策树、支持向量机、神经网络等。  
- **无监督学习(Unsupervised Learning):** 从无标注数据中学习预测模型的机器学习问题。无标注数据是自然得到的数据，预测模型表示数据的类别、转换或概率。无监督学习的本质是学习数据中的统计规律或潜在结构。  
    - **无监督学习的实例:** 聚类分析、主成分分析、高斯混合模型、自编码等。 
- **强化学习(Reinforcement Learning):** 指智能系统在与环境的连续互动中学习最优行为策略的机器学习问题。假设智能系统与环境的互动基于马尔可夫决策过程(Markov decision process)，智能系统能观测到的是与环境互动得到的数据序列。强化学习的本质是学习最优的序贯决策。  
    - **强化学习的实例:** Q-学习、深度Q网络、蒙特卡洛树搜索等。  

&emsp;&emsp;(2) 按模型分类  

- **概率模型与非概率模型**  
    - **概率模型:** 模型的形式为条件概率分布$P(y\vert x)$.  
        - **概率模型的实例:** 决策树、朴素贝叶斯、隐马尔可夫模型、条件随机场、概率图模型等。
    - **非概率模型:** 模型的形式为决策函数$y = f(x)$.  
        - **非概率模型的实例:** 感知机、支持向量机、AdaBoost、神经网络等。 
- **线性模型与非线性模型** 
    - **线性模型:** 非概率模型中，决策函数$y=f(x)$是线性函数的模型。  
        - **线性模型的实例:** 感知机、线性支持向量机、k近邻、潜在语义分析等。  
    - **非线性模型:** 非概率模型中，决策函数$y=f(x)$是非线性函数的模型。
        - **非线性模型的实例:** 核函数支持向量机、AdaBoost、神经网络等。  
- **参数化模型与非参数化模型**
    - **参数化模型:** 假设模型参数的维数是固定的，模型可以由有限维参数完全刻画。  
        - **参数化模型的实例:**  感知机、朴素贝叶斯、逻辑回归、高斯混合模型等。
    - **非参数化模型:** 假设模型参数的维度不固定或者说是无穷大，随着训练数据量的增加而不断增大。
        - **非参数化模型的实例:** 决策树、支持向量机、AdaBoost、k近邻等。  

&emsp;&emsp;(3) 按算法分类  

- **在线学习(Online Learning):** 指每次接受一个样本，进行预测，之后学习模型，并不断重复该操作的机器学习。  
    - **在线学习的实例:** 在线线性回归、在线逻辑回归、在线决策树、在线神经网络等。  
- **批量学习(Batch Learning):** 一次性接受所有数据，学习模型，之后进行预测。
    - **批量学习的实例:** 线性回归、逻辑回归、决策树、神经网络、支持向量机等。 

 &emsp;&emsp;在监督学习方法又可以生成方法与判别方法：  

- **生成方法:** 原理上由数据学习联合概率分布$P(X,Y)$，然后再求出条件概率分布$P(Y \vert X)$作为预测的模型，即生成模型。
    - **生成方法的特点:** 生成方法可以还原出联合概率分布$P(X,Y)$，而判别方法则不能；生成方法的学习收敛速度更快，即当样本容量增加时，学到的模型可以更块地收敛于真实模型；当存在隐变量时，仍可以用生成方法学习，此时判别方法就不能用；生成模型的输入和输出变量均为随机变量；生成方法所需的数据量较大。  
    - **生成模型的实例:** 朴素贝叶斯模型、隐马尔可夫模型等。  
- **判别方法:** 由数据直接学习决策函数$f(X)$或条件概率分布$P(Y \vert X)$作为预测模型，即判别模型。  
    - **判别方法的特点:** 判别方法直接学习的是条件概率分布$P(Y \vert X)$或决策函数$f(X)$，直接面对预测，往往学习的准确率更高；由于直接学习$P(Y \vert X)$和$f(X)$，可以对数据进行各种程度上的抽象、定义特征并使用特征，因此可以简化学习问题；判别方法不需要输入和输出变量均为随机变量；判别方法所需的样本量少于生成方法。  
    - **判别模型的实例:** 感知机、Logistic回归模型、支持向量机、条件随机场等。

### 基本分类的相关概念  

#### 监督学习  

- **输入空间:** 输入的所有可能取值的集合，记为$\mathcal{X}$.  
- **输出空间:** 输出的所有可能取值的集合，记为$\mathcal{Y}$.  
- **实例:** 每一个具体的输入，通常由特征向量表示：  

$$
x_i = \begin{bmatrix}
    x_{i}^{(1)}, x_{i}^{(2)}, \dots, x_{i}^{(n)}
\end{bmatrix}^T
$$  

&emsp;&emsp;其中$x_{i}^{(j)}$表示第$i$个实例的第$j$个特征。  

- **特征空间:** 所有特征向量存在的空间称为特征空间，记为$\mathcal{F}$。  
- **训练集:** 模型用于学习的数据集合，样本容量为$N$的训练集记为：  

$$
T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

&emsp;&emsp;输入变量$X$和输出变量$Y$有不同的类型，可以是连续的，也可以是离散的。人们根据输入输出变量的不同类型，对预测任务给予不同的名称：  

- **回归问题:** 输入变量与输出变量均为连续变量的预测问题。  
- **分类问题:** 输出变量为有限个离散变量的预测问题。  
- **标注问题:** 输入变量与输出变量均为变量序列的预测问题。  

&emsp;&emsp;对于监督学习，主要就是研究输入到输出之间的统计规律，所以要有关于输入和输出和模型的基本假设。  

- **监督学习的基本假设:** 输入变量$X$与输出变量$Y$具有联合概率分布$P(X,Y)$。训练数据与测试数据被看作是依联合概率分布$P(X,Y)$独立同分布产生的。  
- **监督学习的目的:** 学习一个输入空间到输出空间的映射，这一映射以模型的形式表示。  
- **模型形式:** 监督学习的模型可以是概率模型或非概率模型，由条件概率分布$P(Y\vert X)$或决策函数$Y=f(X)$表示，随具体学习方法而定。对具体的输入实例进行相应的输出预测时写作$P(y \vert x)$或决策函数$y=f(x)$.  
- **假设空间:** 所有可能的模型的集合，记为$\mathcal{H}$.  

- **监督学习的过程:** 监督学习分为学习和预测两个过程，由学习系统和预测系统完成。在学习过程中，学习系统利用给定的训练数据集，通过学习(或训练)得到一个模型，表示为条件概率分布$\hat{P}(Y \vert X)$或决策函数$Y=\hat{f}(X)$。条件概率分布$\hat{P}(Y \vert X)$或决策函数$Y=\hat{f}(X)$描述输入与输出变量之间的映射关系。在预测过程中，预测系统对于给定的测试样本集中的输入$X_{N+1}$，由模型$\hat{y}_{N+1}=arg \space max \space \hat{P}(y \vert x_{N+1})$或$\hat{y}_{N+1}=\hat{f}(x_{N+1})$给出相应的输出$\hat{y}_{N+1}$。监督学习的过程可以用图1来描述。  

<center>
    <img src="https://s2.loli.net/2023/09/25/6FAu9srNYcwDSkh.jpg" width=60% height=60%>
    <div align="center">Image1: 监督学习的过程</div>
</center>  

#### 无监督学习  

&emsp;&emsp;无监督学习中的大部分概率与监督学习中的一致，不同点主要在于以下几个方面：  

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<p align="center"><font face="黑体" size=2.>表1 监督学习与无监督学习的区别</font></p>
<div class="center">

||监督学习|无监督学习|
|:---:|:---:|:---:|
|数据|标注数据|无标注数据|
|输出|输出空间$\mathcal{Y}$|隐式结构空间$\mathcal{Z}$|
|预测模型|对给定输入产生相应的输出|表示数据的类别、转换或概率|
|本质|学习输入到输出的映射的统计规律|学习数据中的统计规律或潜在结构|
</div>  


 - **无监督学习的过程:** 和监督学习类似，无监督学习由学习系统与预测系统构成。在学习过程中，学习系统从训练数据集学习，得到一个最优模型，表示为$z=\hat{g}(x)$，条件概率分布$\hat{P}(z\vert x)$或者条件概率分布$\hat{P}(x \vert z)$。在预测过程中，预测系统对于给定的输入$x_{N+1}$，由模型$\hat{z}_{N+1}=\hat{g}(x_{N+1})$或$\hat{z}_{N+1}=\arg\max\limits_{z} \hat{P}(z \vert x_{N+1})$给出相应的输出$\hat{z}_{N+1}$，进行聚类或降维，或者由模型 $\hat{P}(x \vert z)$给出输入的概率$\hat{P}(x_{N+1} \vert z_{N+1})$，进行概率估计。无监督学习的过程可以用图2来描述。  

<center>
    <img src="https://s2.loli.net/2023/09/27/itEGIlOMS9mWbhQ.jpg" width=60% height=60%>
    <div align="center">Image2: 无监督学习的过程</div>
</center>  

#### 强化学习  

&emsp;&emsp;强化学习其实要强调一个互动，互动是指的是智能系统与环境之间的一个连续互动，通过这个互动，学习一个最优的行为策略。  

- **强化学习的过程:** 在每一步$t$，智能系统从环境中观测到一个状态$(state)s_t$，与一个奖励$(reward)r_t$，采取一个动作$(action) a_t$。环境根据智能系统选择的动作，决定下一步$t+1$的状态$s_{t+1}$与奖励$r_{t+1}$。要学习的策略表示为给定的状态下采取的动作。智能系统的目标不是短期奖励的最大化，而是长期累积奖励的最大化。强化学习过程中，智能系统不断试错$(trial \space\ and \space error)$，以达到学习最优策略的目的。强化学习的过程可以用图3来描述。  

<center>
    <img src="https://s2.loli.net/2023/09/27/x4InRw56OMVubzt.jpg" width=60% height=60%>
    <div align="center">Image3: 强化学习的过程</div>
</center>  

## 机器学习方法的三要素  

&emsp;&emsp;机器学习方法都是由模型、策略和算法构成，即机器学习方法由三要素构成，可以简单地表示为：  

$$
方法(Method)=模型(Model)+策略(Strategy)+算法(Algorithm)
$$  

&emsp;&emsp;下面主要介绍监督学习的三要素。  

### 监督学习的三要素  

#### 模型  
&emsp;&emsp;对于监督学习，模型主要可以表示成两种形式，一种是条件概率形式$P(y \vert x)$，另一种是决策函数的形式$y=f(x)$。  
&emsp;&emsp;如果模型是由参数向量$\theta$决定的，我们称所有可能的参数向量组成的空间为参数空间$\Theta$，那么这个假设空间$\mathcal{H}$就应该是由参数空间$\Theta$决定的了。  
&emsp;&emsp;(1) 当用决策函数的形式表示模型时，定义：  

- **决策函数:** $Y = f(X)$
- **假设空间:** 决策函数的集合$\mathcal{H} = \{f \vert Y=f(X) \}$.  
- **函数族:** 假设空间$\mathcal{H}$是由一个参数向量$\theta$决定的函数族构成: $\mathcal{H}=\{f \vert Y=f_{\theta}(X),\theta \in \mathbb{R}^{n}\}$.  
- **参数空间:** $\Theta = \{\theta \vert \theta \in \mathbb{R}^{n} \}$.  

&emsp;&emsp;以线性回归问题为例，这个问题的模型可以表示为决策函数的形式。  

- **决策函数:** $f(x) = w^{T}x+b, x \in \mathbb{R}^{n},w \in \mathbb{R}^{n}, b \in \mathbb{R}$.  
- **假设空间:** $\mathcal{H}=\{f \vert y = f_{\theta}(x), \theta=(w,b), \theta \in \mathbb{R}^{n+1}\}$  
- **参数空间:** $\Theta = \{\theta \vert \theta = (w,b),\theta \in \mathbb{R}^{n+1}\}$.  

&emsp;&emsp;(2) 当用条件概率的形式表示模型时，定义:  

- **条件概率分布:** $P(Y \vert X)$.  
- **假设空间:** 条件概率分布的集合$\mathcal{H}=\{P \vert P(Y \vert X)\}$.  
- **分布族:** 假设空间$\mathcal{H}$是由一个参数向量$\theta$决定的分布族构成: $\mathcal{H}=\{P\vert P_{\theta}(Y\vert X), \theta \in \mathbb{R}^{n}\}$.  
- **参数空间:** $\Theta = \{\theta \vert \theta \in \mathbb{R}^{n}\}$.  

&emsp;&emsp;以二项Logistic回归为例，这个问题的模型可以用条件概率分布的形式表示。  

- **条件概率分布:**  

$$
P(Y=1 \vert x) = \frac{exp(wx+b)}{1+exp(wx+b)}
$$  

$$
P(Y=0 \vert x) = \frac{1}{1+exp(wx+b)}
$$  

&emsp;&emsp;其中，$x \in \mathbb{R}^{n}, Y \in \{0,1\}, w \in \mathbb{R}^{n}, b \in \mathbb{R}$.  

- **假设空间:** $\mathcal{H} = \{P \vert P_{\theta}(Y\vert X), \theta = (w,b), \theta \in \mathbb{R}^{n+1} \}$.  
- **参数空间:** $\Theta = \{\theta \vert \theta = (w,b), \theta \in \mathbb{R}^{n+1} \}$.  

#### 策略  

&emsp;&emsp;有了模型的假设空间，机器学习接着需要考虑的是按照什么样的准则学习或者选择最优模型。机器学习的目标在于从假设空间中选取最优模型。所谓策略其实就是一种学习准则，用来选择最优模型的。想要选择模型，那么一定要知道如何度量模型的好坏。所以，这里先要引入几个概念。  

- **损失函数(Loss function):** 度量模型一次预测的好坏，记作$L(Y,f(X))$.几种常用的损失函数如下：  

&emsp;&emsp;(1)**0-1损失函数:** 主要用于分类问题。  

$$
L(Y,f(X))= \left \{
\begin{array}{rcl}
1, & {Y \neq f(X)}\\
0,& {Y = f(X)}\\
\end{array} \right.
$$ 
    
&emsp;&emsp;(2)**平方损失函数:** 主要用于回归问题。  

$$
L(Y,f(X))=(Y-f(X))^2
$$  

&emsp;&emsp;(3)**绝对损失函数:** 主要用于回归问题。  

$$
L(Y,f(X))= \vert Y-f(X) \vert
$$  

&emsp;&emsp;(4) **对数损失函数:** 主要针对概率模型。  

$$
L(Y,P(Y \vert X))= - \log P(Y \vert X)
$$  

&emsp;&emsp;损失函数越小说明模型的预测值与真实值越接近，模型的预测效果越好。但仅衡量一次预测的好坏是不够的，我们更想要知道模型在整个联合分布上的表现，这时就需要风险函数。  

- **期望风险(Expexted risk):** 度量平均意义下模型预测的好坏。  

$$
R_{exp}(f)=E_{P}[L(Y,f(X))]=\int_{\mathcal{X} \times \mathcal{Y}}L(y,f(x))P(x,y)dxdy
$$  

&emsp;&emsp;理论上我们需要计算期望风险来评估模型的好坏，但在实际中联合概率分布$P(X,Y)$一般是未知的，期望风险无法直接计算，因此我们引入经验风险用于估计风险函数.  

- **经验风险(Empirical risk):** 模型关于数据集的平均损失.  

$$
R_{emp}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))
$$  

&emsp;&emsp;其中，训练集$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$.  

&emsp;&emsp;期望风险$R_{exp}(f)$是模型关于联合分布的期望损失，经验风险$R_{emp}(f)$是模型关于训练集的平均损失。我们知道训练集中的样本$(x_i,y_i)$都是独立同分布的，其损失函数的期望$L(Y,f(X))$存在，由大数定律可知，当$n \rightarrow \infty$时，有：  

$$
R_{emp}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i)) \rightarrow R_{exp}(f)=E_{P}[L(Y,f(X))]
$$  

&emsp;&emsp;可是在现实生活中，样本容量$N$一般是有限的，有的时候甚至会很小。因此，仅仅用经验风险来估计期望风险效果并不理想，需要对其进行一定的矫正。这里就涉及到监督学习的两个基本策略，一个是经验风险最小化策略，一个是结构风险最小化策略。  

- **经验风险最小化策略:** 求解使得经验风险$R_{emp}(f)$最小的模型$f^{*}$，模型$f^{*}$即为最优的模型。根据这一策略，求解最优模型的问题就转化为求解最优化问题：  

$$
\min_{f \in \mathcal{H}} \frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))
$$  

&emsp;&emsp;当样本容量足够大时，经验风险最小化能保证有很好的学习效果。但是，当样本容量很小时，经验风险最小化学习的效果就未必很好，会产生“过拟合”(over-fitting)现象。  
&emsp;&emsp;结构风险最小化策略是为了防止过拟合现象而提出的策略。结构风险在经验风险上加上表示模型复杂度的正则化项。  

- **结构风险(Structural risk):**  

$$
R_{srm}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f)
$$  

&emsp;&emsp;其中，正则化项$J(f)$为模型复杂度，是定义在假设空间$\mathcal{H}$上的泛函。模型$f$越复杂，正则化项$J(f)$就越大，反之，模型$f$越简单，正则化项$J(f)$就越小。$\lambda > 0$是用于权衡经验风险与模型复杂度的系数。  

- **结构风险最小化策略:** 求解使得结构风险$R_{srm}(f)$最小的模型$f^{*}$，模型$f^{*}$即为最优模型。这一策略同样可以转化为求解最优化问题：  

$$
\min_{f \in \mathcal{H}} \frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f)
$$

&emsp;&emsp;关于监督学习的策略，追根究底，就是选取一个目标函数，可以是经验风险，或者是结构风险，然后通过优化这个目标函数，达到学习模型的目的。  

#### 算法  

&emsp;&emsp;在假设空间里面，根据策略去选择最优模型，需要一个具体的操作方案，操作方案也就是算法，是用来求解最优模型的。如果这个最优模型存在显式解析解，那么简单了，直接把这个结果写出来即可。但是往往这个显式解是不存在的，所以需要一定的数值计算方法，比如梯度下降法。  

## 模型评估与模型选择  

### 模型评估  

&emsp;&emsp;在通过策略训练完模型后，我们需要评估得到的模型在已知数据和未知数据上的预测效果。我们称模型在训练集上的预测误差为训练误差，在测试集上的误差为测试误差。  

#### 训练误差  

&emsp;&emsp;假设我们定义完假设空间$\mathcal{H}$后，通过策略与算法学习到的模型为:  

$$
Y = \hat{f(X)}
$$  

&emsp;&emsp;训练数据集为：  

$$
T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_{N_{1}}.y_{N_{1}})\}
$$  

&emsp;&emsp;其中$N_1$表示训练样本容量。由前文的概念可知，我们可以使用模型在测试集上的经验风险$R_{emp}$来表示训练误差：  

$$
e_{train}(\hat{f})=\frac{1}{N_1}\sum_{i=1}^{N_1}L(y_i,\hat{f}(x_i))
$$  

&emsp;&emsp;训练误差$e_{train}$表示模型$\hat{f}$在训练集上预测效果的好坏，训练误差越小，说明模型在训练集上的预测效果越好。然而，仅仅获得模型对已知数据的预测效果并不足以能够正确评价模型性能，我们还需要评估模型对未知数据的预测效果，这就需要计算模型在测试集上的预测误差，即测试误差。  

#### 测试误差  

&emsp;&emsp;若测试数据集为：  

$$
T_{test}=\{(x_1,y_1),(x_2,y_2),\dots,(x_{N_{2}}.y_{N_{2}})\}
$$  

&emsp;&emsp;其中$N_2$表示测试样本容量。同样使用模型在测试集上的经验风险$R_{emp}$来表示测试误差：  

$$
e_{test}(\hat{f})=\frac{1}{N_2}\sum_{i=1}^{N_2}L(y_i,\hat{f}(x_i))
$$  

&emsp;&emsp;测试误差$e_{test}$表示模型$\hat{f}$在测试集上的预测效果的好坏，测试误差越小，说明模型在测试集上的预测效果越好。训练误差反映了学习方法对未知的测试数据集的预测能力，是评估模型性能的重要指标。通常，我们将学习方法对未知数据的预测能力称为**泛化能力(Generalization ability)**。  

### 模型选择  

&emsp;&emsp;在得到模型的训练误差和测试误差后，我们便需要据此评估模型的性能，选择对实际问题性能最出众的模型。最理想的情况是，我们得到的模型在测试集和训练集上的误差都较小，这样我们可以认为模型对于该问题的效果是较好的。但在实际情况中，我们经常会发现模型在训练集上的误差较小，而在测试集上的误差较大，这表明模型的泛化能力交叉，我们称这样的现象为**过拟合(Over-fitting)**。  

#### 过拟合  

&emsp;&emsp;过拟合是指学习时选择的模型所包含的参数过多，模型复杂度过高，以至于出现这一模型对已知数据预测得很好，但对未知数据预测得很差的现象。  
&emsp;&emsp;模型选择的目的实际上就是避免过拟合现象并提高模型的泛化能力。这里我们通过一个多项式函数拟合的问题来说明过拟合与模型选择。  

#### 关于模型选择的实例——多项式拟合问题  

&emsp;&emsp;假设输入空间$\mathcal{X}=(0,1)$到输出空间$\mathcal{Y}=[-1,1]$的真实映射为:  

$$
Y = \sin(2\pi X)
$$  

&emsp;&emsp;由于误差的影响，抽取的样本为真实值加上一个随机误差，我们假设这个随机误差满足均值为0，方差为0.1的正态分布：  

$$
y=\sin(2 \pi x)+\varepsilon, \varepsilon \sim N(0,0.1)
$$  

&emsp;&emsp;我们的样本数据集为：  

$$
T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

&emsp;&emsp;其中，样本容量$N=30$.将样本数据集划分为样本容量为20的训练数据集与样本容量为10的测试数据集。以下给出了生成数成以及测试数据的散点图的代码。  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline

# Data Generation
np.random.seed(1314) # set random seed
x = np.random.random((30,)) # x belong to (0,1)
r = np.random.normal(0,0.1,30) # r ~ N(0,0.1)
y = np.sin(2*(math.pi)*x)+r
data = {"y":y,"x":x}
Data = pd.DataFrame(data,columns=["y","x"]) # data set
train_data = Data[:20] # Train data
test_data = Data[20:] # Test data

# Tradin data scatter plot
plt.scatter(train_data["x"],train_data["y"],marker="o",c="black",label="noise")
# Real line
def real_func(x):
    return np.sin(2*np.pi*x)
x_points = np.linspace(0,1,1000)
plt.plot(x_points,real_func(x_points),label="real curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
```

&emsp;&emsp;得到的散点图如下图4所示：  
<center>
    <img src="https://s2.loli.net/2023/09/30/TlHEjSPh5BRZOzN.png" width=60% height=60%>
    <div align="center">Image4: 训练数据散点图</div>
</center>  

&emsp;&emsp;图4中的黑色散点代表训练数据，蓝色曲线为输入空间到输出空间的真实映射。下面，我们便要建立模型去拟合数据。我们使用多项式函数进行拟合，模型形式为：  

$$
f_{M}(x,w)=w_0+w_{1}x+w_{2}x^2+\dots+w_{M}x^{M}=\sum_{j=1}^{M}w_{j}x_{j} = w^{T}x
$$  

&emsp;&emsp;其中，$x=\begin{bmatrix}1,x,x^2,\dots,x^M\end{bmatrix}^{T}$为输入变量的不同次方，且$x \in (0,1)$，$w=\begin{bmatrix}w_0,w_1,\dots,w_M\end{bmatrix}^{T}$ 为多项式的参数向量。则该问题的假设空间为：  

$$
\mathcal{H}=\{f \vert y=f_{M}(x,w)\}
$$  

&emsp;&emsp;由于多项式函数$f_{M}(x,w)$是由参数向量$w$决定的，因此假设空间$\mathcal{H}$等价于想要的参数空间$\Theta$：  

$$
\Theta = \{w \vert w \in \mathbb{R}^{M+1} \}
$$  

&emsp;&emsp;在拟合的时候，我们采取使经验风险最小化的策略。由于这个问题为回归问题，经验风险中的损失函数可以选用平方损失函数，即预测值与真实值平方差。此时模型的经验风险可以表示为训练误差：  

$$
L(w)=\frac{1}{2}\sum_{i=1}^{N_1}(f_{M}(x_i,w)-y_i)^2=\frac{1}{2}\sum_{i=1}^{N_1}(w^{T}x_i-y_i)^2=\frac{1}{2}(Xw-y)^{T}(Xw-y)
$$  

$$
X=\begin{bmatrix}
    1 & x_1 & x_1^2 & \dots & x_1^M \\
    1 & x_2 & x_2^2 & \dots & x_2^M \\
    \vdots & \vdots & \vdots & & \vdots \\
    1 & x_{N_1} & x_{N_1}^2 & \dots & x_{N_1}^M \\
\end{bmatrix},w = \begin{bmatrix}
    w_0 \\
    w_1 \\
    \vdots \\
    w_M \\
\end{bmatrix}, y = \begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_{N_1} \\
\end{bmatrix}
$$  

&emsp;&emsp;采取使得经验风险最小化的策略，则从假设空间中求解最优模型的问题就转化成了一个最优化问题：  

$$
\min_{w} \frac{1}{2} (Xw-y)^{T}(Xw-y)
$$  

&emsp;&emsp;要求$L(w)$的最小值点，我们需要对$w$求偏导，并令偏导为$0$：  

$$
\frac{\partial L(w)}{\partial w}=X^TXw-X^Ty=0
$$  

$$
\Rightarrow \hat{w} = (X^TX)^{-1}X^Ty
$$  

&emsp;&emsp;其中$N_1 = 20$。  
&emsp;&emsp;在估计完参数后，我们便得到了该问题的预测模型：  

$$
\hat{f}_{M}(x,\hat{w})=\hat{w}^Tx
$$  

&emsp;&emsp;对于测试集中的实例$x_i$，可以利用模型计算$y_i$的预测值：  

$$
\hat{y_i} = \hat{f}_M(x_i,\hat{w})=\hat{w}^Tx_i=y^TX(X^TX)^{-1}x_i
$$  

&emsp;&emsp;要评估模型的性能我们可以计算模型在训练集与测试机上的误差：  

$$
e_{train}(\hat{f}_M)=\frac{1}{N_1}\sum_{i=1}^{N_1}L(y_i,\hat{f}_M(x_i,\hat{w}))
$$

$$
e_{test}(\hat{f}_M)=\frac{1}{N_2}\sum_{i=1}^{N_2}L(y_i,\hat{f}_M(x_i,\hat{w}))
$$  

&emsp;&emsp;其中，$N_2=10$，改变多项式的次数$M$可以得到不同的模型形式，进而得到不同的$e_{train}$和$e_{test}$，我们可以通过比较这两个指标选择最优的多项式次数。下面是多项式回归的代码实现。  

&emsp;&emsp;通过以下这段代码中的`ployreg()`函数，我们可以计算多项式回归的参数向量：  

```python
# Ploy-regression
def ployreg(M):  
    global train_data
    x = train_data["x"].values
    X = np.ones((len(x),M+1))  
    for i in range(M):  
        X[:,i+1]=np.power(x,i+1)
    y = train_data["y"].values
    X_T = np.transpose(X)
    X_TX = np.dot(X_T,X)
    w = np.dot(np.linalg.inv(X_TX),np.dot(X_T,y))
    return w.round(4)
```
&emsp;&emsp;当我们设置多项式的次数$M=2$时，得到的最优多项式函数为：  

$$
\hat{f}(x)=0.6325-1.0391x-0.5826x^2
$$  

&emsp;&emsp;在得到最优模型后，我们可以计算模型在训练集与测试集上的误差。通过设置不同的多项式次数$M$，我们可以比较不同模型的性能，从而选择最优的模型形式。以下代码中的`calerror()`函数便是用于计算模型在训练集与测试集上的误差。  

```python
# calculate error
def calerror(M):
    global train_data,test_data
    error = {}
    w = ployreg(M)
    # train error
    x = train_data["x"].values
    X = np.ones((len(x),M+1))  
    for i in range(M):  
        X[:,i+1]=np.power(x,i+1)
    y = train_data["y"].values
    e = np.dot(X,w)-y
    train_error = (np.dot(np.transpose(e),e))/len(x)
    # test error
    x = test_data["x"].values
    X = np.ones((len(x),M+1))  
    for i in range(M):  
        X[:,i+1]=np.power(x,i+1)
    y = test_data["y"].values
    e = np.dot(X,w)-y
    test_error = (np.dot(np.transpose(e),e))/len(x)
    error["train error"] = train_error.round(4)
    error["test error"] = test_error.round(4)
    return error
```

&emsp;&emsp;设置多项式次数为$M \in [0,9]$，我们可以得到下表2所示的计算结果：  

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<p align="center"><font face="黑体" size=2.>表2 模型误差</font></p>
<div class="center">

|   M     | Train error | Test error |
|  :---:  |  :------:   |  :--:      |  
|    0    |  0.413      | 0.562      |  
|    1    |  0.1392     | 0.2668     |
|    2    |  0.1372     | 0.2398     |
|    3    |  0.0144     | 0.0182     |
|    4    |  0.0137     | 0.0151     |
|    5    |  0.0098     | 0.0065     |
|    6    |  0.0098     | 0.0065     |
|    7    |  0.0087     | 0.0074     |
|    8    |  0.0084     | 0.0121     |
|    9    |  0.0049     | 0.0388     |
</div>  

&emsp;&emsp;为了更直观的观察模型误差的变化，我们可以利用上表的数据绘制误差曲线图。通过以下代码中的`errorplot()`函数来绘制图像：  

```python
# learning curve
def errorplot():
    train_error = []
    test_error = []
    for M in range(10):
        error = calerror(M)
        train_error.append(error['train error'])
        test_error.append(error['test error'])
    M = range(10)
    # train error curve
    plt.plot(M,train_error,c="blue",label="train error")
    # test error curve
    plt.plot(M,test_error,c="green",label="test error")
    plt.xlabel("M")
    plt.ylabel("error")
    plt.legend()
```

&emsp;&emsp;误差图如下图5所示：  

<center>
    <img src="https://s2.loli.net/2023/10/01/yCiIKm39GYcWXFO.png" width=60% height=60%>
    <div align="center">Image5: 模型误差</div>
</center>  

&emsp;&emsp;结合表1和图5，我们可以发现随着多项式次数$M$的增加，训练误差是逐渐减小的，而测试误差会先减小后增大。我们知道模型的复杂度随着$M$的增加而增加，实际上，当$M$增加到7之后，虽然训练误差仍在下降，但测试误差不降反升，此时模型已经出现了过拟合现象，模型的泛化能力下降。当$M=5$或$M=6$时，模型的训练误差与测试误差均达到较低水平，此时模型达到最优。在同样的误差下，我们应当选择复杂度较小的模型，以降低预测时的计算量，因此我们选择$M=5$时的模型作为该问题的最优模型：  

$$
\hat{f}(x)=0.0320+4.6026x+15.8081x^2-107.3805x^3+142.5954x^4-55.7287x^5
$$  

&emsp;&emsp;通过以下代码中的`regcurve()`函数我们可以画出相应的多项式函数的曲线：  

```python
# draw regression curve
def regcurve(M):
    global train_data
    # Tradin data scatter plot
    plt.scatter(train_data["x"],train_data["y"],marker="o",c="black",label="train data")
    # real line
    x_points = np.linspace(0,1,1000)
    plt.plot(x_points,real_func(x_points),label="real curve")
    # regression curve
    w = ployreg(M)
    X_points = np.ones((len(x_points),M+1))
    for i in range(M):
        X_points[:,i+1] = np.power(x_points,i+1)
    y_points = np.dot(X_points,w)
    plt.plot(x_points,y_points,c="red",label="regression curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"M={M}")
    plt.legend()
```

&emsp;&emsp;画出$M=0,1,5,9$时的拟合曲线的图像：  

<center>
    <img src="https://s2.loli.net/2023/10/01/3zGbMkHDVx75t91.png" width=80% height=80%>
    <div align="center">Image6: 拟合曲线</div>
</center>  

&emsp;&emsp;从图6可以看出，当$M=0,1$时，模型过于简单，与真实曲线相差很大，模型误差也可以看出此时模型在训练集与测试集上的误差均较大，我们称模型出现了**欠拟合**现象。当$M=5$时，拟合曲线与真实曲线几乎一致，且模型在训练集上的误差也很小，模型效果很好。当$M=9$时，模型几乎能够预测所有的训练数据，模型的训练误差非常小，但拟合曲线与真实曲线相差较大，模型在测试集上的误差较大，泛化能力较差，此时模型出现了**过拟合**现象。  

## 正则化与交叉验证  

### 正则化  

&emsp;&emsp;上文中我们介绍了进行模型选择时主要评估模型在已知数据与未知数据上的预测能力，即训练误差与测试误差。在进行模型选择时，我们需要综合考虑这两个指标，以避免过拟合与欠拟合现象。  
&emsp;&emsp;为了综合考虑这两个指标，模型选择的一种经典方法是**正则化(regularization)**。正则化是结构风险最小化策略的实现，是在经验风险上加上一个正则化项(regularizer)或罚项(penalty term)。正则化项一般是模型复杂度的单调递增函数，模型越复杂，正则化值越大。比如，正则化项可以是模型参数向量的范数。  
&emsp;&emsp;正则化一般具有如下形式：  

$$
\min_{f \in \mathcal{H}} \frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f)
$$  

&emsp;&emsp;其中，第1项是经验风险，第2项是正则化项，$\lambda \geq 0$为调整两者关系的系数。  
&emsp;&emsp;当模型处于欠拟合状态时，虽然模型复杂度较低，正则化值较低，但模型在训练集上的误差会较大，即经验风险较大；当模型处于过拟合状态时，虽然模型在训练集上的误差会较小，即经验风险较小，但模型复杂度会较高，即正则化值较大。正则化的作用是选择经验风险与模型复杂度同时较小的模型，通过优化正则化表达式，能同时减弱模型的欠拟合与过拟合现象，找到最优模型所处的中间状态。  
&emsp;&emsp;正则化项可以取不同的形式。在回归问题中，损失函数是平方损失，正则化项可以是参数向量的$L_2$范数，以减轻模型的过拟合现象：  

$$
L(w)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\frac{\lambda}{2} \Vert w \Vert_{2}^{2}
$$  

&emsp;&emsp;正则化项也可以是参数向量的$L_1$范数，以得到一个较为稀疏的参数向量：  

$$
L(w)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+ \lambda \Vert w \Vert_{1}
$$

**注: 奥卡姆剃刀原理**  
&emsp;&emsp;正则化符合奥卡姆剃刀原理。奥卡姆剃刀原理应用于模型选择时便为以下想法：在所有可能选择的模型中，能够很好解释已知数据并且十分简单的模型才是最好的模型。  

### 交叉验证  

&emsp;&emsp;另一种常用的模型选择方法是交叉验证(cross validation).如果给定的样本数据充足，进行模型选择的一种简单方法是随机地将数据集切分成三部分，分别为**训练集、验证集、测试集**。训练集用于训练模型，验证集用于模型的选择，而测试集用于最终对学习方法的评估。我们最终选择对验证集有最小预测误差的模型。  
&emsp;&emsp;但是，在许多实际应用中数据是不充足的，为了选择好的模型，可以采用交叉验证方法。**交叉验证的基本想法是重复地使用数据；把给定的数据进行切分，将切分的数据组合为训练集与测试集，在此基础上反复地进行训练、测试以及模型选择。**  
&emsp;&emsp;简单介绍几种主要的交叉验证的方法：  

- **简单交叉验证:** 首先随机地将已给数据分为两部分，一部分作为训练集，另一部分作为测试集(例如，70%的数据为训练集，30%的数据为测试集)；然后用训练集在各种条件下(例如，不同的参数个数)训练模型，从而得到不同的模型；在测试集上评价各个模型的测试误差，选出测试误差最小的模型。  
- **S折交叉验证:** 首先随机地将已给数据切分为S个互不相交、大小相同的子集；然后利用S-1个子集的数据训练模型，利用余下的子集测试模型；将这一过程对可能的S种选择重复进行；最后选出在S次评测中平均测试误差最小的模型。  
-  **留一交叉验证:** 令S折交叉验证中的S=N，称为留一交叉验证，往往在数据缺乏的情况下使用。这里，N是给定数据集的容量。 

## 泛化能力  

&emsp;&emsp;学习方法发泛化能力是指由该方法学习到的模型对未知数据的预测能力。通常，测试数据集用以评价训练所得模型的泛化能力。由于测试数据集包含的样本有限，仅仅通过测试数据集去评价泛化能力，有时并不可靠，此时需要我们从机器学习理论出发，对模型的泛化能力进行评价。  

### 泛化误差  

&emsp;&emsp;如果学习到的模型是$\hat{f}$，那么用这个模型对未知数据预测的期望风险即为泛化误差(generalization ability)：  

$$
R_{exp}(\hat{f})=E_{P}[L(Y,\hat{f}(X))]=\int_{\mathcal{X} \times \mathcal{Y}}L(y,\hat{f}(x))P(x,y)dxdy
$$  

&emsp;&emsp;计算测试误差所使用的样本为测试集，而泛化误差则是基于整个样本空间的。所以泛化误差是度量对未知数据预测能力的理论值，测试误差则是经验值。如果一种学习方法的模型比另一种方法所学习的模型具有更小的泛化误差，那么这种方法就更有效。  

### 泛化误差上界  

&emsp;&emsp;学习方法的泛化能力分析往往是通过研究泛化误差的概率上界进行的，简称泛化误差上界(generalization error bound)。泛化误差上界通常具有以下性质：  

- **泛化误差上界是样本容量$N$的函数，当样本容量$N$增加时，泛化误差上界趋于0**；
- **泛化误差上界是假设空间容量(capacity)的函数，假设空间容量越大，模型就越难学，泛化误差上界越大**。  

&emsp;&emsp;第一条性质，可以从泛化误差的概念出发辅助理解。泛化误差定义为一个期望值，那么经验值就是以全样本空间作为测试集计算所得的测试误差，它表示为一个平均值。在平均值中，样本量位于分母的位置，那么随着样本量的增加，平均值则趋于零。  
&emsp;&emsp;第二条性质，泛化误差上界是假设空间容量的函数，不同的模型，对应不同的泛化误差上界。假设空间是所有可能的模型的集合，那么假设空间容量越大，所有可能的模型类型就会越多，模型也就愈加难以学习，与之对应的泛化误差上界就会越大。  
&emsp;&emsp;下面给出一个简单的泛化误差上界的例子：二分类问题的泛化误差上界。  

#### 二分类问题的泛化误差上界

&emsp;&emsp;已知二分类问题的数据集为$T$：  

$$
T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

&emsp;&emsp;其中，$N$为样本容量。假设数据集$T$是从联合概率分布$P(X,Y)$独立同分布产生的，$X \in \mathbb{R}^N, Y \in \{-1,1\}$。模型的假设空间$\mathcal{H}$是函数的集合：  

$$
\mathcal{H}=\{f_1,f_2,\dots,f_d\},f_i: \mathbb{R}^N \rightarrow \{-1,1\}
$$  

&emsp;&emsp;由于是分类问题，我们可以设置损失函数为0-1损失函数:  

$$
L(Y,f(X))= \left \{
\begin{array}{rcl}
1, & {Y \neq f(X)}\\
0,& {Y = f(X)}\\
\end{array} \right.
$$  

&emsp;&emsp;设$f$是从假设空间$\mathcal{H}$中选取的函数，则$f$的期望风险$R(f)$与经验风险$\hat{R}(f)$分别为：  

$$
R(f)=E[L(Y,f(X))]
$$

$$
\hat{R}(f)=\frac{1}{N} \sum_{i=1}^{N}L(y_i,f(x_i))
$$  

&emsp;&emsp;为了从假设空间$\mathcal{H}$中选取最优模型，我们可以采取经验风险最小化策略，设得到的最优模型为$f_N$:  

$$
f_N = \arg \min_{f \in \mathcal{H}} \hat{R}(f)
$$

&emsp;&emsp;$f_N$依赖训练数据集的样本容量$N$，$f_N$的泛化能力即其在整个样本空间上的期望风险：  

$$
R(f_N)=E[L(Y,f_{N}(X))]
$$  

&emsp;&emsp;下面，首先给出关于假设空间$\mathcal{H}$中任意函数$f$的泛化误差上界的定理，再对定理进行证明。  

**定理**  

&emsp;&emsp;对二分类问题，当假设空间是有限个函数的集合$\mathcal{H}=\{f_1,f_2,\dots,f_d\}$时，对$\forall f \in \mathcal{H}$，其泛化误差$R(f)$有以下不等式成立：  

$$
P(R(f) \leq \hat{R}(f)+\varepsilon(d,N,\delta)) \geq 1-\delta, \delta \in (0,1)
$$  

$$
\varepsilon(d,N,\delta)=\sqrt{\frac{1}{2N}(\log{d}+\log{\frac{1}{\delta}})}
$$

&emsp;&emsp;该不等式的左端$R(f)$是泛化误差，右端即为泛化误差的上界。从泛化误差上界的表达时中，我们可以发现：  

- 当模型$f$的经验风险，即训练误差$\hat{R}(f)$越小时，泛化误差上界也越小。  
- 当样本容量$N \rightarrow \infty$时，$\varepsilon(d,N,\delta) \rightarrow 0$，泛化误差上界下降。  
- 当假设空间$\mathcal{H}$中的函数越多，即$d$越大时，$\varepsilon(d,N,\delta)$越大，泛化误差上界越大。  

&emsp;&emsp;下面对该定理进行证明，证明中需要用到 $Hoeffiding$ 不等式，这里先给出其结论。  

**$Hoeffding$不等式**  

&emsp;&emsp;设 $X_1,X_2,\dots,X_N$ 是独立随机变量，且 $X_i \in [a_i,b_i], i = 1,2,\dots,N$；$\bar{X}$ 是 $X_1,X_2,\dots,X_N$的均值，即$\bar{X}=\frac{1}{N}\sum_{i=1}^{N}X_i$，则对 $\forall t > 0$，有以下不等式成立：  

$$
P \left(  \bar{X}-E(\bar{X})  \ge t \right) \leq exp \left( -\frac{2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2} \right)
$$

$$
P \left( | \bar{X}-E(\bar{X}) | \ge t \right) \leq 2exp \left( -\frac{2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2} \right)
$$  

&emsp;&emsp;$Hoeffding$不等式的证明放在附录中，感兴趣的读者可以自行阅读，下面我们来正式证明关于二分类问题的误差上界的定理。  

**证明**  
&emsp;&emsp;令$X_i = L(y_i,f(x_i))$，由于样本$(x_i,y_i)$是从联合概率分布$P(X,Y)$中产生的，故有$X_1,X_2,\dotsb,X_N$独立同分布，且对 $\forall i, [a_i,b_i]=[0,1]$,  

$$
\bar{X}=\frac{1}{N}\sum_{i=1}^{N}X_i=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))=\hat{R}(f)
$$  

$$
\begin{align*}
    E(\bar{X}) &= E \left( \frac{1}{N} \sum_{i=1}^{N}X_i \right)= \frac{1}{N} \sum_{i=1}^{N}E(X_i) =E(X) \\
    &= E \left( L(Y,f(X)) \right) = R(f) \\
\end{align*}
$$  

&emsp;&emsp;根据 $Hoeffding$不等式可知：对$\forall \varepsilon > 0$，有：  

$$
P(\bar{X}-E(\bar{X}) \ge \varepsilon)=P(R(f)-\hat{R}(f) \ge \varepsilon) \leq exp(-2N \varepsilon^2)
$$  

&emsp;&emsp; $\because$事件 $\{R{f}-\hat{R}{f} \ge \varepsilon\} \Leftrightarrow$ 事件 $\{ \exists f \in \mathcal{H}: R(f)-\hat{R}(f) \ge \varepsilon \}$，故有：  

$$
\begin{align*}
    P(R(f)-\hat{R}(f) \ge \varepsilon) &= P(\exists f \in \mathcal{H}: R(f)-\hat{R}(f) \ge \varepsilon)=P \left( \bigcup_{f \in \mathcal{H}}\{ R(f)-\hat{R}(f) \ge \varepsilon \} \right) \\
    &\leq \sum_{f \in \mathcal{H}} P(R(f)-\hat{R}(f) \ge \varepsilon) \leq dexp(-2N\varepsilon^2)
\end{align*}
$$  

&emsp;&emsp; $\because$ 事件$\{ \exists f \in \mathcal{H}: R(f)-\hat{R}(f) \ge \varepsilon \}$与事件$\{\forall f \in \mathcal{H}: R(f)-\hat{R}(f) < \varepsilon\}$互补，故对$\forall f \in \mathcal{H}$有：  

$$
P(R(f)-\hat{R}(f) < \varepsilon)=P(R(f)-\hat{R}(f) \leq \varepsilon) \ge 1-dexp(-2N\varepsilon^2)
$$  

&emsp;&emsp;令$\delta = dexp(-2N\varepsilon^2) \in (0,1)$，即：  

$$
P(R(f)-\hat{R}(f) \leq \varepsilon) \ge 1-\delta
$$  

$$
\delta = dexp(-2N\varepsilon^2) \Rightarrow \varepsilon = \sqrt{\frac{1}{2N}(\log{d}+\log{\frac{1}{\delta}})}
$$  

&emsp;&emsp;证毕.  

## 附录  

### 关于 $Hoeffding$ 不等式的证明  

&emsp;&emsp;$Hoeffding$ 不等式的证明需要使用 $Markov$ 不等式与 $Hoeffding$ 不等式引理，这里我们首先证明这两个结论。  

#### $Markov$ 不等式  

**结论**  
&emsp;&emsp;对任意非负随变量$X$，$\varepsilon > 0$，有：  

$$
P(X \ge \varepsilon) \leq \frac{E(X)}{\varepsilon}
$$  

**证明:**  
&emsp;&emsp;设随机变量$X$的概率密度函数为$f(x)$，则有：  

$$
E(X)=\int_{0}^{\infty}xf(x) \leq \int_{\varepsilon}^{\infty}xf(x)dx \leq \varepsilon \int_{\varepsilon}^{\infty}f(x)dx=\varepsilon P(X \ge \varepsilon)
$$  

$$
\Rightarrow P(X \ge \varepsilon) \leq \frac{E(X)}{\varepsilon}
$$  

&emsp;&emsp;证毕.  

#### $Hoeffding$ 不等式引理  

**结论**  
&emsp;&emsp;对任意随机变量$X$，若其满足$P(X \in [a,b])=1$和$E(X)=0$，则有以下不等式成立：  

$$
E(e^{sX}) \leq e^{\frac{s^2(b-a)^2}{8}}
$$  

**证明:**  
&emsp;&emsp;设$f(x)=e^{sx}$，且$dom f = [a,b]$,  
&emsp;&emsp;$\because f(x)$二阶可微，$dom f$为凸集，且$f^{''}(x)=s^2e^{sx} \ge 0$,  
&emsp;&emsp;由凸函数的二阶条件可知：$f(x)$为凸函数,  
&emsp;&emsp;由凸函数的性质可知：对$\forall X_1,X_2 \in [a,b]$，有以下不等式成立：  

$$
f(\alpha X_1+\beta X_2) \leq \alpha f(X_1)+\beta f(X_2),\space \forall \alpha,\beta > 0, \alpha+\beta=1
$$  

&emsp;&emsp;令$\alpha=\frac{b-X}{b-a}, \beta=\frac{X-a}{b-a}, X_1=a, X_2=b$，则有：  

$$
f(\alpha X_1+\beta X_2)=f(X)=e^{sX}
$$

$$
\alpha f(X_1)+\beta f(X_2)=\frac{b-X}{b-a}e^{sa}+\frac{X-a}{b-a}e^{sb}
$$  

&emsp;&emsp;由凸函数的性质的得：  

$$
e^{sX} \leq \frac{b-X}{b-a}e^{sa}+\frac{X-a}{b-a}e^{sb}
$$  

&emsp;&emsp;两边同时对$X$取期望，由$E(X)=0$，得：  

$$
0 \leq E(e^{sX}) \leq \frac{be^{sa}-ae^{sb}}{b-a}
$$  

&emsp;&emsp;现要证明以下不等式成立：  

$$
\frac{be^{sa}-ae^{sb}}{b-a} \leq e^{\frac{s^2(b-a)^2}{8}}
$$  

&emsp;&emsp;两边同时取对数，则等价于证明以下不等式成立：  

$$
\log \left( \frac{be^{sa}-ae^{sb}}{b-a} \right) \leq \frac{s^2(b-a)^2}{8}
$$  

&emsp;&emsp;令$t=b-a$，则$b=t+a$，则不等式可作如下变换：  

$$
\begin{align*}
    \log \left( \frac{be^{sa}-ae^{sb}}{b-a} \right) &= \log(be^{sa}-ae^{sb})-\log(b-a) \\
    &= \log(te^{sa}+ae^{sa}-ae^{s(t+a)})-\log t  \\
    &= sa + \log(t+a-ae^{st}) - \log t \\
    &\leq \frac{s^2t^2}{8} \\
\end{align*}
$$  

&emsp;&emsp;令$u=st, g(u)=sa + \log(t+a-ae^{st})-\log t$，则以上不等式等价于：  

$$
g(u) \leq \frac{1}{8}u^2
$$  

&emsp;&emsp;对$g(u)$求一阶导数和二阶导数得：  

$$
g^{'}(u)=\frac{-ae^u}{t+a-ae^u},\space g^{''}(x)=\frac{-ae^u(t+a)}{(t+a-ae^u)^2}
$$  

&emsp;&emsp;将$g(u)$在$u=0$处进行泰勒展开：  

$$
g(u)=g(0)+g^{'}(0)u+\frac{1}{2}g^{''}(\varepsilon)u^2=\frac{1}{2}g^{''}(\varepsilon)u^2
$$  

$$
\begin{align*}
    g^{''}(\varepsilon) &= \frac{-ae^{\varepsilon}(t+a)}{(t+a-ae^{\varepsilon})^2}=\frac{t+a}{t+a-ae^{\varepsilon}} \cdot \frac{-ae^{\varepsilon}}{t+a-ae^{\varepsilon}}  \\
    &= \left( \frac{t+a}{t+a-ae^{\varepsilon}} \right) \cdot \left( 1- \frac{t+a}{t+a-ae^{\varepsilon}} \right)  = m(1-m) \leq \frac{1}{4}\\ 
\end{align*}
$$  

&emsp;&emsp;故可证明以下不等式成立：  

$$
g(u) = \frac{1}{2}g^{''}(\varepsilon)u^2 \leq \frac{1}{8}u^2
$$  

&emsp;&emsp;证毕.  

&emsp;&emsp;有了$Markov$不等式与 $Hodeffding$ 不等式引理作为基础，下面我们开始正式证明 $Hoeffding$ 不等式。  

**证明:**  
&emsp;&emsp;取 $s > 0, t > 0$，由$Markov$不等式得：  

$$
P(\bar{X}-E(\bar{X}) \ge t)=P \left( e^{s(\bar{X}-E(\bar{X}))} \ge e^{st} \right) \leq e^{-st}E \left( e^{s(\bar{X}-E(\bar{X}))} \right)
$$  

&emsp;&emsp; $\because \bar{X}=\frac{1}{N}\sum_{i=1}^{N}X_i, \space E(\bar{X})=\frac{1}{N}\sum_{i=1}^{N}E(X_i)$，故有：  

$$
E \left( e^{s(\bar{X}-E(\bar{X}))} \right) = E \left( e^{s \sum_{i=1}^{N} \frac{X_i-E(X_i)}{N}} \right) = \prod_{i=1}^{N}E \left( e^{s \frac{X_i-E(X_i)}{N}} \right)
$$  

&emsp;&emsp;令$Y_i = \frac{X_i-E(X_i)}{N}$，则$Y_i$相互独立，且有$E(Y_i)=0, Y_i \in [\frac{a_i-E(X_i)}{N},\frac{b_i-E(X_i)}{N}]$，由 $Hoeffding$ 不等式引理得：  

$$
E \left( e^{sY_i} \right) \leq e^{s \frac{(b_i-a_i)^2}{8N^2}}, i=1,2,\dots,N
$$  

&emsp;&emsp;由以上不等式可得：  

$$
E \left( e^{s(\bar{X}-E(\bar{X}))} \right) = \prod_{i=1}^{N}E \left( e^{sY_i} \right) \leq \prod_{i=1}^{N} e^{s^2 \frac{(b_i-a_i)^2}{8N^2}} = e^{s^2 \sum_{i=1}^{N} \frac{(b_i-a_i)^2}{N}}
$$  

&emsp;&emsp;则有：  

$$
P(\bar{X}-E(\bar{X}) \ge t) \leq e^{-st}E \left( e^{s(\bar{X}-E(\bar{X}))} \right) \leq e^{-st+s^2 \sum_{i=1}^{N} \frac{(b_i-a_i)^2}{N}}
$$  

&emsp;&emsp;对 $\forall s > 0$ 均有以上不等式成立，故不等式右端对$s$取最小值，不等式仍然成立：  
&emsp;&emsp;令$h(s) = -st+s^2 \sum_{i=1}^{N} \frac{(b_i-a_i)^2}{N}$，对其求一阶导数并令其为零：  

$$
h^{'}(s)=-t+s \frac{\sum_{i=1}^{N}(b_i-a_i)^2}{4N^2}=0 \Rightarrow s^{'}=\frac{4N^2t}{\sum_{i=1}^{N}(b_i-a_i)^2}
$$  

&emsp;&emsp;代入$s^{'}$可以得到$h(s)$的最小值：  

$$
h_{min}(s)=h(s^{'})=\frac{-2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2}
$$  

&emsp;&emsp;由于$e^x$在$\mathbb{R}$上的单调性可知：  

$$
e^{-st+s^2 \sum_{i=1}^{N} \frac{(b_i-a_i)^2}{N}}=e^{h(s)} \ge e^{h_{min}(s)}=e^{\frac{-2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2}}
$$  

&emsp;&emsp;故有以下不等式成立：  

$$
P(\bar{X}-E(\bar{X}) \ge t) \leq exp \left( \frac{-2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2} \right)
$$  

&emsp;&emsp;同时：  

$$
P(E(\bar{X})-\bar{X} \ge t) = P(-\bar{X}-E(-\bar{X}) \ge t)
$$  

&emsp;&emsp;令$\bar{Z}=-\bar{X}, E(\bar{Z})=E(-\bar{X})$，同上可证：  

$$
P(E(\bar{X})-\bar{X} \ge t) = P(\bar{Z}-E(\bar{Z}) \ge t) \leq exp \left( \frac{-2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2} \right)
$$

&emsp;&emsp;综上所述，有以下不等式成立：  

$$
P \left( | \bar{X}-E(\bar{X}) | \ge t \right) \leq 2exp \left( -\frac{2N^2t^2}{\sum_{i=1}^{N}(b_i-a_i)^2} \right)
$$

&emsp;&emsp;证毕.  

## 参考  

**[1] Book: 李航.(2019).统计学习方法**  
**[2] Video: B站数学up主——简博士**  
**[3] Github: Dod-o/Statistical-Learning-Method_Code**  
**[4] Blog: 知乎,ziyoufeixiang,Hoeffding 不等式证明**  
**[5] Blog: CSDN,吃吃今天努力学习了吗,[Python]多项式曲线拟合(Polynomial Curve Fitting)**
