---
title: 隐马尔可夫模型
tags:
    - Machine Learning
---

## toc

&emsp;&emsp;隐马尔可夫模型(hidden Markov model,HMM)是一种用于对时许数据建模的概率图模型。它主要应用于对观察序列的概率分布进行建模，这些观察序列背后存在一个不可见的状态序列。HMM的主要思想可以总结如下：  

- **状态和观察:** HMM包含两种类型的变量，即隐藏的状态序列和可见的观察序列。状态序列表示系统内部的状态，而观察序列是我们可以观察的外部现象。  
- **马尔可夫性质:** HMM假设状态序列是一个马尔可夫链，即系统的未来状态只依赖于当前状态，而与过去的状态无关。这意味着在给定当前当前状态下，未来状态与过去状态的信息是独立的。
- **状态转移概率:** HMM用状态转移概率描述系统从一个状态转移到另一个状态的可能性。这些概率被组织成状态转移矩阵，矩阵的元素表示从一个状态转移到另一个状态的概率。
- **观察概率:** 对于每个状态，HMM定义了生成每个观察值的概率分布。这些概率被组织成观察概率矩阵。
- **初始概率:** HMM还需要定义系统在初始时刻处于每个状态的概率，这些概率称为初始概率。
- **前向算法和后向算法:** HMM使用前向算法和后向算法来计算给定观测序列的概率。  
- **Baum-Welch算法:** 用于无监督学习的算法，通过观察序列来调整模型参数，使其更好地匹配观察数据。Baum-Welch算法本质上就是EM算法。  

&emsp;&emsp;隐马尔可夫模型在各个领域都具有重要的应用，包括**时序数据建模，语音识别，自然语言处理，生物信息学等**。总体而言，HMM在多个领域中都发挥着关键的作用，为时序数据建模和分析提供了灵活而强大的工具。  

## 基本概念  

### 马尔可夫链(MC)  

&emsp;&emsp;设有随机序列 $S =\{ S_1,S_2,\dots,S_t,S_{t+1},\dots \}$，若 $S_{t+1}$ 只依赖于前一时刻 $S_t$，不依赖于 $S_1,S_2,\dots,S_{t-1}$，即：  

$$
P(S_{t+1} | S_1,S_2,\dots,S_t)=P(S_{t+1} | S_t)
$$  

&emsp;&emsp;则称随机序列$S$为马尔可夫链(Markov Chain)。  

### 隐马尔可夫模型的定义  

&emsp;&emsp;**定义1(隐马尔可夫模型)**  隐马尔可夫模型是关于时序数据的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态序列，再由各个状态生成一个观测从而产生观测序列的过程。隐藏的马尔可夫链随机生成的状态的序列，称为状态序列(state sequence)；每个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列(observation sequence)。序列的每一个位置又可以看作是一个时刻。  

&emsp;&emsp;HMM模型可以用如下的概率图表示：  

<center>
    <img src="https://s2.loli.net/2023/11/26/w4LvBDr9m7oXkf3.jpg" width=80% height=60%>
    <div align="center">Image1: HMM模型的概率图</div>
</center>  

### 模型参数  

&emsp;&emsp;隐马尔可夫模型由**初始概率分布**、**状态转移概率分布**、以及**观测概率分布**确定。下面我们来介绍这些模型参数的含义，在这之前先做一些符号定义。  

**状态序列**  
&emsp;&emsp;设 $I$为状态序列，$Q$ 是所有可能状态的集合，记为：  

$$
Q=\{q_1,q_2,\dots,q_N\},\quad I=\{i_1,i_2,\dots,i_{T}\}, \forall i \in Q
$$

其中，$N$是可能的状态数。  

**观测序列**  
&emsp;&emsp;设 $O$ 是 状态序列 $I$ 所对应的观测序列，$V$ 是所有可能的观测的集合，记为：  

$$
V = \{v_1,v_2,\dots,v_{M}\},\quad O=\{o_1,o_2,\dots,o_{T}\},\forall o \in V
$$  

**状态转移概率矩阵**  
&emsp;&emsp;设 $A$ 为状态转移概率矩阵：  

$$
A = [a_{ij}]_{N \times N}
$$  

其中，  

$$
a_{ij} = P(i_{t+1}=q_{j} | i_{t}=q_{i}),\quad i,j=1,2,\dots,N
$$  

即系统在时刻 $t$ 处于状态 $q_{i}$ 的条件下在时刻 $t+1$ 转移到状态 $q_{j}$ 的概率。  

**观测概率矩阵**  
&emsp;&emsp;设 $B$ 是观测概率矩阵：  

$$
B=[b_{j}(k)]_{N \times M}
$$  

其中，  

$$
b_{j}(k)=P(o_{t}=v_{k} | i_{t}=q_{j}),\quad k=1,2,\dots,M;\quad j=1,2,\dots,N
$$  

即系统在时刻 $t$ 处于状态 $q_{j}$ 的条件下生成观测 $v_{k}$ 的概率。  

**初始状态概率向量**  
&emsp;&emsp;设 $\pi$ 是初始状态概率向量：  

$$
\pi = \begin{bmatrix}
    \pi_1,\pi_2,\dots,\pi_N
\end{bmatrix}^{T}
$$  

其中，  

$$
\pi_{i}=P(i_{1}=q_{i}),\quad i=1,2,\dots,N
$$  

&emsp;&emsp;隐马尔可夫模型由**初始状态概率向量$\pi$**、**状态转移概率矩阵$A$**、**观测概率矩阵$B$** 决定。因此，隐马尔可夫模型的参数$\lambda$可用三元符号表示，即：  

$$
\lambda = (A,B,\pi)
$$  

&emsp;&emsp;$A,B,\pi$ 称为隐马尔可夫模型的三要素。  

## 模型假设  

&emsp;&emsp;隐马尔可夫模型有两个基本假设：  
&emsp;&emsp;(1) 齐次马尔可夫性假设，即假设隐藏的马尔可夫链在任意时刻 $t$ 的状态只依赖于前一时刻的状态，与其他时刻的状态及观测无关，也与时刻 $t$ 无关：  

$$
P(i_{t} | i_{t-1},\dots,i_{1};o_{t-1},\dots,o_{1})=P(i_{t} | i_{t-1}),\quad t=1,2,\dots,T
$$  

&emsp;&emsp;(2) 观测独立性假设，即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他观测及状态无关：  

$$
P(o_{t} | i_{t},\dots,i_{1};o_{t-1},\dots,o_{1})=P(o_{t} | i_{t})
$$  

## 基本问题  

&emsp;&emsp;隐马尔可夫模型有3个基本问题，包括概率计算问题、学习问题、解码问题。  

### (1) 概率计算问题  

&emsp;&emsp;给定模型参数 $\lambda=(A,B,\pi)$ 和观测序列 $O=\{o_1,o_2,\dots,o_{T}\}$，计算在给定模型参数 $\lambda$ 的条件下观测序列 $O$ 出现的概率 $P(O | \lambda)$。  

### (2) 学习问题  

&emsp;&emsp;已知观测序列 $O=\{o_1,o_2,\dots,o_{T}\}$，估计模型参数 $\lambda=(A,B,\pi)$，使得在该模型下观测序列概率 $P(O | \lambda)$ 最大，即：  

$$
\hat{\lambda}=\arg\max_{\lambda} P(O | \lambda)
$$  

即用极大似然估计的方法估计参数。  

### (3) 解码问题  

&emsp;&emsp;已知模型参数 $\lambda=(A,B,\pi)$ 和观测序列 $O=\{o_1,o_2,\dots,o_{T}\}$，求给定观测序列条件下概率 $P(I | O)$ 最大的状态序列 $I=\{i_1,i_2,\dots,i_{T}\}$，即：  

$$
\hat{I} = \arg\max_{I} P(I | O)
$$

根据所预测的$I$的时刻不同，解码问题又可分为预测问题与滤波问题：  

- 预测问题：$\hat{i}_{t+1} = \arg\max P(i_{t+1} | o_1,\dots,o_{t})$  
- 滤波问题：$\hat{i}_{t} = \arg\max P(i_{t} | o_1,\dots,o_{t})$  

## 概率计算问题  

&emsp;&emsp;已知模型参数 $\lambda=(A,B,\pi)$ 和观测序列 $O=\{o_1,o_2,\dots,o_{T}\}$，计算 $P(O | \lambda)$。 概率计算问题主要有**直接计算法、前向计算法、后向计算法**。  

### 直接计算法  

&emsp;&emsp;直接计算法的思路是通过列举所有可能的长度为 $T$ 状态序列 $I=\{i_1,i_2,\dots,i_{T}\}$，求各个状态序列 $I$ 与观测序列 $O=\{o_1,o_2,\dots,o_{T}\}$ 的联合概率 $P(O,I | \lambda)$，然后对所有可能的状态序列求和，得到 $P(O | \lambda)$。计算过程如下：  

$$
P(O | \lambda) = \sum_{I}P(I,O | \lambda)=\sum_{I}P(O | I,\lambda)P(I | \lambda)
$$  

$$
\begin{split}
    P(I | \lambda) &= P(i_1,i_2,\dots,i_{T} | \lambda) \\
    &= P(i_{T} | i_1,\dots,i_{T-1};\lambda)P(i_1,\dots,i_{T-1} | \lambda) \\
    &= P(i_{T} | i_{T-1};\lambda)P(i_1,\dots,i_{T-1} | \lambda) \\
    &= \left( a_{i_{T-1}i_{T}} \right) \left( \pi_{i_1}a_{i_{1}i_{2}} \dotsb a_{i_{T-2}i_{T-1}} \right) \\
    &= \pi_{i_1}a_{i_{1}i_{2}}a_{i_{2}i_{3}} \dotsb a_{i_{T-1}i_{T}}
\end{split}
$$  

$$
\begin{split}
    P(O | I,\lambda) &= P(o_1,o_2,\dots,o_{T} | i_1,i_2,\dots,i_{T};\lambda) \\
    &= b_{i_1}(o_1)b_{i_2}(o_2) \dotsb b_{i_T}(o_T)
\end{split}
$$

$$
\begin{split}
    P(O | \lambda) &= \sum_{I}P(O | I,\lambda)P(I | \lambda) \\
    &= \sum_{i_1,i_2,\dots,i_T}\pi_{i1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2) \dotsb a_{i_{T-1}i_{T}}b_{i_{T}}(o_{T})
\end{split}
$$  

&emsp;&emsp;直接计算法的思路非常直观，容易理解，但缺点是计算量很大，是 $O(TN^{T})$ 阶的，随着时间的推移呈指数型增长，这种算法在实际中是不可取的。实际上，在概率计算问题中，我们更常用的是前向计算法和后向计算法。  

### 前向计算法  

&emsp;&emsp;在导出前向算法之前，我们首先来定义**前向概率:**  

$$
\alpha_{t}(i)=P(o_1,o_2,\dots,o_{t};i_{t}=q_{i} | \lambda)
$$  

前向概率表示在给定模型参数 $\lambda$ 的条件下，到时刻 $t$ 部分观测序列为 $o_1,o_2,\dots,o_{t}$ 且状态为 $q_{i}$ 的概率。则观测序列概率可以表示为：  

$$
P(O | \lambda)=\sum_{i=1}^{N}P(o_1,o_2,\dots,o_{T};i_{T}=q_{k} | \lambda)=\sum_{i=1}^{N}\alpha_{T}(k)
$$

&emsp;&emsp;前向计算法的主要思想是递推地求得前向概率 $\alpha_{t}(i)$ 及观测序列概率 $P(O | \lambda)$，前向计算法的过程可用下图理解：  

<center>
    <img src="https://s2.loli.net/2023/11/27/ry1pAg8RZwhTGkV.jpg" width=60% height=50%>
    <div align="center">Image2: 前向递推</div>
</center>  

&emsp;&emsp;接下来我们需要找到 $\alpha_{t}(i)$ 和 $\alpha_{t+1}(j)$ 之间的递推关系式：  

$$
\begin{split}
    \alpha_{t+1}(j) &= P(o_1,\dots,o_{t},o_{t+1};i_{t+1}=q_{j} | \lambda) \\
    &= \sum_{i=1}^{N}P(o_1,\dots,o_{t},o_{t+1};i_{t}=q_{i},i_{t+1}=q_{j} | \lambda) \\
    &= \sum_{i=1}^{N} P(o_{t+1} | o_1,\dots,o_{t};i_{t}=q_{i},i_{t+1}=q_{j};\lambda)P(o_1,\dots,o_{t};i_{t}=q_{i},i_{t+1}=q_{j} | \lambda) \\
    &= \sum_{i=1}^{N}P(o_{t+1}|i_{t+1}=q_{j};\lambda)P(i_{t+1}=q_{j}|o_1,\dots,o_{t};i_{t}=q_{i};\lambda)P(o_1,\dots,o_{t};i_{t}=q_{i} | \lambda) \\
    &= \sum_{i=1}^{N}P(o_{t+1}|i_{t+1}=q_{j};\lambda)P(i_{t+1}=q_{j} | i_{t}=q_i;\lambda)P(o_1,\dots,o_{t};i_{t}=q_{i} | \lambda) \\
    &= \sum_{i=1}^{N}b_{j}(o_{t+1})a_{ij}\alpha_{t}(i)
\end{split}
$$  

&emsp;&emsp;当 $t=1$ 时，有：  

$$
\alpha_{1}(i)=\pi_{i}b_{i}(o_1)
$$  

&emsp;&emsp;综上所述，前向计算法的递推关系式可以总计为：  

$$
\begin{split}
    \alpha_{1}(i) &= \pi_{i}b_{i}(o_1),\quad i=1,2,\dots,N \\
    \alpha_{t+1}(j) &= b_{j}(o_{t+1})\sum_{i=1}^{N}a_{ij}\alpha_{t}(i),\quad j=1,2,\dots,N
\end{split}
$$  

#### 前向算法  

&emsp;&emsp;**观测序列概率的前向算法**  
&emsp;&emsp;输入：隐马尔可夫模型的参数 $\lambda=(A,B,\pi)$，观测序列 O；  
&emsp;&emsp;输出：观测序列概率 $P(O | \lambda)$。  
&emsp;&emsp;(1) 初值  

$$
\alpha_{1}(i)=\pi_{i}b_{i}(o_1),\quad i=1,2,\dots,N
$$  

&emsp;&emsp;(2) 递推 $\quad$ 对 $t=1,2,\dots,T-1,$  

$$
\alpha_{t+1}(j) = b_{j}(o_{t+1})\sum_{i=1}^{N}a_{ij}\alpha_{t}(i),\quad j=1,2,\dots,N
$$  

&emsp;&emsp;(3) 终止  

$$
P(O | \lambda)=\sum_{i=1}^{N}\alpha_{T}(i)
$$

&emsp;&emsp;前向算法的计算复杂度为 $O(TN^{2})$，优于直接计算法。  

### 后向计算法  

&emsp;&emsp;后向计算法与前向计算法大致相同，不同点在于后向计算法是从后向前递推。我们首先来定义**后向概率:**  

$$
\beta_{t}(i) = P(o_{t+1},o_{t+2},\dots,o_{T} | i_{t}=q_{i};\lambda)
$$  

后向概率表示在给定模型参数 $\lambda$，系统到时刻 $t$ 状态为 $q_{i}$ 的条件下，从 $t+1$ 到 $T$ 的部分观测序列为 $o_{t+1},o_{t+2},\dots,o_{T}$ 的概率。则观测序列概率可以表示为：  

$$
\begin{split}
    P(O | \lambda) &= P(o_1,o_2,\dots,o_{T} | \lambda) \\
    &= \sum_{i=1}^{N} P(o_1,o_2,\dots,o_{T};i_{1}=q_{i} | \lambda) \\
    &= \sum_{i=1}^{N} P(o_1,o_2,\dots,o_{T} | i_{1}=q_{i};\lambda)P(i_1=q_{i} | \lambda) \\
    &= \sum_{i=1}^{N} P(o_1 | o_2,\dots,o_{T};i_{1}=q_{i};\lambda)P(o_2,\dots,o_{T} | i_{1}=q_{i};\lambda)\pi_{i} \\  
    &= \sum_{i=1}^{N}P(o_1 | i_{1}=q_{i};\lambda)\beta_{1}(i)\pi_{i} \\
    &= \sum_{i=1}^{N} \pi_{i} b_{i}(o_1) \beta_{1}(i)
\end{split}
$$
 
&emsp;&emsp;后向计算法的主要思想是递推地求得后向概率 $\beta_{t}(i)$ 及观测序列概率 $P(O | \lambda)$，后向计算法的过程可用下图理解：  

<center>
    <img src="https://s2.loli.net/2023/11/27/9jd7FOnlkuo2rM4.jpg" width=60% height=60%>
    <div align="center">Image3: 后向递推</div>
</center>  

&emsp;&emsp;之后我们来导出 $\beta_{t}(i)$ 和 $\beta_{t+1}(j)$ 之间的递推关系式：  

$$
\begin{split}
    \beta_{t}(i) &= P(o_{t+1},\dots,o_{T} | i_{t}=q_{i};\lambda) \\
    &= \sum_{j=1}^{N}P(o_{t+1},\dots,o_{T};i_{t+1}=q_{j} | i_{t}=q_{i};\lambda) \\
    &= \sum_{j=1}^{N}P(o_{t+1},\dots,o_{T} | i_{t+1}=q_{j},i_{t}=q_{i};\lambda)P(i_{t+1}=q_{j} | i_{t}=q_{i};\lambda) \\
    &= \sum_{j=1}^{N}P(o_{t+1},\dots,o_{T} | i_{t+1}=q_{j};\lambda)a_{ij} \\
    &= \sum_{j=1}^{N}P(o_{t+1} | o_{t+2},\dots,o_{T};i_{t+1}=q_{j};\lambda)P(o_{t+2},\dots,o_{T} | i_{t+1}=q_{j};\lambda)a_{ij} \\
    &= \sum_{j=1}^{N}P(o_{t+1} | i_{t+1}=q_{j};\lambda) \beta_{t+1}(j) a_{ij}  \\
    &= \sum_{j=1}^{N} b_{j}(o_{t+1})a_{ij} \beta_{t+1}(j)
\end{split}
$$  

&emsp;&emsp;当 $t=T$ 时，给定初始的后向概率：  

$$
\beta_{T}(i) = 1,\quad i=1,2,\dots,N
$$  

&emsp;&emsp;综上所述，后向计算法的递推关系式可以总计为：  

$$
\begin{split}
    \beta_{T}(i) &= 1,\quad i=1,2,\dots,N \\
    \beta_{t}(i) &= \sum_{j=1}^{N} b_{j}(o_{t+1})a_{ij} \beta_{t+1}(j),\quad t=T-1,\dots,1;i=1,\dots,N
\end{split}
$$  

#### 后向算法  

**观测序列概率的后向算法**  
&emsp;&emsp;输入：隐马尔可夫模型参数 $\lambda=(A,B,\pi)$，观测序列 $O$;  
&emsp;&emsp;输出：观测序列概率 $P(O | \lambda)$。  
&emsp;&emsp;(1) 初值  

$$
\beta_{T}(i) = 1,\quad i=1,2,\dots,N
$$  

&emsp;&emsp;(2) 递推 $\quad$ 对 $t=T-1,T-2,\dots,1$  

$$
\beta_{t}(i) = \sum_{j=1}^{N} b_{j}(o_{t+1})a_{ij} \beta_{t+1}(j),\quad i=1,\dots,N
$$  

&emsp;&emsp;(3) 终止  

$$
P(O | \lambda) = \sum_{i=1}^{N} \pi_{i} b_{i}(o_1) \beta_{1}(i)
$$  

&emsp;&emsp;后向算法的计算复杂度为 $O(TN^{2})$，与前向算法相同，优于直接计算法。

## 学习问题  

&emsp;&emsp;隐马尔可夫模型的学习，根据训练数据是包括观测序列和对应的状态序列还是只有观测序列，可以分为监督学习与无监督学习。监督学习主要利用极大似然法来估计模型参数，无监督学习则是利用 $Baum-Welch$ 算法，也就是 $EM$ 算法来估计参数。  

### 监督学习方法  

&emsp;&emsp;假设已给训练数据包含 $S$ 个长度相同的观测序列和对应的状态序列 $\{(O_1,I_1),(O_2,I_2),\dotsb,(O_S,I_S)\}$，可以利用极大似然估计法来估计隐马尔可夫模型的参数。  

&emsp;&emsp;**1.转移概率 $a_{ij}$ 的估计**  
&emsp;&emsp;设样本中时刻 $t$ 处于状态 $q_i$ 且时刻 $t+1$ 转移到状态 $q_j$ 的频数为 $A_{ij}$，那么状态转移概率 $a_{ij}$ 的估计为：  

$$
\hat{a}_{ij}=\frac{A_{ij}}{\sum_{j=1}^{N}A_{ij}},\quad i=1,2,\dotsb,N;\quad j=1,2,\dotsb,N
$$  

&emsp;&emsp;**2.观测概率 $b_{j}(k)$ 的估计**  
&emsp;&emsp;设样本中状态为 $q_j$ 并观测为 $v_k$ 的频数是 $B_{jk}$，那么状态为 $q_j$ 观测为 $v_k$ 的概率 $b_{j}(k)$ 的估计为  

$$
\hat{b}_{j}(k)=\frac{B_{jk}}{\sum_{k=1}^{M}B_{jk}},\quad j=1,2,\quad,N;\quad k=1,2,\dotsb,M
$$  

&emsp;&emsp;**3.初始状态概率 $\pi_{i}$ 的估计**  
&emsp;&emsp;设样本中初始状态为 $q_i$ 的频数为 $Q_i$，则初始状态概率 $\pi_i$ 的估计为  

$$
\hat{\pi}_{i}=\frac{Q_{i}}{S},\quad i=1,2,\dotsb,N
$$  

### 无监督学习方法  

&emsp;&emsp;虽然监督学习的方法操作十分简便，也非常容易理解，但监督学习需要对训练数据进行标注，而人工标注训练数据往往代价很高，因此，有时就会利用无监督学习的方法。无监督学习所使用的算法为 $Baum-Welch$，实际上为 $EM$ 算法。 
&emsp;&emsp;假设给定训练数据只包含 $S$ 个长度为 $T$ 的观测序列 $\{O_1,O_2,\dotsb,O_S\}$ 而没有对应的状态序列，目标是学习隐马尔可夫模型 $\lambda=(A,B,\pi)$ 的参数。我们将观测序列数据看作观测数据 $O$，状态序列数据看作不可观测的隐数据 $I$，那么隐马尔可夫模型实际上是一个含有隐变量的概率模型  

$$
P(O | \lambda)=\sum_{I}P(O | I,\lambda)P(I | \lambda)
$$  

它的参数学习可以由 $EM$ 算法实现。  

&emsp;&emsp;**1.确定完全数据的对数似然函数**  
&emsp;&emsp;所有的观测数据写成 $O=(o_1,o_2,\dotsb,o_{T})$，所有隐数据写成 $I=(i_1,i_2,\dotsb,i_{T})$，完全数据为 $(O,I)=(o_1,o_2,\dotsb,o_{T};i_1,i_2,\dotsb,i_{T})$。完全数据的对数似然函数为：  

$$
L(\lambda) = \log{P(O,I | \lambda)}
$$  

&emsp;&emsp;**2.EM 算法的 E 步：求 $Q$ 函数 $Q(\lambda,\bar{\lambda})$**  
&emsp;&emsp;由 $Q$ 函数的定义得  

$$
\begin{split}
    Q(\lambda,\bar{\lambda}) &= E_{I}[\log{P(O,I|\lambda)} | O,\bar{\lambda}]  \\
    &=\sum_{I}\frac{\log{P(O,I | \lambda)}P(O,I|\bar{\lambda})}{P(O|\bar{\lambda})}
\end{split}
$$  

其中，$\bar{\lambda}$ 是隐马尔可夫模型当前的估计值，$\lambda$ 是下一步要极大化的隐马尔可夫模型参数。由于 $P(O | \bar{\lambda})$ 为常数，对优化没有影响，可以舍去；同时由概率计算中的直接计算法可得：  

$$
P(O,I | \lambda)=\pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2) \dotsb a_{i_{T-1}i_{T}}b_{i_{T}}(o_{T})
$$  

于是函数 $Q(\lambda,\bar{\lambda})$ 可以写成：  

$$
\begin{split}
    Q(\lambda,\bar{\lambda}) &= \sum_{I}\log{P(O,I|\lambda)P(O,I|\bar{\lambda})} \\
    &= \sum_{I} P(O,I | \bar{\lambda}) \log{ [\pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2) \dotsb a_{i_{T-1}i_{T}}b_{i_{T}}(o_{T}) ]}  \\
    &= \sum_{I} P(O,I | \bar{\lambda}) \log{\left[ \pi_{i_1}\left(\prod_{t=1}^{T-1}a_{i_{t}i_{t+1}}\right)\left( \prod_{t=1}^{T}b_{i_t}(o_t) \right) \right]} \\
    &=\sum_{I}P(O,I | \bar{\lambda}) \left[ \log(\pi_{i})+\sum_{t=1}^{T-1}\log(a_{i_{t}i_{t+1}})+\sum_{t=1}^{T}\log(b_{i_t}(o_t)) \right] \\
    &=\sum_{I}\log(\pi_{i})P(O,I | \bar{\lambda})+\sum_{I}\left( \sum_{t=1}^{T-1}\log(a_{i_{t}i_{t+1}}) \right)P(O,I | \bar{\lambda}) \\
    &+ \sum_{I}\left( \sum_{t=1}^{T}\log(b_{i_t}(o_t)) \right)P(O,I | \bar{\lambda})
\end{split}
$$

式中求和都是对所有数据的序列总长度 $T$ 进行的。通过观察 $Q(\lambda,\bar{\lambda})$ 的计算式可以看出 $Q(\lambda,\bar{\lambda})$ 的第一项 $\sum_{I}\log(\pi_{i})P(O,I | \bar{\lambda})$ 只与参数 $\lambda$ 中的初始概率 $\pi$ 有关；第二项  $\sum_{I}\left( \sum_{t=1}^{T-1}\log(a_{i_{t}i_{t+1}}) \right)P(O,I | \bar{\lambda})$ 只与参数 $\lambda$ 中的状态转移概率矩阵 $A$ 有关；第三项 $\sum_{I}\left( \sum_{t=1}^{T}\log(b_{i_t}(o_t)) \right)P(O,I | \bar{\lambda})$ 只与参数 $\lambda$ 中的观测概率矩阵 $B$ 有关。因此，可以令：  

$$
\begin{split}
    Q_{1}(\pi,\bar{\lambda}) &= \sum_{I}\log(\pi_{i})P(O,I | \bar{\lambda}) \\
    Q_{2}(A,\bar{\lambda}) &= \sum_{I}\left( \sum_{t=1}^{T-1}\log(a_{i_{t}i_{t+1}}) \right)P(O,I | \bar{\lambda}) \\
    Q_{3}(B,\bar{\lambda}) &= \sum_{I}\left( \sum_{t=1}^{T}\log(b_{i_t}(o_t)) \right)P(O,I | \bar{\lambda})
\end{split}
$$  

则 $Q(\lambda,\bar{\lambda})$ 可以写成：  

$$
Q(\lambda,\bar{\lambda})=Q_{1}(\pi,\bar{\lambda})+Q_{2}(A,\bar{\lambda})+Q_{3}(B,\bar{\lambda})
$$  

&emsp;&emsp;**3.EM 算法的 M 步：极大化 $Q$ 函数 $Q(\lambda,\bar{\lambda})$ 求模型参数 $A,B,\pi$**  
&emsp;&emsp;通过极大化 $Q(\lambda,\bar{\lambda})$ 可以得到模型参数 $A,B,\pi$ 的估计值。  
&emsp;&emsp;**(1) 估计初始状态概率 $\pi$**  
&emsp;&emsp;$Q(\lambda,\bar{\lambda})$ 中只有 $Q_{1}(\pi,\bar{\lambda})$ 与初始状态概率 $\pi$ 有关，因此优化问题可以写成：

$$
\begin{split}
    & \max_{\pi} \quad Q_{1}(\pi,\bar{\lambda}) \\
    & \space s.t. \quad \sum_{i=1}^{N} \pi_{i}=1 \\
\end{split}
$$  

优化问题的拉格朗日函数：  

$$
L(\pi,\gamma)=\sum_{I}\log(\pi_{i})P(O,I | \bar{\lambda})+\gamma \left( \sum_{i=1}^{N}\pi_{i}-1 \right)
$$  

由费马定理得：  

$$
\frac{\partial{L(\pi,\gamma)}}{\partial{\pi_{i}}} = \frac{P(O,i_1=i | \bar{\lambda})}{\pi_{i}}+\gamma=0,\quad i=1,2,\dotsb,N
$$  

得到：  

$$
\gamma\pi_{i}=-P(O,i_1=i|\bar{\lambda})
$$  

两边同时对 $i$ 求和得：  

$$
\begin{split}
    & \gamma\sum_{i=1}^{N}\pi_{i} = -\sum_{i=1}^{N}P(O,i_{1}=i|\bar{\lambda})=P(O | \bar{\lambda}) \\
    & \Rightarrow \gamma = P(O | \bar{\lambda}) \\
\end{split}
$$  

从而得到 $\pi_{i}$ 的估计值为：  

$$
\hat{\pi}_{i}=\frac{P(O,i_1=i|\bar{\lambda})}{P(O | \bar{\lambda})}
$$  

&emsp;&emsp;**(2) 估计状态转移矩阵 $A$**  
&emsp;&emsp;$Q(\lambda,\bar{\lambda})$ 中只有 $Q_{2}(A,\bar{\lambda})$ 与状态转移概率矩阵 $A$ 有关，因此优化问题可以写成：

$$
\begin{split}
    & \max_{A} \quad Q_{2}(A,\bar{\lambda}) \\
    & \space s.t. \quad \sum_{j=1}^{N}a_{ij}=1 \\
\end{split}
$$  

优化问题的拉格朗日函数：  

$$
\begin{split}
    L(A,\gamma) &= \sum_{I}\left( \sum_{t=1}^{T-1}\log(a_{i_{t}i_{t+1}}) \right)P(O,I | \bar{\lambda})+\gamma\left( \sum_{j=1}^{N}a_{ij}-1 \right) \\
    &=\sum_{t=1}^{T-1}\sum_{i=1}^{N}\sum_{j=1}^{N}\left( P(O,i_{t}=i,i_{t+1}=j|\bar{\lambda})\log{a_{ij}} \right)+\gamma\left( \sum_{j=1}^{N}a_{ij}-1 \right)
\end{split}
$$  

由费马定理得：  

$$
\frac{\partial{L(A,\gamma)}}{\partial{a_{ij}}}=\frac{\sum_{t=1}^{T-1}P(O,i_{t}=i,i_{t+1}=j|\bar{\lambda})}{a_{ij}}+\gamma=0,\quad i,j=1,\dotsb,N
$$

得到：  

$$
\gamma a_{ij}=-\sum_{t=1}^{T-1}P(O,i_{t}=i,i_{t+1}=j|\bar{\lambda})
$$

两边同时对 $j$ 求和：  

$$
\begin{split}
    & \gamma\sum_{j=1}^{N}a_{ij}=-\sum_{t=1}^{T-1}\sum_{j=1}^{N}P(O,i_{t}=i,i_{t+1}=j|\bar{\lambda}) \\
    & \Rightarrow \gamma = -\sum_{t=1}^{T-1}P(O,i_{t}=i|\bar{\lambda})
\end{split}
$$  

从而得到状态转移概率矩阵元素 $a_{ij}$ 的估计值为：  

$$
\hat{a}_{ij}=\frac{\sum_{t=1}^{T-1}P(O,i_{t}=i,i_{t+1}=j|\bar{\lambda})}{\sum_{t=1}^{T-1}P(O,i_{t}=i|\bar{\lambda})}
$$  

&emsp;&emsp;**(3) 估计观测概率矩阵 $B$**  
&emsp;&emsp;$Q(\lambda,\bar{\lambda})$ 中只有 $Q_{3}(B,\bar{\lambda})$ 与观测概率矩阵 $B$ 有关，因此优化问题可以写成：

$$
\begin{split}
    & \max_{B} \quad Q_{3}(B,\bar{\lambda}) \\
    & \space s.t. \quad \sum_{k=1}^{M}b_{j}(v_{k})=1 \\
\end{split}
$$  

优化问题的拉格朗日函数：  

$$
\begin{split}
    L(B,\gamma) &= \sum_{I}\left( \sum_{t=1}^{T}\log(b_{i_t}(o_t)) \right)P(O,I | \bar{\lambda})+\gamma\left( \sum_{k=1}^{M}b_{j}(v_{k})-1 \right) \\
    &=\sum_{t=1}^{T}\sum_{j=1}^{N}\left( P(O,i_{t}=j|\bar{\lambda})\log{b_{j}(o_{t})} \right)+\gamma\left( \sum_{k=1}^{M}b_{j}(v_{k})-1 \right)
\end{split}
$$  

由费马定理得：  

$$
\frac{\partial{L(B,\gamma)}}{\partial{b_{j}(v_{k})}}=\frac{\sum_{t=1}^{T}P(O,i_{t}=j|\bar{\lambda})I(o_{t}=v_{k})}{b_{j}(v_{k})}+\gamma=0,\quad j=1,\dotsb,N;k=1,\dotsb,M
$$  

得到：  

$$
\gamma b_{j}(v_{k})=-\sum_{t=1}^{T}P(O,i_{t}=j|\bar{\lambda})I(o_{t}=v_{k})
$$  

两边同时对 $k$ 求和：  

$$
\begin{split}
    & \gamma\sum_{k=1}^{M}b_{j}(v_{k})=-\sum_{k=1}^{M}\sum_{t=1}^{T}P(O,i_{t}=j|\bar{\lambda})I(o_{t}=v_{k}) \\
    & \Rightarrow \gamma = -\sum_{t=1}^{T}P(O,i_{t}=j|\bar{\lambda})
\end{split}
$$  

从而得到观测概率矩阵元素的 $b_{j}(k)$ 的估计值为：  

$$
\hat{b}_{j}(k)=\frac{\sum_{t=1}^{T}P(O,i_{t}=j|\bar{\lambda})I(o_{t}=v_{k})}{\sum_{t=1}^{T}P(O,i_{t}=j|\bar{\lambda})}
$$  

#### Baum-Welch 算法  

&emsp;&emsp;通过以上的推导，我们可以总结出无监督学习下隐马尔可夫参数估计的一种算法，其被称为 Baum-Welch 算法，本质上是 EM 算法在隐马尔可夫模型学习中的具体实现。  

&emsp;&emsp;**Baum-Welch 算法**  
&emsp;&emsp;输入：观测数据 $O=(o_1,o_2,\dotsb,o_{T})$；  
&emsp;&emsp;输出：隐马尔可夫模型参数 $\lambda=(A,B,\pi)$。  
&emsp;&emsp;(1) 初始化。对 $n=0$，选取 $a_{ij}^{(0)},b_{j}^{(0)}(k),\pi_{i}^{(0)}$，得到模型 $\lambda^{(0)}=(A^{(0)},B^{(0)},\pi^{(0)})$。  
&emsp;&emsp;(2) 递推。对 $n=1,2,\dotsb$，  

$$
a_{ij}^{(n+1)}=\frac{\sum_{t=1}^{T-1}P(O,i_{t}=i,i_{t+1}=j|\lambda^{(n)})}{\sum_{t=1}^{T-1}P(O,i_{t}=i|\lambda^{(n)})}
$$  

$$
b_{j}^{(n+1)}(k)=\frac{\sum_{t=1}^{T}P(O,i_{t}=j|\lambda^{(n)})I(o_{t}=v_{k})}{\sum_{t=1}^{T}P(O,i_{t}=j|\lambda^{(n)})}
$$  

$$
\pi_{i}^{(n+1)}=\frac{P(O,i_1=i|\lambda^{(n)})}{P(O | \lambda^{(n)})}
$$  

&emsp;&emsp;(3) 终止。得到模型参数 $\lambda^{(n+1)}=(A^{(n+1)},B^{(n+1)},\pi^{(n+1)})$  

## 解码问题  

&emsp;&emsp;解码问题主要是研究给定观测序列下最有可能出现的状态序列。已知模型参数 $\lambda=(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\dotsb,o_{T})$，求某一观测序列 $I=(i_1,i_2,\dotsb,i_{T})$ 使得条件概率 $P(I | O)$ 最大。隐马尔可夫模型中的解码问题主要使用维特比算法。  

### 维特比算法  

&emsp;&emsp;维特比算法实际是用动态规划(dynamic programming)解隐马尔可夫模型解码问题，即用动态规划求概率最大路径(最优路径)。这时一条路径对应着一个状态序列。维特比算法的思想可以用下图表示：  

<center>
    <img src="https://s2.loli.net/2023/12/06/E87VbUDGMmWHFJp.jpg" width=60% height=60%>
    <div align="center">Image3: 最优路径</div>
</center>  

&emsp;&emsp;根据动态规划的原理，最优路径具有这样的特性：**如果最优路径在 $t$ 时刻通过结点 $i_{t}^{*}$，那么这一路径从结点 $i_{t}^{*}$ 到终点 $i_{T}^{*}$ 的部分路径，对于从 $i_{t}^{*}$ 到 $i_{T}^{*}$ 的所有可能的部分路径来说，必须是最优的。** 依据这一原理，我们只需要从时刻 $t=1$ 开始，递推地计算在时刻 $t$ 状态为 $i$ 的各条部分路径的最大概率，直至得到时刻 $t=T$ 状态为 $i$ 的各条路径的最大概率。时刻 $t=T$ 的最大概率即为最优路径的概率 $P^{*}$，最优路径的终结点 $i_{T}^{*}$ 也同时得到。之后，为了找出最优路径的各个结点，从终结点 $i_{T}^{*}$ 开始，由后向前逐步求得结点 $i_{T-1}^{*},\dotsb,i_{1}^{*}$，得到最优路径 $I^{*}=(i_{1}^{*},i_{2}^{*},\dotsb,i_{T}^{*})$。这就是维特比算法。  

&emsp;&emsp;首先导入两个变量 $\delta$ 和 $\psi$。定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,\dotsb,i_{t})$ 中概率最大值为：  

$$
\delta_{t}(i)=\max_{i_1,i_2,\dotsb,i_{t-1}}P(i_{t}=i,i_{t-1},\dotsb,i_1;o_{t},\dotsb,o_1 | \lambda),\quad i=1,2,\dotsb,N
$$  

&emsp;&emsp;由定义可得变量 $\delta$ 的递推公式：  

$$
\begin{split}
    \delta_{t+1}(i) &= \max_{i_1,i_2,\dotsb,i_{t}}P(i_{t+1}=i,i_{t},\dotsb,i_1;o_{t+1},\dotsb,o_1 | \lambda) \\
    &= \max_{1 \leq j \leq N}[\delta_{t}a_{ji}]b_{i}(o_{t+1}),\quad i=1,2,\dotsb,N; \space t=1,2,\dotsb,T-1
\end{split}
$$  

&emsp;&emsp;定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,\dotsb,i_{t-1},i)$ 中概率最大的路径的第 $t-1$ 个结点为  

$$
\psi_{t}(i)=\arg\max_{1 \leq j \leq N}[\delta_{t-1}(j)a_{ji}],\quad i=1,2,\dotsb,N
$$


&emsp;&emsp;**维特比算法**  
&emsp;&emsp;输入：模型参数 $\lambda=(A,B,\pi)$ 和观测 $O=(o_1,o_2,\dotsb,o_{T})$；  
&emsp;&emsp;输出：最优路径 $I^{*}=(i_{1}^{*},i_{2}^{*},\dotsb,i_{T}^{*})$.  
&emsp;&emsp;(1) 初始化  

$$
\delta_{1}(i)=\pi_{i}b_{i}(o_1),\quad i=1,2,\dotsb,N
$$  

$$
\psi_{1}(i)=0,\quad i=1,2,\dotsb,N
$$  

&emsp;&emsp;(2) 递推。对 $t=2,3,\dotsb,T$  

$$
\delta_{t}(i)=\max_{1 \leq j \leq N}[\delta_{t-1}(j)a_{ji}]b_{i}(o_{t}),\quad i=1,2,\dotsb,N
$$  

$$
\psi_{t}(i)=\arg\max_{1 \leq j \leq N}[\delta_{t-1}(j)a_{ji}],\quad i=1,2,\dotsb,N
$$  

&emsp;&emsp;(3) 终止  

$$
P^{*}=\max_{1 \leq i \leq N} \delta_{T}(i)
$$  

$$
i_{T}^{*}=\arg\max_{1 \leq i \leq N}[\delta_{T}(i)]
$$  

&emsp;&emsp;(4) 最优路径回溯。对 $t=T-1,T-2,\dotsb,1$  

$$
i_{t}^{*}=\psi_{t+1}(i_{t+1}^{*})
$$  

求得最优路径 $I^{*}=(i_{1}^{*},i_{2}^{*},\dotsb,i_{T}^{*})$。  

## 参考

**[1] Book: 李航,统计学习方法(第2版)**  
**[2] Book: 董平,机器学习中的统计思维(Python实现)**  
**[3] Video: bilibili,简博士,隐马尔可夫系列**  
**[4] Video: bilibili,shuhuai008,隐马尔可夫系列**  