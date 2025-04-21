---
title: 感知机
tags:
    - Machine Learning
---  

## toc

## 感知机  

&emsp;&emsp;感知机(perception)是一个处理二分类问题的线性分类模型。感知机模型旨在寻找一个合适的超平面，将特征空间中的示例进行正确分类。为了找到这个超平面，需要设定基于误分类的损失函数，并利用梯度下降方法对损失函数进行极小化，求得感知机模型。感知机学习算法具有简单与易于实现的优点，分为原始形式和对偶形式。  
&emsp;&emsp;感知机模型由$Rosenblatt$于1957年提出，尽管感知机在处理复杂问题方面存在局限性，但它为神经网络和深度学习的发展铺平了道路。感知机模型的思想和学习算法为后来更复杂的神经网络提供了基础，尤其是多层感知机和其他深度学习模型。感知机的历史地位在机器学习领域被广泛认可，被视为神经网络的早期里程碑。  

### 基本思想  

&emsp;&emsp;感知机的基本思想为：在$n$维特征空间中，寻找一个能够使训练数据集中的正实例点与负实例点完全分开的超平面，对于新的实例，利用该超平面进行分类。为了找到这个超平面，感知机设置了以误分类样本数为基础的损失函数，通过梯度下降算法最小化损失函数，更新超平面，最终使得所有的训练实例均被正确分类，此时便找到了分类所需的超平面。感知机的基本思想可以用如下图1来描述：  

<center>
    <img src="https://s2.loli.net/2023/10/08/r9CDXyRLPI5kdN6.jpg" width=60% height=60%>
    <div align="center">Image1: 感知机的基本思想</div>
</center>  

### 模型  

**输入**  

- **输入空间:** $\mathcal{X} = \mathbb{R}^{n}$   
- **输入实例:** $x = \begin{bmatrix}
    x^{(1)},x^{(2)},\dots,x^{(n)} \\
\end{bmatrix} \in \mathcal{X}$  

&emsp;&emsp;其中$\mathcal{X}$代表$n$维实数空间，输入实例$x$为$n$维特征向量。  

**输出**  

- **输出空间:** $\mathcal{Y} = \{-1,+1\}$  
- **输出实例:** $y \in \mathcal{Y}$  

&emsp;&emsp;其中输出空间$\mathcal{Y}$只包含+1和-1的一个集合，+1与-1分别代表二分类问题中的正类与负类。输出实例$y$代表输出实例$x$的类别。  

**数据集**  
&emsp;&emsp;感知机模型的训练数据集$T_{train}$为：  

$$
T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

&emsp;&emsp;其中，$x_i \in \mathcal{X}=\mathbb{R}^{n},y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\dots,N$，$N$表示训练数据的样本容量。  

**模型**  
&emsp;&emsp;设输入空间$\mathcal{X}$到输出空间$\mathcal{Y}$的映射为：  

$$
f(x)=sign(w^{T}x+b)
$$  

&emsp;&emsp;称模型$f$为感知机，其中，$w$和$b$为感知机模型的参数，$w \in \mathbb{R}^{n}$为权重向量，$b \in \mathbb{R}$为偏置。$sign(\cdot)$为符号函数，即：  

$$
sign(x)= \left \{
\begin{array}{rcl}
+1, & x \ge 0\\
-1, & x < 0  \\
\end{array} \right.
$$  

&emsp;&emsp;感知机是一种线性分类模型，属于判别模型。  

**假设空间**  
&emsp;&emsp;感知机模型的假设空间$\mathcal{H}$为：  

$$
\mathcal{H}=\{f \vert f(x)=w^{T}x+b\}
$$  

&emsp;&emsp;感知机模型的假设空间实际上特征空间中超平面的集合。


**参数空间**  
&emsp;&emsp;令$\theta=(w,b)$，则模型的假设空间$\mathcal{H}$等价于参数空间$\Theta$：  

$$
\Theta=\{\theta \vert \theta \in \mathbb{R}^{n+1}\}
$$


### 策略  

&emsp;&emsp;感知机学习的基本假设是训练数据集是线性可分的，这里先给出数据集线性可分性的定义。  

**数据集的线性可分性**  
&emsp;&emsp;给定一个数据集$T$：  

$$
T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

其中，$x_i \in \mathcal{X}=\mathbb{R}^{n},y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\dots,N$，如果存在某个超平面$S$能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧，即 $\exists \theta=(w,b) \in \mathbb{R}^{n+1}$，对所有 $y_i=+1$ 的实例$i$，有 $w^{T}x_i+b > 0$ ；对所有 $y_i=-1$ 的实例$i$，有 $w^{T}x_i+b < 0$，则称数据集$T$为线性可分数据集(linearly separable data set)；否则，称数据集$T$线性不可分。  

**损失函数**
&emsp;&emsp;为了寻找到合适的超平面，需要设置损失函数，一个很自然的想法是将损失函数设置为误分类点的个数，对于误分类数据$(x_i,y_i)$有：  

$$
-y_i(w^{T}x_i+b) > 0
$$  

这是因为误分类的两种情况: $w^{T}x_i+b > 0,y_i=-1$或$w^{T}x_i+b < 0,y_i=+1$ 均可由上式表示，则损失函数可以表示为：  

$$
L(x,y)=\sum_{i=1}^{N}I_{\{-y_i(w^{T}x_i+b)>0\}}
$$  

$$
I_{\{-y_i(w^{T}x_i+b)>0\}}=\left \{
\begin{array}{rcl}
1, & {-y_i(w^{T}x_i+b)>0}\\
0, & {-y_i(w^{T}x_i+b) \leq 0}\\
\end{array} \right.
$$  

&emsp;&emsp;这样设置损失函数的思路非常直观，但是，这样的损失函数不是参数$w,b$的连续可导函数，不易进行优化，因此我们一般不将损失函数设置为这种形式。  
&emsp;&emsp;**感知机所采用的损失函数是误分类点到超平面的总距离**，其与误分类点的个数相关，当损失函数降至0时，意味着没有误分类点了，我们也就找到了可以将训练数据集完全正确分类的超平面了。
&emsp;&emsp;为此，首先给出输入空间$\mathcal{X}$中任意一点$x_0$到超平面$S=\{x \vert w^{T}x+b=0\}$的距离：  

$$
d(x_0,S)=\frac{|w^{T}x_0+b|}{|| w ||_{2} }
$$  

&emsp;&emsp;关于$n$维空间中点到超平面的距离公式的证明可参见附录，这里不多做赘述。对于误分类点$x_i$，其到初始超平面$S$的距离为：  

$$
d(x_i,S)=\frac{-y_i(w^{T}x_i+b)}{||w||_2}
$$  

&emsp;&emsp;假设在初始超平面$S$的分类下，误分类点的集合为$M$，则误分类点到超平面的总距离为：  

$$
d(M,S)=-\frac{1}{||w||_2}\sum_{x_i \in M}y_i(w^{T}x_i+b)
$$  

&emsp;&emsp;在设定损失函数时，我们可以不考虑$\frac{1}{||w||_2}$，因为其既不影响损失函数的正负，也不影响感知机算法的最终结果。这样，对于线性可分的训练数据集：  

$$
T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

&emsp;&emsp;感知机学习的损失函数一般设定为：  

$$
L(w,b)=-\sum_{x_i \in M}y_i(w^{T}x_i+b)
$$  

&emsp;&emsp;损失函数$L(w,b)$是参数$w,b$的连续可导函数，便于进行优化。当损失函数减小时，误分类点的数量也会减小，当$L(w,b)=0$时，意味着训练数据集中所有的实例都被正确分类了，此时得到的超平面即是我们要寻找的可以完全正确分类的超平面。  

### 算法  

&emsp;&emsp;感知机算法分为原始形似与对偶形式，这里分别来介绍这两种形式。  

#### 原始形式  

&emsp;&emsp;在策略里我们已经确定了感知机学习的损失函数，我们现在要做的便是最小化损失函数，即：  

$$
\min_{w,b} -\sum_{x_i \in M}y_i(w^{T}x_i+b)
$$  

&emsp;&emsp;我们采用随机梯度下降法(stochastic gradient descent)来优化损失函数。  
&emsp;&emsp;首先，随机选取一个初始超平面$S_0=\{x \vert w_0^{T}x+b_0=0\}$，此时误分类点的集合为$M_0$，则损失函数$L(w,b)$关于参数$w_0,b_0$的梯度为：  

$$
\nabla_{w_0}L(w_0,b_0)=-\sum_{x_i \in M_0}y_ix_i
$$  

$$
\nabla_{b_0}L(w_0,b_0)=-\sum_{x_i \in M_0}y_i
$$  

&emsp;&emsp;在$M_0$随机选取一个误分类点$(x_0,y_0)$，对$w_0,b_0$进行更新：  

$$
w_1 \leftarrow w_0+\eta y_ix_i
$$  

$$
b_1 \leftarrow b_0+\eta y_i
$$  

&emsp;&emsp;其中，$\eta$为梯度下降的步长，在机器学习中也称为学习率(learning rate)。更新后我们会得到一个新的超平面$S_1=\{x \vert w_1^{T} x + b_1=0\}$，此时损失函数的值会减小。这样通过迭代可以期待损失函数$L(w,b)$不断减小，直到为0，此时误分类集合$M=\varnothing$，我们便找到了可以对训练集完全正确分类的超平面$S$.  
&emsp;&emsp;综上所述，我们写出感知机算法的原始形式。  

##### 感知机学习算法的原始形式  

&emsp;&emsp;输入：训练数据集$T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$，其中$x_i \in \mathcal{X}=\mathbb{R}^{n}，y \in \mathcal{Y}=\{-1,+1\},i=1,2,\dots,N$；学习率$\eta (0 < \eta \leq 1)$.  
&emsp;&emsp;输出：$w,b$：感知机模型 $f(x)=sign(w^{T}x+b)$  
&emsp;&emsp;(1) 随机选取初始参数$w_0,b_0$；  
&emsp;&emsp;(2) 在训练数据集中选取一个数据$(x_i,y_i)$；  
&emsp;&emsp;(3) $if \space y_i(w_0^{T}x_i+b_0) \leq 0:$  

$$
w_1 \leftarrow w_0+\eta y_ix_i
$$  

$$
b_1 \leftarrow b_0+\eta y_i
$$  

&emsp;&emsp;(4) 转至 (2)，直至训练集中没有误分类点。  

&emsp;&emsp;这种学习算法直观上有如下解释：**当一个实例点被误分类，即位于分离超平面的错误一侧时，则调整$w,b$的值，使超平面向该误分类点的一侧移动，以减小误分类点与超平面间的距离，直至超平面越过该误分类点使其被正确分类。**  

#### 对偶形式  

&emsp;&emsp;在感知机算法的原始形式中，如果实例点$(x_i,y_i)$是误分类点，则可以用该点更新参数，即：  

$$
w \leftarrow w + \eta y_ix_i
$$  

$$
b \leftarrow b + \eta y_i
$$  

&emsp;&emsp;现在设想一下，如果在参数更新的过程中，点$(x_i,y_i)$被误分类了$n_i$次，则在从初始参数$w_0,b_0$到最终参数$w,b$中，点$(x_i,y_i)$贡献的增量为$n_i \eta y_ix_i$和$n_i \eta y_i$.  
&emsp;&emsp;假设在迭代过程中，训练集中的每一个实例点$(x_i,y_i)$被误分类的次数为$n_i, i=1,2,\dots,N$，取初始的参数向量为零向量，则通过迭代最终学习到的次数可以表示为：  

$$
w=\sum_{i=1}^{N}\alpha_iy_ix_i
$$  

$$
b = \sum_{i=1}^{N}\alpha_iy_i
$$  

&emsp;&emsp;其中，$\alpha_i=n_i \eta$，若$\eta=1$，则$\alpha_i$表示训练集中的实例点$(x_i,y_i)$由于被误分类而用于更新参数的次数。  
 &emsp;&emsp;在使用对偶算法进行迭代时，若$(x_i,y_i)$被误分类，则我们便在其对应的$\alpha_i$上加上增量$\eta$，从而得到新的超平面，循环迭代过程，直至没有误分类点为止。
&emsp;&emsp;综上所述，我们可以写出感知机算法的对偶形式。  

##### 感知机学习算法的对偶形式  

&emsp;&emsp;输入：训练数据集$T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$，其中$x_i \in \mathcal{X}=\mathbb{R}^{n}，y \in \mathcal{Y}=\{-1,+1\},i=1,2,\dots,N$；学习率$\eta (0 < \eta \leq 1)$.  
&emsp;&emsp;输出：$\alpha$：感知机模型 $f(x)=sign \left( (\sum_{i=1}^{N}\alpha_iy_ix_i) \cdot x+(\sum_{i=1}^{N}\alpha_iy_i) \right), \alpha=\begin{bmatrix}
    \alpha_1, \alpha_2, \dots, \alpha_N \\
\end{bmatrix}^T$  
&emsp;&emsp;(1) $\alpha \leftarrow 0$;  
&emsp;&emsp;(2) 在训练数据集$T_{train}$中选取实例点$(x_j,y_j)$;  
&emsp;&emsp;(3) $if \space y_i \left( \sum_{i=1}^{N}\alpha_iy_ix_i \cdot x_j+\sum_{i=1}^{N}\alpha_iy_i \right) \leq 0：$  

$$
\alpha_i \leftarrow \alpha_i + \eta
$$  

&emsp;&emsp;(4) 转至(2)直到没有误分类数据。  

&emsp;&emsp;值得注意的是，在对偶算法的每次迭代中，训练实例仅以内积的形式出现。为了方便计算，可以预先将训练数据中的实例间的内积计算出来，并以矩阵的形式储存，这个矩阵便是$Gram$矩阵：  

$$
G=[x_i \cdot x_j]_{N \times N}
$$  

&emsp;&emsp;可以证明感知机算法是收敛的，即经过有限次迭代可以得到一个能将训练数据集完全正确分类的分离超平面。这部分证明放在附录里，感兴趣的读者可自行阅读。  

### 感知机算法的实例及Python实现  

&emsp;&emsp;首先，我们需要创造一个线性可分的二分类数据。为了便于可视化，设输入实例$x \in \mathcal{X}=\mathbb{R}^2，y \in \mathcal{Y}= \{-1,1\}$，以下是用于产生数据的Python代码：

```python
# Data Generation
np.random.seed(1314)
def data_generation(n,basepoint,ylabel):
    x1 = np.random.random((n,))+np.array([basepoint[0]]*n)
    x2 = np.random.random((n,))+np.array([basepoint[1]]*n)
    y = np.array([ylabel]*n)
    data = pd.DataFrame({"x1":x1,"x2":x2,"y":y})
    return data

# positive data
positivedata = data_generation(n=30,basepoint=(1,2),ylabel=1)
# negative data
negativedata = data_generation(n=20,basepoint=(2,1),ylabel=-1)
# train data
train_data = pd.concat([positivedata,negativedata])
train_data = shuffle(train_data)
train_data.index = range(train_data.shape[0])
train_data.head(5)
```

&emsp;&emsp;在这段代码里，我利用均匀分布在$\{x \vert x_1 \in (1,2);x_2 \in (2,3)\}$里产生了30个正类实例，在$\{x \vert x_1 \in (2,3);x_2 \in (1,2)\}$里产生了20个负类实例。得到的实例数据如下表1：  

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<p align="center"><font face="黑体" size=2.>table1 训练数据(部分)</font></p>
<div class="center">

| index | x1 | x2 | y |
| :---: | :------: | :--:| :---: |
|    0    |  1.749421 | 2.193847 | 1  |
|    1    |  2.057071 | 1.983485 | -1 |
|    2    |  2.830457 | 1.026819 | -1 |
|    3    |  1.437378 | 2.345612 | 1  |
|    4    |  2.730692 | 1.493602 | -1 |
</div>  

&emsp;&emsp;利用以下代码将训练数据进行可视化：  

```python
# train data scatter plot
plt.scatter(x=positivedata["x1"],y=positivedata["x2"],marker="o",c="green",label="positive data")
plt.scatter(x=negativedata["x1"],y=negativedata["x2"],marker="x",c="red",label="negative data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim((0,4))
plt.ylim((0,4))
plt.xticks(np.arange(0,4,0.5))
plt.yticks(np.arange(0,4,0.5))
plt.legend()
```

&emsp;&emsp;得到的训练数据散点图为下图2：  

<center>
    <img src="https://s2.loli.net/2023/10/12/b3KfhBve5VYl6Sw.png" width=60% height=60%>
    <div align="center">Image2: 训练数据散点图</div>
</center>  

&emsp;&emsp;从图中我们可以看出，训练数据是线性可分的，满足感知机学习的假设。  
&emsp;&emsp;之后我们可以根据感知机学习的原始算法来求解分离超平面的参数$w,b$，参数求解的代码如下：  

```python
def param_solving(train_data,lr,start_w):
    # initial parameter
    w = np.array(start_w) # w = [w1,w2,b]
    x = train_data[["x1","x2"]].values
    onecolumn = np.ones((train_data.shape[0],))
    X = np.column_stack((x,onecolumn))
    y = train_data["y"].values
    k = 0
    inter_data = {"k":[],"Parameter":[],"Loss Function":[],"Misclassifications":[],"Point Used for Update":[]}
    # stochastic gradient descent
    while np.any((np.dot(X,w)*y<0) | (np.dot(X,w)*y==0)):
        k += 1
        falseindex = np.where((np.dot(X,w)*y<0) | (np.dot(X,w)*y==0))

        # storing information about the iterative process
        if k%5 == 0:
            inter_data["k"].append(k)
            inter_data["Parameter"].append(w.round(4))
            falsedata = train_data.iloc[falseindex[0],:]
            x_falseclassified = falsedata[["x1","x2"]].values
            X_falseclassified = np.column_stack((x_falseclassified,np.ones((falsedata.shape[0],))))
            y_falseclassified = falsedata["y"].values
            loss = -sum(np.dot(X_falseclassified,w)*y_falseclassified)
            inter_data["Loss Function"].append(loss.round(4))
            inter_data["Misclassifications"].append(falseindex[0])
            inter_data["Point Used for Update"].append(falseindex[0][0])
        
        x_forupdate = train_data.iloc[falseindex[0][0],[0,1]].values
        y_forupdate = train_data.iloc[falseindex[0][0],[2]].values
        delta = np.append(lr*y_forupdate*x_forupdate,lr*y_forupdate)
        w = w + delta
    inter_information = pd.DataFrame(inter_data)

    result_data = {}
    result_data["final parameters"] = w.round(4)
    result_data["interative information"] = inter_information
    result_data["number of iterations"] = k
 
    return result_data
```

&emsp;&emsp;若我们将初始的参数均设置为0，则我们最终得到的分离超平面参数为：  

$$
w,b = \begin{bmatrix}
    -0.1117,0.1103 \\
\end{bmatrix}^T,0.01
$$  

&emsp;&emsp;总共的迭代次数为$k=85$，以下表是迭代过程中的一些信息：  

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>
<p align="center"><font face="黑体" size=2.>table2 迭代过程信息(部分)</font></p>
<div class="center">

| k | Parameter | Loss Function | Misclassifications | Point Used for Update |
| :---: | :------: | :--:| :---: |:---:|
|    5    |  [-0.0062, 0.0042, 0.0]	 | 0.0161 | [0, 5, 7, 9, 24, 28, 36, 43, 45, 49] | 0 |
|    10    |  [0.0052, 0.0304, 0.01] | 1.3912 | [ 1,  2,  4, 10, 12, 13, 15, 18, 22, 29, 30, 32, 33, 37, 38, 39, 40,42, 44, 46] | 1 |
|    15    |  [-0.0215, 0.0147, 0.0] | 0.0564 | [0, 5, 7, 9, 24, 28, 36, 43, 45, 49] | 0 |
|    20    |  [-0.0102, 0.0409, 0.01]	 | 0.9020 | [ 1,  2,  4, 10, 12, 13, 15, 18, 22, 29, 30, 32, 33, 37, 38, 39, 40,42, 44, 46]  | 1 |
|    25    |  [-0.0369, 0.0252, 0.0] | 0.0967 | [0, 5, 7, 9, 24, 28, 36, 43, 45, 49] | 0 |
</div>  

&emsp;&emsp;表中每一列的含义为：  

- **k:** 表示第$k$次更新参数前。  
- **Parameter:** 表示第$k$次更新参数前参数向量$\hat{w}=(w^T,b)^T$.  
- **Loss Function:** 表示第$k$次更新参数前损失函数$L(w,b)$的值。  
- **Misclassifications:** 表示第$k$次更新参数前训练数据中被误分类的示例点的索引。  
- **Point Used for Update:** 表示第$k$次更新所用的误分类示例点的索引。  

&emsp;&emsp;利用以下代码画出分类超平面：  

```python
result = param_solving(train_data=train_data,lr=0.01,start_w=[0,0,0])
w = result["final parameters"]
x1_line = np.linspace(0,4,1000)
x2_line = (-w[0]*x1_line - w[2])/w[1]
plt.scatter(x=positivedata["x1"],y=positivedata["x2"],marker="o",c="green",label="positive data")
plt.scatter(x=negativedata["x1"],y=negativedata["x2"],marker="x",c="red",label="negative data")
plt.plot(x1_line,x2_line,c="blue",label="Classification Hyperplane")
plt.xlim((0,4))
plt.ylim((0,4))
plt.xticks(np.arange(0,4,0.5))
plt.yticks(np.arange(0,4,0.5))
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
```

&emsp;&emsp;得到的分类超平面如下图所示：  

<center>
    <img src="https://s2.loli.net/2023/10/12/8tjLUPw4aqFr29O.png" width=60% height=60%>
    <div align="center">Image3: 分类超平面</div>
</center>  

&emsp;&emsp;如果我们改变初始参数的值，我们可能会得到不同的分类超平面，比如我们将初始参数设置为$w=\begin{bmatrix}
    1,1 \\
\end{bmatrix}^T,b=1$，则我们最终得到的分类超平面如下图4所示：  

<center>
    <img src="https://s2.loli.net/2023/10/12/Hmcf6eJ13yoFWwh.png" width=60% height=60%>
    <div align="center">Image4: 分类超平面</div>
</center>  

&emsp;&emsp;此时，分类超平面的参数为：  

$$
w,b = \begin{bmatrix}
    -0.2049,-0.0153 \\
\end{bmatrix}^T,0.45
$$  

## 附录  

### 空间中点到超平面的距离  

**结论**
&emsp;&emsp;设$n$维空间中存在某超平面$S=\{x \vert w^{T}x+b=0;x,w \in \mathbb{R}^n,b \in \mathbb{R}\}$，已知点$x_0 \in \mathbb{R}^{n},x_0 \notin S$，则点$x_0$到超平面$S$的距离为：  

$$
d(x_0,S)=\frac{|w^{T}x_0+b|}{||w||}
$$  

**证明:**  

<center>
    <img src="https://s2.loli.net/2023/10/11/mKVMIlyDUipd45k.jpg" width=60% height=60%>
    <div align="center">Image6: 点到超平面的距离</div>
</center>  

&emsp;&emsp;设$x_1$为$x_0$在超平面$S$上的投影，$x_2$为$S$上另外任意一点，向量$\overrightarrow{x_0x_1}$与向量$\overrightarrow{x_0x_2}$的夹角为$\theta$，则有：  

$$
d(x_0,S)=||\overrightarrow{x_0x_1}||=||\overrightarrow{x_0x_2}||\cos{\theta}
$$  

&emsp;&emsp;由向量夹角公式可知：  

$$
\cos{\theta}=\frac{\overrightarrow{x_0x_1} \cdot \overrightarrow{x_0x_2}}{||\overrightarrow{x_0x_1}|| \cdot ||\overrightarrow{x_0x_2}||}
$$  

&emsp;&emsp;将其代入到$d(x_0,S)$表达式中可得：  

$$
d(x_0,S)=\frac{\overrightarrow{x_0x_1} \cdot \overrightarrow{x_0x_2}}{||\overrightarrow{x_0x_1}||}
$$  

&emsp;&emsp;由于$x_1,x_2$是超平面$S$上的点，故有：  

$$
w^{T}x_1+b=0
$$  

$$
w^{T}x_2+b=0
$$  

$$
\Rightarrow w^{T}(x_1-x_2)=0 \Rightarrow w \cdot \overrightarrow{x_2x_1}=0
$$  

&emsp;&emsp;由于$\overrightarrow{x_0x_1}$与$\overrightarrow{x_2x_1}$垂直，因而有：  

$$
\overrightarrow{x_0x_1} \cdot \overrightarrow{x_2x_1}=0
$$

&emsp;&emsp;由于法向量$w$与向量$\overrightarrow{x_0x_1}$均垂直于超平面$S$，故可设$\overrightarrow{x_0x_1}=kw,k \in \mathbb{R}$，将其代入$d(x_0,S)$可得：  

$$
d(x_0,S)=\frac{|k| \cdot |w \cdot \overrightarrow{x_0x_2}|}{|k| \cdot ||w||}=\frac{|w \cdot (x_2-x_0)|}{||w||}
$$  

&emsp;&emsp;由于$w \cdot x_2 + b=0$可得$w \cdot x_2 = -b$，代入上式可得：  

$$
d(x_0,S)=\frac{|-b-w \cdot x_0|}{||w||}=\frac{|w^{T}x_0+b|}{||w||}
$$  

&emsp;&emsp;证毕.  

### 感知机算法收敛性证明  

&emsp;&emsp;现在证明，对于线性可分数据集，感知机学习算法原始形式收敛，即经过有限次迭代可以得到一个将训练数据集完全正确划分的分离超平面及感知机模型。  
&emsp;&emsp;为了便于叙述和推导，将偏置$b$并入权重向量$w$，记作$\hat{w}=(w^T,b)^T$，同样也将输入向量$x$进行扩充，加进常数1，记作$\hat{x}=(x^T,1)^T$。这样，$\hat{x} \in \mathbb{R}^{n+1},\hat{w} \in \mathbb{R}^{n+1}$。显然，$\hat{w}^T \hat{x}=w^{T}x+b$。  

#### $Novikoff$定理  

**结论**  
&emsp;&emsp;设训练数据集$T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$是线性可分的，其中$x_i \in \mathcal{X} = \mathbb{R}^{n},y_i \in \mathcal{Y} = \{-1,+1\},i=1,2,\dots,N$，则有：  
&emsp;&emsp;(1) 存在超平面$S=\{x \vert \hat{w}_{opt}^Tx=w_{opt}^Tx+b_{opt}=0, ||\hat{w}_{opt}||=1\}$可将训练数据集$T_{train}$完全正确分类；且$\exists \gamma >0$，对所有的$x_i,i=1,2,\dots,N$，有  

$$
y_i(\hat{w}_{opt}^{T}\hat{x}_i)=y_i(w_{opt}^{T}x_i+b_{opt}) \ge \gamma
$$  

&emsp;&emsp;(2) 令 $R=\max \{ ||\hat{x_i}|| \space \vert i=1,2,\dots,N\}$，则感知机学习的原始算法在训练数据集$T_{train}$上的误分类次数$k$满足不等式：  

$$
k \leq \left( \frac{R}{\gamma} \right)^2
$$  

&emsp;&emsp;$Novikoff$定理说明感知机学习算法能够经过有限次迭代得到一个可以将训练数据集完全正确分类的超平面，即算法收敛。下面，给出这个定理的证明。  

**证明:**  
&emsp;&emsp;首先来证明(1):  
&emsp;&emsp;由于训练数据集$T_{train}$是线性可分的，由数据集线性可分的定义可知，一定存在某个超平面能够将训练数据集完全正确分类，不妨设这个超平面为$S=\{x \vert \hat{w}_{opt}^Tx=w_{opt}^Tx+b_{opt}=0, ||\hat{w}_{opt}||=1\}$，则对于训练数据集中的每个实例，若$y_i = +1$,则$\hat{w}_{opt}^Tx_i > 0$；若$y_i=-1$，则$\hat{w}_{opt}^Tx_i < 0$，因此有以下不等式成立：

$$
y_i(\hat{w}_{opt}^{T}\hat{x}_i)=y_i(w_{opt}^{T}x_i+b_{opt}) > 0
$$  

&emsp;&emsp;故存在：  

$$
\gamma = \min_{i} \{y_i(w_{opt}^{T}x_i+b_{opt})\}
$$  

&emsp;&emsp;使得：  

$$
y_i(\hat{w}_{opt}^{T}\hat{x}_i)=y_i(w_{opt}^{T}x_i+b_{opt}) \ge \gamma
$$

&emsp;&emsp;再来证明(2):  

&emsp;&emsp;不妨设感知机算法的初始权重向量$\hat{w}_{0}=0$，如果发现一个实例被误分类，则更新一次权重。令$\hat{w}_{k-1}$是发现的第$i$个误分类实例之前的权重向量，即：  

$$
\hat{w}_{k-1}=(w_{k-1}^{T},b_{k-1})^T
$$  

&emsp;&emsp;判断训练集中的实例$(x_i,y_i)$被误分类的条件为：  

$$
y_i(\hat{w}_{opt}^{T}\hat{x}_i)=y_i(w_{opt}^{T}x_i+b_{opt}) \leq 0
$$  

&emsp;&emsp;若实例$(x_i,y_i)$是被$\hat{w}_{k-1}=(w_{k-1}^{T},b_{k-1})^T$误分类的数据，则$w$和$b$的更新过程为：  

$$
w_{k} \leftarrow w_{k-1} + \eta y_ix_i
$$  

$$
b_{k} \leftarrow b_{k-1} + \eta y_i
$$  

&emsp;&emsp;则：  

$$
\hat{w}_{k}= \hat{w}_{k-1}+\eta y_i \hat{x}_i
$$  

&emsp;&emsp;由此可得：  

$$
\begin{align*}
    \hat{w}_{k} \cdot \hat{w}_{opt} &= \hat{w}_{k-1} \cdot \hat{w}_{opt}+\eta (y_i \hat{w}_{opt}^T \hat{x}_i) \\
    & \ge \hat{w}_{k-1} \cdot \hat{w}_{opt} + \eta \gamma  \\
    & \ge \hat{w}_{k-2} \cdot \hat{w}_{opt} + 2\eta \gamma  \\  
    & \space \vdots  \\
    & \ge k \eta \gamma \\
\end{align*}
$$  

&emsp;&emsp;同时：  

$$
\begin{align*}
    ||\hat{w}_{k}||^{2} &= ||\hat{w}_{k-1}+\eta y_i \hat{x}_i||^2 \\
    &= ||\hat{w}_{k-1}||^2 + 2 \eta (y_i \hat{w}_{k-1}^T \hat{x}_i)+\eta^2 ||\hat{x}_i||^2 \\
    & \leq ||\hat{w}_{k-1}||^{2} + \eta^2||\hat{x_i}||^2 \\
    & \leq ||\hat{w}_{k-1}||^{2} + \eta^2R^2 \\
    & \leq ||\hat{w}_{k-2}||^{2} + 2\eta^2R^2 \\  
    & \space \vdots \\
    & \leq k \eta^2R^2
\end{align*}
$$  

&emsp;&emsp;由内积不等式可知：  

$$
k \eta \gamma \leq \hat{w}_k \cdot \hat{w}_{opt} \leq ||\hat{w}_k||\space ||\hat{w}_{opt}|| \leq \sqrt{k} \eta R
$$  

$$
\Rightarrow k \leq \left( \frac{R}{\gamma} \right)^2
$$  

&emsp;&emsp;证毕.  

## 参考  

**[1] Book: 李航.(2019).统计学习方法**  
**[2] Video: bilibili,简博士,感知机系列**  
**[3] Blog: 知乎,夜魔刀,点到超平面距离的公式推导**




