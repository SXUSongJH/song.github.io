---
title: 线性判别分析
tags:
    - Machine Learning
---

## toc

&emsp;&emsp;线性判别分析(Linear Discriminant Analysis，简称LDA)是一种在机器学习和统计学中常用于分类和降维的方法。它的主要目标是在特征空间中找到一个合适的投影方向，将高维数据点投影到低维空间中，使得这些数据点易于分类。LDA在特征选择、降维和模式识别等领域都有广泛的应用。
&emsp;&emsp;线性判别分析最早由著名统计学家$RA.Fisher$于1936年提出，他的工作被认为是$LDA$的奠基。线性判别分析经过多个阶段的发展，从最初的二分类问题到多分类问题，以及对不同数据类型的适应，一直在模式识别与机器学习领域中发挥着重要作用。同时，他也启发了其他降维和分类方法的发展。  

## 基本思想  

&emsp;&emsp;线性判别分析的基本思想为：在$n$维特征空间中，找到一个最佳的投影方向，使得在将训练集中的数据点投影到该方向上后，类别间的散度较大，类别内的散点较小，这样我们可以在该投影方向上找到一个分界点，能够对训练数据集完全正确分类。对于新的实例，将其投影到最佳投影方向上，利用分界点对其进行分类。为了寻找到最佳的投影方向，需要设置与类间散度和类内散度有关的损失函数，使得在最小化损失函数的过程中，类间散度增大而类内散度减小，这样最终得到的投影方向便是最佳投影方向。线性判别分析的基本思想可用下图1来描述：  

<center>
    <img src="https://s2.loli.net/2023/10/09/g2xVFLyehT3D8NS.jpg" width=60% height=60%>
    <div align="center">Image1: 线性判别分析的基本思想</div>
</center>  

## 模型  

&emsp;&emsp;线性判别分析可以应用于二分类问题或者多分类问题，这里我们主要讨论呢二分类问题下的线性判别模型。  

**输入**  

- **输入空间:** $\mathcal{X} = \mathbb{R}^{n}$   
- **输入实例:** $x = \begin{bmatrix}
    x^{(1)},x^{(2)},\dots,x^{(n)} \\
\end{bmatrix} \in \mathcal{X}$  

**输出**  

- **输出空间:** $\mathcal{Y} = \{-1,+1\}$  
- **输出实例:** $y \in \mathcal{Y}$  

&emsp;&emsp;其中输出空间$\mathcal{Y}$只包含+1和-1的一个集合，+1与-1分别代表二分类问题中的正类$C_1$与负类$C_2$。输出实例$y$代表输出实例$x$的类别。  

**数据集**  
&emsp;&emsp;感知机模型的训练数据集$T_{train}$为：  

$$
T_{train}=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}
$$  

&emsp;&emsp;其中，$x_i \in \mathcal{X}=\mathbb{R}^{n},y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\dots,N$，$N$表示训练数据的样本容量。  

**模型**  
&emsp;&emsp;设最佳投影方向为$\hat{w}$，且$||\hat{w}||_{2}=1$，正类与负类的样本均值点分别为：  

$$
\bar{x}_{C_1}=\frac{1}{N_1}\sum_{x_i \in C_1}x_i,\space \bar{x}_{C_2}=\frac{1}{N_2}\sum_{x_i \in C_2}x_i
$$  

&emsp;&emsp;其中，$N_1$表示训练集中正类样本的样本容量，$N_2$表示训练集中负类样本的样本容量。将$\bar{x}_{C_1}$与$\bar{x}_{C_2}$投影到最佳投影方向$\hat{w}$后的投影距离分别为$\hat{w}^{T}\bar{x}_{C_1}$和$\hat{w}^{T}\bar{x}_{C_2}$，设置分界点$threshold$为：  

$$
threshold = \frac{\hat{w}^{T}\bar{x}_{C_1}+\hat{w}^{T}\bar{x}_{C_2}}{2}
$$

&emsp;&emsp;对于新的实例点$x$，同样将其投影到最佳投影方向$\hat{w}$，则投影距离为$\hat{w}^{T}x$，其类别的判断准则为：  

$$
\hat{y} = \left \{
\begin{array}{rcl}
+1, & {\hat{w}^{T}x > threshold}\\
-1, & {\hat{w}^{T}x \leq threshold}\\
\end{array} \right.
$$  

**假设空间**  
&emsp;&emsp;模型的假设空间$\mathcal{H}$实际上是特征空间中所有的投影方向：  

$$
\mathcal{H} = \{w \vert w \in \mathbb{R}^{n}\}
$$  

&emsp;&emsp;此时模型的参数空间$\Theta = \mathcal{H}$.  

## 策略  

&emsp;&emsp;前文提到，线性判别分析的最佳投影方向要满足使得训练集样本在投影后有类间散度大，而类内散度小的特点，我们可以据此来设定损失函数。  
&emsp;&emsp;设特征空间中任意投影方向为$w$，且$||w||_{2}=1$，则特征空间中的数据点$x$在$w$上的投影距离为$w^{T}x$.将训练集中的数据点投影到该方向上，令$z_i=w^{T}x_i$，则训练数据集中正类与负类在投影到$w$后的平均投影距离分别为：  

$$
\bar{z}_1 = \frac{1}{N_1}\sum_{i=1}^{N_1}z_i=\frac{1}{N_1}\sum_{x_i \in C_1}w^{T}x_i
$$

$$
\bar{z}_2 = \frac{1}{N_2}\sum_{i=1}^{N_2}z_i=\frac{1}{N_2}\sum_{x_i \in C_2}w^{T}x_i
$$  

&emsp;&emsp;$\bar{z}_1$与$\bar{z}_2$的差的绝对值表示投影后正类数据与负类数据的中心点之间的距离，我们可以据此来表示类间散度，这个距离越大，说明投影后两个数据整体分离地越远。为了求导的方便，我们用平方代替绝对值，这样我们可以定义类间散度：  

$$
S_{be}=(\bar{z}_1-\bar{z}_2)^2
$$  

&emsp;&emsp;在考虑类间散度的同时，我们也希望投影后，同一个类别的数据尽量聚拢，即类内散度较小，我们可以用投影距离的组内方差来描述类内散度。投影后正类和负类数据点的投影距离的组内方差分别为：  

$$
S_1 = \frac{1}{N_1}\sum_{i=1}^{N_1}(z_i-\bar{z}_1)(z_i-\bar{z}_1)^T=\frac{1}{N_1}\sum_{x_i \in C_1}(w^T x_i-\frac{1}{N_1}\sum_{x_i \in C_1}w^{T}x_i)(w^T x_i-\frac{1}{N_1}\sum_{x_i \in C_1}w^{T}x_i)^T
$$

$$
S_2 = \frac{1}{N_2}\sum_{i=1}^{N_2}(z_i-\bar{z}_2)(z_i-\bar{z}_2)^T=\frac{1}{N_2}\sum_{x_i \in C_2}(w^T x_i-\frac{1}{N_2}\sum_{x_i \in C_2}w^{T}x_i)(w^T x_i-\frac{1}{N_2}\sum_{x_i \in C_2}w^{T}x_i)^T
$$  

&emsp;&emsp;我们希望正类与负类样本在投影后的组内方差均较小，因此我们可以将类内散度定义为：  

$$
S_{in}=S_1+S_2
$$  

&emsp;&emsp;根据判别分析的基本思想，我们希望投影后数据点有类间散度大，类内散度小的特点，因此我们可以将损失函数定义为：  

$$
L(w)=-\frac{(\bar{z}_1-\bar{z}_2)^2}{S_1+S_2}
$$  

&emsp;&emsp;当我们最小化损失函数$L(w)$时，可以在增大类间散度的同时，减小类内散度。损失函数的最小值点$\hat{w}$便是我们要寻找的最佳投影方向。  
&emsp;&emsp;我们对类间散度与类内散度做一下化简，以简化损失函数，便于优化。  

$$
\begin{align*}
    \bar{z}_1-\bar{z}_2 &=  \frac{1}{N_1}\sum_{x_i \in C_1}w^{T}x_i-\frac{1}{N_2}\sum_{x_i \in C_2}w^{T}x_i  \\
    &= w^T \left( \frac{1}{N_1}\sum_{i=1}^{N_1}x_i-\frac{1}{N_2}\sum_{i=1}^{N_2}x_i \right) \\
    &= w^T(\bar{x}_{C_1}-\bar{x}_{C_2})
\end{align*}
$$  

$$
\begin{align*}
    S_1 &= \frac{1}{N_1}\sum_{x_i \in C_1}(w^T x_i-\frac{1}{N_1}\sum_{x_i \in C_1}w^{T}x_i)(w^T x_i-\frac{1}{N_1}\sum_{x_i \in C_1}w^{T}x_i)^T \\
    &= \frac{1}{N_1}\sum_{i=1}^{N_1}w^{T}(x_i-\bar{x}_{C_1})(x_i-\bar{x}_{C_1})^{T}w  \\
    &= w^{T} \left( \frac{1}{N_1}\sum_{i=1}^{N_1}(x_i-\bar{x}_{C_1})(x_i-\bar{x}_{C_1})^{T} \right)w
\end{align*}
$$  

&emsp;&emsp;记$S_{C_1}=\frac{1}{N_1}\sum_{i=1}^{N_1}(x_i-\bar{x}_{C_1})(x_i-\bar{x}_{C_1})^{T}$，表示投影前正类的组内方差，则投影后正类的组内方差为：  

$$
S_1=w^{T}S_{C_1}w
$$  

&emsp;&emsp;同理可得：  

$$
S_2=w^{T}S_{C_2}w
$$  

$$
S_{C_2}=\frac{1}{N_1}\sum_{i=1}^{N_1}(x_i-\bar{x}_{C_1})(x_i-\bar{x}_{C_1})^{T}
$$  

&emsp;&emsp;则类内散度可以化简为：  

$$
S_1+S_2=w^{T}S_{C_1}w+w^{T}S_{C_2}w=w^{T}(S_{C_1}+S_{C_2})w
$$  

**损失函数**
&emsp;&emsp;化简后，最终得到的损失函数为：  

$$
\begin{align*}
    L(w)&=-\frac{w^{T}(\bar{x}_{C_1}-\bar{x}_{C_2})(\bar{x}_{C_1}-\bar{x}_{C_2})^{T}w}{w^{T}(S_{C_1}+S_{C_2})w} \\
    &= -\frac{w^{T}Aw}{w^{T}Bw}  \\
\end{align*}
$$  

&emsp;&emsp;其中，$A=(\bar{x}_{C_1}-\bar{x}_{C_2})(\bar{x}_{C_1}-\bar{x}_{C_2})^{T},B=S_{C_1}+S_{C_2}$.  

## 算法  

&emsp;&emsp;我们需要解决的优化问题为：  

$$
\min_{w} L(w)=-\frac{w^{T}Aw}{w^{T}Bw}
$$  

&emsp;&emsp;对$w$求一阶偏导：并令其为零：  

$$
\begin{align*}
    \frac{\partial L(w)}{\partial w} &= -\frac{\partial (w^{T}Aw)(w^{T}Bw)^{-1}}{\partial w} \\
    &= (2Aw)(w^{T}Bw)^{-1}-(w^{T}Aw)(w^{T}Bw)^{-2}(2Bw)=0  \\
\end{align*}
$$  

$$
\Rightarrow Aw(w^{T}Bw)-(w^{T}Aw)(Bw)=0
$$  

$$
\Rightarrow (w^{T}Aw)Bw=Aw(w^{T}Bw)
$$  

$$
\Rightarrow w= \frac{w^{T}Bw}{w^{T}Aw}B^{-1}Aw
$$

&emsp;&emsp;由于我们只需要求得最佳投影方向，而不需要关系其长度，因此有：  

$$
w \varpropto B^{-1}Aw
$$  

&emsp;&emsp;表示$w$的方向与$B^{-1}Aw$一致。由于$Aw=(\bar{x}_{C_1}-\bar{x}_{C_2})(\bar{x}_{C_1}-\bar{x}_{C_2})^{T}w$，而$(\bar{x}_{C_1}-\bar{x}_{C_2})^{T}w \in \mathbb{R}$为标量，故有：  

$$
w \varpropto B^{-1}(\bar{x}_{C_1}-\bar{x}_{C_2})
$$  

&emsp;&emsp;$\because B=S_{C_1}+S_{C_2}$，因此我们求得的最佳投影方向为：  

$$
w \varpropto (S_{C_1}+S_{C_2})^{-1}(\bar{x}_{C_1}-\bar{x}_{C_2})
$$  
 
$$
\hat{w} = \frac{(S_{C_1}+S_{C_2})^{-1}(\bar{x}_{C_1}-\bar{x}_{C_2})}{||(S_{C_1}+S_{C_2})^{-1}(\bar{x}_{C_1}-\bar{x}_{C_2})||_2}
$$  

## 线性判别分析实例及Python实现  

&emsp;&emsp;首先生成训练数据。

```python
# Data Generation
np.random.seed(520)
def data_generation(n,basepoint,ylabel):
    x1 = np.random.random((n,))+np.array([basepoint[0]]*n)
    x2 = np.random.random((n,))+np.array([basepoint[1]]*n)
    y = np.array([ylabel]*n)
    data = pd.DataFrame({"x1":x1,"x2":x2,"y":y})
    return data

# positive data
positivedata = data_generation(n=30,basepoint=(1,2),ylabel=1)
# negative data
negativedata = data_generation(n=20,basepoint=(2,1),ylabel=0)
# train data
train_data = pd.concat([positivedata,negativedata])
train_data = shuffle(train_data)
train_data.index = range(train_data.shape[0])
train_data.head(5)

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
<center>
    <img src="https://s2.loli.net/2024/02/03/YtmOUQ8PNdGD1Mi.png" width=60% height=60%>
    <div align="center">Image2: 训练数据</div>
</center>  

&emsp;&emsp;利用前文所提出的算法，计算线性判别模型的参数 $w$ .

```python
def LDA_param_solving(train_data):
    positive_data = train_data[train_data["y"]==1]
    negative_data = train_data[train_data["y"]==0]

    pos_X = np.array(positive_data.iloc[:,:-1])
    neg_X = np.array(negative_data.iloc[:,:-1])

    pos_X_mean = np.mean(pos_X,axis=0)
    pos_X_var = np.cov(pos_X,rowvar=False)
    neg_X_mean = np.mean(neg_X,axis=0)
    neg_X_var = np.cov(neg_X,rowvar=False)

    w = np.dot(np.linalg.inv(pos_X_var+neg_X_var),pos_X_mean-neg_X_mean)
    w = w/np.linalg.norm(w)

    return w.round(8)

def LDA_threshold(train_data):
    w = LDA_param_solving(train_data)

    positive_data = train_data[train_data["y"]==1]
    negative_data = train_data[train_data["y"]==0]
    pos_X = np.array(positive_data.iloc[:,:-1])
    neg_X = np.array(negative_data.iloc[:,:-1])
    pos_X_mean = np.mean(pos_X,axis=0)
    neg_X_mean = np.mean(neg_X,axis=0)

    t = (np.dot(w,pos_X_mean)+np.dot(w,neg_X_mean))/2

    return round(t,8)

w_hat = LDA_param_solving(train_data=train_data)
threshold = LDA_threshold(train_data)
```

&emsp;&emsp;得到的模型参数及threshold为:  

$$
w = \begin{bmatrix}
    -0.8235 \\
    0.5673 \\
\end{bmatrix}, \quad threshlod=-0.4689
$$

&emsp;&emsp;画出模型的决策边界：

```python
x1_line = np.linspace(-3,3,1000)
x2_line = (w_hat[1]/w_hat[0])*x1_line
theta = math.atan(w_hat[1]/w_hat[0])
threshold_x1 = -math.cos(-theta)*threshold
threshold_x2 = math.sin(-theta)*threshold
decision_boundary_x2 = (-w_hat[0]/w_hat[1])*x1_line+(threshold_x2+(w_hat[0]/w_hat[1])*threshold_x1)
plt.figure(figsize=(7, 7))
plt.scatter(x=threshold_x1,y=threshold_x2,marker="p",c="black",label="threshold",s=100,alpha=1)
plt.scatter(x=positivedata["x1"],y=positivedata["x2"],marker="o",c="green",label="positive data")
plt.scatter(x=negativedata["x1"],y=negativedata["x2"],marker="x",c="red",label="negative data")
plt.plot(x1_line,x2_line,c="blue",label="Best Project Direction")
plt.plot(x1_line,decision_boundary_x2,c="purple",label="decision boundary")
ax = plt.subplot()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.xlim((-3,3))
plt.ylim((-3,3))
plt.xticks(np.arange(-3,3,1))
plt.yticks(np.arange(-3,3,1))
plt.legend(loc="lower left")
```

<center>
    <img src="https://s2.loli.net/2024/02/03/v5zwfZsA6dYotO1.png" width=60% height=60%>
    <div align="center">Image3: 决策边界</div>
</center>  

## 参考

**[1] Video: bilibili,shuhuai008,线性判别分析**  
**[2] Blog: CSDN,SongGu1996,线性判别分析(Linear Discriminant Analysis，LDA)**


