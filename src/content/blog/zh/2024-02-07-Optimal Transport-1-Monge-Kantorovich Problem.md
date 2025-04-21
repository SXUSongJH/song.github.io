---
title: Optimal Transport-1.Monge-Kantorovich Problem
tags:
  - Optimal Transport
  - Math
---


## toc

&emsp;&emsp;最优传输理论(Optimal Transport Theroy)是应用数学的一个分支，主要研究的是概率分布之间的最优转移方式以及相关的距离度量。它的核心思想是通过最小化两个概率分布之间的转移成本，来定义这两个分布之间的距离。  
&emsp;&emsp;最优传输问题的历史可以追溯到18世纪。著名法国数学家 Monge 在1781年提出了最优传输问题的早期形式，即 Monge-Problem。他关注的是如何以最小的成本将一个土堆移动到另一个位置，这被看作是最优传输理论的起源。时至今日，很多介绍最优传输理论的科普文章依然喜欢使用“土堆移动、推土机”等例子来论述最优传输理论。Monge的工作奠定了这一理论的基础。另外一个对最优传输理论的发展具有关键作用的是前苏联数学家 Kantorovich。Kantorovich在20世纪40年代从实际经济资源分配问题中重新导出了最优传输问题，他基于测度论对最优传输问题进行了严格的定义，并推导出了一系列重要的理论成果。Kantorovich的工作对最优传输理论的数学形式和理论基础的建立起到了关键作用。由于在线性规划和资源分配方面的重要贡献，Kantorovich于1975年获得诺贝尔经济学奖。此外还有多位数学家对最优传输理论的发展做出了重要贡献，Wassertein、Villani、Berg等数学家从理论、应用、数值计算等方面丰富了最优传输理论。  
&emsp;&emsp;最优传输理论的应用涵盖多个领域：  

- **1.图像处理和计算机视觉:** 在图像处理中，最优传输理论被用来衡量图像之间的相似性，从而进行图像匹配、图像检索等任务。在计算机视觉领域，它被用于解决图像生成和变换的问题。  
- **2.机器学习:** 在机器学习中，最优传输理论被应用于生成模型、领域自适应等问题，以提高模型的泛化性能。  
- **3.经济学和金融学:** 在经济学中，最优传输理论被用于研究资源的分配和经济结构的演变。在金融学中，它可以用来量化不同资产之间的差异和联系。  
- **4.统计学和信息论:** 最优传输理论在统计学和信息论中也有广泛的应用，尤其是在测度空间中定义概率分布之间的距离。  

&emsp;&emsp;总的来说，最优传输理论在多个领域都展现了广泛的应用，其在处理概率分布之间的关系和相似性的能力为研究者提供了有力的工具。  

## Monge Problem  

&emsp;&emsp;Gaspard Monge(1746-1818)是法国著名的数学家和物理学家，他在数学、物理学和工程学领域做出了突出的贡献，被认为是现代微分几何的奠基人之一，对现代科学和数学的发展产生了深远的影响。  
&emsp;&emsp;在18世纪，第一次工业革命正在欧洲如火如荼地进行着，经济与军事活动催生了大量的数学、物理理论的诞生。最优传输理论的萌芽便来源于军事活动中所遇到的实际问题。彼时，英国与法国正在争夺欧洲霸权，为此进行了一系列的战争。在过去的战争中，防御工事的修建至关重要，修建防御工事需要将大量的土堆从某地转移到阵地，并堆砌成确定的形状。修建工事需要消耗大量的人力物力，这便催生了一个实际问题，**如何将某一形状的土堆移动到另一个地方并堆成另一种形状，使得这整个过程中所消耗的资源最少。**  
&emsp;&emsp;Monge此时正在法国军队中担任工程研究人员，负责为军队提供有关土木工程和防御工事的重要建议。Monge敏锐地注意到了这个问题，并对其展开了研究。1781年，Monge发表了著作——**Mémoire sur la théorie de déblais et de remblais(关于挖掘和填充的备忘录).** Monge在著作中对该问题进行了论述，并提出了一种解决方案。  
&emsp;&emsp;Monge首先对所要研究的问题做出了一系列假设：  

- **所要运输的土堆都是由质量相等的不可再分的分子所构成的，且土堆中分子的分布是均匀的。**
- **每个分子的运输价格都是相等的，与该分子的重量和它所被传输的距离成正比。**
- **总的运输价格是所运输的每个分子的质量乘以分子所传输的距离的总和。**  

基于以上的假设，Monge提出了"前向运输法"，并给出了从一维到三维的实例。  

### 点到点的最优传输  

&emsp;&emsp;我们首先来考虑最简单的两点运输情形，如下图1所示：  

<center>
    <img src="https://s2.loli.net/2024/02/09/HvhSCsApP1a5Q38.jpg" width=40% height=60%>
    <div align="center">Image1: 两点运输</div>
</center>  

A区域中有两个分子需要运输到B区域的指定位置，总共有两种运输方案，图1中分别用绿色与橙色的箭头表示。由三角形两边之和大于第三边可知，橙色方案的运输距离要大于绿色方案，又因为分子的质量是相同的，所有我们可以很容易得到，绿色方案的运输效率要高于橙色方案。  
&emsp;&emsp;再来考虑多点的情形，如下图2所示

<center>
    <img src="https://s2.loli.net/2024/02/09/vkGiCydSJEOT7BF.jpg" width=60% height=60%>
    <div align="center">Image2: 多点运输</div>
</center>  

我们需要将点1、2、3运输到点4、5、6处，图2中给出了A、B、C三种运输方案。首先来比较方案A、B，方案A，B的不同在于点1、3到点5、6的运输路线，基于上文两点运输的思考，我们可以得知B方案的运输效率要高于方案A，同理，C方案的运输效率要高于B方案。  
&emsp;&emsp;基于以上的思考，Monge认为点到点的运输要遵循"前向法"，即运输路线不存在交叉。Monge同时将这种思想拓展到更高维的情形。  

### 平面到平面的最优运输  

&emsp;&emsp;现在考虑将某一平面区域运输到另一个平面区域，基于“前向法”，Monge给出了如下图3所示的运输方案：  

<center>
    <img src="https://s2.loli.net/2024/02/09/2JhmZywrsRd9zpF.jpg" width=60% height=60%>
    <div align="center">Image3: 平面运输</div>
</center>  

现在需要将平面区域Ⅰ中的分子运输到平面区域Ⅱ，**Monge认为区域Ⅰ中的A、C两点要运输到区域Ⅱ的B、D两点，直线AB、CD相较于O点，由O点出发可以确定两条边界射线$l_3,l_4$。在锥型区域中，可以用很多条射线将运输平面进行分割。例如夹角"非常小"的射线$l_1,l_2$与区域Ⅰ相交于$M_1、M_3、H_1、H_3$，与区域Ⅱ相交于$M_4、M_6、H_4、H_6$，分割出区域$M_1M_3H_1H_3$和$M_4M_6H_4H_6$，根据"前向法"，区域Ⅰ中$M_1M_3H_1H_3$的分子要运输到区域Ⅱ中$M_4M_6H_4H_6$中，同时运输时也要遵循“前向法”，即橙色区域要运输到橙色区域，绿色区域要运送到绿色区域。由无数个分割出的小区域依据“前向法”进行运输，这种运输方式的所消耗的资源最少。**  
&emsp;&emsp;Monge借助微分几何的知识证明了这些从O点出发的运输射线都垂直于某一曲线R，并通过解析几何的知识求出了曲线R的解析式，以及最优运输的映射方程$y = T(x)$。基于“前向法”，Monge同时也对三维运输的映射方程进行了求解，具体的求解过程这里不作详细的介绍，有兴趣的读者可自行查阅相关资料。  

<center>
    <img src="https://s2.loli.net/2024/02/11/FG4H59woPsLAJMy.png" width=60% height=60%>
    <div align="center">Image4: Monge著作中的作图</div>
</center>  

### 待解决的问题  
&emsp;&emsp;虽然Monge借助“前向法”给出了最优传输问题的一种解决方法，但他同时也在著作中承认，实际的运输问题是非常复杂的，他的解法只是一种非常理想化的方法，并且这种方法也存在着缺陷，主要有以下几个方面:  

- **1.被运输的每个分子的质量可能是不同的。**
- **2.每个区域的密度有可能不同，即面积相同的区域所含有的分子数量可能不相等。**
- **3.当目标区域是非凸区域时，“前向法”在中间区域的映射方程是无解的。**

&emsp;&emsp;在Monge-Problem被提出后，后世的学者也不断地对这一问题进行研究，但并没有取得突破性的进展。  

## Kantorovich Relaxation  

&emsp;&emsp;19世纪-20世纪初，集合论、测度论、概率论等数学分支得到了充分的发展壮大，Monge-Problem的解决也迎来了转机。前苏联数学家kantorovich在解决Monge-Problem中起到了不可忽视的作用。然而，有趣的是，Kantorovich最初在解决这个问题时，并没有意识到自己所面对的问题与Monge-Problem之间的关联，直到将近10年后，Kantorovich才在一篇新的论文中阐述了自己的方法可以解决Monge-Problem。  
&emsp;&emsp;1938年，圣彼得堡胶合板托拉斯的研究人员找到圣彼得堡大学数学系，想要圣彼得堡大学的数学家们帮助解决在生产中所遇到的一个实际问题。**托拉斯有八台不同类型的机器，每台机器能够生产五种不同型号的胶合板。每种类型的机器生产这五种胶合板的效率不同，各种不同的胶合板的产量是由现有的需求量决定的。为了在最短的时间内生产这些胶合板，应该如何给每种类型的机器分配生产任务？**  
&emsp;&emsp;与高深的数学理论相比，这看起来是一个非常简单的数学问题，仅仅只需要求解方程组而已，然而要求出这个问题的最优解却并不是一个简单的工作，因为当时有关线性规划的理论还没有被提出。彼时，26岁的Kantorovich正在圣彼得堡大学担任教职，这个问题最终交由他来研究。1939年，Kantorovich发表了论文《数学方法中的问题理论》，以经济资源分配与运输问题为背景，正式提出了"最优传输问题"，这一工作被认为是线性规划理论的先驱性工作。1947年，kantorovich发表论文《关于数学规划问题的一般理论》，正式提出了线性规划理论及其一般解法。  

### 最优传输问题  

&emsp;&emsp;我们来考虑一个实际问题，有 n 个生产木材的工厂，生产的木材需要供应给 m 个城市，每个工厂每个月的产能是固定的，记为 $\boldsymbol{a}, \boldsymbol{a} \in \mathbb{R}^{n}$；每个城市每个月木材的需求量也是固定的，记为 $\boldsymbol{b}, \boldsymbol{b} \in \mathbb{R}^{m}$。记 $\boldsymbol{P} = [p_{ij}]_{n \times m} \in \mathbb{R}^{n \times m}_{+}$ 表示运输方案，其中 $p_{ij}$ 表示第 i 个工厂运输到第 j 个城市的木材量。记 $\boldsymbol{C} = [c_{ij}]_{n \times m} \in \mathbb{R}^{n \times m}_{+}$ 表示运输成本，其中 $c_{ij}$ 表示将单位木材从第i个工厂运输到第j个城市的成本。记：  

$$
\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b}) = \{ \boldsymbol{P} \in \mathbb{R}^{n \times m}_{+}: \boldsymbol{P} \mathbf{1}_{m}=\boldsymbol{a} \quad and \quad \boldsymbol{P^{T}} \mathbf{1}_{n}=\boldsymbol{b} \}
$$  

&emsp;&emsp;则最优传输问题可以被定义为：  

$$
L_{\boldsymbol{C}}(\boldsymbol{a}, \boldsymbol{b}) \quad \overset{\text{def.}}{=} \quad \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})} \left< \boldsymbol{C},\boldsymbol{P} \right>_{F}=\sum_{i,j}p_{ij}c_{ij}
$$  

$L_{\boldsymbol{C}}(\boldsymbol{a}, \boldsymbol{b})$即为最优运输方案。由于目标函数是线性函数，且 $\boldsymbol{U}(\boldsymbol{a},\boldsymbol{b})$ 为 $n+m$ 个等式定义的凸多胞体，故最优传输问题是一个典型的线性规划问题。  

### 概率视角  

&emsp;&emsp;Kantorovich使用测度论定义了被运输的对象。在Monge的理论中，运输是确定性的，即每个分子都不可再分，且会被整体被运输到另一个位置，**Kantorovich放松了这种确定性条件，他认为每个分子是可以进行切分的，一个分子所包含的质量可以被运输到不同的位置，这种运输是具有概率性的**。  

#### 离散概率分布运输问题  

&emsp;&emsp;设 $X, Y$ 是两个服从多项分布的随机变量，取值于 $\{ 1,2,\dotsb, d \}$。$X, Y$的概率分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 取自概率单纯形 $\sum_{d}:=\{ x \in \mathbb{R}^{d}_{+}: \boldsymbol{x^{T}}\mathbf{1}_{d}=1 \}$。运输矩阵 $\boldsymbol{P} \in \mathbb{R}^{d \times d}_{+}$。记:  

$$
\boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \{ \boldsymbol{P} \in \mathbb{R}^{d \times d}_{+} : \boldsymbol{P}\mathbf{1}_{d}=\boldsymbol{\alpha} \quad and \quad \boldsymbol{P^{T}}\mathbf{1}_d=\boldsymbol{\beta} \}
$$  

从概率视角看，集合 $\boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})$ 包含了随机变量$X, Y$所有可能的联合分布$\pi(X,Y)$，即矩阵$\boldsymbol{P}=[p_{ij}]_{d \times d} = [\pi(x=i,y=j)]$。设成本矩阵为$\boldsymbol{C} \in \mathbb{R}^{d \times d}_{+}$。则多项分布 $\boldsymbol{\alpha},\boldsymbol{\beta}$ 之间的最优传输问题可以被定义为：  

$$
L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \min_{\boldsymbol{P} \in \boldsymbol{U}(\boldsymbol{\alpha},\boldsymbol{\beta})} \left< \boldsymbol{P},\boldsymbol{C} \right> = \min_{(X,Y)} \{ \mathbb{E}_{(X,Y)}(c(X,Y)): X \sim \boldsymbol{\alpha}, Y \sim \boldsymbol{\beta} \}
$$  

$(X,Y)$表示取值于$\mathcal{X} \times \mathcal{Y}$的联合分布。  

#### 连续概率分布运输问题  

&emsp;&emsp;连续分布的运输问题与离散问题相似，不同点在于随机变量 $X,Y$ 服从的是连续分布 $\boldsymbol{\alpha}, \boldsymbol{\beta}$，最优运输同样可以定义为：  

$$
L_{\boldsymbol{C}}(\boldsymbol{\alpha},\boldsymbol{\beta}) := \min_{(X,Y)} \{ \mathbb{E}_{(X,Y)}(c(X,Y)): X \sim \boldsymbol{\alpha}, Y \sim \boldsymbol{\beta} \}
$$  

### 局部前向法  

&emsp;&emsp;通过求解离散分布和连续分布的最优传输问题，我们可以发现最优传输方案与Monge的"前向法"存在联系，下图5是最优传输的结果实例：  

<center>
    <img src="https://s2.loli.net/2024/02/12/BZ8uThM5CcbQzm4.png" width=60% height=60%>
    <div align="center">Image5: 概率分布最优传输结果实例</div>
</center>  

从右图离散分布的最优运输结果来看，最优运输是满足“局部前向法"的，即在满足边界分布的条件下，遵循前向运输法则。  

## References  

- **[1] Book: Peyré G, Cuturi M. Computational optimal transport[J]. Center for Research in Economics and Statistics Working Papers, 2017 (2017-86).**  
- **[2] Lecture: 李向东. 最优传输理论及其应用. BIMSA**  
- **[3] Paper: Cuturi M. Sinkhorn distances: Lightspeed computation of optimal transport[J]. Advances in neural information processing systems, 2013, 26.**







