# 搭建[AlexNet](https://so.csdn.net/so/search?q=AlexNet&spm=1001.2101.3001.7020)并训练花分类数据集C

## AlexNet模型流程

1. model.py ——定义AlexNet网络模型
2. train.py ——加载数据集并训练，训练集计算loss，测试集计算accuracy，保存训练好的网络参数
3. predict.py——得到训练好的网络参数后，用自己找的图像进行分类测试

[TOC]





## 数据集处理

### 数据集下载

http://download.tensorflow.org/example_images/flower_photos.tgz
包含 5 中类型的花，每种类型有600~900张图像不等CD

![image-20220907120152094](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120152094.png)





### 训练集与测试集划分

#### 数据集划分操作

由于此数据集不像 CIFAR10 那样下载时就划分好了训练集和测试集，因此需要自己划分。

`shift + 右键` 打开 PowerShell ，执行 “split_data.py” 分类脚本自动将数据集划分成 训练集train 和 验证集val。

![image-20220907120212489](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120212489.png)

#### 文件夹划分结构

完整的目录结构如下：

```
|-- flower_data
	|-- flower_photos
		|-- daisy
		|-- dandelion
		|-- roses
		|-- sunflowers
		|-- tulips
		|-- LICENSE.txt
	|-- train
		|-- daisy
		|-- dandelion
		|-- roses
		|-- sunflowers
		|-- tulips
	|-- val
		|-- daisy
		|-- dandelion
		|-- roses
		|-- sunflowers
		|-- tulips
	|-- flower_photos.tgz
|-- flower_link.txt
|-- README.md
|-- split_data.py

```



**数据集划分代码**

用到自己的数据集时，可以简单修改代码中的文件夹名称进行数据集的划分

```python
#导入库
import os
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file): #判断文件是否存在
        os.makedirs(file) ## 创建的目录（文件夹）
        
# 获取 flower_photos 文件夹下除 .txt 文件以外所有文件夹名（即5种花的类名）
file_path = 'flower_data/flower_photos'
#os.listdir() 用于返回指定文件夹的文件夹的文件或文件夹的名字的方法列表
flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla] #结果返回五类文件夹

# 创建 训练集train 文件夹，并由5种类名在其目录下创建5个子目录
mkfile('flower_data/train')
for cla in flower_class:
    mkfile('flower_data/train/'+cla)
    
# 创建 验证集val 文件夹，并由5种类名在其目录下创建5个子目录
mkfile('flower_data/val')
for cla in flower_class:
    mkfile('flower_data/val/'+cla)

# 划分比例，训练集 : 验证集 = 9 : 1
split_rate = 0.1

# 遍历5种花的全部图像并按比例分成训练集和验证集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别花的子目录
    images = os.listdir(cla_path)		    # iamges 列表存储了该目录下所有图像的名称
    num = len(images)  #图像的总数量
    eval_index = random.sample(images, k=int(num*split_rate)) # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        '''
        >>>季节= [ 'Spring' , 'Summer' , 'Fall' , 'Winter' ]
		>>> list ( enumerate ( seasons ) )
		[ ( 0 , 'Spring' ) , ( 1 , 'Summer' ) , ( 2 , 'Fall' ) , ( 3 , 'Winter' ) ]
        '''
    	# eval_index 中保存验证集val的图像名称
        if image in eval_index:					
            image_path = cla_path + image
            new_path = 'flower_data/val/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径
           
        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'flower_data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

print("processing done!")

```



## **AlexNet详解**

AlexNet 是2012年 ISLVRC ( ImageNet Large Scale Visual Recognition Challenge)竞赛的冠军网络，分类准确率由传统的70%+提升到80%+。它是由Hinton和他的学生Alex Krizhevsky设计的。 也是在那年之后，深度学习开始迅速发展。

![image-20220907120523327](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120523327.png)





使用Dropout的方式在网络正向传播过程中随机失活一部分神经元，以减少过拟合

![image-20220907120557692](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120557692.png)





### Conv1



![image-20220409211856668](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220409211856668.png)





注意：原作者实验时用了两块GPU并行计算，上下两组图的结构是一样的。

输入：input_size(图片大小) = [224, 224, 3]
卷积层：
			kernels = 48 * 2 = 96 组卷积核---------因为使用的是两块GPU，单个结构48个过滤器
			kernel_size（过滤器大小） = 11*11
			padding（填充） = [1, 2] （左上围加半圈0，右下围加2倍的半圈0）-------左边上边加1列（0），右边下边加1列（0）
			stride（步长） = 4
输出：output_size = [55, 55, 96]
**经 Conv1 卷积后的输出层尺寸为：**
![image-20220907120616875](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120616875.png)

- ```tiddlywiki
  - 输入图片大小 W×W（一般情况下Width=Height）
  - Filter大小F×F
  - 步长 S
  - padding的像素数 P
  ```

  

### Maxpool1



![image-20220907120629240](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120629240.png)



输入：input_size = [55, 55, 96]
池化层：（**只改变尺寸，不改变深度channel**）
kernel_size = 3
padding = 0
stride = 2
输出：output_size = [27, 27, 96]
**经 Maxpool1 后的输出层尺寸为：**

![image-20220907120655804](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120655804.png)
$$

$$

### Conv2

![image-20220907120752131](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120752131.png)

输入：input_size = [27, 27, 96]
卷积层：
kernels = 128 * 2 = 256 组卷积核
kernel_size = 5
padding = [2, 2]
stride = 1
输出：output_size = [27, 27, 256]
**经 Conv2 卷积后的输出层尺寸为：**
![image-20220907120805642](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120805642.png)

### Maxpool2

![image-20220409212025325](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220409212025325.png)



输入：input_size = [27, 27, 256]
池化层：（只改变尺寸，不改变深度channel）
kernel_size = 3
padding = 0
stride = 2
输出：output_size = [13, 13, 256]
**经 Maxpool2 后的输出层尺寸为：**
![image-20220907120925273](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120925273.png)

### Conv3

![image-20220907120936196](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120936196.png)

输入：input_size = [13, 13, 256]
卷积层：
kernels = 192* 2 = 384 组卷积核
kernel_size = 3
padding = [1, 1]
stride = 1
输出：output_size = [13, 13, 384]
**经 Conv3 卷积后的输出层尺寸为：**

![image-20220907120951633](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907120951633.png)

### Conv4

![image-20220907121030663](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907121030663.png)



输入：input_size = [13, 13, 384]
卷积层：
kernels = 192* 2 = 384 组卷积核
kernel_size = 3
padding = [1, 1]
stride = 1
输出：output_size = [13, 13, 384]
**经 Conv4 卷积后的输出层尺寸为：**

![image-20220907121109914](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907121109914.png)



### Conv5

![image-20220907121127350](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907121127350.png)

输入：input_size = [13, 13, 384]
卷积层：
kernels = 128* 2 = 256 组卷积核
kernel_size = 3
padding = [1, 1]
stride = 1
输出：output_size = [13, 13, 256]
**经 Conv5 卷积后的输出层尺寸为：**

![image-20220907121150530](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907121150530.png)



### Maxpool3

![image-20220409212202999](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220409212202999.png)



输入：input_size = [13, 13, 256]
池化层：（只改变尺寸，不改变深度channel）
kernel_size = 3
padding = 0
stride = 2
输出：output_size = [6, 6, 256]
**经 Maxpool3 后的输出层尺寸为：**

![image-20220907121210799](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907121210799.png)

### FC1、FC2、FC3

Maxpool3 → (6*6 *256) → FC1 → 2048 → FC2 → 2048 → FC3 → 1000
最终的1000可以根据数据集的类别数进行修改。



### 总结

<img src="C:/Users/1/AppData/Roaming/Typora/typora-user-images/image-20220409212225678.png" alt="image-20220409212225678" style="zoom:67%;" />







**分析可以发现，除 Conv1 外，AlexNet 的其余卷积层都是在改变特征矩阵的深度，而池化层则只改变（减小）其尺寸。**



## 1. model.py

Pytorch 中 Tensor 参数的顺序为 (**batch, channel, height, width)** ，下面代码中没有写batch

卷积的参数为Conv2d(**in_channels, out_channels, kernel_size, stride, padding,** ...)，一般关心这5个参数即可

```python
卷积池化层提取图像特征，全连接层进行图像分类，代码中写成两个模块，方便调用
```

为了加快训练，代码只使用了一半的网络参数，相当于只用了原论文中网络结构的下半部分（正好原论文中用的双GPU，我的电脑只有一块GPU）（后来我又用完整网络跑了遍，发现一半参数跟完整参数的训练结果acc相差无几）



### 1.0导入包

```python
#导入库
import torch.nn as nn #使用torch.nn包来构建神经网络.
import torch
```



### 1.1 卷积层，池化层结构创建并进行打包

-    `self.features = nn.Sequential()`  函数能够将一系列的层结构（卷积层、池化层、全连接层）打包成一个新的结构features

#### 1.1.1 卷积 Conv2d

我们常用的卷积（Conv2d）在pytorch中对应的函数是：

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```

一般使用时关注以下几个参数即可：

| 名词         | 定义                                                         |
| ------------ | ------------------------------------------------------------ |
| in_channels  | 输入特征矩阵的深度。如输入一张RGB彩色图像，那in_channels=3(通道) |
| out_channels | 输出特征矩阵的深度。也等于卷积核的个数，使用n个卷积核输出的特征矩阵深度就是n |
| kernel_size  | 卷积核的尺寸。可以是int类型，如3 代表卷积核的height=width=3，也可以是tuple类型如(3, 5)代表卷积核的height=3，width=5 |
| stride       | 卷积核的步长。默认为1，和kernel_size一样输入可以是int型，也可以是tuple类型 |
| padding      | 补零操作，默认为0。可以为int型如1即补一圈0，如果输入为tuple型如(2, 1) 代表在上下补2行，左右补1列。 |



经卷积后的输出层尺寸计算公式为：
![image-20220907121226548](https://raw.githubusercontent.com/GISer-longchaohuo/Images/master/img/image-20220907121226548.png)

- 输入图片大小 W×W（一般情况下Width=Height）
- Filter大小 F×F
- 步长 S
- padding的像素数 P



#### 1.1.2 池化 MaxPool2d

最大池化（MaxPool2d）在 pytorch 中对应的函数是：

```python
MaxPool2d(kernel_size, stride)
```

#### 1.1.3 激活函数

```python
nn.ReLU(inplace=True), #rulu激活函数，inplace=True直接修改覆盖原值，节省运算内存
```



******************************************************************************************************************************************************************************************************************************************

```python
      #卷积层，池化层创建并进行打包  
        self.features = nn.Sequential(  #nn.Sequential函数能够将一系列的层结构（卷积层、池化层、全连接层）打包成一个新的结构features
            #nn.Sequential函数使用的必要性:对于层数较多的神经网络来说，减少代码，降低工作量
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]，第一个卷积层（深度为3（通道），48个卷积盒，卷积核的尺寸11*11的大小,步长4，填充2）
            nn.ReLU(inplace=True), #relu激活函数，inplace=True直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2), # output[48, 27, 27]，第一个池化层（尺寸为3*3最大值，步距为2）---下采样层
            nn.Conv2d(48, 128, kernel_size=5, padding=2),# output[128, 27, 27]，第二个卷积层（深度为48（通道），128个卷积盒，卷积核的尺寸5*5的大小,步长5，填充2）
            nn.ReLU(inplace=True),#relu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2), # output[128, 13, 13]，第二个池化层（尺寸为3*3最大值，步距为2）---下采样层
            nn.Conv2d(128, 192, kernel_size=3, padding=1), # output[192, 13, 13]，第三个卷积层（深度为128（通道），192个卷积盒，卷积核的尺寸3*3的大小,步长5，填充1）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]，第四个卷积层（深度为192（通道），192个卷积盒，卷积核的尺寸3*3的大小,步长3，填充1）
            nn.ReLU(inplace=True),#relu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.Conv2d(192, 128, kernel_size=3, padding=1), # output[128, 13, 13]，第五个卷积层（深度为192（通道），128个卷积盒，卷积核的尺寸3*3的大小,步长3，填充1）
            nn.ReLU(inplace=True),#relu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2), # output[128, 6, 6]，第二个池化层（尺寸为3*3最大值，步距为2）---下采样层
        )
```



### 1.2 全连接层结构创建并进行打包

- `self.features = nn.Sequential()`  函数能够将一系列的层结构（卷积层、池化层、全连接层）打包成一个新的结构features

- Dropout 随机失活神经元,`nn.Dropout(p=0.5),# Dropout 随机失活神经元，默认比例为0.5`

- `nn.ReLU(inplace=True),`#relu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存

  #### 1.2.1全连接（ Linear）在 pytorch 中对应的函数是

  ```python
  Linear(in_features, out_features, bias=True)
  ```

  *********************************************************************************************************************************************************************************************************************************

```python
    #全连接层创建并进行打包  
    self.classifier = nn.Sequential(
        nn.Dropout(p=0.5),# Dropout 随机失活神经元，默认比例为0.5
        nn.Linear(128 * 6 * 6, 2048),#第一个全连接（全连接是一纬的向量，节点数为2048个）
        nn.ReLU(inplace=True),#relu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
        nn.Dropout(p=0.5),,# Dropout 随机失活神经元，默认比例为0.5
        nn.Linear(2048, 2048),#第二个全连接（全连接是一纬的向量，节点数为2048个）
        nn.ReLU(inplace=True),#relu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
        nn.Linear(2048, num_classes),#第三个全连接（全连接是一纬的向量，节点数为1000个）
```



### 1.3 前向传播过程

#### 1.3.1 Tensor的展平

```python
    x = torch.flatten(x, start_dim=1)# 展平后再传入全连接层,x是需要展平的数据，
```



```python
# 前向传播过程
def forward(self, x):#x代表输入的数据（batch[一批图像的个数],channel,height,width）
    x = self.features(x)#传入卷积层、池化层
    x = torch.flatten(x, start_dim=1)# 展平后再传入全连接层
    x = self.classifier(x)#传入全连接层
    return x

```

 

### 1.4 网络权重初始化

注意：**实际上 pytorch 在构建网络时会自动初始化权重**

```python
def _initialize_weights(self):
    for m in self.modules():#遍历self.modules#（遍历神经网络的全部层结构）
        if isinstance(m, nn.Conv2d):# 若是卷积层
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')# 用（何）kaiming_normal_法初始化权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)# 初始化偏重为0
        elif isinstance(m, nn.Linear): # 若是全连接层
            nn.init.normal_(m.weight, 0, 0.01)# 正态分布初始化
            nn.init.constant_(m.bias, 0)# 初始化偏重为0
```



### 1.5model.py 完整代码

```python
#导入库
import torch.nn as nn #使用torch.nn包来构建神经网络.
import torch

#定义一个类（AlexNet），继承来自nn.Module这个父类
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):# 定义初始化函数
        super(AlexNet, self).__init__() # 多继承需用到super函数（super()继承父类的构造函数）
      #卷积层，池化层创建并进行打包  
        self.features = nn.Sequential(  #nn.Sequential函数能够将一系列的层结构（卷积层、池化层、全连接层）打包成一个新的结构features
            #nn.Sequential函数使用的必要性:对于层数较多的神经网络来说，减少代码，降低工作量
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]，第一个卷积层（深度为3（通道），48个卷积盒，卷积核的尺寸11*11的大小,步长4，填充2）
            nn.ReLU(inplace=True), #rulu激活函数，inplace=True直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2), # output[48, 27, 27]，第一个池化层（尺寸为3*3最大值，步距为2）---下采样层
            nn.Conv2d(48, 128, kernel_size=5, padding=2),# output[128, 27, 27]，第二个卷积层（深度为48（通道），128个卷积盒，卷积核的尺寸5*5的大小,步长5，填充2）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2), # output[128, 13, 13]，第二个池化层（尺寸为3*3最大值，步距为2）---下采样层
            nn.Conv2d(128, 192, kernel_size=3, padding=1), # output[192, 13, 13]，第三个卷积层（深度为128（通道），192个卷积盒，卷积核的尺寸3*3的大小,步长5，填充1）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]，第四个卷积层（深度为192（通道），192个卷积盒，卷积核的尺寸3*3的大小,步长3，填充1）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.Conv2d(192, 128, kernel_size=3, padding=1), # output[128, 13, 13]，第五个卷积层（深度为192（通道），128个卷积盒，卷积核的尺寸3*3的大小,步长3，填充1）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2), # output[128, 6, 6]，第二个池化层（尺寸为3*3最大值，步距为2）---下采样层
        )
        #全连接层创建并进行打包  
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),# Dropout 随机失活神经元，默认比例为0.5
            nn.Linear(128 * 6 * 6, 2048),#第一个全连接（全连接是一纬的向量，节点数为2048个）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.Dropout(p=0.5),,# Dropout 随机失活神经元，默认比例为0.5
            nn.Linear(2048, 2048),#第二个全连接（全连接是一纬的向量，节点数为2048个）
            nn.ReLU(inplace=True),#rulu激活函数，inplace=True则是直接修改覆盖原值，节省运算内存
            nn.Linear(2048, num_classes),#第三个全连接（全连接是一纬的向量，节点数为1000个）
        )
        if init_weights:
            self._initialize_weights()
	# 前向传播过程
    def forward(self, x):#x代表输入的数据（batch[一批图像的个数],channel,height,width）
        x = self.features(x)#传入卷积层、池化层
        x = torch.flatten(x, start_dim=1)# 展平后再传入全连接层
        x = self.classifier(x)#传入全连接层
        return x
	# 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():遍历self.modules#（遍历神经网络的全部层结构）
            if isinstance(m, nn.Conv2d):# 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')# 用（何）kaiming_normal_法初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)# 初始化偏重为0
            elif isinstance(m, nn.Linear): # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)# 正态分布初始化
                nn.init.constant_(m.bias, 0)# 初始化偏重为0
```



## 2. train.py

### 2.0导入包及使用GPU训练

- 使用GPU训练`torch.device("cuda" if torch.cuda.is_available() else "cpu")`

```python
# 导入包
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

# 使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

```



### 2.1 数据预处理

需要注意的是，对训练集的预处理，多了**随机裁剪**和**水平翻转**这两个步骤。可以起到扩充数据集的作用，增强模型泛化能力。

- 随机裁剪`transforms.RandomResizedCrop(224)`，随机裁剪，再缩放成 224×224

- 水平翻转`transforms.RandomHorizontalFlip(p=0.5)`, 水平方向随机翻转，概率为 0.5

  

```python
#**随机裁剪**和**水平翻转**
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),       # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


```

### 2.2 导入、加载 训练集

[LeNet网络搭建](中是使用的`torchvision.datasets.CIFAR10`和`torch.utils.data.DataLoader()`来导入和加载数据集。

```python
#LeNet
# 导入训练集
train_set = torchvision.datasets.CIFAR10(root='./data', 	 # 数据集存放目录
										 train=True,		 # 表示是数据集中的训练集
                                        download=True,  	 # 第一次运行时为True，下载数据集，下载完成后改为False
                                        transform=transform) # 预处理过程
# 加载训练集                              
train_loader = torch.utils.data.DataLoader(train_set, 	  # 导入的训练集
										   batch_size=50, # 每批训练的样本数
                                          shuffle=False,  # 是否打乱训练集
                                          num_workers=0)  # num_workers在windows下设置为0

```

但是 花分类数据集 并不在 pytorch 的 torchvision.datasets. 中，因此需要用到`datasets.ImageFolder()`来导入。

`ImageFolder()`**返回的对象是一个包含数据集所有图像及对应标签构成的二维元组容器**，支持索引和迭代，可作为torch.utils.data.DataLoader的输入。具体可参考：pytorch ImageFolder和Dataloader加载自制图像数据集

```python
# 获取图像数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  		# get data root path 返回上上层目录
image_path = data_root + "/data_set/flower_data/"  				 		# flower data_set path

# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "/train",		
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,	# 导入的训练集
                                           batch_size=32, 	# 每批训练的样本数
                                           shuffle=True,	# 是否打乱训练集
                                           num_workers=0)	# 使用线程数，在windows下设置为0

```

### 2.3 导入、加载 验证集

```python
# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,	# 导入的验证集
                                              batch_size=32, 
                                              shuffle=True,
                                              num_workers=0)

```

### 2.4 存储 索引：标签 的字典

为了方便在 predict 时读取信息，将 索引：标签 存入到一个 `json` 文件中

```python
# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

```

`class_indices.json` 文件内容如下:

```python
{
    "0": "daisy",
    "1": "dandelion",
    "2": "roses",
    "3": "sunflowers",
    "4": "tulips"
}

```



### 2.5 训练过程

训练过程中需要注意：

- `net.train()`：训练过程中开启 Dropout

- `net.eval()`： 验证过程关闭 Dropout

  

#### 2.5.1实例化AlexNe网络模型（损失函数、优化器）

- `nn.CrossEntropyLoss()` 函数：结合了nn.LogSoftmax()和nn.NLLLoss()两个函数

- `optim.Adam(net.parameters(), 0.0002)`  函数：net.parameters代表训练参数即LetNet网络模型的可训练全部参数，lr代表学习率

  ```python
  net = AlexNet(num_classes=5, init_weights=True)  	  # 实例化网络（输出类型为5，初始化权重）
  net.to(device)									 	  # 分配网络到指定的设备（GPU/CPU）训练
  loss_function = nn.CrossEntropyLoss()			 	  # 交叉熵损失
  optimizer = optim.Adam(net.parameters(), lr=0.0002)	  # 优化器（训练参数，学习率）
  
  ```

  

#### 2.5.2 进入模型训练过程

- 设置训练集进行多少次训练

- 遍历训练集，获取训练集的图像和标签

- 清除历史梯度

- 正向传播

- 计算损失

- 反向传播

- 优化器更新参数

  ```python
  for epoch in range(10):
      ########################################## train ###############################################
      net.train()     					# 训练过程中开启 Dropout
      running_loss = 0.0					# 每个 epoch 都会对 running_loss  清零
      time_start = time.perf_counter()	# 对训练一个 epoch 计时
      
      for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
          images, labels = data   # 获取训练集的图像和标签
          optimizer.zero_grad()	# 清除历史梯度
          
          outputs = net(images.to(device))				 # 正向传播
          loss = loss_function(outputs, labels.to(device)) # 计算损失
          loss.backward()								     # 反向传播
          optimizer.step()								 # 优化器更新参数
          running_loss += loss.item()
          
  ```

  

#### 2.5.3 打印训练进度（使训练过程可视化）



```python
    rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
    a = "*" * int(rate * 50)
    b = "." * int((1 - rate) * 50)
    print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
print()
print('%f s' % (time.perf_counter()-time_start))
```



#### 2.5.4 验证训练精度、损失、准确率等数据（结果）

```python
########################################### validate ###########################################
    net.eval()    # 验证过程中关闭 Dropout
    acc = 0.0  
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()    
        val_accurate = acc / val_num
```



#### 2.5.5 保存训练参数

-  `torch.save(net.state_dict(), save_path)`

​		net.state_dict()：以字典的形式

​		save_path：保存参数的路径

```python
save_path = './AlexNet.pth'#保存参数路径
# 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

```



### 2.6trian.py 完整代码

```python
# 导入包
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet

net = AlexNet(num_classes=5, init_weights=True)  	  # 实例化网络（输出类型为5，初始化权重）
net.to(device)									 	  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()			 	  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.0002)	  # 优化器（训练参数，学习率）

save_path = './AlexNet.pth'
best_acc = 0.0

for epoch in range(10):
    ########################################## train ###############################################
    net.train()     					# 训练过程中开启 Dropout
    running_loss = 0.0					# 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()	# 对训练一个 epoch 计时
    
    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        images, labels = data   # 获取训练集的图像和标签
        optimizer.zero_grad()	# 清除历史梯度
        
        outputs = net(images.to(device))				 # 正向传播
        loss = loss_function(outputs, labels.to(device)) # 计算损失
        loss.backward()								     # 反向传播
        optimizer.step()								 # 优化器更新参数
        running_loss += loss.item()
        
        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print('%f s' % (time.perf_counter()-time_start))

    ########################################### validate ###########################################
    net.eval()    # 验证过程中关闭 Dropout
    acc = 0.0  
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()    
        val_accurate = acc / val_num
        
        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

```



训练打印信息如下：

```python
cuda
train loss: 100%[**************************************************->]1.566
27.450399 s
[epoch 1] train_loss: 1.413  test_accuracy: 0.404 

train loss: 100%[**************************************************->]1.412
27.897467399999996 s
[epoch 2] train_loss: 1.211  test_accuracy: 0.503 

train loss: 100%[**************************************************->]1.412
28.665594 s
[epoch 3] train_loss: 1.138  test_accuracy: 0.544 

train loss: 100%[**************************************************->]0.924
28.6858524 s
[epoch 4] train_loss: 1.075  test_accuracy: 0.621 

train loss: 100%[**************************************************->]1.200
28.020624199999986 s
[epoch 5] train_loss: 1.009  test_accuracy: 0.621 

train loss: 100%[**************************************************->]0.985
27.973145999999986 s
[epoch 6] train_loss: 0.948  test_accuracy: 0.607 

train loss: 100%[**************************************************->]0.583
28.290610200000003 s
[epoch 7] train_loss: 0.914  test_accuracy: 0.670 

train loss: 100%[**************************************************->]0.930
28.51416950000001 s
[epoch 8] train_loss: 0.912  test_accuracy: 0.621 

train loss: 100%[**************************************************->]1.210
28.98158360000002 s
[epoch 9] train_loss: 0.840  test_accuracy: 0.668 

train loss: 100%[**************************************************->]0.961
28.330670499999997 sp
[epoch 10] train_loss: 0.833  test_accuracy: 0.684 

Finished Training

```



## 3. predict.py

### 3.1 导入包

```python
#导入包
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
```



### 3.2 数据预处理

#### 3.2.1 图像数据缩放及归一化、标准化

```python
# 图像数据缩放及归一化、标准化
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),# 首先需resize成跟训练集图像一样的大小（对图像进行缩放成224*224）
         transforms.ToTensor(),#然后转化成Tensor（张量）
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #接着进行标准化处理
    
```



#### 3.2.2 导入要测试/识别的图像（自己找的，不在数据集中)

```python
# 导入要测试的图像（自己找的，不在数据集中），放在源文件目录下
img = Image.open("蒲公英.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)#数据预处理
# expand batch dimension
img = torch.unsqueeze(img, dim=0)# 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]
```



#### 3.2.3 读取在train.py中存储的图像对于标签的json文件

```python
# read class_indict
try:
    json_file = open('./class_indices.json', 'r')#对json文件进行读取
    class_indict = json.load(json_file)#duijson文件进行解密，变成我们可以用的字典
except Exception as e:
    print(e)
    exit(-1)
```



#### 3.2.4 实例化网络，加载训练好的模型参数

```python
# create model，实例化网络，加载训练好的模型参数
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))#把权重载入网络模型

```



#### 3.2.5 对输入的图片（需要预测的图片）不求其损失梯度

```python
# 关闭 Dropout
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()

```



### **3.3 predict.py.完整代码**    

```python
#导入包
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

# 预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),# 首先需resize成跟训练集图像一样的大小
     transforms.ToTensor(),#转化为张量（Tensor）,是将输入的数据shape W，H，C ——> C，W，H,将所有数除以255，将数据归一化到【0，1】
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#标准化处理x = (x - mean) / std

# 导入要测试的图像（自己找的，不在数据集中），放在源文件目录下
img = Image.open("蒲公英.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)#数据预处理
# expand batch dimension
img = torch.unsqueeze(img, dim=0)# 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')#对json文件进行读取
    class_indict = json.load(json_file)#duijson文件进行解密，变成我们可以用的字典
except Exception as e:
    print(e)
    exit(-1)

# create model，实例化网络，加载训练好的模型参数
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))#把权重载入网络模型

# 关闭 Dropout
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()

```

打印出预测的标签以及概率值：

```python
dandelion 0.7221569418907166
```

