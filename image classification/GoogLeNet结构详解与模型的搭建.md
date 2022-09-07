# GoogLeNet结构详解与模型的搭建

## 1. GoogLeNet网络详解

GoogLeNet在2014年由Google团队提出（与VGG网络同年，注意GoogLeNet中的L大写是为了致敬LeNet），斩获当年ImageNet竞赛中Classification Task (分类任务) 第一名。

**GoogLeNet 的创新点：**

- 引入了 Inception 结构（融合不同尺度的特征信息）
- 使用1x1的卷积核进行降维以及映射处理 （虽然VGG网络中也有，但该论文介绍的更详细）
- 添加两个辅助分类器帮助训练
- 丢弃全连接层，使用平均池化层（大大减少模型参数，除去两个辅助分类器，网络大小只有vgg的1/20）



### 1.1 inception 结构

#### 1.1.0 传统的CNN结构

传统的CNN结构如AlexNet、VggNet（下图）都是==**串联**==的结构，即将一系列的卷积层和池化层进行串联得到的结构。

![image-20220907141921659](https://gitee.com/long_chaohuo/images/raw/master/image-20220907141921659.png)









#### 1.1.1 **inception原始结构**

GoogLeNet 提出了一种==**并联**==结构，下图是论文中提出的inception原始结构，将特征矩阵同时输入到==**多个分支**==进行处理，并将输出的特征矩阵按深度进行拼接，得到最终输出。

![image-20220907141934237](https://gitee.com/long_chaohuo/images/raw/master/image-20220907141934237.png)





- **==inception的作用：增加网络深度和宽度的同时减少参数。==**

注意：每个分支所得特征矩阵的高和宽必须相同（通过调整stride和padding），以保证输出特征能在深度上进行拼接。



#### 1.1.2 inception + 降维

在 inception 的基础上，还可以加上降维功能的结构，如下图所示，在原始 inception 结构的基础上，在**分支2，3，4上加入了卷积核大小为1x1的卷积层，目的是为了降维（减小深度）**，减少模型训练参数，减少计算量。![image-20220907141948669](https://gitee.com/long_chaohuo/images/raw/master/image-20220907141948669.png)





```toml
1×1卷积核的降维功能
同样是对一个深度为512的特征矩阵使用64个大小为5x5的卷积核进行卷积，不使用1x1卷积核进行降维的 话一共需要819200个参数，如果使用1x1卷积核进行降维一共需要50688个参数，明显少了很多。
```

![image-20220907142002728](https://gitee.com/long_chaohuo/images/raw/master/image-20220907142002728.png)



==注：CNN参数个数 = 卷积核尺寸×卷积核深度 × 卷积核组数 = 卷积核尺寸 × 输入特征矩阵深度 × 输出特征矩阵深度==



### 1.2 辅助分类器（Auxiliary Classifier）

AlexNet 和 VGG 都只有1个输出层，GoogLeNet 有3个输出层，其中的两个是辅助分类层。

如下图所示，网络主干右边的 两个分支 就是 辅助分类器，其结构一模一样。
在训练模型时，将两个辅助分类器的损失乘以权重（论文中是0.3）加到网络的整体损失上，再进行反向传播。

​                                                                                          ![image-20220907142017962](https://gitee.com/long_chaohuo/images/raw/master/image-20220907142017962.png)





```tsx
引用：GoogLeNet(Inception V1)
辅助分类器的两个分支有什么用呢？
作用一：可以把他看做inception网络中的一个小细节，它确保了即便是隐藏单元和中间层也参与了特征计算，他们也能预测图片的类别，他在inception网络中起到一种调整的效果，并且能防止网络发生过拟合。

作用二：给定深度相对较大的网络，有效传播梯度反向通过所有层的能力是一个问题。通过将辅助分类器添加到这些中间层，可以期望较低阶段分类器的判别力。在训练期间，它们的损失以折扣权重（辅助分类器损失的权重是0.3）加到网络的整个损失上。
```



### 1.3 GoogLeNet 网络参数

下面是原论文中给出的网络参数列表，配合上图查看

![image-20220907142045511](https://gitee.com/long_chaohuo/images/raw/master/image-20220907142045511.png)

- 对于Inception模块，所需要使用到参数有`#1x1`, `#3x3reduce`, `#3x3`, `#5x5reduce`, `#5x5`, `poolproj`，这6个参数，分别对应着所使用的卷积核个数。

![image-20220907142057024](https://gitee.com/long_chaohuo/images/raw/master/image-20220907142057024.png)



- `#1x1`对应着分支1上1x1的卷积核个数

- #3x3reduce`对应着分支2上1x1的卷积核个数

- `#3x3`对应着分支2上3x3的卷积核个数

- `#5x5reduce`对应着分支3上1x1的卷积核个数

- `#5x5`对应着分支3上5x5的卷积核个数

- `poolproj`对应着分支4上1x1的卷积核个数。

  

## 2. pytorch搭建GoogLeNet

### 1.model.py

相比于 AlexNet 和 VggNet 只有卷积层和全连接层这两种结构，GoogLeNet多了 ==inception== 和 ==辅助分类器（Auxiliary Classifier）==，而 inception 和 辅助分类器 也是由多个卷积层和全连接层组合的，因此在定义模型时可以将 卷积、inception 、辅助分类器定义成不同的类，调用时更加方便。

#### 1.0 导入包

```python
#导入包
import torch.nn as nn
import torch
import torch.nn.functional as F
```



#### 2.0创建模板文件（inception、辅助分类器），即类class

##### 提取特征网络结构模板（卷积层+ReLU+正向传播）

```python
# 基础卷积层（卷积+ReLU）
class BasicConv2d(nn.Module):#定义一个类（BasicConv2d），继承来自nn.Module这个父类
    def __init__(self, in_channels, out_channels, **kwargs):# 定义初始化函数（in_channels(输入通道数)，out_channels（输出通道数（深度）））
        super(BasicConv2d, self).__init__# 多继承需用到super函数（super()继承父类的构造函数）
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)#卷积层
        self.relu = nn.ReLU(inplace=True)#激活函数ReLUinplace=True则是直接修改覆盖原值，节省运算内存
        # 前向传播过程
        def forward(self, x):#x代表输入的数据（batch[一批图像的个数],channel,height,width）
        x = self.conv(x)#传入卷积层
        x = self.relu(x)#传入激活函数
        return x
```



#####  Inception结构模板

```python
# Inception结构
class Inception(nn.Module):#定义一个类（Inception），继承来自nn.Module这个父类
    
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):# 定义初始化函数
    '''
    	in_channels:输入特征矩阵的深度
    	ch1x1:对应着分支1上1x1的卷积核个数
    	ch3x3red:对应着分支2上1x1的卷积核个数
    	ch3x3:对应着分支2上3x3的卷积核个数
		ch5x5red:对应着分支3上1x1的卷积核个数
		ch5x5:对应着分支3上5x5的卷积核个数
		pool_proj:对应着分支4上1x1的卷积核个数。
    '''
        super(Inception, self).__init__# 多继承需用到super函数（super()继承父类的构造函数）
		#分支1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)#BasicConv2d（in_channels输入特征矩阵的深度，ch1x1输出特征矩阵深度，卷积核大小）
		#分支2
        #nn.Sequential将一系列的层结构（卷积层、池化层、全连接层）打包成一个新的结构
        self.branch2 = nn.Sequential(
            
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),#BasicConv2d（in_channels输入特征矩阵的深度，ch3x3red输出特征矩阵深度，卷积核大小）
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )
		#分支3
         #nn.Sequential将一系列的层结构（卷积层、池化层、全连接层）打包成一个新的结构
        self.branch3 = nn.Sequential(
            #BasicConv2d（in_channels输入特征矩阵的深度，ch5x5red输出特征矩阵深度，卷积核大小）
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )
		#分支4
        self.branch4 = nn.Sequential(
            #max池化层（卷积核大小，步距，填充）
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
	# 前向传播过程
    def forward(self, x):	#x代表输入的数据（batch[一批图像的个数],channel,height,width）
        branch1 = self.branch1(x)#传入分支
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
		#输出放入列表中
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1) # 按 channel 对四个分支拼接 ，维度为1 （0是batch）

```



##### 辅助分类器

```python
# 辅助分类器
class InceptionAux(nn.Module):
    #(in_channels(输入特征矩阵的深度), num_classes（输出特征矩阵的个数（深度））)
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
		#全连接层
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
	#正向传播
    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)#展平
        x = F.dropout(x, 0.5, training=self.training# Dropout 随机失活神经元，默认比例为0.5
                      '''
                      	当我们实例化一个mode后，可以通过mode.train()和mode.eval()来控制模型的状态，在mode.train()模式							下self.trainimg=True,在mode.eval()模式下self.trainimg=False    
                      '''
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x
```





#### 3.0定义GoogLeNet网络模型

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
![image-20220907142120020](https://gitee.com/long_chaohuo/images/raw/master/image-20220907142120020.png)

- 输入图片大小 W×W（一般情况下Width=Height）
- Filter大小 F×F
- 步长 S
- padding的像素数 P



### 1.2 池化 MaxPool2d

最大池化（MaxPool2d）在 pytorch 中对应的函数是：

```python
MaxPool2d(kernel_size, stride)
```





```python
#定义GoogLeNet网络模型
#定义一个类（AlexNet），继承来自nn.Module这个父类
class GoogLeNet(nn.Module):
	# 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False# 定义初始化函数
        super(GoogLeNet, self).__init__# 多继承需用到super函数（super()继承父类的构造函数）
        self.aux_logits = aux_logits
		#卷积层
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
         #池化层
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)#ceil_mode（True向上取整，False向下取整）
                 ##############这里不使用locaoRsepNorm##############
		#卷积层       
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
         #卷积层
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
         #池化层
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
		#inception3a结构
        '''
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        192=输入特征矩阵的深度
    	64对应着分支1上1x1的卷积核个数
    	96对应着分支2上1x1的卷积核个数
    	128对应着分支2上3x3的卷积核个数
		16对应着分支3上1x1的卷积核个数
		32对应着分支3上5x5的卷积核个数
		32:对应着分支4上1x1的卷积核个数。
        '''
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
		#辅助分类器
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
		#平均值池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#nn.AdaptiveAvgPool2d((1, 1))---输出为1*1的矩阵
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
	#正向传播
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```







































```python
import torch.nn as nn
import torch
import torch.nn.functional as F

class GoogLeNet(nn.Module):
	# 传入的参数中aux_logits=True表示训练过程用到辅助分类器，aux_logits=False表示验证过程不用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1) # 按 channel 对四个分支拼接  

# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x

# 基础卷积层（卷积+ReLU）
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

```



### 2.train.py

### 3.predict.py