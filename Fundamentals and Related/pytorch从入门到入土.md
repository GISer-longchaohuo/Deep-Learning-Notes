# pytorch从入门到入土

## 一、python package学习的法宝

python package学习的两大法宝`dir( )`和`help（）`

```python
dir(pytorch)#打开、看见package
help(pytorch)#关于package的说明
```

在遇到函数需要填写哪些参数的时候：按一下`ctrl+P`









## 二、Pycharm及Jupyter使用及对比

```python
#打开Jupyter notebook，在你需要打开的虚拟环境下，输入以下命令
jupyter notebook

```

![image-20220927142549869](https://gitee.com/long_chaohuo/images/raw/master/image-20220927142549869.png)



各种方式的对比

| 区别                     | python文件                     | python控制台         | Jupyter            |
| ------------------------ | ------------------------------ | -------------------- | ------------------ |
| 代码是以块为一个整体运行 | 块就是所有行的代码             | 以每一行为块运行     | 以任意行执行       |
| 优点                     | 通用、传播方便，适用于大型醒目 | 显示每个变量属性     | 利于代码阅读及修改 |
| 缺点                     | 需要从头运行                   | 不利于代码阅读及修改 | 环境需要配置       |







## 三、pytorch加载数据初认识

由`Dataset`类和`Dataloader`类两个类函数组成

![image-20220927151608592](https://gitee.com/long_chaohuo/images/raw/master/image-20220927151608592.png)

### Dataset类

```
提供一种方式获取数据及其label(从数据集当中)
```

#### 作用

- 如何获取每一个数据及其label

- 数据集中总共有多少个数据

  

### Dataloader类

```
为后面的网络提供不同的数据形式
```





### 实例示意

```
数据集结构
--hymenoptera_data
	————train
        --bees
        --ants
    ————val
    	--bees
        --ants
```

<img src="https://gitee.com/long_chaohuo/images/raw/master/image-20220928133951977.png" alt="image-20220928133951977" style="zoom:33%;" />

<img src="https://gitee.com/long_chaohuo/images/raw/master/image-20220928134017497.png" alt="image-20220928134017497" style="zoom:33%;" />

```python

from torch.utils.data import Dataset
from PIL import Image
import os


# 创建一个数据集类,继承父类Dataset
class MyDatat(Dataset):
    # 初始化函数
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


# 实例化
# 根目录
root_dir = r'D:\BaiduNetdiskDownload\PyTorch 深度学习快速入门教程\数据集\hymenoptera_data\hymenoptera_data\train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
# label文件夹
ants_dataset = MyDatat(root_dir,ants_label_dir)
bees_dataset = MyDatat(root_dir,bees_label_dir)

train_dataset = ants_dataset +bees_dataset
```





## 四、TensorBoard的使用

### 1.writer.add_scalar( )添加数据折线图

```python
from torch.utils.tensorboard import SummaryWriter
# 创建文件夹
writer = SummaryWriter('logs')
# writer.add_image()
for i in range(int(100)):
    sum += i
    writer.add_scalar('test',sum,i)
    '''  tag (string): 标题
         scalar_value y轴
         global_step (int): x轴
            '''

writer.close()
```



![image-20220928140903347](https://gitee.com/long_chaohuo/images/raw/master/image-20220928140903347.png)



```python
#tensorboard文件的命令
tensorboard --logdir=[event文件所在的文件夹]
#指定端口号命令
tensorboard --logdir=[event文件所在的文件夹] --port=[指定的端口号]
```



![image-20220928141755710](https://gitee.com/long_chaohuo/images/raw/master/image-20220928141755710.png)



### 2.writer.add_image()常用来观察训练结果

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
# 创建文件夹
writer = SummaryWriter('logs')
img_path = r'C:\Users\1\Desktop\417306271310413824-17.tif'
img_plt = Image.open(img_path)
img_array = np.array(img_plt)
writer.add_image('test',img_plt,dataformats='HWC')
''' 
tag (string): 标题
img_tensor:图片

 dataformats='CHW'即格式
'''

writer.close()
```

![image-20220928144759532](https://gitee.com/long_chaohuo/images/raw/master/image-20220928144759532.png)

![image-20220929135903927](https://gitee.com/long_chaohuo/images/raw/master/image-20220929135903927.png)

![image-20220929141302547](https://gitee.com/long_chaohuo/images/raw/master/image-20220929141302547.png)

![image-20220929142356727](https://gitee.com/long_chaohuo/images/raw/master/image-20220929142356727.png)

![image-20220929145130158](https://gitee.com/long_chaohuo/images/raw/master/image-20220929145130158.png)



![image-20220929141902286](https://gitee.com/long_chaohuo/images/raw/master/image-20220929141902286.png)





![image-20220929150110463](https://gitee.com/long_chaohuo/images/raw/master/image-20220929150110463.png)





```python
#使用方法总结
1.关注函数的输入和输出类型
2.多看官方文档
3.关注方法需要什么参数
4.反正不懂时就debug
```







## 五、torchvision中的数据集使用

```python
import torchvision
# 数据预处理
from torch.utils.tensorboard import SummaryWriter

dataset_trainsform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 使用CIFAR10数据集
train_set = torchvision.datasets.CIFAR10(root='./datatset',train=True,transform=dataset_trainsform,download=True)
val_set = torchvision.datasets.CIFAR10(root='./datatset',train=False,download=True)
print(train_set[0])
# 添加图片数据到tensorboard
writer = SummaryWriter('log')
for i in range(100):
    img,target = train_set[i]
    writer.add_image('image',img,i)
writer.close()
    
```







## 六、Dataloader的使用

![image-20220930140903312](https://gitee.com/long_chaohuo/images/raw/master/image-20220930140903312.png)

```python
import torchvision
# 准备测试数据集
from torch.utils.data import DataLoader
test_data = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor)
# drop_last不足够bach_size时就舍去
test_loader = DataLoader(datatset=test_data,batch_size=2,shuffle=True,num_workers=0,drop_last=True)
# # 测试数据集中的第一张图片及target
# img,target = test_data[0]
# print(img.shape)
# print(target)
for data in test_loader:
    img,targets = data
    print(img.shape)
    print(targets)
```
