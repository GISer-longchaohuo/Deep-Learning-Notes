# FCN网络结构详解(语义分割)

![image-20220907152459165](https://gitee.com/long_chaohuo/images/raw/master/image-20220907152459165.png)

> `Fully convolutional Networks for Semantic Segmentation`  2015CVPR
>
> ==首个==端到端的针对像素级预测的==全卷积==网络
>
> 所谓的全卷积就是将网络的==全连接层==全部换为卷积层（作用：对输入的图片大小无限制）
>
> <img src="https://gitee.com/long_chaohuo/images/raw/master/image-20220907154332480.png" alt="image-20220907154332480" style="zoom:33%;" />



## 一、简介

**该网络的与之前的对比**

![image-20220907153845084](https://gitee.com/long_chaohuo/images/raw/master/image-20220907153845084.png)



回归一下VGG16

![image-20220907155038172](https://gitee.com/long_chaohuo/images/raw/master/image-20220907155038172.png)



使用全连接层和使用卷积层代替的比较

![image-20220907160156287](https://gitee.com/long_chaohuo/images/raw/master/image-20220907160156287.png)



这里的32s、16s等代表的是上采样的倍数，如下图

![image-20220907165514562](https://gitee.com/long_chaohuo/images/raw/master/image-20220907165514562.png)





## 二、各网络结构详解

### 1. FCN-32S

![image-20220907170234591](https://gitee.com/long_chaohuo/images/raw/master/image-20220907170234591.png)



```
原论文的padding为100：为了适宜不同尺寸的图片
```

