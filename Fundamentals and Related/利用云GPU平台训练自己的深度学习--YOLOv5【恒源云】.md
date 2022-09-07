

# 利用云GPU平台训练自己的深度学习--YOLOv5【恒源云】

## 一 前言

用[yolov5](https://so.csdn.net/so/search?q=yolov5&spm=1001.2101.3001.7020)v5训练自己的数据集,本地电脑配置不足，决定采用云平台的方式进行训练

![image-20220907135037003](https://gitee.com/long_chaohuo/images/raw/master/image-20220907135037003.png)



## 二 训练前的准备

### 注册账号

输入恒源云https://gpushare.com/，按照要求进行注册并登录

```
如果是学生党的话，记得进行学生身份认证，有优惠及代金券！！！
```

![image-20220505164114334](https://gitee.com/long_chaohuo/images/raw/master/image-20220505164114334.png)

```
注册用户也有新手大礼包！！！而且平时优惠多多
```



### 上传个人数据

#### 个人数据预览

```
上传数据到个人空间中的过程中。可以在开启实例前进行操作，节约了实例运行的费用！！！！先上传数据再创建实例
```

点击右上角的控制台，进入【控制台】---【我的数据】，本案例使用的是OSS命令

<img src="https://gitee.com/long_chaohuo/images/raw/master/image-20220505164612952.png" alt="image-20220505164612952" style="zoom:67%;" />

<img src="https://gitee.com/long_chaohuo/images/raw/master/image-20220505164837481.png" alt="image-20220505164837481" style="zoom:67%;" />

![image-20220505165056646](C:/Users/1/AppData/Roaming/Typora/typora-user-images/image-20220505165056646.png)



#### 使用oss命令行工具进行数据上传

##### 下载oss命令工具

这里使用Windows系统作为本地环境。如果是其他系统可以参考[oss命令行工具](https://gpushare.com/docs/data/storage/#oss)进行安装。下载[oss_windows_x86_64.exe](https://gpucloud-static-public-prod.gpushare.com/installation/oss/oss_windows_x86_64.exe)可执行文件，下载完成将其放到合适位置后将其重命名为oss.exe

![image-20220505165337108](https://gitee.com/long_chaohuo/images/raw/master/image-20220505165337108.png)

##### 登录oss工具

1.进入cmd,运行oss.exe工具,第一种方式：按`win+r`键进入cmd命令程序，利用命令`cd F:\云GPU上传`进入文件夹

```
#cd oss工具所在的文件夹
cd F:\云GPU上传
```

![image-20220505165956543](https://gitee.com/long_chaohuo/images/raw/master/image-20220505165956543.png)

第二种方式：直接到`oss.exe`程序所在的文件目录，在上方的路径处输入`cmd.`然后`回车`也可获得一样的效果

<img src="C:/Users/1/AppData/Roaming/Typora/typora-user-images/image-20220505170407956.png" alt="image-20220505170407956" style="zoom:67%;" />







2.在当前目录进入cmd后，执行`.\oss login`或`oss.exe login`命令，输入恒源云平台账号和密码登录，账号为手机号!，成功登录后会返回一个`1521****320 login successfully!`

```
#登录命令
oss.exe login
.\oss login
```

![image-20220505170656428](https://gitee.com/long_chaohuo/images/raw/master/image-20220505170656428.png)



3.执行`.\oss cp 或oss cp 压缩文件所在目录\xxx.zip oss://` 命令，上传本地当前目录数据（将数据打包成 **zip、tar.gz** 常用格式的压缩包)到个人数据根目录

```
#oss cp 压缩文件所在目录\xxx.zip 个人数据库的文件夹（oss://为头目录）
oss cp 压缩文件所在目录\xxx.zip oss://
```

![image-20220505172726060](https://gitee.com/long_chaohuo/images/raw/master/image-20220505172726060.png)



```
#文件目录操作
# 查看当前所在目录位置
~# pwd
/root

# 进入其他目录中
~# cd /hy-nas

/hy-nas# pwd
/hy-nas

# 创建一个空文件
~# touch emptyfile

# 查看当前目录下的文件
~# ls
emptyfile

# 在当前目录下创建一个目录
~# mkdir directory

~# cd directory
~/directory# pwd
/root/directory

# 删除文件
~# rm emptyfile

# 删除文件夹
~# rm -rf directory
```

> 数据保存期限：
> 按量付费关机或包周包月到期后，超过10天后实例会被删除
> 按量付费关机或包周包月到期后，超过24小时后实例本地盘 /hy-tmp 数据会被删除，但是平台提供了自动化上传训练数据的方案，来规避该问题，查看执行训练并自动上传结果后关机了解
> 如果发生欠费，从欠费算起第15天中午12点会删除个人数据、共享存储 /hy-nas、自定义镜像







### 创建实例

##### 购买实例

在恒源云的[云市场](https://gpushare.com/store)中筛选需要的主机配置，点击 **立即租** 进入购买界面

![image-20220907135245570](https://gitee.com/long_chaohuo/images/raw/master/image-20220907135245570.png)



在购买实例页面选择计费模式、实例配置和系统环境镜像，点击 **创建实例** 进行租用。

![image-20220907135300184](https://gitee.com/long_chaohuo/images/raw/master/image-20220907135300184.png)

##### 使用实例

在 **我的实例** 中可以看到刚刚购买的实例，等待创建完成即可使用

实例启动完成后可以通过复制登陆指令，或打开 **JupyterLab** 链接来运行代码或登陆实例。

![image-20220907135318559](https://gitee.com/long_chaohuo/images/raw/master/image-20220907135318559.png)



在 JupyterLab 上点击终端新建一个终端窗口

![image-20220907135331043](https://gitee.com/long_chaohuo/images/raw/master/image-20220907135331043.png)

##### 下载个人数据

在终端中使用 `oss` 命令下载刚刚上传到个人数据空间中的压缩包。

先执行 `cd /hy-tmp` 进入 `/hy-tmp` 目录。再执行 `oss login` 命令登陆个人数据空间，使用恒源云的账号名与密码，账号名为手机号。







## 三 训练模型

