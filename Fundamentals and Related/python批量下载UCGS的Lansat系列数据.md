# python批量下载UCGS的Lansat系列数据

## 一、前言

闲来无事，本人没有下载需求。无聊时在搜索引擎上搜索关键字：`python批量下载Lansa数据`，弹出有一堆下载教程，但是，这些代码要么不能跑通（官网API有变化），要么就是不够贴心（研究范围要手动输入经纬度），所以我就来捣鼓一下。

<img src="https://gitee.com/long_chaohuo/images/raw/master/image-20221008135443077.png" alt="image-20221008135443077" style="zoom: 150%;" />



> 下载主要依赖landsatxplore包，先贴出项目地址
>
> **landsatxplore package项目地址**：https://github.com/yannforget/landsatxplore
>
> **landsatxplore package项目说明：**https://pypi.org/project/landsatxplore/
>
> <img src="https://gitee.com/long_chaohuo/images/raw/master/image-20221008135929846.png" style="zoom:50%;" />



因为USGS:https://earthexplorer.usgs.gov/ 官方时不时对API等相关信息有改动，所以你直接`pip install landsatxplore`安装好的包就不能用（包没有进行更新），作者估计是忘记账号了（我猜的）。目前更新主要靠广大的用户朋友们自己动手维护！！！所以遇到什么问题直接在git上进行提问（毕竟大佬们多）

![image-20221008140534226](https://gitee.com/long_chaohuo/images/raw/master/image-20221008140534226.png)

因为需要改动的代码有点多且乱。所以我直接把整个项目克隆下载进行修改。

> 运行环境要求：
>
> osgeo库
>
> os库
>
> 一个USGS网站:https://earthexplorer.usgs.gov/账号



**能下载的数据产品**

| Dataset Name                        | Dataset ID          |
| ----------------------------------- | ------------------- |
| Landsat 5 TM Collection 1 Level 1   | `landsat_tm_c1`     |
| Landsat 5 TM Collection 2 Level 1   | `landsat_tm_c2_l1`  |
| Landsat 5 TM Collection 2 Level 2   | `landsat_tm_c2_l2`  |
| Landsat 7 ETM+ Collection 1 Level 1 | `landsat_etm_c1`    |
| Landsat 7 ETM+ Collection 2 Level 1 | `landsat_etm_c2_l1` |
| Landsat 7 ETM+ Collection 2 Level 2 | `landsat_etm_c2_l2` |
| Landsat 8 Collection 1 Level 1      | `landsat_8_c1`      |
| Landsat 8 Collection 2 Level 1      | `landsat_ot_c2_l1`  |
| Landsat 8 Collection 2 Level 2      | `landsat_ot_c2_l2`  |
| Landsat 9 Collection 2 Level 1      | `landsat_ot_c2_l1`  |
| Landsat 9 Collection 2 Level 2      | `landsat_ot_c2_l2`  |
| Sentinel 2A                         | `sentinel_2a`       |



## 二、代码思路

函数部分我就巴拉巴拉了，直接说说主函数思路

1.为了考虑代码的复用性，我们要确定有几个形参。如账号密码、产品类型、时间间隔、云量、输入、输出等；

2.因为该包的输出只支持点和矩形的经纬度输入（其实可以魔改，但是我懒），这样输入不方便，我就写了一个求json（面要素）文件的最小外接矩形的函数

如果你的研究区范围文件是shp的话就可以直接转json格式（注意一定要地理坐标！！！），或者直接在https://geojson.io/#map=2/20.0/0.0画范围保存。

![image-20221008142057526](https://gitee.com/long_chaohuo/images/raw/master/image-20221008142057526.png)

```python
# 求geojson中图形的最小外接矩形'
def circumscribed_rectangle(geojson_path):
    json_driver = ogr.GetDriverByName('GeoJSON')
    json_ds = json_driver.Open(geojson_path)
    cs=json_ds.GetLayerByIndex(0)
    for row in cs:
        k=row.geometry()
        data_num= k.GetEnvelope()
        xmin = data_num[0]
        xmax =data_num[1]
        ymin = data_num[2]
        ymax = data_num[3]
    return xmin, xmax, ymin, ymax
```



3.编写一个UCGS网站的API登录及响应函数,主要使用的是 `landsatxplore.api.API（）`类函数（超级简单的）

```python
#登录UCGS及参数设置
def request_UCGS(username, password, product, xmin, ymin, xmax, ymax, start_date, end_date, cloud_max):
    #API连接
    api = landsatxplore.api.API(username, password)

    scenes = api.search(
        dataset=product,
        bbox=(xmin, ymin, xmax, ymax),
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=cloud_max,)
    print(f'共有产品数量{len(scenes)}')
    api.logout()
    return scenes
```



4.再编写一个主函数用于处理API响应回来的函数，主要使用的是`Earth_Down.download（）`函数（超级简单的）

```python
# 下载主函数
def download_UCGS_data(username, password, Landsat_name, output_dir):
    Earth_Down = EarthExplorer(username, password)
    for scene in Landsat_name:
        # print(scene[landsat_product_id])
        ID = scene['display_id']
        # print(f"下载的数据为{scene['landsat_product_id']}")
        if not os.path.isfile(output_dir + ID+".zip"):
            print(f'本地无:{output_dir + ID}.zip 的完整文件，尝试下载')
            Earth_Down.download(identifier=ID, output_dir=output_dir)
    Earth_Down.logout()
```



**主入门代码完整代码**

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@version: Anaconda
@author: longchaohuo
@contact: 460958592@qq.com
@software: PyCharm
@file: UCGS_Landsat_Download
@time: 2022/9/30  下午 14:23
参考文档：https://github.com/yannforget/landsatxplore；https://pypi.org/project/landsatxplore/
"""
from osgeo import ogr
import os
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer

# 求geojson中图形的最小外接矩形'
def circumscribed_rectangle(geojson_path):
    json_driver = ogr.GetDriverByName('GeoJSON')
    json_ds = json_driver.Open(geojson_path)
    cs=json_ds.GetLayerByIndex(0)
    for row in cs:
        k=row.geometry()
        data_num= k.GetEnvelope()
        xmin = data_num[0]
        xmax =data_num[1]
        ymin = data_num[2]
        ymax = data_num[3]
    return xmin, xmax, ymin, ymax
#登录UCGS及参数设置
def request_UCGS(username, password, product, xmin, ymin, xmax, ymax, start_date, end_date, cloud_max):
    #API连接
    api = landsatxplore.api.API(username, password)

    scenes = api.search(
        dataset=product,
        bbox=(xmin, ymin, xmax, ymax),
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=cloud_max,)
    print(f'共有产品数量{len(scenes)}')
    api.logout()
    return scenes
# 下载主函数
def download_UCGS_data(username, password, Landsat_name, output_dir):
    Earth_Down = EarthExplorer(username, password)
    for scene in Landsat_name:
        # print(scene[landsat_product_id])
        ID = scene['display_id']
        # print(f"下载的数据为{scene['landsat_product_id']}")
        if not os.path.isfile(output_dir + ID+".zip") or os.path.isfile(output_dir + ID+".tar") or os.path.isfile(output_dir + ID+".tar.gz"):
            print(f'本地无:{output_dir + ID}的完整文件，尝试下载')
            Earth_Down.download(identifier=ID, output_dir=output_dir)
    Earth_Down.logout()



if __name__ == '__main__':
    #https://earthexplorer.usgs.gov/的账号及密码
    username = '***********'
    password = '************'
    # 产品类别（详细请看表）
    product = 'landsat_ot_c2_l2'
    # 下载范围https://geojson.io/#map=2/20.0/0.0
    geojson_path = r"G:\哨兵数据库\珠江口\2016\map.geojson"
    # 起始及结束时间
    start_date = '2000-01-01'
    end_date = '2022-12-31'
    # 云量大小（<=cloud_max%）
    cloud_max = 90
    # 输出文件夹
    output_dir = r'G:\Lansat数据下载'
    #------------------------------**********----------------------------
    xmin, xmax, ymin, ymax = circumscribed_rectangle(geojson_path)
    Landsat_name = request_UCGS(username, password, product, xmin, ymin, xmax, ymax, start_date, end_date, cloud_max)
    download_UCGS_data(username, password, Landsat_name, output_dir)
```



## 三、后记

下载示例如下图

![image-20221008145219244](https://gitee.com/long_chaohuo/images/raw/master/image-20221008145219244.png)



### 使用

下载及解压该文件后以项目形式打开，在主函数入口`main.py`填写相关参数即可运行（跑不通的找我！！！）

![image-20221008145315830](https://gitee.com/long_chaohuo/images/raw/master/image-20221008145315830.png)

![image-20221008145151649](https://gitee.com/long_chaohuo/images/raw/master/image-20221008145151649.png)



若想获取python源脚本和栅格示例数据，可关注公众号，后台回复【landsatxplore】即可获得下载链接。

**参考：**

【1】https://gdal.org/index.html 

【2】https://github.com/yannforget/landsatxplore

