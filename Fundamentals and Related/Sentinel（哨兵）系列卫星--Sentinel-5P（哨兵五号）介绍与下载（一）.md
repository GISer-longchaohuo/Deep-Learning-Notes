## Sentinel（哨兵）系列卫星--Sentinel-5P（哨兵五号）介绍与下载（一）

![image-20221029113911521](https://gitee.com/long_chaohuo/images_1/raw/master/image-20221029113911521.png)

## 1. 概述

Sentinel-5 Precursor也称为Sentinel-5P，于2017年10月13日发射的一颗全球大气污染监测卫星（低地球轨道卫星，轨道高度为824km）。该卫星搭载了最先进的对流层观测仪（Tropomi传感器），以大范围的天底模式观测地球。 其主要目标是以高时空分辨率进行大气观测，用于空气质量，臭氧和紫外线辐射以及气候监测和预测，并每日提供全球覆盖数据。

> Sentinel-5P 旨在减少 Envisat 卫星（尤其是 Sciamachy 仪器）与 Sentinel-5 发射之间的数据差距，并在 MetOp 上补充 GOME-2。未来，地球静止 Sentinel-4 和极地轨道 Sentinel-5 任务将监测哥白尼大气服务的大气成分。 这两项任务都将在 Eumetsat 运营的气象卫星上进行，在此之前，Sentinel-5P 任务在监测和跟踪空气污染方面发挥着关键作用。
>



![image-20221029131052395](../AppData/Roaming/Typora/typora-user-images/image-20221029131052395.png)



##  2. 传感器

TROPOMI是目前世界上技术===最先进==、==空间分辨率最高==的==大气监测光谱==仪。成像幅宽达2600km，每日覆盖全球各地，成像分辨率达==7km×3.5km==。

![](https://gitee.com/long_chaohuo/images_1/raw/master/image-20221029111048374.png)

| 参数               | 详情                                                         |
| ------------------ | ------------------------------------------------------------ |
| 类型               | 无源光栅成像光谱仪（ passive grating imaging spectrometer）  |
| 条带宽度           | 2600km                                                       |
| 空间采样           | 7×7km²                                                       |
| 光谱               | 4 个光谱仪，每个光谱仪以电子方式分为两个波段。紫外线、可见光（270-495nm）、近红外（675-775nm）、短波红外（2305-2385nm）。注意：每个波段各两个 |
| 辐射精度（绝对值） | 测量地球光谱反射率的 1.6% （SWIR） 至 1.9% （UV）            |
| 总质量             | 204.3千克                                                    |
| 尺寸               | 1.40 x 0.65 x 0.75 m                                         |
| 设计寿命           | 7年（携带燃料约为10年）                                      |
| 平均功耗           | 155 W                                                        |
| 生成的数据量       | 每个完整轨道 139 Gb。                                        |





## 3. 可供用户使用的产品

![image-20221029114346790](https://gitee.com/long_chaohuo/images_1/raw/master/image-20221029114346790.png)



### 0 级产品

时间顺序、无时间重叠的原始卫星遥测，包括 4 台光谱仪的传感器数据，用于大气和校准测量。还包括工程（例如内务管理）数据、卫星辅助（例如位置、姿态）数据和地面处理和监测任务所需的采集元数据。==0 级数据不向公众提供==。



### 1B 级产品

地理定位和辐射校正的大气层顶部所有光谱波段的地球辐射，以及太阳辐照度。

| 文件类型   | 光谱仪（波段） | 波度光谱范围（nm） | 用户文档 |
| ---------- | -------------- | ------------------ | -------- |
| L1B_RA_BD1 | UV             | 270 - 300          |          |
| L1B_RA_BD2 | UV             | 300 - 320          |          |
| L1B_RA_BD3 | UVIS           | 320 - 405          |          |
| L1B_RA_BD4 | UVIS           | 405 - 500          |          |
| L1B_RA_BD5 | NIR            | 675 - 725          |          |
| L1B_RA_BD6 | NIR            | 725 - 775          |          |
| L1B_RA_BD7 | SWIR           | 2305-2345          |          |
| L1B_RA_BD8 | SWIR           | 2345-2385          |          |
| L1B_IR_UVN | UVN            | 270-775            |          |
| L1B_IR_SIR | SWIR           | 2305-2385          |          |

> **级别 1B – 用户技术文档**
>
> - **MDS**（元数据规范）：包含 TROPOMI 1b 级数据产品的元数据规范
> - **IODS**（输入输出数据规范）：从 0 级到 1b 级处理产生的产品描述
> - **ATBD**（算法理论基础文档）：0 级到 1b 级数据处理中使用的算法的高级描述
> - **PRF**（产品自述文件）：不同产品版本之间的更改描述和整体质量信息（发布后几个月提供）





### 2 级产品

- 臭氧、二氧化硫、二氧化氮、一氧化碳、甲醛和甲烷的地理定位总柱数

- 臭氧对流层的地理定位柱

- 臭氧的地理定位垂直剖面

- 地理定位云和气溶胶信息（例如吸收气溶胶指数和气溶胶层高度）

  

> L2数据产品又分为了三种数据流：近实时数据流（near-real-time, **NRTI**），卫星成像3小时后即可获取，数据可能不完整或存在质量缺陷；
>
> 离线数据流（Offline, OFFL），不同数据获取时间不一致；
>
> - 12小时内对1B级数据处理
> - 对于甲烷、对流层臭氧和总二氧化氮柱在传感后约5天内校正。
>
> 再次处理数据流（Reprocessing, RPRO），有些数据可能经过了多次处理，获得的最新的质量最佳的版本。一般情况下，长期的时序变化研究不可以混用不同级别的数据流，推荐使用最新的RPRO数据以保证数据质量。



| **产品类型**            | **参数**                                                     | **用户文档**                                                 |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| L2__O3____              | Ozone (O3) total column                                      | [PRF-O3-NRTI](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Total-Ozone-Level-2-Product-Readme-File), [PRF-03-OFFL](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Readme-OFFL-Total-Ozone.pdf), [PUM-O3](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Ozone-Total-Column), [ATBD-O3](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-Total-Ozone), [IODD-UPAS](https://sentinels.copernicus.eu/documents/247904/2506504/S5P-input-output-data-spec-TROPOMI-L2-data-processor.pdf) |
| L2__O3_TCL              | Ozone (O3) tropospheric column                               | [PRF-03-T](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-OFFL-Tropospheric-Ozone-Product-Readme-File.pdf), [PUM-O3_T](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-OFFL-Tropospheric-Ozone-Product-Readme-File.pdf), [ATBD-O3_T](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-ATBD-TROPOMI-Tropospheric-Ozone), [IODD-UPAS](https://sentinels.copernicus.eu/documents/247904/2506504/S5P-input-output-data-spec-TROPOMI-L2-data-processor.pdf) |
| L2__O3__PR              | Ozone (O3) profile                                           | [PRF-O3_PR](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Ozone-profile-Product-Readme-File.pdf), [IODD-NL](https://sentinels.copernicus.eu/documents/247904/3119978/Sentinel-5P-Level-2-Input-Output-Data-Definition), [PUM-O3_PR](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-TROPOMI-Level-2-Product-User-Manual-Ozone-profiles.pdf), [ATBD-03_PR](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-Ozone-Profile.pdf) |
| L2__NO2___              | Nitrogen Dioxide (NO2), total and tropospheric columns       | [PRF-NO2](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Nitrogen-Dioxide-Level-2-Product-Readme-File), [PUM-NO2](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Nitrogen-Dioxide.pdf), [ATBD-NO2](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-NO2-data-products), [IODD-NL](https://sentinels.copernicus.eu/documents/247904/3119978/Sentinel-5P-Level-2-Input-Output-Data-Definition) |
| L2__SO2___              | Sulfur Dioxide (SO2) total column                            | [PRF-SO2](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Sulphur-Dioxide-Readme.pdf), [PUM-SO2](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Sulphur-Dioxide), [ATBD-SO2](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-ATBD-SO2-TROPOMI), [IODD-UPAS](https://sentinels.copernicus.eu/documents/247904/2506504/S5P-input-output-data-spec-TROPOMI-L2-data-processor.pdf) |
| L2__CO____              | Carbon Monoxide (CO) total column                            | [PRF-CO](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Carbon-Monoxide-Level-2-Product-Readme-File), [PUM-CO](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Carbon-Monoxide.pdf), [ATBD-CO](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-Carbon-Monoxide-Total-Column-Retrieval.pdf), [IODD-NL](https://sentinels.copernicus.eu/documents/247904/3119978/Sentinel-5P-Level-2-Input-Output-Data-Definition) |
| L2__CH4___              | Methane (CH4) total column                                   | [PRF-CH4](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Methane-Product-Readme-File), [PUM-CH4](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Methane.pdf), [ATBD-CH4](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-Methane-retrieval.pdf), [IODD-NL](https://sentinels.copernicus.eu/documents/247904/3119978/Sentinel-5P-Level-2-Input-Output-Data-Definition) |
| L2__HCHO__              | Formaldehyde (HCHO) total column                             | [PRF-HCHO](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Formaldehyde-Readme.pdf), [PUM-HCHO](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Formaldehyde) , [ATBD-HCHO ](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-ATBD-HCHO-TROPOMI), [IODD-UPAS](https://sentinels.copernicus.eu/documents/247904/2506504/S5P-input-output-data-spec-TROPOMI-L2-data-processor.pdf) |
| L2__CLOUD_              | Cloud fraction, albedo, top pressure                         | [PRF-CL](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Cloud-Level-2-Product-Readme-File), [PUM-CL](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Cloud), [ATBD-CL](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-Clouds), [IODD-UPAS](https://sentinels.copernicus.eu/documents/247904/2506504/S5P-input-output-data-spec-TROPOMI-L2-data-processor.pdf) |
| L2__AER_AI              | UV Aerosol Index                                             | [PRF-AI](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Aerosol-Level-2-Product-Readme-File), [PUM-AI](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Aerosol-Index-product), [ATBD-AI](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-UV-Aerosol-Index.pdf), [IODD-NL](https://sentinels.copernicus.eu/documents/247904/3119978/Sentinel-5P-Level-2-Input-Output-Data-Definition) |
| L2__AER_LH              | Aerosol Layer Height (mid-level pressure)                    | [PRF-LH](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Aerosol-Layer-Height-Product-Readme-File.pdf), [PUM-LH](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-Aerosol-Layer-Height.pdf), [ATBD-LH ](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-TROPOMI-ATBD-Aerosol-Height), [IODD-NL](https://sentinels.copernicus.eu/documents/247904/3119978/Sentinel-5P-Level-2-Input-Output-Data-Definition) |
| UV product1             | Surface Irradiance/erythemal dose                            | -                                                            |
| L2__NP_BDx, x=3, 6, 7 2 | Suomi-NPP VIIRS Clouds                                       | [PRF-NPP](https://sentinels.copernicus.eu/documents/247904/3541451/Sentinel-5P-Mission-Performance-Centre-NPP-Cloud-Readme), [PUM-NPP](https://sentinels.copernicus.eu/documents/247904/2474726/Sentinel-5P-Level-2-Product-User-Manual-NPP-Cloud-product), [ATBD-NPP](https://sentinels.copernicus.eu/documents/247904/2476257/Sentinel-5P-NPP-ATBD-NPP-Clouds) |
| AUX_CTMFC AUX_CTMANA    | A-priori profile shapes for the NO2, HCHO and SO2 vertical column retrievals | [PUM](https://sentinels.copernicus.eu/documents/247904/2474726/PUM-for-the-TM5-NO2-SO2-and-HCHO-profile-auxiliary-support-product.pdf/de18a67f-feca-1424-0195-756c5a3df8df) |

>  **级别 2 – 用户技术文档**
>
> - **PUM**（产品用户信息）：有关 S5P/TROPOMI 2 级产品的技术特性信息
> - **ATBD**（算法理论基础文档）：有关检索算法的详细信息
> - **IODD**（输入输出数据定义）：S5P/TROPOMI 2 级产品的输入和输出数据描述
> - **PRF** - 描述不同产品版本和整体质量信息之间的变化



## 4.  应用领域

大气观测，用于空气质量，臭氧和紫外线辐射以及气候监测和预测



## 5. 下载方式

**目前所知道的下载渠道**

- 欧空局官网 ：https://s5phub.copernicus.eu/dhus/
- Google Earth Engine ：https://code.earthengine.google.com/
- PIE Engine ：https://engine.piesat.cn/engine/home
- sentinel-hub：https://apps.sentinel-hub.com/
- python API下载



### 5.1 欧空局官网 

在https://s5phub.copernicus.eu/dhus/网站上，帐号密码均为：s5pguest



### 5.2 Google Earth Engine 

下载方式参考哨兵2



### 5.3 PIE Engine 

https://engine.piesat.cn/dataset-list

### 5.4 python API下载

```python
# -*- coding: utf-8 -*-
'''
@author: LongChaohuo
@contact: 460958592@qq.com
@software: PyCharm
@file: Sentinel5P_Download
@time: 2022/08/29 晚上 22:37

'''
# 导入相应的模块
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import time
import datetime
import os
import pathlib

# sentinel-5p数据下载函数
def sentinel_5_data(user_name,user_password,sentinel_url,start_date,end_date):
    '''
    :param user_name: 哥白尼数据访问中心账号
    :param user_password:哥白尼数据访问中心账号密码
    :param sentinel_url: 哥白尼数据访问中心URL
    :param start_date: 数据搜索起始时间
    :param end_date: 数据搜索结束时间
    :param platformname: 卫星（sentinel卫星）
    :param max_cloud: 最大云量
    :param download_dir 数据下载到本地地址
    '''
    # 获取文件下全部json文件(绝对路径)
    # 兴趣区（https://geojson.io/）
    while True:
        areas = list(pathlib.Path(path).glob('*.geojson'))
        print(areas)
        #遍历json文件
        for AOI in areas:
            # 拼接路径
            filename = os.path.join(path, AOI)
            print(f'兴趣区路径为：{filename}')
            #获取兴趣区名称
            out_file = filename.split(".geojson")[0]
            print(f'输出路径为：{out_file}')
            # 根据文件名创建文件夹
            if not os.path.exists(out_file):
                os.makedirs(out_file)
            download_area = geojson_to_wkt(read_geojson(filename))
            #  设置连接sentinel数据中心的相关参数
            sentinel_API = SentinelAPI(user_name,user_password,sentinel_url)
            sentinel_products = sentinel_API.query(
                    area=download_area,#下载兴趣区域
                    date=(start_date,end_date),#时间间隔
                    # platformname= platformname, #卫星
                    producttype = producttype,#数据等级
                    # cloudcoverpercentage = (0, max_cloud) #云量设置
            )
            print(f'共有产品{len(sentinel_products)}个')

            #遍历查询列表的每一个数据产品
            for product in sentinel_products:
                #根据产品元数据字典
                product_dict = sentinel_API.get_product_odata(product)
                # print(product_dict)
                #获取产品id
                product_ID = product_dict['id']
                # 获取产品文件title
                product_title = product_dict['title']
                print(f'产品名称为{product_title}')
                #通过产品id查询产品
                product_info = sentinel_API.get_product_odata(product_ID)
                #获取产品的在线及离线状态信息
                product_online = product_info['Online']
                #判断产品是否在线
                if product_online: #在线
                    print(f'产品为{product_title}在线')
                    #判断本地是否有完整产品
                    if not os.path.isfile(out_file + os.sep + product_title + ".nc"):
                        print(f'本地无 {product_title}.zip 的完整文件')
                        #通过产品id下载产品
                        sentinel_API.download(product_ID,directory_path= out_file)
                else:#产品不在线
                    print(f'产品为{product_title}不在线')
                    # 判断本地是否有完整产品
                    if not os.path.isfile(out_file + os.sep + product_title + ".zip"):
                        print(f'本地无{product_title}.zip 的完整文件，尝试触发 LongTermArchive 库')
                        try: # 尝试触发
                            sentinel_API.download(product_info['id'], directory_path= out_file)
                            sentinel_API.trigger_offline_retrieval(product_ID)
                            break  #成功则跳出
                        except Exception as e:
                            print(f'[错误]请求失败,休眠 15分钟后重试（当前时间：{datetime.datetime.now()}')
            # 每隔15min才能提交一次请求
            time.sleep(60 * 15)


if __name__ == '__main__':
    # 存放数据地址
    path = r'G:\sentinel_data download\research area'
    # 账号及密码（固定的,全世界都一样）
    user_name='s5pguest'
    user_password = "s5pguest"
    # 哥白尼数据访问5p的URL（与1/2/3不同）
    sentinel_url = 'https://s5phub.copernicus.eu/dhus'
    # 时间间隔
    start_date ='20220815'
    end_date ='20220928'
    # 下载Sentinel卫星
    # platformname = 'Sentinel-5P'
    # 数据类型
    '''
    可选                       
    L1B_IR_SIR, L1B_IR_UVN, L1B_RA_BD1, L1B_RA_BD2, L1B_RA_BD3, L1B_RA_BD4, L1B_RA_BD5, L1B_RA_BD6, L1B_RA_BD7, 
    L1B_RA_BD8, L2__AER_AI, L2__AER_LH, L2__CH4, L2__CLOUD_, L2__CO____, L2__HCHO__, L2__NO2___, L2__NP_BD3, L2__NP_BD6,
    L2__NP_BD7, L2__O3_TCL, L2__O3____, L2__SO2___, AUX_CTMFCT、AUX_CTMANA
     
    '''
    producttype = 'L2__NO2___'
    sentinel_5_data(user_name, user_password, sentinel_url, start_date, end_date)
```