# -Spatiotemporal-data-processing-and-organization
时空数据处理与组织实习
## 1. 开发
本程序基于的Windows下的Spark环境，采用Python编程。
数据采用微软的车辆轨迹数据-Geolife Trajectories中的编号为170号的车辆轨迹数据。

## 2. 功能
- **车辆速率计算**
- **车辆停留点分析**
- **车辆加减速分析**
- **车辆轨迹可视化**

## 3. 安装
首先应安装jdk1.8, Scala-2.12.5, Spark2.3.2, 搭建好开发环境。

## 4. 使用
本程序基于python 3.6.13构建，使用了findspark, Pyspark, Itertools, Folium, Os等第三方库或模块。要在Python中使用，请运行以下命令:
```
pip install findspark
pip install Itertools
pip install Folium
```

## 5. 函数说明
- cal_speed(lat1, lon1, lat2, lon2, dtime)：通过前后轨迹点的经纬度以及时间差计算速速
- cal_acceleration(speed1, speed2, dtime): 通过前后轨迹点的速度计算加速度
- df_init(path): 读取文件，做出一定处理，转为DataFrame
- determine_threshold(df, rate, default_thres): 确定停留点速度阈值
- array_to_string(my_list): 将列表转为字符串
- draw_gps(trajectory, color): 绘制轨迹图
- question1_speed(df): 计算速度，并添加到DataFrame中
- question2_stop_pts(df, speed_threshold): 分析停留点，并添加到DataFrame中
- question3_acceleration(df): 分析加速度片段，并添加到DataFrame中
- clustering(df): 将同一加速度片段里的点聚合成列表

## 6. 项目负责人
lty[muzol132126](https://github.com/muzol132126)
