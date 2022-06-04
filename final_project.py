import findspark

findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Window, Row
import pyspark.sql.functions as func
from itertools import islice
from pyspark.sql.types import StringType
import folium
import os

# spark 入口
conf = SparkConf().setMaster("local").setAppName("Final_Project"). \
    set("spark.executor.heartbeatInterval", "200000"). \
    set("spark.network.timeout", "300000")

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# 建立窗口对象，因源数据来自同一车辆，不需要进行分类处理，partitionBy设为空。
my_window = Window.partitionBy().orderBy("date-time")


# 速度计算函数, 直接在窗口上进行操作
def cal_speed(lat1, lon1, lat2, lon2, dtime):
    """
    :param lat1: 前面一个点的纬度
    :param lon1: 前面一个点的经度
    :param lat2: 后面一个点的纬度
    :param lon2: 后面一个点的经度
    :param dtime: 前后两点的时间差
    :return: 保留五位小数的速度
    """
    pi = 3.1415926
    lat1 = lat1 / 180 * pi
    lon1 = lon1 / 180 * pi
    lat2 = lat2 / 180 * pi
    lon2 = lon2 / 180 * pi

    earth_r = 6371393
    dlat = func.abs(lat2 - lat1)  # 纬度差
    dlon = func.abs(lon2 - lon1)  # 经度差
    # 半正矢公式
    a = (func.sin(dlat / 2)) ** 2 + func.cos(lat1) * func.cos(
        lat2) * (func.sin(dlon / 2)) ** 2
    # c = 2 * func.atan2(func.sqrt(a), func.sqrt(1 - a))
    c = 2 * func.asin(func.sqrt(a))
    dist = earth_r * c
    # 速度保留五位小数
    speed = func.round(dist / dtime, 5)

    return speed


# 加速度计算函数
def cal_acceleration(speed1, speed2, dtime):
    """
    :param speed1: 前一个点的速度
    :param speed2: 后一个点的速度
    :param dtime: 前后点的时间差
    :return: 保留五位小数的加速度
    """
    return func.round((speed2 - speed1) / dtime, 5)


# 初始化df, 包括读取文件, 列处理等步骤
def df_init(path):
    # 读取文件
    data_lines = sc.textFile(path)

    # 读取了经度、纬度、日期和时间，并将时间和日期合并为一列
    dataSet = data_lines.map(lambda line: line.split(",")).mapPartitionsWithIndex(
        lambda i, test_iter: islice(test_iter, 6, None) if i == 0 else test_iter). \
        map(lambda x: {"latitude": float(x[0]), "longitude": float(x[1]),
                       "date-time": f"{x[5]} {x[6]}"}). \
        map(lambda p: Row(**p)).toDF()

    # 添加时间戳列, 序号列
    dataSet = dataSet. \
        withColumn("unix_time", func.unix_timestamp("date-time", "yyyy-MM-dd HH:mm:ss")). \
        withColumn("id", func.monotonically_increasing_id())
    # 计算前后点的时间差，并添加至df中
    dataSet = dataSet.withColumn("dtime", dataSet.unix_time - func.lag("unix_time", 1).over(my_window))
    # 将时间差为空的值赋值0
    dataSet = dataSet.fillna(0, subset=["dtime"])

    return dataSet


# 确定停留点速度阈值
def determine_threshold(df, rate=0.0, default_thres=0.4):
    """
    :param df: 输入的DataFrmae
    :param rate: 速度排名(从低到高)所占百分比
    :param default_thres: 默认的速度阈值为0.4
    :return: 返回速度阈值
    """
    # 若rate不为0, 根据rate确定速度阈值
    if rate != 0.0:
        df_len = df.count() # df长度
        speed_num = int(df_len * (1 - rate)) # 速度阈值的排名
        # 将速度自慢到快的排名
        df = df.select("speed", func.rank().over(my_window).alias("rank"))
        rank = df.select("speed").collect()
        rank_list = [row[0] for row in rank] # 速度排名列表
        threshold = rank_list[speed_num]
        return threshold
    # 若rate等于0, 返回给定的速度阈值
    else:
        return default_thres


# 将df中的列表转为string, 否则无法直接写入csv
def array_to_string(my_list):
    return '[' + ','.join([str(elem) for elem in my_list]) + ']'


# 绘制坐标点的轨迹图
def draw_gps(trajectory, color):
    # 中心区域的确定
    m1 = folium.Map(trajectory[0], zoom_start=15, attr='default')

    folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
        trajectory,  # 将坐标点连接起来
        weight=3,  # 线的粗细为3
        color=color,  # 线的颜色为橙色
        opacity=0.8  # 线的透明度
    ).add_to(m1)  # 将这条线添加到刚才的区域m1内

    # 起点，终点
    folium.Marker(trajectory[0], popup='<b>Starting Point</b>').add_to(m1)
    folium.Marker(trajectory[-1], popup='<b>Starting Point</b>').add_to(m1)

    # 将结果以HTML形式保存到指定路径
    m1.save(os.path.join('data/', 'acceleration_trajectory.HTML'))


# 添加速度列(将起点和终点速度置为0)
def question1_speed(df):
    row_number = df.count() # df总行数
    df = df.withColumn("speed",
        func.when(((func.col("id") < row_number - 1) & (func.col("id") > 0)),
        cal_speed(func.lag("latitude", 1).over(my_window),
            func.lag("longitude", 1).over(my_window),
            df.latitude, df.longitude, df.dtime)). \
        otherwise(0))
    return df


# 停留点分析
def question2_stop_pts(df, speed_threshold=0.4):
    # 添加用于判断是否是停留点的列
    df = df.withColumn("is_stop",
        ((func.lead("speed", 1).over(my_window) < speed_threshold)|
            (func.col("speed") < speed_threshold)))

    return df


# 加速区间分析
def question3_acceleration(df, speed_threshold=0.4):
    # # 加速度计算
    # df = df.withColumn("acceleration",
    #     func.when(((func.col("id") > 0)&(~(func.lag("is_stop").over(my_window)))),
    #         cal_acceleration(func.lag("speed", 1).over(my_window), df.speed, df.dtime)).\
    #     otherwise(0))

    # 加速度计算
    df = df.withColumn("acceleration",
        func.when(((func.col("speed")>=speed_threshold) & (~((func.lead("is_stop", 1).over(my_window)) &
            (func.lag("is_stop", 1).over(my_window))))),
            cal_acceleration(func.lag("speed", 1).over(my_window), df.speed, df.dtime)).\
        otherwise(0))

    # 删除时间戳和时间差两列
    df = df.drop("dtime").drop("unix_time")

    # 判断加速度的正负, 正的为1, 负的为-1
    df = df.withColumn("flag",
        func.when(df.acceleration > 0, 1).
        otherwise(-1))

    # 判断加速度是否连续(符号是否连续)
    df = df.withColumn("flagChange",
        (func.col("flag") != func.lag("flag").over(my_window)).cast("int")
    ).fillna(
        0,
        subset=["flagChange"]
    )

    # 判断是否是加速或减速区间
    df = df.withColumn("indicator",
        func.when((func.col("id") < df.count() - 1),
            (~((func.lead("flagChange", 1).over(my_window) == 1) & (
                 func.col("flagchange") == 1)))). \
        otherwise(False))

    # # 若要进行加减速区间聚类, 应当使用下面注释的部分
    # df = df.filter(df["indicator"])

    # 增加两列，判断是加速度区间还是减速度区间
    df = df.withColumn("is_positive", (func.col("acceleration") > 0) & (func.col("indicator"))).\
        withColumn("is_negative", (func.col("acceleration") < 0) & (func.col("indicator")))

    return df


# 区间聚类(将加速点合并为一个列表)
def clustering(df):
    # 标记区间号
    df = df.withColumn(
        "interval_number",
        func.when((func.col("flag") == func.lead("flag", 1).over(my_window))|\
             (func.col("flag") == func.lag("flag", 1).over(my_window)),
             func.sum(func.col("flagChange")). \
             over(my_window.rangeBetween(Window.unboundedPreceding, 0))).otherwise(0))
    # 筛除区间号为0的点
    df = df.filter(df["interval_number"] != 0)

    # 根据区间号，将区间号合并为列表
    df = df.groupby('interval_number').\
        agg(func.collect_list('id').alias("id"),
        func.min("date-time").alias("开始时间"),
        func.max("date-time").alias("结束时间"),
        func.sum("flag").alias("加速点(该列绝对值为点数,正负代表加速度符号)"))

    # 将列表转为字符串类型
    array_to_string_udf = func.udf(array_to_string, StringType())

    dataSet = df.withColumn('区间包含的点', array_to_string_udf(df["id"]))
    path = "data/clustering.csv"
    dataSet.drop("id").write.mode("overwrite").option("header", "true"). \
        option("encoding", "utf-8").csv(path)

    return df


# 读取文件的路径
file_path = "data/20080428112704.plt"
# 保存的csv路径
csv_path = "data/test.csv"

# 初始化df
dataSet = df_init(file_path)

# 第一问
dataSet = question1_speed(dataSet)

# 速度阈值
speed_threshold = determine_threshold(dataSet, 0.0, 0.4)

# 第二问
dataSet = question2_stop_pts(dataSet, speed_threshold)

# 第三问
dataSet = question3_acceleration(dataSet, speed_threshold)

# clustering(dataSet)

# 将加速和减速的坐标取出
a_list = dataSet. \
    select(func.col("latitude"), func.col("longitude")) \
    .rdd.map(lambda x: [x[0], x[1]]).collect()

# 绘制轨迹图
draw_gps(a_list, 'red')


# 删除无用的列
resultDF = dataSet.drop("stop_flag").drop("flag").drop("flagChange").drop("indicator")
resultDF.show()

# 保存结果到csv(修改列的名称为中文)
resultDF.drop("id"). \
    withColumnRenamed("date-time", "时间"). \
    withColumnRenamed("speed", "速度"). \
    withColumnRenamed("is_stop", "是否是停留点"). \
    withColumnRenamed("is_positive", "是否在加速区间"). \
    withColumnRenamed("is_negative", "是否在减速区间"). \
    withColumnRenamed("acceleration", "加速度"). \
    withColumnRenamed("latitude", "纬度"). \
    withColumnRenamed("longitude", "经度"). \
    write.mode("append").option("header", "true").option("encoding", "utf-8").csv(csv_path)
