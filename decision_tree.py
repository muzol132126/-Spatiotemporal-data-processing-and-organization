import findspark

findspark.init()
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row, SQLContext
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from itertools import islice


# 为每一个特征的不同属性赋值
work_type = {'Private': 1,
             'Self-emp-not-inc': 2,
             'Self-emp-inc': 3,
             'Federal-gov': 4,
             'Local-gov': 5,
             'State-gov': 6,
             'Without-pay': 7,
             'Never-worked': 8,
             '?': -1}
education = {'Bachelors': 1,
             'Some-college': 2,
             '11th': 3,
             'HS-grad': 4,
             'Prof-school': 5,
             'Assoc-acdm': 6,
             'Assoc-voc': 7,
             '9th': 8,
             '7th-8th': 9,
             '12th': 10,
             'Masters': 11,
             '1st-4th': 12,
             '10th': 13,
             'Doctorate': 14,
             '5th-6th': 15,
             'Preschool': 16,
             '?': -1}
marital_status = {'Married-civ-spouse': 1,
                  'Divorced': 2,
                  'Never-married': 3,
                  'Separated': 4,
                  'Widowed': 5,
                  'Married-spouse-absent': 6,
                  'Married-AF-spouse': 7,
                  '?': -1}
occupation = {'Tech-support': 1,
              'Craft-repair': 2,
              'Other-service': 3,
              'Sales': 4,
              'Exec-managerial': 5,
              'Prof-specialty': 6,
              'Handlers-cleaners': 7,
              'Machine-op-inspct': 8,
              'Adm-clerical': 9,
              'Farming-fishing': 10,
              'Transport-moving': 11,
              'Priv-house-serv': 12,
              'Protective-serv': 13,
              'Armed-Forces': 14,
              '?': -1}
relationship = {'Wife': 1,
                'Own-child': 2,
                'Husband': 3,
                'Not-in-family': 4,
                'Other-relative': 5,
                'Unmarried': 6,
                '?': -1}
race = {'White': 1,
        'Asian-Pac-Islander': 2,
        'Amer-Indian-Eskimo': 3,
        'Other': 4,
        'Black': 5,
        '?': -1}
sex = {'Female': 1,
       'Male': 2,
       '?': -1}
native_country = {'United-States': 1,
                  'Cambodia': 2,
                  'England': 3,
                  'Puerto-Rico': 4,
                  'Canada': 5,
                  'Germany': 6,
                  'Outlying-US(Guam-USVI-etc)': 7,
                  'India': 8,
                  'Japan': 9,
                  'Greece': 10,
                  'South': 11,
                  'China': 12,
                  'Cuba': 13,
                  'Iran': 14,
                  'Honduras': 15,
                  'Philippines': 16,
                  'Italy': 17,
                  'Poland': 18,
                  'Jamaica': 19,
                  'Vietnam': 20,
                  'Mexico': 21,
                  'Portugal': 22,
                  'Ireland': 23,
                  'France': 24,
                  'Dominican-Republic': 25,
                  'Laos': 26,
                  'Ecuador': 27,
                  'Taiwan': 28,
                  'Haiti': 29,
                  'Columbia': 30,
                  'Hungary': 31,
                  'Guatemala': 32,
                  'Nicaragua': 33,
                  'Scotland': 34,
                  'Thailand': 35,
                  'Yugoslavia': 36,
                  'El-Salvador': 37,
                  'Trinadad&Tobago': 38,
                  'Peru': 39,
                  'Hong': 40,
                  'Holand-Netherlands': 41,
                  '?': -1}

# 将数据转换为所需格式
def toDataModel(x):
    rel = {
        'features': Vectors.dense(int(x[0]),
                                  int(work_type[x[1]]),
                                  int(x[2]),
                                  int(education[x[3]]),
                                  int(x[4]),
                                  int(marital_status[x[5]]),
                                  int(occupation[x[6]]),
                                  int(relationship[x[7]]),
                                  int(race[x[8]]),
                                  int(sex[x[9]]),
                                  int(x[10]),
                                  int(x[11]),
                                  int(x[12]),
                                  int(native_country[x[13]])
                                  ),
        # 特征值索引
        'label': str(x[14])}
    return rel


# spark 入口
conf = SparkConf().setMaster("local").setAppName("Decision_Tree")
sc = SparkContext(conf=conf)  
sqlContext = SQLContext(sc)

# 读取训练样本，使用**把字典变成关键字参数传递，并转为DataFrame
# 读取3000行后的所有数据
data = sc.textFile("data/adult/adult_c.data").map(lambda line: line.split(', ')).mapPartitionsWithIndex(
    lambda i, test_iter: islice(test_iter, 3000, None) if i == 0 else test_iter).map(
    lambda p: Row(**toDataModel(p))).toDF()
# 读取前3000行
first_part_data = sc.textFile("data/adult/adult_c.data").map(lambda line: line.split(', ')).map(
    lambda p: Row(**toDataModel(p))).toDF().limit(3000)

# 将两个df合并
data = data.union(first_part_data)

# 字符串标签转索引标签
labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
# 将训练数据的属性值转化成数字编码index，以便后续其他数字编码的算法使用
featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
# 将索引标签转为原始字符串标签
labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
trainingData = data

# 读取测试数据并转为DataFrame
testData = sc.textFile("data/adult/adult_c.test").map(lambda line: line.split(', ')).mapPartitionsWithIndex(
    lambda i, test_iter: islice(test_iter, 10000, None) if i == 0 else test_iter).map(
    lambda p: Row(**toDataModel(p))).toDF()

# 定义分类器
dtClassifier = DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
# 构建机器学习流水线，训练
dtPipeline = Pipeline().setStages([labelIndexer, featureIndexer, dtClassifier, labelConverter])
dtPipelineModel = dtPipeline.fit(trainingData)
# 进行预测
dtPredictions = dtPipelineModel.transform(testData)
# 显示前30条数据
dtPredictions.select("predictedLabel", "label", "features").show(30)
# 结果评估
evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
# 分类精度
dtAccuracy = evaluator.evaluate(dtPredictions)
print(f"精度为: {dtAccuracy}")