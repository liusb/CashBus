package liusb.github.com

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{DecisionTreeClassifier,
GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidatorModel,
ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.{PipelineStage, Pipeline}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}

object RunApp {

  def runCrossValidator(data: DataFrame, classifier: PipelineStage,
                        paramGrid: Array[ParamMap]):
  CrossValidatorModel = {
    // 训练数据
    val trainData = data.filter("label >= 0")

    // 分类特征索引化处理
    val featuresIndexer = new VectorIndexer()
      .setInputCol("features").setOutputCol("indexedFeatures")
      .setMaxCategories(20).fit(data)

    // 标签索引化处理
    val labelIndexer = new StringIndexer()
      .setInputCol("label").setOutputCol("indexedLabel").fit(trainData)

    // 索引化标签还原
    val labelConverter = new IndexToString()
      .setInputCol("prediction").setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // 串成管道
    val pipeline = new Pipeline().setStages(
      Array(featuresIndexer, labelIndexer, classifier, labelConverter))

    // 生成交叉验证
    val crossValidator = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(3)

    // 交叉验证训练模型
    val cvModel = crossValidator.fit(trainData)

    cvModel
  }

  def runCvModel(data: DataFrame, cvModel: CrossValidatorModel):
  DataFrame = {
    // 分开训练数据和测试数据
    val testData = data.filter("label < 0")

    // 使用模型预测结果
    val predictions = cvModel.transform(testData)

    val result = predictions.select("uid", "probability")
    result.show()

    result
  }

  def cvDecisionTree(data: DataFrame): DataFrame = {
    // 决策树模型分类器
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    // 生成参数表格
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxBins, Array(32, 40, 48))
      .addGrid(dt.impurity, Array("entropy", "gini"))
      .addGrid(dt.maxDepth, Array(5, 10, 15)).build()

    // 建立交叉验证模型
    val cvModel = runCrossValidator(data, dt, paramGrid)

    // 运行模型
    val result = runCvModel(data, cvModel)

    result
  }

  def cvRandomForest(data: DataFrame): DataFrame = {
    // 随机森林分类器
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    // 生成参数表格
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(32, 40, 48))
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .addGrid(rf.numTrees, Array(15, 20, 25))
      .addGrid(rf.maxDepth, Array(5, 10, 15)).build()

    // 建立交叉验证模型
    val cvModel = runCrossValidator(data, rf, paramGrid)

    // 运行模型
    val result = runCvModel(data, cvModel)

    result
  }

  def cvGBT(data: DataFrame): DataFrame = {
    // 梯度提升树分类器
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    // 生成参数表格
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxBins, Array(32, 40, 48))
      .addGrid(gbt.impurity, Array("entropy", "gini"))
      .addGrid(gbt.maxDepth, Array(5, 10, 15)).build()

    //建立交叉验证模型
    val cvModel = runCrossValidator(data, gbt, paramGrid)

    // 运行模型
    val result = runCvModel(data, cvModel)

    result
  }

  def main(args: Array[String]) {

    // 设置日志为警告级别，以免打印过多信息
    Logger.getLogger("org").setLevel(Level.WARN)

    // 配置并初始化上下文
    val conf = new SparkConf().setMaster("local[3]").setAppName("Spark")
    conf.setJars(Seq("./out\\artifacts\\Spark_jar\\Spark.jar"))
    System.setProperty("hadoop.home.dir", "F:\\RunEnv\\hadoop-2.7.1")
    System.setProperty("user.name", "hadoop")
    val spark = new SparkContext(conf)
    val sqlContext = new SQLContext(spark)
    val dataHelper = new DataHelper(sqlContext)

    // 加载数据
    val rawData = dataHelper.loadData()

    // 格式化数据
    val mlData = dataHelper.formatData(rawData)

    // 使用决策树预测结果
    val dtResult = cvDecisionTree(mlData)
    val dtResultDf = sqlContext.createDataFrame(dtResult.map(row =>
      (row.getInt(0), row.getAs[DenseVector](1)(0))
      )).toDF("uid", "score")
    dataHelper.saveData(dtResultDf, "dtResult")

    // 使用随机森林预测结果
    val rfResult = cvRandomForest(mlData)
    // 保存结果
    val rfResultDF = sqlContext.createDataFrame(rfResult.map(row =>
      (row.getInt(0), row.getAs[DenseVector](1)(0))
      )).toDF("uid", "score")
    dataHelper.saveData(rfResultDF, "rfResult")

    // 使用梯度上升预测结果
    val gbtResult = cvGBT(mlData)
    // 保存结果
    val gbtResultDF = sqlContext.createDataFrame(gbtResult.map(row =>
      (row.getInt(0), row.getAs[DenseVector](1)(0))
      )).toDF("uid", "score")
    dataHelper.saveData(gbtResultDF, "gbtResult")

    spark.stop()
  }

}
