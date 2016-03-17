package liusb.github.com

import org.apache.log4j.{Level, Logger}
import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.{Normalizer, StandardScaler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.sql.types.{IntegerType, DoubleType, StructField, StructType}


object LinerClassifier {

  //    // Normalize each Vector using $L^1$ norm.
  //    var trainData = new Normalizer()
  //      .setInputCol("indexedFeatures")
  //      .setOutputCol("normFeatures")
  //      .setP(1.0).transform(trainRawData)
  //    trainData = new StandardScaler()
  //      .setInputCol("normFeatures")
  //      .setOutputCol("scaledFeatures")
  //      .setWithStd(true).setWithMean(false).
  //      fit(trainData).transform(trainData)
  //
  //    var testData = new Normalizer()
  //      .setInputCol("indexedFeatures")
  //      .setOutputCol("normFeatures")
  //      .setP(1.0).transform(testRawData)
  //    testData = new StandardScaler()
  //      .setInputCol("normFeatures")
  //      .setOutputCol("scaledFeatures")
  //      .setWithStd(true).setWithMean(false).
  //      fit(testData).transform(testData)

  def run(args: Array[String]) {

    // 设置日志为警告级别，以免打印过多信息
    Logger.getLogger("org").setLevel(Level.WARN)

    // 配置并初始化上下文
    val conf = new SparkConf().setMaster("local[3]").setAppName("CashBus")
    conf.setJars(Seq("./out\\artifacts\\Spark_jar\\Spark.jar"))
    System.setProperty("hadoop.home.dir", "F:\\RunEnv\\hadoop-2.7.1")
    System.setProperty("user.name", "hadoop")
    val spark = new SparkContext(conf)

    // 从csv文件读取数据
    val sqlContext= new SQLContext(spark)
    val train_x = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")//.option("inferSchema", "true")
      .load("F:\\Workspace\\IdeaWorkspace\\Spark\\data\\train_x.csv")
    val train_y = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")//.option("inferSchema", "true")
      .load("F:\\Workspace\\IdeaWorkspace\\Spark\\data\\train_y.csv")
    val test_x = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")//.option("inferSchema", "true")
      .load("F:\\Workspace\\IdeaWorkspace\\Spark\\data\\test_x.csv")
    val features_type = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")//.option("inferSchema", "true")
      .load("F:\\Workspace\\IdeaWorkspace\\Spark\\data\\features_type.csv")

    // 合并测试数据和训练数据
    val allData = train_x.unionAll(test_x)

    // 处理类别特征
    val categoryFeatures = features_type.filter(features_type("type") === "category")
      .map(row => row(0).toString).collect()
    val indexTransformers: Array[org.apache.spark.ml.PipelineStage] = categoryFeatures.map(
      feature => new StringIndexer()
        .setInputCol(feature)
        .setOutputCol(s"${feature}Index")
    )
    val indexPipeline = new Pipeline().setStages(indexTransformers)
    val allDataIndexed = indexPipeline.fit(allData).transform(allData)

    val indexColumns  = allDataIndexed.columns.filter(x => x contains "Index")
    val oneHotEncoders: Array[org.apache.spark.ml.PipelineStage] = indexColumns.map(
      feature => new OneHotEncoder()
        .setInputCol(feature)
        .setOutputCol(s"${feature}Vec")
    )
    val oneHotPipeline = new Pipeline().setStages(oneHotEncoders)
    val allDataEncoded = oneHotPipeline.fit(allDataIndexed).transform(allDataIndexed)
    allDataEncoded.show()

    val joinData = allDataEncoded.join(train_y, Seq("uid"), "left_outer")
    // 将训练数据合成一个表
    val trainData = joinData.filter("y is not null")
    val testData = joinData.filter("y is null")

    // 忽略的列索引
    val trainIgnoredInd = List("uid", "y")
    // 特征列索引
    val trainFeatureInd = trainData.columns.diff(trainIgnoredInd).map(trainData.columns.indexOf(_))
    val trainLabelInd = trainData.columns.indexOf("y")
    val labeledTrainData = trainData.rdd.map(row => LabeledPoint(
      row.getString(trainLabelInd).toDouble,
      Vectors.dense(trainFeatureInd.map(row.get(_).asInstanceOf[Double]))))

    val testIgnoredInd = List("uid", "y")
    val testFeatureInd = testData.columns.diff(testIgnoredInd).map(testData.columns.indexOf(_))
    val testLabelInd = testData.columns.indexOf("uid")
    val labeledTestData = testData.rdd.map(row => LabeledPoint(
      row.getString(testLabelInd).toDouble,
      Vectors.dense(testFeatureInd.map(row.get(_).asInstanceOf[Double]))))

    // 特征归一化
    val normalizer = new Normalizer()
    // 特征标准化
    val trainVectors = labeledTrainData.map(p => p.features)
    val trainStandScaler = new StandardScaler(withMean = true, withStd = true).fit(trainVectors)
    val standTrain = labeledTrainData.map(row =>
      LabeledPoint(row.label, normalizer.transform(trainStandScaler.transform(row.features)))
    )
    val testVectors = labeledTestData.map(p => p.features)
    val testStandScaler = new StandardScaler(withMean = true, withStd = true).fit(testVectors)
    val standTest = labeledTestData.map(row =>
      LabeledPoint(row.label, normalizer.transform(testStandScaler.transform(row.features)))
    )

    // 逻辑回归
    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(standTrain)
    // 清除阀值
    lrModel.clearThreshold()

    // 对测试样本进行测试
    val schema = StructType(Seq(
      StructField("uid", IntegerType, true),
      StructField("score", DoubleType, true)))
    val result = standTest.map(row => Row(
      row.label.toInt, lrModel.predict(row.features)))
    val resultDF = sqlContext.createDataFrame(result, schema)
    resultDF.registerTempTable("result")
    resultDF.repartition(1).write
      .format("com.databricks.spark.csv")
      .mode("overwrite")
      .option("header", "true")
      .save("F:\\Workspace\\IdeaWorkspace\\Spark\\data\\result")

//    // 对模型评估
//    val scoreAndLabels = result.map(row => (row.getInt(0).toDouble, row.getDouble(1)))
//    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//    println(metrics.areaUnderROC())

    spark.stop()
  }

}