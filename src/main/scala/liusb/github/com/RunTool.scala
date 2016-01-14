package liusb.github.com


import org.apache.log4j.{Level, Logger}
import org.apache.spark._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.sql.types.{IntegerType, DoubleType, StructField, StructType}


object RunTool {

  def main(args: Array[String]) {

    // 设置日志为警告级别，以免打印过多信息
    Logger.getLogger("org").setLevel(Level.WARN)

    val conf = new SparkConf().setMaster("local[3]").setAppName("CashBus")
    conf.setJars(Seq("./out\\artifacts\\Spark_jar\\Spark.jar"))
    System.setProperty("hadoop.home.dir", "F:\\RunEnv\\hadoop-2.7.1")
    System.setProperty("user.name", "hadoop")
    val spark = new SparkContext(conf)

    val sqlContext = new SQLContext(spark)

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

    // 将训练数据合成一个表
    val trainData = train_x.join(train_y, "uid")


    // 忽略的列索引
    val trainIgnoredInd = List("uid", "y")
    // 特征列索引
    val trainFeatureInd = trainData.columns.diff(trainIgnoredInd).map(trainData.columns.indexOf(_))
    val trainLabelInd = trainData.columns.indexOf("y")
    val labeledTrainData = trainData.rdd.map(row => LabeledPoint(
      row.getString(trainLabelInd).toDouble,
      Vectors.dense(trainFeatureInd.map(row.getString(_).toDouble))))

    val testIgnoredInd = List("uid")
    val testFeatureInd = test_x.columns.diff(testIgnoredInd).map(test_x.columns.indexOf(_))
    val testLabelInd = test_x.columns.indexOf("uid")
    val labeledTestData = test_x.rdd.map(row => LabeledPoint(
      row.getString(testLabelInd).toDouble,
      Vectors.dense(testFeatureInd.map(row.getString(_).toDouble))))

    // 特征标准化
    val trainVectors = labeledTrainData.map(p => p.features)
    val trainStandScaler = new StandardScaler(withMean = true, withStd = true).fit(trainVectors)
    val standTrain = labeledTrainData.map(row =>
      LabeledPoint(row.label, trainStandScaler.transform(row.features))
    )
    val testVectors = labeledTestData.map(p => p.features)
    val testStandScaler = new StandardScaler(withMean = true, withStd = true).fit(testVectors)
    val standTest = labeledTestData.map(row =>
      LabeledPoint(row.label, testStandScaler.transform(row.features))
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
    resultDF.write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("F:\\Workspace\\IdeaWorkspace\\Spark\\data\\result.csv")

    spark.stop()
  }

}