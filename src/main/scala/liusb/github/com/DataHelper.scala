package liusb.github.com

import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.{Row, SQLContext, DataFrame}

class DataHelper(sqlContext: SQLContext) {

  def loadData(): DataFrame = {
    // 从csv文件读取数据
    val train_x = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").load("./data\\train_x.csv")
    val test_x = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").load("./data\\test_x.csv")
    val train_y = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").load("./data\\train_y.csv")

    // 合并测试数据和训练数据
    val allData = train_x.unionAll(test_x)
    // 加入标签数据
    val joinData = allData.join(train_y, Seq("uid"), "left_outer")
    joinData
  }

  def formatData(data: DataFrame): DataFrame = {
    val uidIndex = data.columns.indexOf("uid")
    val labelIndex = data.columns.indexOf("y")
    // 获取特征列索引值
    val featureIndex = data.columns.map(
      data.columns.indexOf(_)).diff(Seq(uidIndex, labelIndex))
    // 将特征列合并成一个向量列，并对数据进行类型转换
    val fixData = data.map(row =>
      (row.getString(uidIndex).toInt,
        if (row.getString(labelIndex) == null) -1L
        else row.getString(labelIndex).toDouble,
        Vectors.dense(featureIndex.map(row.getString(_).toDouble)))
    )
    sqlContext.createDataFrame(fixData).toDF("uid", "label", "features")
  }

  def loadFeaturesInfo(): DataFrame = {
    val features_type = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").load("./data\\features_type.csv")
    features_type
  }

  def saveData(data: DataFrame, name: String): Unit = {
    // 保存数据到文件
    data.registerTempTable(name)
    data.repartition(1).write.mode("overwrite")
      .format("com.databricks.spark.csv")
      .option("header", "false").save("./data\\" + name)
  }

}