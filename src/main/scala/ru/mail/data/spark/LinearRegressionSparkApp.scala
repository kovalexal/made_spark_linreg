package ru.mail.data.spark

import breeze.linalg.DenseVector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.made.{LinearRegressionGD, LinearRegressionModel}


object LinearRegressionSparkApp {
  def main(args: Array[String]): Unit = {
    // Create SparkSession
    val spark = SparkSession.builder()
      .appName("linregsgd")
      .master("local[*]")
      .getOrCreate()

    // Add sintactic sugar
    import spark.implicits._

    // Set random number generator to match results between multiple runs
    val randomizer = new scala.util.Random(1)
    def randomDouble = randomizer.nextDouble

    // Create RDD from random numbers
    val randomRDD: RDD[(Double, Double, Double)] = spark.sparkContext.parallelize(
      Seq.fill(100000){(randomDouble, randomDouble, randomDouble)}
    )

    // Create dataframe from random RDD
    val df: DataFrame = spark.createDataFrame(randomRDD)
      .toDF("X", "Y", "Z")
      .withColumn(
        "F",
        lit(1.5) * $"X" + lit(0.3) * $"Y" + lit(-0.7) * $"Z" + lit(10)
      )

    // Create vector Assembler
    val assembler = new VectorAssembler()
      .setInputCols(Array("X", "Y", "Z"))
      .setOutputCol("features")

    // Transform data
    val output = assembler.transform(df)

    // Create our LinearRegressionGD model
    val lrSolver = new LinearRegressionGD()
      .setFeaturesCol("features")
      .setLabelCol("F")
      .setPredictionCol("prediction")
      .setLR(1e-1)
      .setMaxIter(1000)
      .setTol(1e-5)

    // Fit model
    val lrModel = lrSolver.fit(output)

    // Save model to disk
    lrModel.write.overwrite().save("linear_model")

    // Make predictions
    lrModel.transform(output).show(10)

    // Print results
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
  }
}
