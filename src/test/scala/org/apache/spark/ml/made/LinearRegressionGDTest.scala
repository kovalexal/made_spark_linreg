package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import org.apache.spark.sql.functions._


class LinearRegressionGDTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.01
  lazy val data: DataFrame = LinearRegressionGDTest._data
  lazy val vectors: Seq[Vector] = LinearRegressionGDTest._vectors
  lazy val lrModel: LinearRegressionModel = {
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
        lit(1.5) * col("X") + lit(0.3) * col("Y") + lit(-0.7) * col("Z") + lit(10)
      )

    // Create vector Assembler
    val assembler = new VectorAssembler()
      .setInputCols(Array("X", "Y", "Z"))
      .setOutputCol("features")

    // Transform data
    val output = assembler.transform(df)

    // Fit spark linear regression
    val lrSolver = new LinearRegressionGD()
      .setFeaturesCol("features")
      .setLabelCol("F")
      .setPredictionCol("prediction")
      .setLR(1e-1)
      .setMaxIter(1000)
      .setTol(1e-5)

    // Fit model
    lrSolver.fit(output)
  }

  "math" should "work" in {
    (1 + 1) should be(2)
  }

  "Model" should "correctly predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      Vectors.dense(1.0, 2.0).toDense,
      1.0
    )

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    for (i <- 0 until vectors.length) {
      model.predict(vectors(i)) should be(((vectors(i).asBreeze dot model.coefficients.asBreeze) + model.intercept) +- delta)
    }
  }

  "Model" should "correctly write/read to/from disk" in {
    // Save model to disk
    val tmpFolder = Files.createTempDir().getAbsolutePath()
    lrModel.write.overwrite().save(tmpFolder)

    // Read model from disk
    val lrModelAfterRead = LinearRegressionModel.load(tmpFolder)

    // Check result
    lrModelAfterRead.coefficients(0) should be(1.5 +- delta)
    lrModelAfterRead.coefficients(1) should be(0.3 +- delta)
    lrModelAfterRead.coefficients(2) should be(-0.7 +- delta)
    lrModelAfterRead.intercept should be(10.0 +- delta)
  }

  "Regressor" should "correctly fit" in {
    // Check result coefficients
    lrModel.coefficients(0) should be(1.5 +- delta)
    lrModel.coefficients(1) should be(0.3 +- delta)
    lrModel.coefficients(2) should be(-0.7 +- delta)
    // Check result intercept
    lrModel.intercept should be(10.0 +- delta)
  }
}

object LinearRegressionGDTest extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(1.0, 2.0),
    Vectors.dense(3.0, 4.0),
    Vectors.dense(5.0, 6.0),
    Vectors.dense(7.0, 8.0),
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
}
