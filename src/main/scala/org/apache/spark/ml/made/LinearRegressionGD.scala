package org.apache.spark.ml.made

import scala.util.control.Breaks._
import breeze.linalg
import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import breeze.stats.mean
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.BLAS.dot
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils}
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.made.LinearRegressionModel.LinearRegressionModelWriter
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.mllib

import scala.collection.mutable.ArrayBuffer


trait HasLR extends Params {
  final val lr: DoubleParam = new DoubleParam(this, "lr", "gd learning rate", ParamValidators.gtEq(0))
  final def getLR: Double = $(lr)
}

trait LinearRegressionParams extends PredictorParams with HasLR with HasMaxIter with HasTol {
  override protected def validateAndTransformSchema(
                                                     schema: StructType,
                                                     fitting: Boolean,
                                                     featuresDataType: DataType): StructType = {
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }

  setDefault(lr -> 1e-4, maxIter -> 1000, tol -> 1e-5)
}

class LinearRegressionGD(override val uid: String)
  extends Regressor[Vector, LinearRegressionGD, LinearRegressionModel]
    with LinearRegressionParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("linRegGD"))

  def setLR(value: Double): this.type = set(lr, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  override def copy(extra: ParamMap): LinearRegressionGD = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))

    // Used to convert untyped dataframes to datasets with vectors
    implicit val vectorEncoder : Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder : Encoder[Double] = ExpressionEncoder()

    // Current coefficients
    var coefficients: BDV[Double] = BDV.ones[Double](numFeatures)
    var intercept: Double = 1.0
    var error: Double = Double.MaxValue

    // Convert input dataset
    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(featuresCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    // Main loop
    breakable { for (i <- 1 to getMaxIter) {
      val (coefficientsSummary, interceptSummary) = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val coefficientsSummarizer = new MultivariateOnlineSummarizer()
        val interceptSummarizer = new MultivariateOnlineSummarizer()

        data.grouped(1000).foreach((r: Seq[(Vector, Double)]) => {
          // Let's try to create matrix from this group

          // Create array of numbers for matrix and vector column
          val (x_, y_) = r.map(x => (
            x._1.toArray.to[ArrayBuffer], Array(x._2).to[ArrayBuffer]
          )).reduce((x, y) => {
            (x._1 ++ y._1, x._2 ++ y._2)
          })

          // Create breeze matrix and dense vector
          val x__ = x_.toArray
          val y__ = y_.toArray
          val X = BDM.create(x__.size / numFeatures, numFeatures, x__, 0, numFeatures, true)
          val Y = BDV(y__)

          // Calculate current values
          var Yhat = (X * coefficients) + intercept

          // Calculate residuals
          val residuals = Y - Yhat

          // Calculate coefficients step
          val c: BDM[Double] = X(::, *) * residuals

          coefficientsSummarizer.add(mllib.linalg.Vectors.fromBreeze(mean(c(::, *)).t))
          interceptSummarizer.add(mllib.linalg.Vectors.dense(mean(residuals)))
        })

        Iterator((coefficientsSummarizer, interceptSummarizer))
      }).reduce((x, y) => {
        (x._1 merge y._1, x._2 merge y._2)
      })

      // Calculate error
      error = interceptSummary.mean(0)
      // If error is small enough, break
      if (error.abs < getTol)
        break

      // Update coefficients
      var dCoeff: BDV[Double] = coefficientsSummary.mean.asBreeze.toDenseVector
      dCoeff :*= (-2.0) * getLR
      coefficients -= dCoeff

      // Update intercept
      var dInter = (-2.0) * getLR * error
      intercept -= dInter
    } }

    // Return fitted model
    val lrModel = copyValues(new LinearRegressionModel(uid, new DenseVector(coefficients.toArray), intercept))
    lrModel
  }
}

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: Vector,
                                           val intercept: Double)
  extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams with MLWritable {

  val brzCoefficients: BV[Double] = coefficients.asBreeze

  private[made] def this(coefficients: Vector, intercept: Double) = this(Identifiable.randomUID("linRegGD"), coefficients.toDense, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(coefficients, intercept))

  override def write: MLWriter = new LinearRegressionModelWriter(this)

  override def predict(features: Vector): Double = {
    // Breeze prediction
    (features.asBreeze dot brzCoefficients) + intercept

    // Spark prediction
    // dot(features, coefficients) + intercept
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  class LinearRegressionModelWriter(instance: LinearRegressionModel) extends MLWriter {
    private case class Data(intercept: Double, coefficients: Vector)

    override def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)

      // Save model data: intercept, coefficients
      val data = Data(instance.intercept, instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  class LinearRegressionModelReader extends MLReader[LinearRegressionModel] {
    private val className = classOf[LinearRegressionModel].getName

    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val Row(intercept: Double, coefficients: Vector) = data.select("intercept", "coefficients").head()
      val model = new LinearRegressionModel(metadata.uid, coefficients, intercept)

      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[LinearRegressionModel] = new LinearRegressionModelReader

  override def load(path: String): LinearRegressionModel = super.load(path)
}
