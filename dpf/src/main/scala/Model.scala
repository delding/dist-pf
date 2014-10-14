package erli.dpf

import java.util.Random
import java.io._
//should use breeze instead, Vector type doesn't have apply and foreach methods
//should dig deep into mllib Vector class source code
//I convert Vector to Array when evaluating value
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

abstract class HMM(
    val dimX: Int,
    val dimY: Int,
    var timeStep: Double,
    val rand: Random) extends Serializable {
  def initialize: Vector
  def propagate(t: Double, x: Vector): Vector
  def observe(t: Double, x: Vector): Vector
  def tranpdf(t: Double, x: Vector, pre_x: Vector): Double
  def obpdf(t: Double, y: Vector, x: Vector): Double

  def generateData(length: Int, obfileName: String, statefileName: String): Unit = {
    val obfile = new File(obfileName)
    val obWriter = new BufferedWriter(new FileWriter(obfile))
    val statefile = new File(statefileName)
    val stateWriter = new BufferedWriter(new FileWriter(statefile))
    if(!obfile.exists()) obfile.createNewFile()
    if(!statefile.exists()) statefile.createNewFile()
    var x = initialize
    for(step <- 1 to length) {
      x = propagate(step * timeStep, x)
      for(x_ <- x.toArray) stateWriter.write(x_.toString + " ")
      stateWriter.write("\n")
      val y = observe(step * timeStep, x)
      for(y_ <- y.toArray) obWriter.write(y_.toString + " ")
      obWriter.write("\n")
    }
    stateWriter.close()
    obWriter.close()
  }
}

class BenchmarkModel(timeStep: Double, rand: Random) extends
    HMM(1, 1, timeStep, rand) with Serializable{
  private def normalPDF(x: Double, mu: Double, sigma: Double) = {
    1 / (sigma * math.sqrt(2 * math.Pi)) * math.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma))
  }
  override def initialize: Vector = {
    Vectors.dense(math.sqrt(10) * rand.nextGaussian())
  }
  override def propagate(t: Double, x: Vector): Vector = {
    val x0 = x.toArray(0)
    Vectors.dense(
      x0 / 2 + 25 * x0 / (1 + x0 * x0) + 8 * math.cos(1.2 * t) + math.sqrt(10) * rand.nextGaussian()
    )
  }
  override def observe(t: Double, x: Vector): Vector = {
    val x0 = x.toArray(0)
    Vectors.dense(
      x0 * x0 / 20 + rand.nextGaussian()
    )
  }
  override def tranpdf(t: Double, x: Vector, pre_x: Vector): Double = {
    val x0 = x.toArray(0)
    val pre_x0 = pre_x.toArray(0)
    normalPDF(x0, pre_x0 / 2 + 25 * pre_x0 / (1 + pre_x0 * pre_x0) + 8 * math.cos(1.2 * t), math.sqrt(10))
  }
  override def obpdf(t: Double, y: Vector, x: Vector): Double = {
    val x0 = x.toArray(0)
    val y0 = y.toArray(0)
    normalPDF(y0, x0 * x0 / 20, 1)
  }
}
