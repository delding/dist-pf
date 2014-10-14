package erli.dpf

// TODO: use tail recursion to avoid stack overflow
import java.util.Random
//import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}

//Consider use template for Model to allow sub-classes of HMM have public methods that HMM doesn't have
//HMM is the place that holds the common code
abstract class ParticleFilter(
    @transient val sc: SparkContext,
    var seed: Long, // if declared as private, subclass must redefine their own
    var numParticles: Int,
    var timeStep: Double,
    var resampler: (Int, Iterator[Double]) => Iterator[Int],
    var numPartitions: Int,
    var model: HMM) extends Serializable {
  //def model: HMM
  def initialize(numSlices: Int = numPartitions): RDD[Particle] = {
    numPartitions = numSlices
    val weight = 1.0 / numParticles
    val states = for(index <- 0 until numParticles) yield model.initialize
    //sc.parallelize will assign same number of elements (particles) to each partition
    val particles = states.map{ x => Particle(weight, x) }
    sc.parallelize(particles, numSlices)
  }
  def propagate(t: Double, y: Vector, par: RDD[Particle]): RDD[Particle]
  def resample(par: RDD[Particle]): RDD[Particle]
  def resample(par: RDD[Particle], otherInfo: Array[(Int, Double)]): RDD[Particle]
  def run(t: Double, y: Vector, par: RDD[Particle]): RDD[Particle]
}
/*
 * Distributed particle filter algorithm
 * All the Spark operations in DPF are transformations, if driver program iterates
 * too many times without calling a Spark action, stack overflow error will occur.
 * Should each computation on each node has their own random number generator?
 */
class DPF(
    sc: SparkContext,
    seed: Int,
    numParticles: Int,
    timeStep: Double,
    resampler: (Int, Iterator[Double]) => Iterator[Int],
    numPartitions: Int,
    model: HMM)
    extends ParticleFilter(
    sc, seed, numParticles, timeStep, resampler, numPartitions, model) with Serializable {

  val rand = new Random(seed)

  override def propagate(t: Double, y: Vector, par: RDD[Particle]) = {
    par.map{ case Particle(w, x) =>
      Particle(model.obpdf(t, y, x) * w, model.propagate(t, x))
    }
  }

  def shuffle(par: RDD[Particle]): RDD[Particle] ={
    //cache() only works when particles are loaded from files
    //if particles are parallelized from driver program, they are always in memory
    //unless checkpoint is done and some partition needs to be recovered from checkpoint file
    par.cache
    //Split RDD into numPartitions number of RDDs
    //Each RDD is obtained by unformly sampling a subset on each partition of original RDD
    val splits = for{ num <- Array.range(0, numPartitions) } yield {
      par.mapPartitions{
        iter =>
        val (iter1, iter2) = iter.duplicate
        val numLocal = iter1.length
        val numPerSplit = numLocal / numPartitions
        val drawIndex = for(i <- 1 to numPerSplit) yield rand.nextInt(numLocal)
        val indexRepNum = Iterator.tabulate(numLocal){ index => drawIndex.count(_ == index) }
        iter2.flatMap { particle => Iterator.fill(indexRepNum.next)(particle) }
      }
    }
    //Coalesce each RDD into one partition
    val coalesce = for(split <- splits) yield split.coalesce(1)
    //merge into one RDD
    coalesce.reduce(_ union _)
  }

  override def resample(par: RDD[Particle], respNumArr: Array[(Int, Double)]) = par
  override def resample(par: RDD[Particle]) = {
    //particle resample number for each partition, should be intger number
    val numResample = numParticles / numPartitions
    val weight = 1.0 / numParticles
    par.mapPartitions {
      iter =>
      val (iter0, iter1) = iter.duplicate
      val (iter2, iter3) = iter0.duplicate
      val weightSum = iter1.foldLeft(0.0){ (sum, par) => sum + par.weight }
      val normedWeights = iter2.map(par => par.weight / weightSum)
      val indexRepNum = resampler(numResample, normedWeights)
      // indexRepNum is created on the same node as RDD partition and will be referenced in flatMap
      // if it is created on driver node, a copy of indexRepNum.next will delivered to work nodes
      // and only the value of first call to indexRepNum.next gets coppied so each fill will
      // have same argument value which is a logical error
      iter3.flatMap {
        case Particle(w, x) => Iterator.fill(indexRepNum.next)(Particle(weight, x))
      }
    }
  }

  override def run(t: Double, y: Vector, par: RDD[Particle]) = {
    val propagated = propagate(t, y, par)
    val shuffled = shuffle(propagated)
    resample(shuffled)
  }
}

class CenBF(
    sc: SparkContext,
    seed: Int,
    numParticles: Int,
    timeStep: Double,
    resampler: (Int, Iterator[Double]) => Iterator[Int],
    numPartitions: Int,
    model: HMM)
    extends ParticleFilter(
    sc, seed, numParticles, timeStep, resampler, numPartitions, model) with Serializable {

  val rand = new Random(seed)

  override def propagate(t: Double, y: Vector, par: RDD[Particle]) = {
    par.map{ case Particle(w, x) =>
      Particle(model.obpdf(t, y, x) * w, model.propagate(t, x))
    }
  }

  override def resample(par: RDD[Particle], respNumArr: Array[(Int, Double)]) = par
  override def resample(par: RDD[Particle]) = {
    val partiles = par.collect
    val weightSum = partiles.foldLeft(0.0)( (w, par) => w + par.weight )
    val normedWeights = partiles.map(par => par.weight / weightSum).toIterator
    val indexRepNum = resampler(numParticles, normedWeights)
    val weight = 1.0 / numParticles
    val resampledPars = partiles.flatMap {
      case Particle(w, x) => Array.fill(indexRepNum.next)(Particle(weight, x))
    }
    sc.parallelize(resampledPars, numPartitions)
  }
  override def run(t: Double, y: Vector, par: RDD[Particle]) = {
    resample(propagate(t, y, par))
  }
}

//Distributed resampling with centralized weight normalization
class DistBF1(
    sc: SparkContext,
    seed: Int,
    numParticles: Int,
    timeStep: Double,
    resampler: (Int, Iterator[Double]) => Iterator[Int],
    numPartitions: Int,
    model: HMM)
    extends ParticleFilter(
    sc, seed, numParticles, timeStep, resampler, numPartitions, model) with Serializable {

  val rand = new Random(seed)

  override def propagate(t: Double, y: Vector, par: RDD[Particle]) = {
    par.map{ case Particle(w, x) =>
      Particle(model.obpdf(t, y, x) * w, model.propagate(t, x))
    }
  }

  override def resample(par: RDD[Particle], respNumArr: Array[(Int, Double)]) = par
  override def resample(par: RDD[Particle]) = {
    val numPerPartition = numParticles / numPartitions
    val totalWeightSum = par.aggregate(0.0)((sum, par) => sum + par.weight, _ + _)
    par.mapPartitions {
      iter =>
      val (iter0, iter1) = iter.duplicate
      val (iter2, iter3) = iter0.duplicate
      val weightSum = iter1.foldLeft(0.0){ (sum, par) => sum + par.weight }
      val normedWeights = iter2.map(par => par.weight / weightSum)
      val indexRepNum = resampler(numPerPartition, normedWeights)
      val weight = weightSum / totalWeightSum / numPerPartition
      iter3.flatMap {
        case Particle(w, x) => Iterator.fill(indexRepNum.next)(Particle(weight, x))
      }
    }
  }

  override def run(t: Double, y: Vector, par: RDD[Particle]) = {
    resample(propagate(t, y, par))
  }
}

//Distributed resampling with centralized sampling of resampling number
class DistBF2(
    sc: SparkContext,
    seed: Int,
    numParticles: Int,
    timeStep: Double,
    resampler: (Int, Iterator[Double]) => Iterator[Int],
    numPartitions: Int,
    model: HMM)
    extends ParticleFilter(
    sc, seed, numParticles, timeStep, resampler, numPartitions, model) with Serializable {

  val rand = new Random(seed)

  override def propagate(t: Double, y: Vector, par: RDD[Particle]) = {
    par.map { case Particle(w, x) =>
      Particle(model.obpdf(t, y, x) * w, model.propagate(t, x))
    }
  }

  //method to sample resampling number of particles on each partition
  //return Array[(respNumRdd, localWeightSum)]
  def sampleRespNum(par: RDD[Particle]) = {
    val localWeightSums = par.glom.map{ parArr => parArr.foldLeft(0.0){ (sum, par) => sum + par.weight } }.collect
    val totalWeightSum = localWeightSums.reduce(_ + _)
    val normedWeights = localWeightSums.map(_ / totalWeightSum).toIterator
    resampler(numParticles, normedWeights).toArray.zip(localWeightSums)
  }

  override def resample(par: RDD[Particle]) = par
  override def resample(par: RDD[Particle], respNumArr: Array[(Int, Double)]) = {
    val weight = 1.0 / numParticles
    val respNumRdd = sc.parallelize(respNumArr, numPartitions)
    //iterRespNum only contains one element, i.e. a Tuple2(respNumRdd, weightSum) for its zipped partition
    par.zipPartitions(respNumRdd){ (iterPar, iterRespNum) =>
      val (respNum, weightSum) = iterRespNum.next
      val (iterPar0, iterPar1) = iterPar.duplicate
      val normedWeights = iterPar0.map(par => par.weight / weightSum)
      val indexRepNum = resampler(respNum, normedWeights)
      iterPar1.flatMap {
        case Particle(w, x) => Iterator.fill(indexRepNum.next)(Particle(weight, x))
      }
    }
  }

  override def run(t: Double, y: Vector, par: RDD[Particle]) = {
    val propagatedPar = propagate(t, y, par)
    val respNumArr = sampleRespNum(propagatedPar)
    resample(propagatedPar, respNumArr)
  }
}

class DistBF3(
    sc: SparkContext,
    seed: Int,
    numParticles: Int,
    timeStep: Double,
    resampler: (Int, Iterator[Double]) => Iterator[Int],
    numPartitions: Int,
    model: HMM)
    extends ParticleFilter(
    sc, seed, numParticles, timeStep, resampler, numPartitions, model) with Serializable {

  val rand = new Random(seed)

  override def propagate(t: Double, y: Vector, par: RDD[Particle]) = {
    par.map { case Particle(w, x) =>
      Particle(model.obpdf(t, y, x) * w, model.propagate(t, x))
    }
  }

  //method to sample resampling number of each partition
  def samplePartition(par: RDD[Particle]) = {
    val localWeightSums = par.mapPartitions {
      iter =>
      val weightSum = iter.foldLeft(0.0){ (sum, par) => sum + par.weight }
      Iterator(weightSum)
    }.collect
    val totalWeightSum = localWeightSums.reduce(_ + _)
    val normedWeights = localWeightSums.map(_ / totalWeightSum).toIterator
    resampler(numPartitions, normedWeights).toArray.zip(localWeightSums)
  }

  override def resample(par: RDD[Particle]) = par
  override def resample(par: RDD[Particle], respPartArr: Array[(Int, Double)]) = {
    val numPerPartition = numParticles / numPartitions
    val weight = 1.0 / numParticles
    //methodglom can provide a similar funtion as method mapPartitions does,
    //allowing partition-wise operations
    val partitionArr = par.glom
    val respPartRdd = sc.parallelize(respPartArr, numPartitions)
    val sampledPartArr = partitionArr.zip(respPartRdd).flatMap{
      case (part, (respNumRdd, weightSum)) => Array.fill(respNumRdd)((part, weightSum))
    }
    val resampledPartArr = sampledPartArr.map { case (part, weightSum) =>
      val normedWeights = part.map{ case Particle(w, x) => w / weightSum }.toIterator
      val indexRepNum = resampler(numPerPartition, normedWeights)
      part.flatMap { case Particle(w, x) =>
        Array.fill(indexRepNum.next)(Particle(weight, x))
      }
    }
    //undo glom, get plain RDD[Particle]
    resampledPartArr.flatMap(part => part)
    //same as:
    //for(part <- resamplePartArr; particle <- part) yield particle
  }

  override def run(t: Double, y: Vector, par: RDD[Particle]) = {
    val propagatedPar = propagate(t, y, par)
    val respPartArr = samplePartition(propagatedPar)
    resample(propagatedPar, respPartArr)
  }
}
