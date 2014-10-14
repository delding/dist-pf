package erli.dpf

import java.util.Random
import org.apache.spark._
import java.io._
import scala.io.Source
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}

//
object Driver {
  val N = 10000 // number of particles
  val numNodes = 4 // number of compute nodes
  val timeStep = 1.0

  def evaluateTime[T](str: String)(thunk: => T): T = {
    print(str + "... ")
    val t1 = System.currentTimeMillis
    val x = thunk
    val t2 = System.currentTimeMillis
    println((t2 - t1) + " msecs")
    val file = new File(str)
    val writer = new BufferedWriter(new FileWriter(file))
    if(!file.exists()) file.createNewFile()
    writer.write(str + ": " + (t2 -t1).toString + "msecs")
    writer.close()
    x
  }

  def writeParticles(p: Array[Particle], fileName: String): Unit = {
    val file = new File(fileName)
    val writer = new BufferedWriter(new FileWriter(file))
    if(!file.exists()) file.createNewFile()
    for(par <- p.toArray) {
      writer.write(par.weight.toString + " " + par.state.toArray.apply(0).toString + "\n")
    }
    writer.close()
  }

  def writeArray[T](arr: Array[T], fileName: String): Unit = {
    val file = new File(fileName)
    val writer = new BufferedWriter(new FileWriter(file))
    if(!file.exists()) file.createNewFile()
    for(x <- arr) {
      writer.write(x.toString + "\n")
    }
    writer.close()
  }

  //compute Monte Carlo variance, resample simNum times from the same particle set, RDD[particle]
  def compareRespVars(particles: RDD[Particle], simNum: Int, dpf: DPF,
      cenbf: CenBF, distbf1: DistBF1, distbf2: DistBF2,
      distbf3: DistBF3): Unit = {
    //DPF
    val parSet = particles.collect
    val number = parSet.length
    val weightSum = parSet.foldLeft(0.0){ (sum, par) => sum + par.weight }
    val meanBeforeResp = parSet.foldLeft(0.0){ (sum, par) =>
      sum + par.weight * par.state.toArray.apply(0) } / weightSum
    var parSample4VarArray = Array.fill(simNum)(particles).map( pars => dpf.resample(dpf.shuffle(pars)) )
    var respMeanArray = parSample4VarArray.map{ pars => pars.collect.foldLeft(0.0){
      (sum, par) => (sum + par.state.toArray.apply(0) * par.weight)} }
    writeArray(respMeanArray, "DpfRespMean")
    val dpfVariance = respMeanArray.map{ mean =>
      (mean - meanBeforeResp) * (mean - meanBeforeResp) }.reduce(_ + _) / number

    //CenBF
    parSample4VarArray = Array.fill(simNum)(particles).map( pars => cenbf.resample(pars) )
    respMeanArray = parSample4VarArray.map{ pars => pars.collect.foldLeft(0.0){
      (sum, par) => (sum + par.state.toArray.apply(0) * par.weight)} }
    writeArray(respMeanArray, "CenBfRespMean")
    val cenbfVariance = respMeanArray.map{ mean =>
      (mean - meanBeforeResp) * (mean - meanBeforeResp) }.reduce(_ + _) / number

    //DistBF1
    parSample4VarArray = Array.fill(simNum)(particles).map( pars => distbf1.resample(pars) )
    respMeanArray = parSample4VarArray.map{ pars => pars.collect.foldLeft(0.0){
      (sum, par) => (sum + par.state.toArray.apply(0) * par.weight)} }
    writeArray(respMeanArray, "DistBf1RespMean")
    val distbf1Variance = respMeanArray.map{ mean =>
      (mean - meanBeforeResp) * (mean - meanBeforeResp) }.reduce(_ + _) / number

    //DistBF2
    parSample4VarArray = Array.fill(simNum)(particles).map( pars =>
      distbf2.resample(pars, distbf2.sampleRespNum(pars)) )
    respMeanArray = parSample4VarArray.map{ pars => pars.collect.foldLeft(0.0){
      (sum, par) => (sum + par.state.toArray.apply(0) * par.weight)} }
    writeArray(respMeanArray, "DistBf2RespMean")
    val distbf2Variance = respMeanArray.map{ mean =>
      (mean - meanBeforeResp) * (mean - meanBeforeResp) }.reduce(_ + _) / number

    //DistBF3
    parSample4VarArray = Array.fill(simNum)(particles).map( pars =>
      distbf3.resample(pars, distbf3.samplePartition(pars)) )
    respMeanArray = parSample4VarArray.map{ pars => pars.collect.foldLeft(0.0){
      (sum, par) => (sum + par.state.toArray.apply(0) * par.weight)} }
    writeArray(respMeanArray, "DistBf3RespMean")
    val distbf3Variance = respMeanArray.map{ mean =>
      (mean - meanBeforeResp) * (mean - meanBeforeResp) }.reduce(_ + _) / number

    val varArr = Array("dpfVar = " + dpfVariance.toString,
      "cenbfVar = " + cenbfVariance.toString,
      "distbf1Var = " + distbf1Variance.toString,
      "distbf2Var = " + distbf2Variance.toString,
      "distbf3Var = " + distbf3Variance.toString,
      "meanBeforeResp = " + meanBeforeResp.toString)
    writeArray(varArr, "resampleVar")
  }

  def computeRespTimes(particles: RDD[Particle], numIterations: Int, dpf: DPF,
      cenbf: CenBF, distbf1: DistBF1, distbf2: DistBF2,
      distbf3: DistBF3): Unit = {
    val numPartitions = particles.partitions.length
    val numParticles = particles.count
    var particle = particles
    evaluateTime(numIterations.toString + " iterations DpfRunningTime " + numParticles.toString + " particles " + numPartitions + " partitions"){
      for(i <- 1 to numIterations){ particle = dpf.resample(dpf.shuffle(particle)) }
      particle.collect
    }
    evaluateTime(numIterations.toString + " iterations CenbfRunningTime " + numParticles.toString + " particles " + numPartitions + " partitions"){
      for(i <- 1 to numIterations){ particle = cenbf.resample(particle) }
      particle.collect
    }
    evaluateTime(numIterations.toString + " iterations Distbf1RunningTime " + numParticles.toString + " particles " + numPartitions + " partitions"){
      for(i <- 1 to numIterations){ particle = distbf1.resample(particle) }
      particle.collect
    }
    evaluateTime(numIterations.toString + " iterations Distbf2RunningTime " + numParticles.toString + " particles " + numPartitions + " partitions"){
      for(i <- 1 to numIterations){ particle = distbf2.resample(
        particle, distbf2.sampleRespNum(particle)) }
      particle.collect
    }
    evaluateTime(numIterations.toString + " iterations Distbf3RunningTime " + numParticles.toString + " particles " + numPartitions + " partitions"){
      for(i <- 1 to numIterations){ particle = distbf3.resample(
        particle, distbf3.samplePartition(particle)) }
      particle.collect
    }
  }

  def runFilters(initPar: RDD[Particle], initTime: Double, timeStep: Double,
      obFile: String, dpf: DPF,
      cenbf: CenBF, distbf1: DistBF1, distbf2: DistBF2,
      distbf3: DistBF3): Unit = {
    var particles = initPar
    var time = initTime + timeStep
    val ob = Source.fromFile(obFile).getLines.toArray
    //DPF
    /*
     * The iterations here only do transformations
     * the real computation is delayed to the execution of collect action
     */
    for (y <- ob) {
      particles = dpf.run(time, Vectors.dense(y.toDouble), particles)
      time += timeStep
    }
    writeParticles(particles.collect, "DpfEstimation")
    //DPF without shuffling
    particles = initPar
    time = initTime + timeStep
    for (y <- ob) {
      particles = dpf.resample( dpf.propagate(time, Vectors.dense(y.toDouble), particles) )
      time += timeStep
    }
    writeParticles(particles.collect, "DpfNoShuffleEstimation")
    //CenBF
    particles = initPar
    time = initTime + timeStep
    for (y <- ob) {
      particles = cenbf.run(time, Vectors.dense(y.toDouble), particles)
      time += timeStep
    }
    writeParticles(particles.collect, "CenBFEstimation")
    //DistBF1
    particles = initPar
    time = initTime + timeStep
    for (y <- ob) {
      particles = distbf1.run(time, Vectors.dense(y.toDouble), particles)
      time += timeStep
    }
    writeParticles(particles.collect, "DistBF1Estimation")
    //DistBF2
    particles = initPar
    time = initTime + timeStep
    for (y <- ob) {
      particles = distbf2.run(time, Vectors.dense(y.toDouble), particles)
      time += timeStep
    }
    writeParticles(particles.collect, "DistBF2Estimation")
    //DistBF3
    particles = initPar
    time = initTime + timeStep
    for (y <- ob) {
      particles = distbf3.run(time, Vectors.dense(y.toDouble), particles)
      time += timeStep
    }
    writeParticles(particles.collect, "DistBF3Estimation")
  }

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("SparkDPF").setMaster("local[4]")
    val sc = new SparkContext(sparkConf)
    val rand0, rand1, rand2, rand3, rand4 = new Random(42)
    val (model0, model1, model2, model3, model4) = ( new BenchmarkModel(timeStep, rand0),
      new BenchmarkModel(timeStep, rand1), new BenchmarkModel(timeStep, rand2), new BenchmarkModel(timeStep, rand3),
      new BenchmarkModel(timeStep, rand4) )
    val sampler = new Resampler(422)
    val dpf= new DPF(sc, 4222, N, timeStep, sampler.systematic, numNodes, model0)
    val cenbf = new CenBF(sc, 4222, N, timeStep, sampler.systematic, numNodes, model1)
    val distbf1 = new DistBF1(sc, 4222, N, timeStep, sampler.systematic, numNodes, model2)
    val distbf2 = new DistBF2(sc, 4222, N, timeStep, sampler.systematic, numNodes, model3)
    val distbf3 = new DistBF3(sc, 4222, N, timeStep, sampler.systematic, numNodes, model4)
    val initParticles = dpf.initialize()

    //run particle filter
    ///runFilters(initParticles, 0.0, 1.0, "20observes", dpf, cenbf, distbf1, distbf2, distbf3)

    // take a particle set sample
    val y = Source.fromFile("20observes").getLines.next.toDouble
    ///val parSet = dpf.propagate(0.0 + timeStep, Vectors.dense(y), initParticles)
    //compare resampling variance
    ///compareRespVars(parSet, 100, dpf, cenbf, distbf1, distbf2, distbf3)
    //compute resampling time
    ///computeRespTimes(parSet, 10, dpf, cenbf, distbf1, distbf2, distbf3)

    val testPar = dpf.propagate(1.0, Vectors.dense(y), initParticles)
    val testParArr = testPar.collect
    val testSum = testParArr.foldLeft(0.0)((sum, par) => sum + par.weight)
    val testNormedPar = testParArr.map{ case Particle(w, x) => Particle(w / testSum, x) }
    writeParticles(testNormedPar, "parBeforeResp")
    val dpfPar = dpf.resample(dpf.shuffle(testPar))
    writeParticles(dpfPar.collect, "parDpf")
    val cenbfPar = cenbf.resample(testPar)
    writeParticles(cenbfPar.collect, "parCenbf")
    val distbf1Par = distbf1.resample(testPar)
    writeParticles(distbf1Par.collect, "parDistbf1")
    val distbf2Par = distbf2.resample(testPar, distbf2.sampleRespNum(testPar))
    writeParticles(distbf2Par.collect, "parDistbf2")
    val distbf3Par = distbf3.resample(testPar, distbf3.samplePartition(testPar))
    writeParticles(distbf3Par.collect, "parDistbf3")
    val (meanBefore, meanDpf, meanCenbf, meanDbf1, meanDbf2, meanDbf3) =
      ( testNormedPar.foldLeft(0.0)((mean, par) => mean + par.weight * par.state.toArray.apply(0)),
        dpfPar.collect.foldLeft(0.0)((mean, par) => mean + par.weight * par.state.toArray.apply(0)),
        cenbfPar.collect.foldLeft(0.0)((mean, par) => mean + par.weight * par.state.toArray.apply(0)),
        distbf1Par.collect.foldLeft(0.0)((mean, par) => mean + par.weight * par.state.toArray.apply(0)),
        distbf2Par.collect.foldLeft(0.0)((mean, par) => mean + par.weight * par.state.toArray.apply(0)),
        distbf3Par.collect.foldLeft(0.0)((mean, par) => mean + par.weight * par.state.toArray.apply(0)) )
    println(meanBefore)
    println(meanDpf)
    println(meanCenbf)
    println(meanDbf1)
    println(meanDbf2)
    println(meanDbf3)


    println("completed, Erli")
    sc.stop()

  }
}
