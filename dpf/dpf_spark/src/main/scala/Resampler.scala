package erli.dpf

import java.util.Random

class Resampler(val seed: Long) extends Serializable{
  val rand = new Random(seed)

  def systematic(numResample: Int, normedWeights: Iterator[Double]) = {
    var random = rand.nextDouble() / numResample
    var weightCumulative = 0.0
    for(w <- normedWeights) yield {
      var count = 0
      weightCumulative += w
      while (weightCumulative > random) {
        count += 1
        random += 1.0 / numResample
      }
      count
    }
  }
}
