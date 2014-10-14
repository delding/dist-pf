package erli.dpf

import org.apache.spark.mllib.linalg.Vector

case class Particle(weight: Double, state: Vector) {
  override def toString: String = {
    "Particle(%s, %s)".format(weight, state)
  }
}
