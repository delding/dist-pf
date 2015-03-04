/*
 * del_data_generator.hpp
 *
 *  Created on: Jan 19, 2013
 *      Author: erli
 */

#ifndef DEL_DATA_GENERATOR_HPP_
#define DEL_DATA_GENERATOR_HPP_

#include <fstream>
#include <Eigen/Dense>
#include "del_particle_system.hpp"
#include "del_filter.hpp"
#include "del_aux_immpf.hpp"
#include "bearing_only_model.hpp"
#include "del_rsgarch_model.hpp"

namespace del {

  template <typename Model>
  void dataGenerator(const del::ModelBase<Model>& model, double time_length, double time_step,
	  	  	 	 	 const char* x_file, const char* q_file, const char* param_file, const char* ob_file) {
	  int steps = time_length / time_step;
	  gsl_rng* rng = gsl_rng_alloc(gsl_rng_ranlux389);
	  ///generate random seed between 0 and 999
	  int seed = rand() % 1000;
	  gsl_rng_set(rng, seed);
	  std::ofstream state_x(x_file);
	  std::ofstream state_q(q_file);
	  std::ofstream parameter(param_file);
	  std::ofstream measure(ob_file);
	  typename ModelTrait<Model>::Params param = model.paramInitialize(rng);
	  typename ModelTrait<Model>::DisVars q = model.qInitialize(rng);
	  typename ModelTrait<Model>::ConVars x = model.xInitialize(rng);
	  parameter << param;
	  parameter << std::endl;
	  for (int t = 0; t < steps; ++t) {
		  q = model.qPropagate((t+1) * time_step, x, q, param, rng);
		  x = model.xPropagate((t+1) * time_step, x, q, param, rng);
		  state_q << q;
		  state_q << std::endl;
		  state_x << x;
		  state_x << std::endl;
		  measure << model.outPut((t+1) * time_step, x, q, param, rng);
		  measure << std::endl;
	  }
	  gsl_rng_free(rng);
	  return;
  }

}

#endif /* DEL_DATA_GENERATOR_HPP_ */
