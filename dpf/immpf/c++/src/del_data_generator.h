/*
 * del_data_generator.h
 *
 *  Created on: Jan 19, 2013
 *      Author: erli
 */

#ifndef DEL_DATA_GENERATOR_H_
#define DEL_DATA_GENERATOR_H_

#include <fstream>
#include <Eigen/Dense>
#include "del_particle_system.h"
#include "del_filter.h"
#include "del_aux_immpf.h"
#include "bearing_only_model.h"
#include "del_rsgarch_model.h"

namespace del {


/*


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
	  ///TODO:particle_system class can employ this switching structure
	  if(ModelTrait<Model>::model_type() == CK) {
		  typename ModelTrait<Model>::ConVars x = model.xInitialize(rng);
		  for (int t = 0; t < steps; ++t) {
			  x = model.xPropagate((t+1) * time_step, x, rng);
			  state_x << x;
			  state_x << std::endl;
			  measure << model.outPut((t+1) * time_step, x, rng);
			  measure << std::endl;
		  }
		  gsl_rng_free(rng);
		  return;
	  }
	  else if(ModelTrait<Model>::model_type() == CU) {
		  typename ModelTrait<Model>::Params param = model.paramInitialize(rng);
		  typename ModelTrait<Model>::ConVars x = model.xInitialize(rng);
		  parameter << param;
		  parameter << std::endl;
		  for (int t = 0; t < steps; ++t) {
			  x = model.xPropagate((t+1) * time_step, x, param, rng);
			  state_x << x;
			  state_x << std::endl;
			  measure << model.outPut((t+1) * time_step, x, param, rng);
			  measure << std::endl;
		  }
		  gsl_rng_free(rng);
		  return;
	  }
	  else if(ModelTrait<Model>::model_type() == HK) {
		  typename ModelTrait<Model>::DisVars q = model.qInitialize(rng);
		  typename ModelTrait<Model>::ConVars x = model.xInitialize(rng);
		  for (int t = 0; t < steps; ++t) {
			  q = model.qPropagate((t+1) * time_step, x, q, rng);
			  x = model.xPropagate((t+1) * time_step, x, q, rng);
			  state_q << q;
			  state_q << std::endl;
			  state_x << x;
			  state_x << std::endl;
			  measure << model.outPut((t+1) * time_step, x, q, rng);
			  measure << std::endl;
		  }
		  gsl_rng_free(rng);
		  return;
	  }
	  else {
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


  */


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

#endif /* DEL_DATA_GENERATOR_H_ */
