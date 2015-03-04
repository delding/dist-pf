/*
 * del_filtering.cpp
 *
 *  Created on: Jan 22, 2013
 *      Author: erli
 */

#include <boost/shared_ptr.hpp>
#include <tr1/tuple>
#include <Eigen/Dense>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include "del_filtering.hpp"
#include "del_model.hpp"
#include "del_rsgarch_model.hpp"
#include "del_particle_system.hpp"
#include "del_aux_immpf.hpp"

///TODO:unify interfaces in particle_system for BF, APF and IMMPF
///TODO:use factory class to produce partilce_system object, filter object, and model object and get the pointers returned
///TODO:encapsulate parameters into structures
namespace del {
void* RsgarchFiltering(int model_index, int filter_index, int resample_index, double time_step, int number,
		  	     const char* data_file, double* weight, double* state,
		  	     double* state_vol, double* p0, double* p1, double* p2, double* p3, double* p4, double* p5,
		  	     double* p6, double* p7, double* p8, double* mean_x, double* mean_vol,
		  	     double* x_95, double* x_5, double* vol_95, double* vol_5,double* prob_q,
		  	     double* mean_params) {


	  //static boost::shared_ptr<del::BearingOnly> model(new del::BearingOnly());
	  static boost::shared_ptr<del::RSGARCH> model(new del::RSGARCH());

	  //static del::ParticleSystem<del::BearingOnly, del::CK> particle_system(model, time_step, number, del::kNotStored);
	  static del::ParticleSystem<del::RSGARCH, del::HU> particle_system(model, time_step, number, del::kNotStored);


	  //static del::Filter<del::Bootstrap, del::ParticleSystem<del::BearingOnly, del::CK> > bootstrap;
	  //static del::Filter<del::APF, del::ParticleSystem<del::BearingOnly, del::CK> > apf(particle_system, del::BearingOnly::predicExp);
	  ///TODO:modify Aux_IMM default constructor, I missed (particle_system) after aux_immpf, making the program get crushed
	  static del::Filter<del::Aux_IMM, del::ParticleSystem<del::RSGARCH, del::HU> > aux_immpf(particle_system);


	  static del::MultinomialResample resample;
	  static del::ResidualResample res_resample;
	  static del::StratifiedResample strat_resample;
	  static del::SystematicResample sys_resample;

	  static bool isInitialized = false;
	  ///TODO:code a reset member for particle_system class
	  if (!isInitialized) {
		  particle_system.set_time(0.0);
		  particle_system.set_time_step(time_step);
		  particle_system.set_data_source(data_file);//TODO:need to close it later
		  if(number != particle_system.get_number()) {
			  particle_system.set_number(number);
			  aux_immpf.set_particle_numbers(number);
		  }

		  //particle_system.initializeConState();
		  //TODO:put the method into particle_system class
		  aux_immpf.initialize(particle_system);
		  isInitialized = true;
	  }


	  //particle_system.iterate(apf, sys_resample);
	  particle_system.iterate(aux_immpf, sys_resample);


	  for(int i = 0; i != number; ++i) {
		  weight[i] = (std::tr1::get<0>(*particle_system.get_particles()))(i);
	  	  state[i] = (std::tr1::get<1>(*particle_system.get_particles()))(0,i);
	  	  state_vol[i] = (std::tr1::get<1>(*particle_system.get_particles()))(1,i);
	  	  p0[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(0,i);
	  	  p1[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(1,i);
	  	  p2[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(2,i);
	  	  p3[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(3,i);
	  	  p4[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(4,i);
	  	  p5[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(5,i);
	  	  p6[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(6,i);
	  	  p7[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(7,i);
	  	  p8[i] =  (std::tr1::get<3>(*particle_system.get_particles()))(8,i);
	  }
	  *mean_x = (std::tr1::get<1>(*particle_system.get_particles())).row(0).matrix() *
			    (std::tr1::get<0>(*particle_system.get_particles())).matrix();
	  *mean_vol = (std::tr1::get<1>(*particle_system.get_particles())).row(1).matrix() *
			      (std::tr1::get<0>(*particle_system.get_particles())).matrix();
	  /*Eigen::ArrayXd temp = (std::tr1::get<1>(*particle_system.get_particles())).matrix()
			                 * (std::tr1::get<0>(*particle_system.get_particles())).matrix();*/
	  ///since not stored as row major, sort the array use specific stride
	  ///TODO:can not sort directly on the original data, because corresponding weight will not be sorted which will make the result of next iteration wrong
	  ///create a temp variable storing ConVars
	  Eigen::Array<double,Eigen::Dynamic, Eigen::Dynamic> temp_states_particles(2, number);
	  temp_states_particles = std::tr1::get<1>(*particle_system.get_particles());
	  double* address_x = &temp_states_particles(0,0);
	  ///sort all particles of x in ascending order, stride = 2
	  gsl_sort(address_x, 2, number);
	  ///sort all particles of vol in ascending order, address_x + 1 is the address of first vol since column major stored by default
	  gsl_sort(address_x + 1, 2, number);
	  ///TODO: followings are not weighted percentiles, need to modify
	  *x_95 = gsl_stats_quantile_from_sorted_data(address_x, 2, number, 0.95);
	  *x_5 = gsl_stats_quantile_from_sorted_data(address_x, 2, number, 0.05);
	  *vol_95 = gsl_stats_quantile_from_sorted_data(address_x + 1 , 2, number, 0.95);
	  *vol_5 = gsl_stats_quantile_from_sorted_data(address_x + 1, 2, number, 0.05);
	  ///TODO:add those functions to particle_system class
	  int number_per_mode = number / 2;
	  double prob_q0 = 0;
	  for(int i = 0; i != number_per_mode; ++i) {
		  prob_q0 += std::tr1::get<0>(*particle_system.get_particles())(i);
	  }
	  *prob_q = prob_q0;
	  for(int i =0; i!=9; ++i) {
		  mean_params[i] = (std::tr1::get<3>(*particle_system.get_particles())).row(i).matrix() *
				         (std::tr1::get<0>(*particle_system.get_particles())).matrix();
	  }
	  //return static_cast<void*>(&particle_system);
	  return static_cast<void*>(&isInitialized);
}

}
