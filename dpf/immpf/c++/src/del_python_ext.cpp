/*
 * del_python_ext.cpp
 *
 *  Created on: Oct 15, 2012
 *      Author: erli
 */

//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>
#include "del_python_ext.h"
#include "del_data_generator.h"
#include "del_filtering.h"

/* //debug
#include <limits>
*/ //debug


//namespace del {
///TODO:use ctypes instead of boost_python
  void generateData(int model_index, double time_length, double time_step,
		  	  	    const char* x_file, const char* q_file, const char* param_file, const char* ob_file) {
	  if (model_index == 0) {
		  del::BearingOnly model;
		  //dataGenerator(model, time_length, time_step, x_file, q_file, param_file, ob_file);
		  return;
	  }
	  else if (model_index == 1) {
		  del::RSGARCH model;
		  del::dataGenerator(model, time_length, time_step, x_file, q_file, param_file, ob_file);
		  return;
	  }
	  else if (model_index == 2) {
		  return;
	  }
	  else
		  return;
  }

  //  extern "C" {
    //TODO: use structure to get parameters together
void* filtering(int model_index, int filter_index, int resample_index, double time_step, int number,
  		  	     const char* data_file, double* weight, double* state,
  		  	     double* state_vol, double* p0, double* p1, double* p2, double* p3, double* p4, double* p5,
  		  	     double* p6, double* p7, double* p8, double* mean_x, double* mean_vol,
  		  	     double* x_95, double* x_5, double* vol_95, double* vol_5,double* prob_q,
  		  	     double* mean_params) {
	void* ptr_isInitialized = del::RsgarchFiltering(model_index, filter_index, resample_index, time_step, number, data_file, weight, state,
  			  	       state_vol, p0, p1, p2, p3, p4, p5, p6, p7, p8, mean_x, mean_vol, x_95, x_5, vol_95, vol_5, prob_q, mean_params);
    return ptr_isInitialized;
}

//TODO:if bool* instead of void* is used to transfer address,
//isInitialized can be set in python code directly
//TODO:reset the new particle numbers, now if set a new number in GUI and restart, program crashes
int initializeSystem(void* ptr_isInitialized) {
	//del::ParticleSystem<del::RSGARCH, del::HU> *p_system = static_cast<del::ParticleSystem<del::RSGARCH, del::HU>* >(ptr_system);
	bool* isInitialized = static_cast<bool*>(ptr_isInitialized);
	*isInitialized = false;
	return 0;
}

 // }

//}

///need to change the generated shared lib name to del_python_ext
/*BOOST_PYTHON_MODULE(del_python_ext)
{
    using namespace boost::python;
    def("generateData", generateData);
    //def("filtering", del::filtering);
}*/
