/*
 * del_python_ext.h
 *
 *  Created on: Jan 23, 2013
 *      Author: erli
 */

#ifndef DEL_PYTHON_EXT_H_
#define DEL_PYTHON_EXT_H_

//namespace del {
void generateData(int model_index, double time_length, double time_step,
		  	  	  const char* x_file, const char* q_file, const char* param_file, const char* ob_file);

extern "C" {
  void* filtering(int model_index, int filter_index, int resample_index, double time_step, int number,
  		  	     const char* data_file, double* weight, double* state,
  		  	     double* state_vol, double* p0, double* p1, double* p2, double* p3, double* p4, double* p5,
  		  	     double* p6, double* p7, double* p8, double* mean_x, double* mean_vol,
  		  	     double* x_95, double* x_5, double* vol_95, double* vol_5,double* prob_q,
  		  	     double* mean_params);

  int initializeSystem(void* ptr_isInitialized);


}


//}

#endif /* DEL_PYTHON_EXT_H_ */
