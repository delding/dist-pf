/*
 * del_model_trait.hpp
 *
 *  Created on: Oct 10, 2012
 *      Author: erli
 */

#ifndef DEL_MODEL_TRAIT_HPP_
#define DEL_MODEL_TRAIT_HPP_

namespace del {
  /*
   * Define scalar type: ConState, DisState, Observe, Param
   * their dimensions: dim_x(), dim_q(), dim_y(), dim_param()
   * and their container type: ConVars, DisVars, ObVars, Params
   * If not appears, define it as void type
   * Particles type also need to be defined
   *
   *  e.g.
   *  typedef Eigen::Array<ConState, dim_x, 1> ConVars;
	  typedef Eigen::Array<DisState, dim_q, 1> DisVars;
	  typedef Eigen::Array<Observe, dim_y, 1> ObVars;
	  typedef Eigen::Array<Param, dim_param, 1> Paramss;
	  typedef std::tr1::tuple<Eigen::ArrayXd,
			  	  	  	  	  Eigen::Array<ConState, dim_x, Eigen::Dynamic>,
			  	  	  	  	  Eigen::Array<DisState, dim_q, Eigen::Dynamic>,
			  	  	  	  	  Eigen::Array<Param, dim_param, Eigen::Dynamic> > Particles;
	  tuple<weights, continuous states, discrete states, parameters>
	  row number represents dimension, column number represents particles number
   *
   */

  enum ModelType {CK = 0, CU, HK, HU};

  //primary trait class
  template <typename Model>
  class ModelTrait {};

}



#endif /* DEL_MODEL_TRAIT_HPP_ */
