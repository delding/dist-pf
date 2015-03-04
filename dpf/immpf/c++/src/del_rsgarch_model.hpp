/*
 * del_rsgarch_model.hpp
 *
 *  Created on: Jan 18, 2013
 *      Author: erli
 */

#ifndef DEL_RSGARCH_MODEL_HPP_
#define DEL_RSGARCH_MODEL_HPP_

#include <cmath>
#include <tr1/tuple>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include "del_model.hpp"

using namespace Eigen;

namespace del {

  class RSGARCH;

  template <>
  class ModelTrait<RSGARCH> {
  public:
	  typedef double Observe;
	  typedef Array<double,2,1> ConVars;
	  typedef Array<int,1,1> DisVars;
	  //parameters: mu, c0, c1, alpha, beta, a00, b00, a11, b11
	  typedef Array<double,9,1> Params;
	  typedef Array<double,1,1> ObVars;
	  typedef std::tr1::tuple<Eigen::ArrayXd,
			  	  	  	  	  Eigen::Array<double, 2, Eigen::Dynamic>,
			  	  	  	  	  Eigen::Array<int, 1, Eigen::Dynamic>,
			  	  	  	  	  Eigen::Array<double, 9, Eigen::Dynamic> > Particles;
	  static ModelType model_type() {return HU;}
	  static int dim_x() {return 2;}
	  //two discrete states represented by one dimensional int supported by {0, 1}, TODO: use get_mode_number() instead
	  static int dim_q() {return 2;}
	  static int dim_param() {return 9;}
	  static int dim_y() {return 1;}
  };

  class RSGARCH : public del::ModelBase<RSGARCH> {
  public:
	  RSGARCH();
	  RSGARCH(double p0, double p1);
	  void set_qInitProb(double p0, double p1) {p0_ = p0; p1_ = p1;}
	  Array<double, 2, 1> get_qInitProb() const {Array<double, 2, 1> temp; temp(0) = p0_; temp(1) = p1_; return temp;}

	  //struct ConPdfParams { double t; double pre_x; int cur_q; double mu; double c0; double c1; double alpha; double beta; };
	  //static double xTimesPx(double *x, size_t dim, void* params);
	  ConVars predicState(double t, const ConVars& pre_x, const DisVars& cur_q, const Params& param) const;
  //!!! temporarily makes all member functions public
  //private:
	  friend class ModelAccessor;
	  //because ModelBase<RSGARCH> is not a dependent base class, its typedef can be used without fully qualified
	  ConVars xInit(gsl_rng* rng) const;
	  DisVars qInit(gsl_rng* rng) const;
	  Params paramInit(gsl_rng* rng) const;
      template<typename PreX, typename CurQ, typename Pa>
	  ConVars conModel(double t, const ArrayBase<PreX>& pre_x, const ArrayBase<CurQ>& cur_q,
			  	  	   const ArrayBase<Pa>& param, gsl_rng* rng) const;
      template<typename PreX, typename PreQ, typename Pa>
      DisVars disModel(double t, const ArrayBase<PreX>& pre_x, const ArrayBase<PreQ>& pre_q,
    		  	  	   const  ArrayBase<Pa>& param, gsl_rng* rng) const;
      template<typename CurX, typename CurQ, typename Pa>
      ObVars obModel(double t, const ArrayBase<CurX>& cur_x, const ArrayBase<CurQ>& cur_q,
    		  	  	 const ArrayBase<Pa>& param, gsl_rng* rng) const;
      template<typename CurX, typename PreX, typename CurQ, typename Pa>
      double conPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const ArrayBase<PreX>& pre_x,
    		  	  	  	   const ArrayBase<CurQ>& cur_q, const ArrayBase<Pa>& param) const;
      template<typename CurQ, typename PreQ, typename PreX, typename Pa>
      double disProb(double t, const ArrayBase<CurQ>& cur_q, const ArrayBase<PreQ>& pre_q,
    		  	  	 const ArrayBase<PreX>& pre_x, const ArrayBase<Pa>& param) const;
      template<typename CurY, typename CurX, typename CurQ, typename Pa>
      double obProb(double t, const ArrayBase<CurY>& cur_y, const ArrayBase<CurX>& cur_x,
    		  	    const ArrayBase<CurQ>& cur_q, const ArrayBase<Pa>& param) const;

      double p0_, p1_;
  };

  template<typename PreX, typename CurQ, typename Pa>
  ModelBase<RSGARCH>::ConVars RSGARCH::conModel(double t, const ArrayBase<PreX>& pre_x, const ArrayBase<CurQ>& cur_q,
		  	  	   const ArrayBase<Pa>& param, gsl_rng* rng) const {
	  ConVars x;
	  double epsilon = pre_x(0) - param(0);
	  if (cur_q(0) == 0) {
		  x(1) = param(1) + param(3) * epsilon * epsilon + param(4) * pre_x(1);
		  x(0) = param(0) + sqrt(static_cast<double>(x(1))) *  gsl_ran_gaussian(rng, 1.0);
		  return x;
	  }
	  x(1) = param(2) + param(3) * epsilon * epsilon + param(4) * pre_x(1);
	  x(0) = param(0) + sqrt(static_cast<double>(x(1))) * gsl_ran_gaussian(rng, 1.0);
	  return x;
  }

  template<typename PreX, typename PreQ, typename Pa>
  ModelBase<RSGARCH>::DisVars RSGARCH::disModel(double t, const ArrayBase<PreX>& pre_x, const ArrayBase<PreQ>& pre_q,
		  	  	   const ArrayBase<Pa>& param, gsl_rng* rng) const {
	  DisVars q;
	  double p00 = 1 / (1 + exp(param(5) + param(7) * pre_x(1)));
	  double p11 = 1 / (1 + exp(param(6) + param(8) * pre_x(1)));
	  double uniform = gsl_ran_flat(rng, 0.0, 1.0);
	  if (pre_q(0) == 0) {
		  if (uniform < p00)
			  q(0) = 0;
		  else q(0) = 1;
		  return q;
	  }
	  if (uniform < p11)
		  q(0) = 1;
	  else q(0) = 0;
	  return q;
  }

  template<typename CurX, typename CurQ, typename Pa>
  ModelBase<RSGARCH>::ObVars RSGARCH::obModel(double t, const ArrayBase<CurX>& cur_x, const ArrayBase<CurQ>& cur_q,
		  	  	 const ArrayBase<Pa>& param, gsl_rng* rng) const {
	 ObVars y;
	 y(0) = cur_x(0) + gsl_ran_gaussian(rng, 1.0);
	 return y;
  }

  template<typename CurX, typename PreX, typename CurQ, typename Pa>
  double RSGARCH::conPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const ArrayBase<PreX>& pre_x,
		  	  	  	   const ArrayBase<CurQ>& cur_q, const ArrayBase<Pa>& param) const {
	  ///given the info of time t-1, vol is known for sure at time t, so given info of time t-1 the only random variable at time t is r
	  double p_vol, p_r_given_vol;
	  int cq = cur_q(0) + 1;
	  /*
	  p_vol = gsl_ran_gaussian_pdf(sqrt((cur_x(1) - param(4) * pre_x(1) - param(cq)) / (param(3) * pre_x(1))), 1.0)
			  * pow(param(3) * pre_x(1), -0.5) * 0.5 * pow(cur_x(1) - param(4) * pre_x(1) - param(cq), -0.5);
	  p_r_given_vol = gsl_ran_gaussian_pdf((cur_x(0) - param(0)) / sqrt(cur_x(1)), 1.0) / sqrt(cur_x(1));
	  return p_r_given_vol * p_vol;
	  */
	  //cur_x(1) is not random only cur_x(0) is random
	  return gsl_ran_gaussian_pdf((cur_x(0) - param(0)) / sqrt(cur_x(1)), 1.0) / sqrt(cur_x(1));
  }

  template<typename CurQ, typename PreQ, typename PreX, typename Pa>
  double RSGARCH::disProb(double t, const ArrayBase<CurQ>& cur_q, const ArrayBase<PreQ>& pre_q,
		  	  	 const ArrayBase<PreX>& pre_x, const ArrayBase<Pa>& param) const {
	  double p00 = 1 / (1 + exp(param(5) + param(7) * pre_x(1)));
	  double p11 = 1 / (1 + exp(param(6) + param(8) * pre_x(1)));
	  if (pre_q(0) == 0 && cur_q(0) == 0)
		  return p00;
	  else if (pre_q(0) == 0 && cur_q(0) == 1)
		  return 1 - p00;
	  else if (pre_q(0) == 1 && cur_q(0) ==1)
		  return p11;
	  else
		  return 1 - p11;
  }

  template<typename CurY, typename CurX, typename CurQ, typename Pa>
  double RSGARCH::obProb(double t, const ArrayBase<CurY>& cur_y, const ArrayBase<CurX>& cur_x,
		  	    const ArrayBase<CurQ>& cur_q, const ArrayBase<Pa>& param) const {
	  double mean = cur_y(0) - cur_x(0);
	  return gsl_ran_gaussian_pdf(mean, 1.0);
  }

}

#endif /* DEL_RSGARCH_MODEL_HPP_ */
