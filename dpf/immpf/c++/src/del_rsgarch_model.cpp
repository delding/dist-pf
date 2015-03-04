/*
 * del_rsgarch_model.cpp
 *
 *  Created on: Jan 18, 2013
 *      Author: erli
 */


#include "del_rsgarch_model.hpp"

namespace del {

  RSGARCH::RSGARCH():p0_(1.0), p1_(0.0) {}

  RSGARCH::RSGARCH(double p0, double p1):p0_(p0), p1_(p1) {}

  ///employ predictive expectation as predictive state,
  ///and in this model predictive expectation is mu since the conPdf here is gaussian with mean mu and variance vol
  ///and of cause vol is no random given the value of vol at time t-1
  ModelBase<RSGARCH>::ConVars RSGARCH::predicState(double t, const ConVars& pre_x, const DisVars& cur_q, const Params& param) const {
	  RSGARCH::ConVars x;
	  int cq = cur_q(0) + 1;
	  double epsilon = pre_x(0) - param(0);
	  x(1) = param(cq) + param(3) * epsilon * epsilon + param(4) * pre_x(1);
	  x(0) = param(0);
	  return x;
  }

  ModelBase<RSGARCH>::ConVars RSGARCH::xInit(gsl_rng* rng) const {
	  ConVars x0;
	  //initialize return with truncated Gaussian on (-5, 5)
	  x0(0) = gsl_cdf_gaussian_Pinv(gsl_cdf_gaussian_P(-5, 1) + gsl_ran_flat(rng, 0, 1) *
			  	  	  	  	  	  	    (gsl_cdf_gaussian_P(5, 1) - gsl_cdf_gaussian_P(-5, 1)), 1);
	  //initialize volatility with Beta(2,5)
	  x0(1) = gsl_ran_beta(rng, 2, 5);
	  return x0;
  }

  ModelBase<RSGARCH>::DisVars RSGARCH::qInit(gsl_rng* rng) const {
	  //initialize regime state with 0 (low vol regime), while 1 corresponding to high vol regime
	  double uniform = gsl_ran_flat(rng, 0.0, 1.0);
	  if (uniform < p0_)
		  return DisVars::Constant(0);
	  return DisVars::Constant(1);
  }

  ModelBase<RSGARCH>::Params RSGARCH::paramInit(gsl_rng* rng) const {
	  Params params;
	  params(0) = gsl_ran_gaussian(rng, 1);//mu
	  params(1) = gsl_ran_gamma(rng, 2, 0.2);//c1
	  params(2) = gsl_ran_gamma(rng, 2, 0.2);//c2
	  if(params(1) > params(2)) {
		  double temp = params(1);
		  params(1) = params(2);
		  params(2) = temp;
	  }
	  params(3) = gsl_ran_beta(rng, 1, 9);//alpha
	  do {
		  params(4) = gsl_ran_beta(rng, 9, 1);
	  }
	  while (params(3) + params(4) > 1);//alpha + beta <1
	  params(5) = gsl_ran_laplace(rng, 1);//a11
	  params(6) = -10 + gsl_ran_laplace(rng, 1);//b11
	  params(7) = -5 + gsl_ran_laplace(rng, 1);//a22
	  params(8) = 10 + gsl_ran_laplace(rng, 1);//b22
	  return params;
  }

}



