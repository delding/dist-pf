/*
 * bearing_only_model.cpp
 *
 *  Created on: Oct 8, 2012
 *      Author: erli
 */

#include "bearing_only_model.h"


namespace del {

  ModelBase<BearingOnly>::ConVars BearingOnly::xInit(gsl_rng* rng) const {
	  return ConVars::Constant(gsl_ran_gaussian(rng, sqrt(10.0)));
  }

  double BearingOnly::xTimesPx(double x, void* params) {
	  ConPdfParams* param = static_cast<ConPdfParams*>(params);
	  double pre_x = param -> pre_x;
	  return x * gsl_ran_gaussian_pdf(
			  (x -((pre_x - 1.0)/2 + 25 * pre_x / (1.0 + pre_x * pre_x) + 8 * cos(1.2 * (param->t-1))))
	                     , sqrt(10.0) );
  }

  ModelBase<BearingOnly>::ConVars BearingOnly::predicExp(double t, const ConVars& pre_x) {
	  ConPdfParams params = {t, pre_x[0]};
	  gsl_function func;
	  func.function = &BearingOnly::xTimesPx;
	  func.params = &params;
	  double result, error;
	  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
	  gsl_integration_qagi(&func, 0, 1e-5, 1000, w, &result, &error);
	  gsl_integration_workspace_free(w);
	  ConVars exp;
	  exp[0] = result;
	  return exp;
  }

}

