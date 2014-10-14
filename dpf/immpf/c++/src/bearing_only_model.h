/*
 * bearing_only_model.h
 *
 *  Created on: Oct 8, 2012
 *      Author: erli
 */

#ifndef BEARING_ONLY_MODEL_H_
#define BEARING_ONLY_MODEL_H_

#include <tr1/tuple>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_integration.h>
#include "del_model.h"

using namespace Eigen;

namespace del {
/*
 * 1) forward declaration of derived model Bearingonly
 * 2) define ModelTrait<BearingOnly>
 * 3) with ModelTrait<BearingOnly> having been defined,
 * 	  BearingOnly can then be defined by deriving from ModelBase<BearingOnly>
 */

  class BearingOnly;
  struct ConPdfParams { double t; double pre_x; };

  template<>
  class ModelTrait<BearingOnly> {
  public:
	  typedef double Observe;
	  typedef Array<double,1,1> ConVars;
	  typedef Array<double,1,1> ObVars;
	  typedef void DisVars;
	  typedef void Params;
	  typedef std::tr1::tuple<Eigen::ArrayXd,
			  	  	  	  	  Eigen::Array<double, 1, Eigen::Dynamic> > Particles;
	  static ModelType model_type() {return CK;}
	  static int dim_x() {return 1;}
	  static int dim_y() {return 1;}
  };

  class BearingOnly : public del::ModelBase<BearingOnly> {
  public:
	  static double xTimesPx(double x, void* params);
	  static ConVars predicExp(double t, const ConVars& pre_x);
  private:
	 friend class ModelAccessor;
	 //because ModelBase<BearingOnly> is not a dependent base class, its typedef can be used without fully qualified
	 ConVars xInit(gsl_rng* rng) const;
	 template <typename PreX>
	 ConVars conModel(double t, const Eigen::ArrayBase<PreX>& pre_x, gsl_rng* rng) const;
	 template <typename CurX>
	 ObVars obModel(double t, const Eigen::ArrayBase<CurX>& cur_x, gsl_rng* rng) const;
	 template <typename CurX, typename PreX>
	 double conPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x) const;
	 template <typename CurY, typename CurX>
	 double obProb(double t, const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<CurX>& cur_x) const;
  };


template <typename PreX>
ModelBase<BearingOnly>::ConVars BearingOnly::conModel(double t, const Eigen::ArrayBase<PreX>& x, gsl_rng* rng) const {
    return (x - 1.0)/2 + 25 * x / (1.0 + x * x) + 8 * cos(1.2 * t) + gsl_ran_gaussian(rng, sqrt(10.0));
}

template <typename CurX>
ModelBase<BearingOnly>::ObVars BearingOnly::obModel(double t, const Eigen::ArrayBase<CurX>& x, gsl_rng* rng) const {
    return (x * x / 20 + gsl_ran_gaussian(rng, 1.0));
}

template <typename CurX, typename PreX>
double BearingOnly::conPdf(double t, const Eigen::ArrayBase<CurX>& x, const Eigen::ArrayBase<PreX>& pre_x) const {
	return gsl_ran_gaussian_pdf( (x -((pre_x - 1.0)/2 + 25 * pre_x / (1.0 + pre_x * pre_x) + 8 * cos(1.2 * (t-1))))[0]
				  , sqrt(10.0) );
}

template <typename CurY, typename CurX>
double BearingOnly::obProb(double t, const Eigen::ArrayBase<CurY>& y, const Eigen::ArrayBase<CurX>& x) const{
	double mean = y[0] - x[0] * x[0] / 20;
	return gsl_ran_gaussian_pdf(mean, 1.0);
}




}


#endif /* BEARING_ONLY_MODEL_H_ */
