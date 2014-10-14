/*
 * del_filter.h
 *
 *  Created on: Sep 28, 2012
 *      Author: erli
 */

#ifndef DEL_FILTER_H_
#define DEL_FILTER_H_

#include <boost/scoped_ptr.hpp>
#include "del_resample.h"


namespace del {

  enum FilterType { Bootstrap, APF, Aux_IMM };
  ///Base class for all filter classes
  //class FilterBase {};
  ///all the filter type is a specification of class Filter
  ///so that they are all friends of class ParticleSystem
  template <FilterType type, typename System>
  class Filter {};
  /*
   * Bootstrap Filter for continuous state with known parameters
   */
  ///TODO:make System as a template argument of member functions
  template<typename System>
  class Filter<Bootstrap, System> {
  public:
	  //instance of BootstrapFilter can not call following member fuctions of FilterBase (method could not be resolved)
	  //TODO:after using qualified name scope FilterBase, why errors of invalid arguments occur??

	  /*
	  using FilterBase::set_observed_data;
	  using FilterBase::initializeConState;
	  using FilterBase::iterate;
	  using FilterBase::get_number;
	  using FilterBase::get_particles;
	  */
	  //typedef is also invisible unless either use qualified name scope FilterBase:: or use using typename FilterBase::ObVar
	  Filter();
	  template <typename ResampleType>
	  void doIterate(const System&, const ResampleType&) const;
  };

  template<typename System>
  Filter<Bootstrap, System>::Filter() {}

  template<typename System>
  template <typename ResampleType>
  void Filter<Bootstrap, System>::doIterate(const System& system, const ResampleType& resample_type) const {
	  ///propagate
	  ///for template inheritance,
	  ///this should be used explicitly when derived class use data member or member function of base class
	  system.propagateCon();
	  ///update weights
	  /* TODO: see if broadcasting works here
	   *
	   * std::tr1::get<0>(*particles_).rowwise() *=
	   * 			likelihood(observed_data_[cur_time_  - 1], std::tr1::get<1>(*particles_).colwise());
	   */
	  for(size_t k = 0; k != system.number_; ++k) {
		  (std::tr1::get<0>(*(system.particles_)))(k) *=
				  system.likelihood(*(system.observed_data_), std::tr1::get<1>(*(system.particles_)).col(k));
	  }
	  ///normalized weights
   	  system.normalizeWeight();
	  ///resampling
	  if (system.computeESS(system.cur_time_) < system.resample_threshold_) {
		  ///set resampling weight
		  *(system.resample_weight_) = std::tr1::get<0>(*(system.particles_));
		  system.resample(resample_type);
		  std::tr1::get<0>(*(system.particles_)) = Eigen::ArrayXd::Constant(
				  	  	  	  	  	  	  	  system.number_, 1.0 / static_cast<double>(system.number_));
	  }
  }

  /*
   * Auxiliary Particle Filter for continuous state with known parameters
   */
  ///using default Trait
  template<typename System>
  class Filter<APF, System> {
  public:
	  typedef typename System::ConVars ConVars;
	  Filter();
	  template <typename Callable>
	  Filter(const System&, const Callable& pre_state_fn);
	  template <typename Callable>
	  void set_pre_state_fn(const Callable& pre_state_fn) {pre_state_fn_ = pre_state_fn;}
	  template <typename ResampleType>
	  void doIterate(const System&, const ResampleType&) const;
  private:
	  ///array storing predictive likelihood for each iteration, row number is decided by number of particles
	  boost::scoped_ptr<Eigen::Array<double, Eigen::Dynamic, 1> > pre_likelihood_;
	  ///numerical integration algorithm to calculate the integral of x*p(x|x(t-1))dx, predictive expectation of x(t-1),
	  ///function x*p(x|x(t-1)) with t  and x(t-1) as inputs, predictive expectation as output
	  ///TODO:can not use template expression as input here, any way to improve?
	  ///the implementation is put in the model class or some where else
	  ///define two static member function in model class
	  ///one is c-type x*p(x|x(t-1)) and the other is ConVars(const ConVars&) using gsl numerical integration inside
	  std::tr1::function<ConVars(double t, const ConVars&)> pre_state_fn_;
  };

  template<typename System>
  Filter<APF, System>::Filter() : pre_likelihood_(new Eigen::Array<double, Eigen::Dynamic, 1>), pre_state_fn_(NULL) {}
  template<typename System>
  template <typename Callable>
  Filter<APF, System>::Filter(const System& system, const Callable& pre_state_fn)
  :pre_likelihood_(new Eigen::Array<double, Eigen::Dynamic, 1>(system.number_)), pre_state_fn_(pre_state_fn) {}

  template<typename System>
  template <typename ResampleType>
  void Filter<APF, System>::doIterate(const System& system, const ResampleType& resample_type) const {
	  //must use this to indicate data member or member function deriving from template base class
	  pre_likelihood_ -> resize(system.number_, Eigen::NoChange);
	  for(size_t k = 0; k != system.number_; ++k) {
		  ///integral_ taking const ConVars& as an argument, the return of col(k) can be passed as const ConVars&
		  ///expression will implicitly be evaluated into a temporary ConVars object
		  ///this means that you lose the benefit of expression templates. Concretely, this has two drawbacks:
		  ///The evaluation into a temporary may be useless and inefficient;
		  ///This only allows the function to read from the expression, not to write to it.
		  (*pre_likelihood_)[k] = system.likelihood(*(system.observed_data_),
				  pre_state_fn_(system.cur_time_, std::tr1::get<1>(*(system.particles_)).col(k)));
	  }
	  ///set resampling weight
	  (*(system.resample_weight_)) = (std::tr1::get<0>(*(system.particles_))) * (*pre_likelihood_);
	  ///normalize resample weight
	  *(system.resample_weight_) /= (*(system.resample_weight_)).sum();
	  ///resampling
	  system.resample(resample_type);

	  ///TODO:following computation could be avoided
	  for(size_t k = 0; k != system.number_; ++k) {
		  (*pre_likelihood_)[k] = system.likelihood(*(system.observed_data_),
				  pre_state_fn_(system.cur_time_, std::tr1::get<1>(*(system.particles_)).col(k)));
	  }

	  ///propagate
	  system.propagateCon();
	  ///update weight
	  for(size_t k = 0; k != system.number_; ++k) {
		  (std::tr1::get<0>(*(system.particles_)))(k) =
				  system.likelihood(*(system.observed_data_),
						  	  	    std::tr1::get<1>(*(system.particles_)).col(k)) / (*pre_likelihood_)(k);
	  }
	  ///normalize weights
   	  system.normalizeWeight();
  }


}


#endif /* DEL_FILTER_H_ */
