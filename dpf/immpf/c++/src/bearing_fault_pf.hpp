/*
 * bearing_fault_pf.hpp
 *
 *  Created on: Feb 12, 2013
 *      Author: erli
 */

#ifndef BEARING_FAULT_PF_HPP_
#define BEARING_FAULT_PF_HPP_


#include <tr1/tuple>
#include "del_filter.hpp"

///TODO:use <limits> tests if at anyplace of algorithm there occurs a Nan
//#include <limits>

namespace del {

//TODO: move this function to a separate header file
// Generate samples from the multivariate Gaussian distribution with specified
// mean and variance using Box-Muller algorithm.
// dim:	the dimension of the distribution
// r:  	pointer to a constant gsl random number generator
// return:   random Gaussian vector
  template <typename Mean, typename Var, typename Normal>
  inline void MultiGaussianRV(const gsl_rng* r, const Eigen::ArrayBase<Mean>& mean,
							const Eigen::MatrixBase<Var>& var, const Eigen::ArrayBase<Normal>& normal_rv) {
	  int dim = mean.rows();
	  Eigen::MatrixXd temp_rv(dim, 1);
	  for (int i = 0; i != dim; ++i) {
		  temp_rv(i, 0) = gsl_ran_gaussian(r, 1);
	  }
	  Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > cholesky_decomposition;
	  cholesky_decomposition.compute(var);
	  const_cast<Eigen::ArrayBase<Normal>& >(normal_rv) = mean.matrix() + (cholesky_decomposition.matrixL() * temp_rv);
  }


/*
 * Auxiliary IMM Particle Filter with particle learning for hybrid states with unknown parameters
 * system used is hybrid system with unknown parameters (HU) but no need to store discrete states
 * assume DisVars is one dimensional
 */

  template<typename System>
  class Filter<Aux_IMM, System> {
  public:
	  Filter();
	  Filter(const System&);
	  double get_a() {return a_;}
	  void set_a(double a) {a_ = a;}
	  void set_particle_numbers(int number);
	  void initialize(System&);//no const since qProb_ will be modified
	  template <typename ResampleType>
	  void doIterate(System&, const ResampleType&);

  private:
	  ///a_ measure the extent of the shrinkage
	  double a_;
	  int mode_number_;
	  int number_per_mode_;
	  ///mode switching probability, row number decided by number of mode
	  Eigen::Array<double, Eigen::Dynamic, 1> qProb_;
	  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> m_params_;
	  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> resampled_m_params_;
	  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> resampled_params_;
	  ///array storing predictive likelihood for each iteration, row number decided by number of particles, column number decided by mode number
	  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> pre_likelihood_;
	  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> resampled_x_;
	  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> resampled_q_;//TODO:seems useless
  };

  template<typename System>
  Filter<Aux_IMM, System>::Filter() : a_(0.98), mode_number_(1), number_per_mode_(0),
  qProb_(Eigen::Array<double, 1, 1>::Constant(0.0)), m_params_(Eigen::Array<double, 1, 1>::Constant(0.0)),
  resampled_m_params_(Eigen::Array<double, 1, 1>::Constant(0.0)), resampled_params_(Eigen::Array<double, 1, 1>::Constant(0.0)),
  pre_likelihood_(Eigen::Array<double, 1, 1>::Constant(0.0)), resampled_x_(Eigen::Array<double, 1, 1>::Constant(0.0)),
  resampled_q_(Eigen::Array<int, 1, 1>::Constant(0)){}

  template<typename System>
  Filter<Aux_IMM, System>::Filter(const System& system) : a_(0.98), mode_number_(system.dim_q()), number_per_mode_(system.number_ / mode_number_) {
	  m_params_.resize(system.dim_param(), system.number_);
	  qProb_.resize(mode_number_, 1);
	  resampled_m_params_.resize(system.dim_param(), system.number_);
	  resampled_params_.resize(system.dim_param(), system.number_);
	  pre_likelihood_.resize(system.number_, mode_number_);
	  resampled_x_.resize(system.dim_x(), system.number_);
	  resampled_q_.resize(1, system.number_);//assume discrete state dimension is one, dim_q()returns the number of mode, TODO: use dim_q() to return dimension of q while get_mode_number() return number of mode
  }

  template<typename System>
  void Filter<Aux_IMM, System>::set_particle_numbers(int number) {
	  number_per_mode_ = number / mode_number_;
	  m_params_.resize(Eigen::NoChange, number);
	  resampled_m_params_.resize(Eigen::NoChange, number);
	  resampled_params_.resize(Eigen::NoChange, number);
	  pre_likelihood_.resize(number, mode_number_);
	  resampled_x_.resize(Eigen::NoChange, number);
	  resampled_q_.resize(1, number);
  }

  template<typename System>
  void Filter<Aux_IMM, System>::initialize(System& system) {
	  system.cur_time_ = 0.0;
	  system.observed_data_.reset(new typename System::ObVars);
	  qProb_ = system.model_ -> get_qInitProb();
	  for(int i = 0; i != mode_number_; ++i) {
		  for(int k = 0; k != number_per_mode_; ++k) {
			  ///model_ -> call doesn't work as member functions are private, need to use model<base> accessor
			  ///TODO:I make those member functions public at the time of testing algo
			  std::tr1::get<3>(*(system.particles_)).col(k + i * number_per_mode_) = system.model_ -> paramInitialize(system.rng_);
			  std::tr1::get<1>(*(system.particles_)).col(k + i * number_per_mode_) = system.model_ -> xInitialize(system.rng_);
			  std::tr1::get<2>(*(system.particles_)).col(k + i * number_per_mode_) = Eigen::Array<int,1,1>::Constant(i);
			  std::tr1::get<0>(*(system.particles_))(k + i * number_per_mode_) = qProb_[i] / number_per_mode_;
		  }
	  }
	  return;
  }

  template<typename System>
  template <typename ResampleType>
  void Filter<Aux_IMM, System>::doIterate(System& system, const ResampleType& resample_type) {
	  ///mode switching
	  for(int i = 0; i != mode_number_; ++i) {
		  qProb_(i) = 0.0;
		  for(int j = 0; j != mode_number_; ++j)
			  for(int k = 0; k != number_per_mode_; ++k) {
			/*	  //debug
				  double tempweight= std::tr1::get<0>(*(system.particles_))(k + j * number_per_mode_);
				  double temptranprob = system.model_ -> disProb(system.cur_time_, Eigen::Array<int,1,1>::Constant(i), Eigen::Array<int,1,1>::Constant(j),
  	  	  	  	  	  	  	std::tr1::get<1>(*(system.particles_)).col(k + j * number_per_mode_),
  	  	  	  	  	  	  	std::tr1::get<3>(*(system.particles_)).col(k + j * number_per_mode_));

				  double tempq = std::tr1::get<0>(*(system.particles_))(k + j * number_per_mode_) *
					       ( system.model_ -> disProb(system.cur_time_, Eigen::Array<int,1,1>::Constant(i), Eigen::Array<int,1,1>::Constant(j),
					  	  	  	  	  	  	  	  	std::tr1::get<1>(*(system.particles_)).col(k + j * number_per_mode_),
					  	  	  	  	  	  	  	  	std::tr1::get<3>(*(system.particles_)).col(k + j * number_per_mode_)) );
				  std::cout<<tempweight<<std::endl<<temptranprob<<std::endl<<tempq<<std::endl;

			*/	  //debug
				  qProb_(i) += std::tr1::get<0>(*(system.particles_))(k + j * number_per_mode_) *
						       ( system.model_ -> disProb(system.cur_time_, Eigen::Array<int,1,1>::Constant(i), Eigen::Array<int,1,1>::Constant(j),
						  	  	  	  	  	  	  	  	std::tr1::get<1>(*(system.particles_)).col(k + j * number_per_mode_),
						  	  	  	  	  	  	  	  	std::tr1::get<3>(*(system.particles_)).col(k + j * number_per_mode_)) );

			  }
	  }

	  //debug
	  //std::cout<<"qProb_(0) = "<< qProb_(0) ;std::cout<<std::endl;;
	  //std::cout<<" qProb_(1)"<< qProb_(1);std::cout <<std::endl;
	  //debug

	  ///interaction resampling
	  typename System::Params params_mean = std::tr1::get<3>(*(system.particles_)).matrix() * std::tr1::get<0>(*(system.particles_)).matrix();
	  m_params_ = (a_ * std::tr1::get<3>(*(system.particles_))).colwise() + (1 - a_) * params_mean;
	  Eigen::MatrixXd params_cov(system.dim_param(), system.dim_param());
	  params_cov = (1 - a_ * a_) * ((std::tr1::get<3>(*(system.particles_)).colwise() - params_mean).rowwise() *
			   	    std::tr1::get<0>(*(system.particles_)).transpose()).matrix() *
			   	   (std::tr1::get<3>(*(system.particles_)).colwise() - params_mean).transpose().matrix();
	  for(int i = 0; i != mode_number_; ++i) {
		  for(int j = 0; j != mode_number_; ++j)
			  for(int k = 0; k != number_per_mode_; ++k) {
				  pre_likelihood_(k + j * number_per_mode_,i) = system.model_ -> obProb(system.cur_time_, *(system.observed_data_),
						                                                                system.model_ -> predicState(system.cur_time_,
						                                                            		                      /*pre_x*/std::tr1::get<1>(*(system.particles_)).col(k + j * number_per_mode_),
						                                                            		                      /*cur_q*/Eigen::Array<int,1,1>::Constant(i),
						                                                            		                      /*params*/std::tr1::get<3>(*(system.particles_)).col(k + j * number_per_mode_)),
						                                                                Eigen::Array<int,1,1>::Constant(i), m_params_.col(k + j * number_per_mode_));

				  //debug
				  //std::cout<<"pre_likelihood_(k + j * number_per_mode_,i) = ";std::cout<<pre_likelihood_(k + j * number_per_mode_,i);std::cout<<std::endl;
				  //debug

		  ///compute resampling weights
				  (*(system.resample_weight_))(k + j * number_per_mode_) = pre_likelihood_(k + j * number_per_mode_, i) *
						  	  	  	  	  	  	  	  	  	  	  	  	   system.tranProbDis(Eigen::Array<int,1,1>::Constant(i), Eigen::Array<int,1,1>::Constant(j),
						  	  	  	  	  	  	  	  	  	  	  	  			   	   	   	  std::tr1::get<1>(*(system.particles_)).col(k + j * number_per_mode_),
						  	  	  	  	  	  	  	  	  	  	  	  			   	   	   	  std::tr1::get<3>(*(system.particles_)).col(k + j * number_per_mode_)) *
						  	  	  	  	  	  	  	  	  	  	  	  	   (std::tr1::get<0>(*(system.particles_)))(k + j * number_per_mode_);
				  //debug
				  //std::cout<<"(*(system.resample_weight_))(k + j * number_per_mode_) = ";std::cout<<(*(system.resample_weight_))(k + j * number_per_mode_);std::cout<<std::endl;
				  //debug
			  }

		  ///normalize
		  ///!!!I put the following normalize statement in the upper inner loop,and make a big mistake generating NANs
		  ///TODO:reexamine algos when done coding them, it takes 10mins checking but saves 10hours debugging
		  ///one NaN in the array, every entry becomes NaN after being normalized
		  *(system.resample_weight_) /= (*(system.resample_weight_)).sum();

		  ///resample number_per_mode particles from number_ particles for mode i
		  system.set_resample_number(number_per_mode_);
		  resample_type.resample(system.rng_, system.number_, *(system.resample_weight_), system.resample_number_, *(system.resample_count_), *(system.resample_index_));
		  for(size_t k = 0, j = 0; k != system.number_; ++k) {
			  while ((*(system.resample_count_))[k] >= 1) {
				  ///using one resampled index array will save much more memory allocation and copy
				  resampled_x_.col(j + i * number_per_mode_) = std::tr1::get<1>(*(system.particles_)).col(k);
				  resampled_q_(j + i * number_per_mode_) = std::tr1::get<2>(*(system.particles_))(k);
				  resampled_params_.col(j + i * number_per_mode_) = std::tr1::get<3>(*(system.particles_)).col(k);
				  ///resample only means sampling some particles more than one times, values such as m_params still work at this step for the corresponding resampled particles
				  resampled_m_params_.col(j + i * number_per_mode_) = m_params_.col(k);
				  --(*(system.resample_count_))[k];
				  ++j;
			  }
		  }
	  }
	  ///propogate parameters, using the same cov of params but use resampled mean of params
	  ///this is where lots of NaNs are borned, say sqrt(negative value)
	  ///the positivity of parameters can be destroyed since the domain of multivariate Gaussian cover the whole axis
	  for(size_t k = 0; k != system.number_; ++k) {
		  MultiGaussianRV(system.rng_, resampled_m_params_.col(k), params_cov, std::tr1::get<3>(*(system.particles_)).col(k));

		  ///test positivity of the value of alpha and if false draw a new one from its initial density(here set it zero)
		  ///TODO: not testing other parameters
		  if(std::tr1::get<3>(*(system.particles_))(3,k) < 0) {
			  std::tr1::get<3>(*(system.particles_))(3,k) = 0.0;
		  }
		  ///test if alpha + beta <=1 to prevent explosion of vol (here, if false set beta = 1 - alpha)
		  ///TODO:improve
		  if(std::tr1::get<3>(*(system.particles_))(3,k) + std::tr1::get<3>(*(system.particles_))(4,k) > 1) {
			  std::tr1::get<3>(*(system.particles_))(4,k) = 1 - std::tr1::get<3>(*(system.particles_))(3,k);
		  }
		  ///test positivity of c0 and c1 to prevent vol changing its sign (if ture, set to 0)
		  if(std::tr1::get<3>(*(system.particles_))(1,k) < 0) {
			  std::tr1::get<3>(*(system.particles_))(1,k) = 0;
		  }
		  if(std::tr1::get<3>(*(system.particles_))(2,k) < 0) {
			  std::tr1::get<3>(*(system.particles_))(2,k) = 0;
		  }
	  }
	  ///propogate x
	  ///particle_system class should incorporate more interface, otherwise it becomes useless
	  for(size_t k = 0; k != system.number_; ++k) {
			  std::tr1::get<1>(*(system.particles_)).col(k) = system.model_ -> conModel(system.cur_time_, resampled_x_.col(k), std::tr1::get<2>(*(system.particles_)).col(k),
					  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	std::tr1::get<3>(*(system.particles_)).col(k), system.rng_);
	  }

	  ///update weights
	  for(int i = 0; i != mode_number_; ++i)
		  for(int k = 0; k != number_per_mode_; ++k) {
			  (std::tr1::get<0>(*(system.particles_)))(k + i * number_per_mode_) = qProb_(i) *
				  system.model_ -> obProb(system.cur_time_, *(system.observed_data_), std::tr1::get<1>(*(system.particles_)).col(k + i * number_per_mode_), Eigen::Array<int,1,1>::Constant(i),
						  	  	    std::tr1::get<3>(*(system.particles_)).col(k + i * number_per_mode_)) /
				  system.model_ -> obProb(system.cur_time_, *(system.observed_data_), system.model_ -> predicState(system.cur_time_, resampled_x_.col(k + i * number_per_mode_),
						  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	   Eigen::Array<int,1,1>::Constant(i), resampled_params_.col(k + i * number_per_mode_)),
						  	  	  	Eigen::Array<int,1,1>::Constant(i), resampled_m_params_.col(k + i * number_per_mode_));

	  }
	  ///!!there needs only one NaN in the weight array, the normalized weight array will have all entries NaNs
	  system.normalizeWeight();
  }

}



#endif /* BEARING_FAULT_PF_HPP_ */
