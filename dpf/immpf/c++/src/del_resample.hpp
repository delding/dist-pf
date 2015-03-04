/*
 * del_resample.hpp
 *
 *  Created on: Sep 28, 2012
 *      Author: erli
 */

#ifndef DEL_RESAMPLE_HPP_
#define DEL_RESAMPLE_HPP_

#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <Eigen/Dense>

/*
 * Weight, Count and Index are Eigen types
 */

namespace del {
  /*
   * notice arguments of resample fuction are nonconst reference to ArrayBase,
   * so only Array object works, template expression doesn't work
   * if use const&, const_cast need to be used inside resample function
   */
  class ResampleBase {};

  class MultinomialResample : ResampleBase {
  public:
	  template<typename Weight, typename Count, typename Index>
	  void resample(const gsl_rng*, size_t, Eigen::ArrayBase<Weight>&, unsigned int,
			  	    Eigen::ArrayBase<Count>&, Eigen::ArrayBase<Index>&) const;
  };

  class ResidualResample : ResampleBase{
  public:
	  template<typename Weight, typename Count, typename Index>
	  void resample(const gsl_rng*, size_t, Eigen::ArrayBase<Weight>&, unsigned int,
			  	    Eigen::ArrayBase<Count>&, Eigen::ArrayBase<Index>&) const;
  };

  class StratifiedResample : ResampleBase{
  public:
	  template<typename Weight, typename Count, typename Index>
	  void resample(const gsl_rng*, size_t, Eigen::ArrayBase<Weight>&, unsigned int,
			  	    Eigen::ArrayBase<Count>&, Eigen::ArrayBase<Index>&) const;
  };

  class SystematicResample : ResampleBase{
  public:
	  template<typename Weight, typename Count, typename Index>
	  void resample(const gsl_rng*, size_t, Eigen::ArrayBase<Weight>&, unsigned int,
			  	    Eigen::ArrayBase<Count>&, Eigen::ArrayBase<Index>&) const;
  };


  template<typename Weight, typename Count, typename Index>
  void MultinomialResample::resample(const gsl_rng* rng, size_t number,
		  	  	  	  	  	  	  	 Eigen::ArrayBase<Weight>& resample_weight,
		  	  	  	  	  	  	  	 unsigned int resample_number,
		  	  	  	  	  	  	  	 Eigen::ArrayBase<Count>& resample_count,
		  	  	  	  	  	  	  	 Eigen::ArrayBase<Index>& resample_index) const {
	  ///take naked pointer from Eigen::Array object
	  gsl_ran_multinomial(rng, number, resample_number, &resample_weight(0), &resample_count(0));
  }

  template<typename Weight, typename Count, typename Index>
  void ResidualResample::resample(const gsl_rng* rng, size_t number,
		  	  	  	  	  	  	  Eigen::ArrayBase<Weight>& resample_weight,
		  	  	  	  	  	  	  unsigned int resample_number,
		  	  	  	  	  	  	  Eigen::ArrayBase<Count>& resample_count,
		  	  	  	  	  	  	  Eigen::ArrayBase<Index>& resample_index) const {
      unsigned int residual_number = resample_number;
	  resample_weight *= resample_number;
      for(size_t k = 0; k != number; ++k) {
		  resample_index[k] = std::floor(resample_weight[k]);
		  resample_weight[k] -= resample_index[k];
	  }
	  residual_number -= resample_index.sum();
	  gsl_ran_multinomial(rng, number, residual_number, &resample_weight(0), &resample_count(0));
	  for(size_t k = 0; k != number; ++k) {
		  resample_count[k] += static_cast<int>(resample_index[k]);//don't have same scalar type, so can't use: resample_count += resample_index;
	  }
  }

  template<typename Weight, typename Count, typename Index>
  void StratifiedResample::resample(const gsl_rng* rng, size_t number,
		  	  	  	  	  	  	    Eigen::ArrayBase<Weight>& resample_weight,
		  	  	  	  	  	  	    unsigned int resample_number,
		  	  	  	  	  	  	    Eigen::ArrayBase<Count>& resample_count,
		  	  	  	  	  	  	    Eigen::ArrayBase<Index>& resample_index) const {
	  for (size_t i = 0; i != resample_number; ++i) {
		  resample_index[i] = gsl_ran_flat(rng, static_cast<double>(i), static_cast<double>(i+1))
										  / static_cast<double>(resample_number);
	  }
	  double weight_cumulative = 0.0;
	  size_t i = 0;
	  for (size_t k = 0; k != number; ++k) {
		  int count = 0;
		  weight_cumulative += resample_weight[k];
		  // (i < resample_number) must be the first condition
		  // because when i = resample_number, resample_index[i] in the second condition will fail due to index out of range
		  while(i < resample_number && weight_cumulative > resample_index[i]) {
			  ++count;
			  ++i;
		  }
		  resample_count[k] = count;
	  }
  }

  template<typename Weight, typename Count, typename Index>
  void SystematicResample::resample(const gsl_rng* rng, size_t number,
		  	  	  	  	  	  	  	Eigen::ArrayBase<Weight>& resample_weight,
		  	  	  	  	  	  	  	unsigned int resample_number,
		  	  	  	  	  	  	  	Eigen::ArrayBase<Count>& resample_count,
		  	  	  	  	  	  	  	Eigen::ArrayBase<Index>& resample_index) const {
	 //Generate a random number between 0 and 1/N times the sum of the weights
	  double rand = gsl_ran_flat(rng, 0.0, 1.0) / static_cast<double>(resample_number);
	  double weight_cumulative = 0.0;
	  //check the index k before use it
	  for (size_t k = 0; k != number; ++k) {
		  int count = 0;
		  weight_cumulative += resample_weight[k];
		  while (weight_cumulative > rand) {
			  ++count;
			  rand += 1 / static_cast<double>(resample_number);
		  }
		  resample_count[k] = count;
	  }
  }
}

#endif /* DEL_RESAMPLE_HPP_ */
