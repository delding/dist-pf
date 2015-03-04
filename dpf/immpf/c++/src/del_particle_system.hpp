/*
 * del_particle_system.hpp
 *
 *  Created on: Sep 28, 2012
 *      Author: erli
 */

#ifndef DEL_PARTICLE_SYSTEM_HPP_
#define DEL_PARTICLE_SYSTEM_HPP_

#include <fstream>
#include <list>
#include <tr1/tuple>
#include <tr1/functional>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <gsl/gsl_rng.h>
#include <Eigen/Dense>
#include "del_filter.hpp"
#include "del_model_trait.hpp"

  /*
   *
   * For Hidden Markov models
   * Eigen library is assumed as basic data structure, so that
   * ConVar, DisVar, Obvar, and Params are restricted to those types derived from EigenBase,
   * e.g. class ConVar : public ArrayBase<ConVar> {}; ConVar = Array<dim_x
   * or more generally, class ConVar : public EigenBase<ConVar> {}
   * even if ConVar is double, it must be Array<double, 1, 1> instead of plain double
   *
   * FOR ParicleSystems
   *  tuple<weights (Eigen::Array<double, number,1>,
   * 	   continuous states (Eigen::Array<ConState, x_dim, number>),
   *       discrete states (Eigen::Array<DisState, q_dim, number>),
   *       parameters (Eigen::Array<Param, param_dim, number>) >
   *  row number represents dimension, column number represents number of particles
   *  void is used for partial specification representing types which do not appear
   *
   */

namespace del {
  ///forward declaration
  template <FilterType type, typename System>
  class Filter;

  enum StorageMode {kNotStored = 0, kStored} ;
  //CK:continuous states with known parameters, CU: continuous states with unknown parameters
  //HK:hybrid states with known parameter, HU:hybrid states with unknown parameters

  //primary class
  template <typename Model, ModelType Type = HU, typename Trait = ModelTrait<Model> >
  class ParticleSystem : private boost::noncopyable {
  public:
	  typedef typename Trait::ConVars ConVars;
	  typedef typename Trait::DisVars DisVars;
	  typedef typename Trait::ObVars ObVars;
	  typedef typename Trait::Params Params;
	  typedef typename Trait::Particles Particles;

	  int dim_x() const {return Trait::dim_x();}
	  //number of discrete states TODO:
	  int dim_q() const {return Trait::dim_q();}
	  int dim_param() const {return Trait::dim_param();}
	  int dim_y() const {return Trait::dim_y();}

	  template <FilterType, typename> friend class Filter;

	  ParticleSystem();
	  ParticleSystem(const boost::shared_ptr<Model>& model, double step, unsigned int number, StorageMode storage_mode = kNotStored);
	  ~ParticleSystem();

	  boost::shared_ptr<Model> get_model() const {return model_;}
	  gsl_rng* get_ran_generator() const {return rng_;}//TODO: set and reset seed of random generator
	  unsigned int get_number() const {return number_;}
	  double get_time() const {return cur_time_;}
	  double get_time_step() const {return step_;}
	  unsigned int get_resample_number() const {return resample_number_;}
	  double get_resample_threshold() const {return resample_threshold_;}
	  //returning scoped_ptr makes it be copied, which is not allowed, so use shared_ptr
	  boost::shared_ptr<Particles> get_particles() const {return particles_;}
	  boost::shared_ptr<std::list<Particles> > get_particles_list() const {return particles_list_;}
	  void set_ran_generator(const gsl_rng_type* type) {
		  gsl_rng_free(rng_); rng_ = gsl_rng_alloc(type);}
	  void set_number(unsigned int number);
	  void set_time(double time) {cur_time_ = time;}
	  void set_time_step(double step) {step_ = step;}
	  void set_resample_number(unsigned int number);
	  void set_resample_threshold(double threshold) {resample_threshold_ = threshold;}
	  void set_data_source(const char* filename) {if(data_source_) data_source_.close(); data_source_.open(filename);}
	  void close_data_source() {data_source_.close();}

	  void readData();
	  double computeESS(double time) const;
	  void initializeConState();//set cur_time = 0
	  void initializeDisState();//set cur_time = 0
	  void initializeParam() const;
	  /// Integrate the supplied function using the generation of particles on this node
	  /// time: time at which the integration is to be carried out
	  /// integrand: function which is to be integrated
	  //TODO:do not use tr1::function
	  template<typename DerivedCon, typename DerivedDis, typename DerivedPa>
	  double integrate(double time, const Eigen::ArrayBase<DerivedCon>& con_state,
			  	  	   const Eigen::ArrayBase<DerivedDis>& dis_state,
			  	  	   const Eigen::ArrayBase<DerivedPa>& param,
			  	  	   std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&,
			  	  			   	   	   	   	   	 const Eigen::ArrayBase<DerivedDis>&,
			  	  			   	   	   	   	   	 const Eigen::ArrayBase<DerivedPa>&)> integrand) const;
	  template <typename Filter, typename ResampleType>
	  void iterate(Filter&,const ResampleType&);//no const before Filter& because aux_IMM modify its data
	  void normalizeWeight() const {std::tr1::get<0>(*particles_) /= std::tr1::get<0>(*particles_).sum();}
	  void propagateCon() const;
	  void propagateDis() const;
	  ///transition density for continuous state
	  template <typename CurCon, typename PreCon, typename PreDis, typename DerivedPa>
	  inline double tranPdfCon(const Eigen::ArrayBase<CurCon>&, const Eigen::ArrayBase<PreCon>&,
			            const Eigen::ArrayBase<PreDis>&, const Eigen::ArrayBase<DerivedPa>&) const;
	  ///transition probability for discrete state
	  template <typename CurDis, typename PreDis, typename PreCon, typename DerivedPa>
	  inline double tranProbDis(const Eigen::ArrayBase<CurDis>&, const Eigen::ArrayBase<PreDis>&,
			  	  	  	 const Eigen::ArrayBase<PreCon>&, const Eigen::ArrayBase<DerivedPa>&) const;
	  template <typename CurOb, typename CurCon, typename CurDis, typename DerivedPa>
  	  inline double likelihood(const Eigen::ArrayBase<CurOb>&, const Eigen::ArrayBase<CurCon>&,
  			  	  	    const Eigen::ArrayBase<CurDis>&, const Eigen::ArrayBase<DerivedPa>&) const;
  	  template <typename ResampleType>
	  void resample(const ResampleType&) const;

  private:
	  gsl_rng* rng_;
	  boost::shared_ptr<Model> model_;
	  ///number of particles
	  unsigned int number_;
	  ///current time
	  double cur_time_;
	  ///time step
	  double step_;
	  StorageMode storage_mode_;
	  double resample_threshold_;
	  unsigned int resample_number_;
	  ///resample weight must be normalized
	  boost::scoped_ptr<Eigen::ArrayXd> resample_weight_;
	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_count_;
	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_index_;
	  ///particles at current time t
	  boost::shared_ptr<Particles> particles_;
	  ///the whole list of partiles if storagemode = 1
	  boost::shared_ptr<std::list<Particles> > particles_list_;
	  //observed_data_[t-1] represents data observed at time t
	  //should be point to constant data since observed data should not be changed
	  std::ifstream data_source_;
	  boost::scoped_ptr<ObVars> observed_data_;
  };

  template <typename Model, ModelType Type, typename Trait>
  ParticleSystem<Model, Type, Trait> :: ParticleSystem()
  : model_(NULL), rng_(NULL), number_(0), cur_time_(0.0), step_(1.0), storage_mode_(kNotStored), resample_threshold_(0),
    resample_number_(number_), resample_weight_(NULL), resample_count_(NULL), resample_index_(NULL),
    particles_(NULL), particles_list_(NULL), data_source_(NULL), observed_data_(NULL) {}

  template <typename Model, ModelType Type, typename Trait>
  ParticleSystem<Model, Type, Trait> :: ParticleSystem(
		  	  	  	  	  	  	  	const boost::shared_ptr<Model>& model, double step,
		  	  	  	  	  	  	  	unsigned int number, StorageMode storage_mode)
  : model_(model), number_(number), cur_time_(0.0), step_(step), storage_mode_(storage_mode),
    resample_threshold_(0.5 * number_), resample_number_(number_),
    resample_weight_(new Eigen::ArrayXd(number_)),
    resample_count_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),///eTypey element is initialized to zero
    resample_index_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),
    particles_(new Particles()), particles_list_(new std::list<Particles>()), data_source_(NULL), observed_data_(NULL) {
	  rng_ = gsl_rng_alloc(gsl_rng_ranlux389);
	  std::tr1::get<0>(*particles_).resize(number_);
	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
	  std::tr1::get<2>(*particles_).resize(Trait::dim_q(), number_);
	  std::tr1::get<3>(*particles_).resize(Trait::dim_param(), number_);
  }

  template <typename Model, ModelType Type, typename Trait>
  ParticleSystem<Model, Type, Trait> :: ~ParticleSystem() {
	  gsl_rng_free(rng_);
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: set_number(unsigned int number) {
	  number_ = number;
	  resample_weight_ -> resize(number_);
	  resample_count_ -> resize(number_, Eigen::NoChange);
	  resample_index_ -> resize(number_);//TODO:test later if still works without Eigen::NoChange
	  std::tr1::get<0>(*particles_).resize(number_);
	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
	  std::tr1::get<2>(*particles_).resize(Trait::dim_q(), number_);
	  std::tr1::get<3>(*particles_).resize(Trait::dim_param(), number_);
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: set_resample_number(unsigned int number) {
	  //TODO: consider put resample_number in Resample class
	  resample_number_ = number;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: initializeConState() {
	  cur_time_ = 0.0;
	  observed_data_.reset(new typename Trait::ObVars);
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<1>(*particles_).col(k) = model_ -> xInitialize(rng_);
	  }
	  //TODO:use trait class to decide it is an ArrayXd or a VectorXd here
	  std::tr1::get<0>(*particles_) = Eigen::ArrayXd::Constant(number_, 1.0 / static_cast<double>(number_));
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: initializeDisState() {
	  cur_time_ = 0.0;
	  if(!observed_data_)
		  observed_data_.reset(new typename Trait::ObVars);
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<2>(*particles_).col(k) = model_ -> qInitialize(rng_);
	  }
	  std::tr1::get<0>(*particles_) = Eigen::ArrayXd::Constant(number_, 1.0 / static_cast<double>(number_));
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: initializeParam() const {
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<3>(*particles_).col(k) = (model_ -> paramInitialize(rng_));
	  	  }
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: readData() {
	  for(int i = 0; i < Trait::dim_y(); ++i)
		  data_source_ >> (*observed_data_)(i);
  }

  template <typename Model, ModelType Type, typename Trait>
  double ParticleSystem<Model, Type, Trait> :: computeESS(double time) const {
  	  if (time == cur_time_)
  		  return std::tr1::get<0>(*particles_).sum() * std::tr1::get<0>(*particles_).sum()
  				  / std::tr1::get<0>(*particles_).matrix().squaredNorm();
	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
	  for(int t = 0; t != static_cast<int>(time / step_); ++t)
		  ++current;
	  return std::tr1::get<0>(*current).sum() * std::tr1::get<0>(*current).sum()
			  / std::tr1::get<0>(*current).matrix().squaredNorm();
  }

  template <typename Model, ModelType Type, typename Trait>
  template <typename DerivedCon, typename DerivedDis, typename DerivedPa>
  double ParticleSystem<Model, Type, Trait> :: integrate(
		  	  	  	  	  	 double time, const Eigen::ArrayBase<DerivedCon>& con_state,
		  	  	  	  	  	 const Eigen::ArrayBase<DerivedDis>& dis_state, const Eigen::ArrayBase<DerivedPa>& param,
		  	  	  	  	  	 std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&,
				  	  	  	 const Eigen::ArrayBase<DerivedDis>&,
				  	  	  	 const Eigen::ArrayBase<DerivedPa>&)> integrand) const {
	  double integral = 0.0;
	  if (time == cur_time_) {
		  for (size_t k = 0; k != number_; ++k) {
			  integral += (std::tr1::get<0>(*particles_))(k) * integrand(time,
					  std::tr1::get<1>(*particles_).col(k), std::tr1::get<2>(*particles_).col(k),
					  std::tr1::get<3>(*particles_).col(k));
		  }
		  return integral;
	  }
	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
	  for(size_t t = 0; t != static_cast<int>(time / step_); ++t)
		  ++current;
	  for (size_t k = 0; k != number_; ++k) {
		  integral += (std::tr1::get<0>(*current))(k) * integrand(time,
		  					  std::tr1::get<1>(*current).col(k), std::tr1::get<2>(*current).col(k),
		  					  std::tr1::get<3>(*current).col(k));
	  }
	  return integral;
  }

  template <typename Model, ModelType Type, typename Trait>
  template <typename Filter, typename ResampleType>
  void ParticleSystem<Model, Type, Trait> :: iterate(Filter& filter, const ResampleType& resample_type) {
	  cur_time_ += step_;//in this function, particles_ represents particles at time t-1
	  readData();
	  if (storage_mode_)
		  (*particles_list_).push_back(*particles_);
	  ///const Filter& is OK, since currently doIterate does not change private member
	  filter.doIterate(*this, resample_type);
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: propagateCon() const {
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<1>(*particles_).col(k) = model_ -> xPropagate(cur_time_,
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<1>(*particles_).col(k),
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<2>(*particles_).col(k),
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<3>(*particles_).col(k), rng_);
	  }
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  void ParticleSystem<Model, Type, Trait> :: propagateDis() const {
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<2>(*particles_).col(k) = model_ -> qPropagate(cur_time_,
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<1>(*particles_).col(k),
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<2>(*particles_).col(k),
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<3>(*particles_).col(k), rng_);
	  }
	  return;
  }

  template <typename Model, ModelType Type, typename Trait>
  template <typename CurCon, typename PreCon, typename PreDis, typename DerivedPa>
  double ParticleSystem<Model, Type, Trait> :: tranPdfCon(
		  const Eigen::ArrayBase<CurCon>& cur_con_state, const Eigen::ArrayBase<PreCon>& pre_con_state,
  		  const Eigen::ArrayBase<PreDis>& pre_dis_state, const Eigen::ArrayBase<DerivedPa>& param) const {
	  return model_ -> xTranPdf(cur_time_, cur_con_state, pre_con_state, pre_dis_state, param);
  }

  template <typename Model, ModelType Type, typename Trait>
  template <typename CurDis, typename PreDis, typename PreCon, typename DerivedPa>
  double ParticleSystem<Model, Type, Trait> :: tranProbDis(
		  const Eigen::ArrayBase<CurDis>& cur_dis_state, const Eigen::ArrayBase<PreDis>& pre_dis_state,
  	      const Eigen::ArrayBase<PreCon>& pre_con_state, const Eigen::ArrayBase<DerivedPa>& param) const {
	  return model_ -> qTranProb(cur_time_, cur_dis_state, pre_dis_state, pre_con_state, param);
  }

  template <typename Model, ModelType Type, typename Trait>
  template <typename CurOb, typename CurCon, typename CurDis, typename DerivedPa>
  double ParticleSystem<Model, Type, Trait> :: likelihood(
		  	   const Eigen::ArrayBase<CurOb>& data, const Eigen::ArrayBase<CurCon>& cur_con_state,
		  	   const Eigen::ArrayBase<CurDis>& cur_dis_state, const Eigen::ArrayBase<DerivedPa>& param) const {
	  return model_ -> likelihood(cur_time_, data, cur_con_state, cur_dis_state, param);
  }

  template <typename Model, ModelType Type, typename Trait>
  template <typename ResampleType>
  void ParticleSystem<Model, Type, Trait> :: resample(const ResampleType& type) const {
	  /// resampling weight must be normalized beforehand
	  ///resample
	  type.resample(rng_, number_, *resample_weight_, resample_number_, *resample_count_, *resample_index_);
	  /*Map counts to indices
	  for (size_t i = 0, j = 0; i != number_; ++i) {
		  while ((*resample_count_)[i] > 0) {
			  (*resample_index_)[j] = i;
			  --(*resample_count_)[i];
			  ++j;
		  }
	  }
	  */
	  ///resample number is equal to particle number, resample_count.sum() == number_, update resampled particles
	  for (size_t i = 0, j = 0; i != number_; ++i) {
		  if ((*resample_count_)[i] == 0) {
			  while ( (*resample_count_)[j] <= 1 )
				  ++j;
			  std::tr1::get<1>(*particles_).col(i) = std::tr1::get<1>(*particles_).col(j);
			  std::tr1::get<2>(*particles_).col(i) = std::tr1::get<2>(*particles_).col(j);
			  std::tr1::get<3>(*particles_).col(i) = std::tr1::get<3>(*particles_).col(j);
			  --(*resample_count_)[j];
			  //++(*resample_count_)[i]; not necessary
		  }
	  }
  }
  /*
   *
   */
    ///partial specification of system with continuous states and known parameters
    /*
     *when use template argument list <Model, CK>
     *this ModelType of partial specification is still more specialized than primer class
     *and will be employed with Trait set as its default type ParticleSystemTrait
     *
     *if a ModelType of partial specification <typename Model>
     *is defined, it will be more specialized than this ModelType if template argument list <Model, CK>
     *is used, and Trait will be set as its default type ParticleSystemTrait
     */
    ///when use template argument list
    ///<Model, CK, Trait>
    ///this ModelType will naturally be employed
  template <typename Model, typename Trait>
  class ParticleSystem<Model, CK, Trait> : private boost::noncopyable {
  public:
	  typedef typename Trait::ConVars ConVars;
	  typedef typename Trait::ObVars ObVars;
	  typedef typename Trait::Particles Particles;

	  template <FilterType, typename> friend class Filter;

	  ParticleSystem();
  	  ParticleSystem(const boost::shared_ptr<Model>& model, double step,
  			  	  	 unsigned int number, StorageMode storage_mode = kNotStored);
  	  ~ParticleSystem();

  	  gsl_rng* get_ran_generator() const {return rng_;}
  	  unsigned int get_number() const {return number_;}
  	  double get_time() const {return cur_time_;}
	  double get_time_step() const {return step_;}
  	  unsigned int get_resample_number() const {return resample_number_;}
  	  double get_resample_threshold() const {return resample_threshold_;}
  	  boost::scoped_ptr<Eigen::ArrayXd> get_resample_weight() const {return resample_weight_;}
  	  boost::shared_ptr<Particles> get_particles() const {return particles_;}
  	  boost::shared_ptr<std::list<Particles> > get_particles_list() const {return particles_list_;}
  	  void set_ran_generator(const gsl_rng_type* type) { gsl_rng_free(rng_); rng_ = gsl_rng_alloc(type);}
  	  void set_number(unsigned int number);
	  void set_time(double time) {cur_time_ = time;}
  	  void set_time_step(double step) {step_ = step;}
  	  void set_resample_number(unsigned int number);
  	  void set_resample_threshold(double threshold) {resample_threshold_ = threshold;}
	  void set_data_source(const char* filename) {if(data_source_) data_source_.close(); data_source_.open(filename);}
	  void close_data_source() {data_source_.close();}

	  void readData();
  	  double computeESS(double time) const;
  	  void initializeConState();
  	  template<typename DerivedCon>
  	  double integrate(double time, const Eigen::ArrayBase<DerivedCon>& con_state,
  			  	  	   std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&)> integrand) const;
  	  template <typename Filter, typename ResampleType>
  	  void iterate(Filter&, const ResampleType&);
  	  void normalizeWeight() const {std::tr1::get<0>(*particles_) /= std::tr1::get<0>(*particles_).sum();}
  	  void propagateCon() const;
  	  template <typename CurCon, typename PreCon>
  	  inline double tranPdfCon(const Eigen::ArrayBase<CurCon>&, const Eigen::ArrayBase<PreCon>&) const;
  	  template <typename CurOb, typename CurCon>
      inline double likelihood(const Eigen::ArrayBase<CurOb>&, const Eigen::ArrayBase<CurCon>&) const;
      template <typename ResampleType>
  	  void resample(const ResampleType&) const;

    protected:
  	  gsl_rng* rng_;
  	  boost::shared_ptr<Model> model_;
  	  unsigned int number_;
  	  double cur_time_;
  	  double step_;
  	  StorageMode storage_mode_;
  	  double resample_threshold_;
  	  unsigned int resample_number_;
  	  boost::scoped_ptr<Eigen::ArrayXd> resample_weight_;
  	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_count_;
  	  boost::scoped_ptr<Eigen::Array<double, Eigen::Dynamic, 1> >resample_index_;
  	  boost::shared_ptr<Particles> particles_;
  	  boost::shared_ptr<std::list<Particles> > particles_list_;
  	  std::ifstream data_source_;
	  boost::scoped_ptr<ObVars> observed_data_;
    };

  template <typename Model, typename Trait>
  ParticleSystem<Model, CK, Trait> :: ParticleSystem()
  : model_(NULL), rng_(NULL), number_(0), cur_time_(0.0), step_(1.0), storage_mode_(kNotStored), resample_threshold_(0),
    resample_number_(number_), resample_weight_(NULL), resample_count_(NULL), resample_index_(NULL),
    particles_(NULL), particles_list_(NULL), data_source_(NULL), observed_data_(NULL) {}

  template <typename Model, typename Trait>
  ParticleSystem<Model, CK, Trait>
  :: ParticleSystem(const boost::shared_ptr<Model>& model, double step, unsigned int number, StorageMode storage_mode)
  : model_(model), number_(number), cur_time_(0.0), step_(step), storage_mode_(storage_mode),
    resample_threshold_(0.5 * number_), resample_number_(number_),
    resample_weight_(new Eigen::ArrayXd(number_)),
    resample_count_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),
    resample_index_(new Eigen::Array<double, Eigen::Dynamic, 1>(number_)),
    particles_(new Particles()), particles_list_(new std::list<Particles>()), data_source_(NULL), observed_data_(NULL) {
    rng_ = gsl_rng_alloc(gsl_rng_ranlux389);
    std::tr1::get<0>(*particles_).resize(number_);
    std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
  }

  template <typename Model, typename Trait>
  ParticleSystem<Model, CK, Trait>
  :: ~ParticleSystem() {
	  gsl_rng_free(rng_);
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CK, Trait>
  :: set_number(unsigned int number) {
	  number_ = number;
	  resample_weight_ -> resize(number_);
	  resample_count_ -> resize(number_);
	  resample_index_ -> resize(number_);
	  std::tr1::get<0>(*particles_).resize(number_);
	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CK, Trait>
  :: set_resample_number(unsigned int number) {
	  resample_number_ = number;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CK, Trait>
  :: initializeConState() {
	  cur_time_ = 0;
	  observed_data_.reset(new typename Trait::ObVars);
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<1>(*particles_).col(k) = model_ -> xInitialize(rng_);
	  }
	  std::tr1::get<0>(*particles_) = Eigen::ArrayXd::Constant(number_, 1.0 / static_cast<double>(number_));
	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CK, Trait> :: readData() {
	  for(int i = 0; i < Trait::dim_y(); ++i)
		  data_source_ >> (*observed_data_)(i);
  }

  template <typename Model, typename Trait>
  double ParticleSystem<Model, CK, Trait>
  :: computeESS(double time) const {
	  if (time == cur_time_)
		  return std::tr1::get<0>(*particles_).sum() * std::tr1::get<0>(*particles_).sum()
     				  / std::tr1::get<0>(*particles_).matrix().squaredNorm();
	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
   	  for(int t = 0; t != static_cast<int>(time / step_); ++t)
   		  ++current;
   	  return std::tr1::get<0>(*current).sum() * std::tr1::get<0>(*current).sum()
   			  / std::tr1::get<0>(*current).matrix().squaredNorm();
  }

  template <typename Model, typename Trait>
  template<typename DerivedCon>
  double ParticleSystem<Model, CK, Trait>
  :: integrate(double time, const Eigen::ArrayBase<DerivedCon>& con_state,
  	  	  	   std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&)> integrand) const {
	  double integral = 0.0;
   	  if (time == cur_time_) {
   		  for (size_t k = 0; k != number_; ++k) {
   			  integral += (std::tr1::get<0>(*particles_))(k) * integrand(time,
   					  	  	  	  	  	  	  	  	  	  	  	  std::tr1::get<1>(*particles_).col(k));
   		  }
   		  return integral;
   	  }
   	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
   	  for(size_t t = 0; t != static_cast<int>(time / step_); ++t)
   		  ++current;
   	  for (size_t k = 0; k != number_; ++k) {
   		  integral += (std::tr1::get<0>(*current))(k) * integrand(time,
   		  					  	  	  	  	  	  	  	  	  	  std::tr1::get<1>(*current).col(k));
   	  }
   	  return integral;
   }

  template <typename Model, typename Trait>
  template <typename Filter, typename ResampleType>
  void ParticleSystem<Model, CK, Trait>
  :: iterate(Filter& filter, const ResampleType& resample_type) {
	  cur_time_ += step_;
	  readData();
   	  if (storage_mode_)
   		  (*particles_list_).push_back(*particles_);
	  filter.doIterate(*this, resample_type);
   	  return;
    }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CK, Trait>
  :: propagateCon() const {
	  for(size_t k = 0; k != number_; ++k) {
   		  std::tr1::get<1>(*particles_).col(k) = model_ -> xPropagate(cur_time_,
   				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<1>(*particles_).col(k), rng_);
   	  }
   	  return;
  }

  template <typename Model, typename Trait>
  template <typename CurCon, typename PreCon>
  double ParticleSystem<Model, CK, Trait>
  :: tranPdfCon(const Eigen::ArrayBase<CurCon>& cur_con_state,
  		  	  	const Eigen::ArrayBase<PreCon>& pre_con_state) const {
   	  return model_ -> xTranPdf(cur_time_, cur_con_state, pre_con_state);
    }

  template <typename Model, typename Trait>
  template <typename CurOb, typename CurCon>
  double ParticleSystem<Model, CK, Trait>
  :: likelihood(const Eigen::ArrayBase<CurOb>& data, const Eigen::ArrayBase<CurCon>& cur_con_state) const {
   	  return model_ -> likelihood(cur_time_, data, cur_con_state);
  }

  template <typename Model, typename Trait>
  template <typename ResampleType>
  void ParticleSystem<Model, CK, Trait>
  :: resample(const ResampleType& type) const {
   	  type.resample(rng_, number_, *resample_weight_, resample_number_, *resample_count_, *resample_index_);
   	  for (size_t i = 0, j = 0; i != number_; ++i) {
  		  if ((*resample_count_)[i] == 0) {
   			  while ( (*resample_count_)[j] <= 1 )
   				  ++j;
   			  std::tr1::get<1>(*particles_).col(i) = std::tr1::get<1>(*particles_).col(j);
   			  --(*resample_count_)[j];
   		  }
   	  }
  }
  /*
   *
   */
  ///partial specification of system with continuous states and unknown parameters
  template <typename Model, typename Trait>
  class ParticleSystem<Model, CU, Trait> : private boost::noncopyable {
  public:
	  typedef typename Trait::ConVars ConVars;
	  typedef typename Trait::ObVars ObVars;
	  typedef typename Trait::Params Params;
	  typedef typename Trait::Particles Particles;

	  template <FilterType, typename> friend class Filter;

	  ParticleSystem();
	  ParticleSystem(const boost::shared_ptr<Model>& model, double step,
			  	  	 unsigned int number, StorageMode storage_mode = kNotStored);
	  ~ParticleSystem();

	  gsl_rng* get_ran_generator() const {return rng_;}
	  unsigned int get_number() const {return number_;}
	  double get_time() const {return cur_time_;}
	  double get_time_step() const {return step_;}
	  unsigned int get_resample_number() const {return resample_number_;}
	  double get_resample_threshold() const {return resample_threshold_;}
	  boost::scoped_ptr<Eigen::ArrayXd> get_resample_weight() const {return resample_weight_;}
	  boost::shared_ptr<Particles> get_particles() const {return particles_;}
	  boost::shared_ptr<std::list<Particles> > get_particles_list() const {return particles_list_;}

	  void set_ran_generator(const gsl_rng_type* type) { gsl_rng_free(rng_); rng_ = gsl_rng_alloc(type);}
	  void set_number(unsigned int number);
	  void set_time(double time) {cur_time_ = time;}
	  void set_time_step(double step) {step_ = step;}
	  void set_resample_number(unsigned int number);
	  void set_resample_threshold(double threshold) {resample_threshold_ = threshold;}
	  void set_data_source(const char* filename) {if(data_source_) data_source_.close(); data_source_.open(filename);}
	  void close_data_source() {data_source_.close();}
	  void readData();
	  double computeESS(double time) const;
	  void initializeConState();
	  void initializeParam() const;
	  template<typename DerivedCon, typename DerivedPa>
	  double integrate(double time, const Eigen::ArrayBase<DerivedCon>& con_state,
			  	  	   const Eigen::ArrayBase<DerivedPa>& param,
			  	  	   std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&,
			  	  			   	   	   	   	   	 const Eigen::ArrayBase<DerivedPa>&)> integrand) const;
	  template <typename Filter, typename ResampleType>
	  void iterate(Filter&, const ResampleType&);
	  void normalizeWeight() const {std::tr1::get<0>(*particles_) /= std::tr1::get<0>(*particles_).sum();}
	  void propagateCon() const;
	  template <typename CurCon, typename PreCon, typename DerivedPa>
	  inline double tranPdfCon(const Eigen::ArrayBase<CurCon>&, const Eigen::ArrayBase<PreCon>&,
			            const Eigen::ArrayBase<DerivedPa>&) const;
	  template <typename CurOb, typename CurCon, typename DerivedPa>
  	  inline double likelihood(const Eigen::ArrayBase<CurOb>&, const Eigen::ArrayBase<CurCon>&,
  			  	  	  	  	   const Eigen::ArrayBase<DerivedPa>&) const;
  	  template <typename ResampleType>
	  void resample(const ResampleType&) const;

  protected:
	  gsl_rng* rng_;
	  boost::shared_ptr<Model> model_;
	  unsigned int number_;
	  double cur_time_;
  	  double step_;
	  StorageMode storage_mode_;
	  double resample_threshold_;
	  unsigned int resample_number_;
	  boost::scoped_ptr<Eigen::ArrayXd> resample_weight_;
	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_count_;
	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_index_;
	  boost::shared_ptr<Particles> particles_;
	  boost::shared_ptr<std::list<Particles> > particles_list_;
	  std::ifstream data_source_;
	  boost::scoped_ptr<ObVars> observed_data_;
  };

  template <typename Model, typename Trait>
  ParticleSystem<Model, CU, Trait> :: ParticleSystem()
  : model_(NULL), rng_(NULL), number_(0), cur_time_(0.0), step_(1.0), storage_mode_(kNotStored), resample_threshold_(0),
    resample_number_(number_), resample_weight_(NULL), resample_count_(NULL), resample_index_(NULL),
    particles_(NULL), particles_list_(NULL), data_source_(NULL), observed_data_(NULL) {}

  template <typename Model, typename Trait>
  ParticleSystem<Model, CU, Trait>
  ::ParticleSystem(const boost::shared_ptr<Model>& model, double step, unsigned int number, StorageMode storage_mode)
  : model_(model), number_(number), cur_time_(0.0), step_(step), storage_mode_(storage_mode),
    resample_threshold_(0.5 * number_), resample_number_(number_),
    resample_weight_(new Eigen::ArrayXd(number_)),
    resample_count_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),
    resample_index_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),
    particles_(new Particles()), particles_list_(new std::list<Particles>()), data_source_(NULL), observed_data_(NULL) {
	  rng_ = gsl_rng_alloc(gsl_rng_ranlux389);
	  std::tr1::get<0>(*particles_).resize(number_);
	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
	  std::tr1::get<2>(*particles_).resize(Trait::dim_param(), number_);
  }

  template <typename Model, typename Trait>
  ParticleSystem<Model, CU, Trait> :: ~ParticleSystem() {
	  gsl_rng_free(rng_);
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CU, Trait> :: set_number(unsigned int number) {
	  number_ = number;
	  resample_weight_ -> resize(number_);
	  resample_count_ -> resize(number_);
	  resample_index_ -> resize(number_);
	  std::tr1::get<0>(*particles_).resize(number_);
	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
	  std::tr1::get<2>(*particles_).resize(Trait::dim_param(), number_);
	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CU, Trait> :: set_resample_number(unsigned int number) {
	  resample_number_ = number;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CU, Trait>
  :: initializeConState() {
	  cur_time_ = 0;
	  observed_data_.reset(new typename Trait::ObVars);
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<1>(*particles_).col(k) = model_ -> xInitialize(rng_);
	  }
	  std::tr1::get<0>(*particles_) = Eigen::ArrayXd::Constant(number_, 1.0 / static_cast<double>(number_));
	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CU, Trait>
  :: initializeParam() const {
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<2>(*particles_).col(k) = (model_ -> paramInitialize(rng_));
	  	  }
	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CU, Trait> :: readData() {
	  for(int i = 0; i < Trait::dim_y(); ++i)
		  data_source_ >> (*observed_data_)(i);
  }

  template <typename Model, typename Trait>
  double ParticleSystem<Model, CU, Trait>
  :: computeESS(double time) const {
  	  if (time == cur_time_)
  		  return std::tr1::get<0>(*particles_).sum() * std::tr1::get<0>(*particles_).sum()
  				  / std::tr1::get<0>(*particles_).matrix().squaredNorm();
	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
	  for(int t = 0; t != static_cast<int>(time / step_); ++t)
		  ++current;
	  return std::tr1::get<0>(*current).sum() * std::tr1::get<0>(*current).sum()
			  / std::tr1::get<0>(*current).matrix().squaredNorm();
  }

  template <typename Model, typename Trait>
  template <typename DerivedCon, typename DerivedPa>
  double ParticleSystem<Model, CU, Trait>
  ::integrate(double time, const Eigen::ArrayBase<DerivedCon>& con_state,
		  	  const Eigen::ArrayBase<DerivedPa>& param,
		  	  std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&,
				  	  	  	  	  	  	const Eigen::ArrayBase<DerivedPa>&)> integrand) const {
	  double integral = 0.0;
	  if (time == cur_time_) {
		  for (size_t k = 0; k != number_; ++k) {
			  integral += (std::tr1::get<0>(*particles_))(k) * integrand(time,
					  std::tr1::get<1>(*particles_).col(k), std::tr1::get<2>(*particles_).col(k));
		  }
		  return integral;
	  }
	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
	  for(size_t t = 0; t != static_cast<int>(time / step_); ++t)
		  ++current;
	  for (size_t k = 0; k != number_; ++k) {
		  integral += (std::tr1::get<0>(*current))(k) * integrand(time,
		  					  std::tr1::get<1>(*current).col(k), std::tr1::get<2>(*current).col(k));
	  }
	  return integral;
  }

  template <typename Model, typename Trait>
  template <typename Filter, typename ResampleType>
  void ParticleSystem<Model, CU, Trait>
  :: iterate(Filter& filter, const ResampleType& resample_type) {
	  cur_time_ += step_;
	  readData();
	  if (storage_mode_)
		  (*particles_list_).push_back(*particles_);
	  filter.doIterate(*this, resample_type);
	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, CU, Trait>
  :: propagateCon() const {
	  for(size_t k = 0; k != number_; ++k) {
		  std::tr1::get<1>(*particles_).col(k) = model_ -> xPropagate(cur_time_,
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<1>(*particles_).col(k),
				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<2>(*particles_).col(k), rng_);
	  }
	  return;
  }

  template <typename Model, typename Trait>
  template <typename CurCon, typename PreCon, typename DerivedPa>
  double ParticleSystem<Model, CU, Trait>
  :: tranPdfCon(const Eigen::ArrayBase<CurCon>& cur_con_state,
		  	    const Eigen::ArrayBase<PreCon>& pre_con_state,
		  	    const Eigen::ArrayBase<DerivedPa>& param) const {
	  return model_ -> xTranPdf(cur_time_, cur_con_state, pre_con_state, param);
  }

  template <typename Model, typename Trait>
  template <typename CurOb, typename CurCon, typename DerivedPa>
  double ParticleSystem<Model, CU, Trait>
  :: likelihood(const Eigen::ArrayBase<CurOb>& data, const Eigen::ArrayBase<CurCon>& cur_con_state,
		  	  	const Eigen::ArrayBase<DerivedPa>& param) const {
	  return model_ -> likelihood(cur_time_, data, cur_con_state, param);
  }

  template <typename Model, typename Trait>
  template <typename ResampleType>
  void ParticleSystem<Model, CU, Trait>
  :: resample(const ResampleType& type) const {
	  type.resample(rng_, number_, *resample_weight_, resample_number_, *resample_count_, *resample_index_);
	  for (size_t i = 0, j = 0; i != number_; ++i) {
		  if ((*resample_count_)[i] == 0) {
			  while ( (*resample_count_)[j] <= 1 )
				  ++j;
			  std::tr1::get<1>(*particles_).col(i) = std::tr1::get<1>(*particles_).col(j);
			  std::tr1::get<2>(*particles_).col(i) = std::tr1::get<2>(*particles_).col(j);
			  --(*resample_count_)[j];
		  }
	  }
  }
   /*
    *
    */
   ///partial specification of system with hybrid states and known parameters
  template <typename Model, typename Trait>
  class ParticleSystem<Model, HK, Trait> : private boost::noncopyable {
  public:
	  typedef typename Trait::ConVars ConVars;
	  typedef typename Trait::DisVars DisVars;
	  typedef typename Trait::ObVars ObVars;
	  typedef typename Trait::Particles Particles;

	  template <FilterType, typename> friend class Filter;

 	  ParticleSystem();
 	  ParticleSystem(const boost::scoped_ptr<Model>& model, double step,
 			  	  	 unsigned int number, StorageMode storage_mode = kNotStored);
 	  ~ParticleSystem();

 	  gsl_rng* get_ran_generator() const {return rng_;}
 	  unsigned int get_number() const {return number_;}
 	  double get_time() const {return cur_time_;}
	  double get_time_step() const {return step_;}
 	  unsigned int get_resample_number() const {return resample_number_;}
 	  double get_resample_threshold() const {return resample_threshold_;}
 	  boost::scoped_ptr<Eigen::ArrayXd> get_resample_weight() const {return resample_weight_;}
 	  boost::shared_ptr<Particles> get_particles() const {return particles_;}
 	  boost::shared_ptr<std::list<Particles> > get_particles_list() const {return particles_list_;}

 	  void set_ran_generator(const gsl_rng_type* type) { gsl_rng_free(rng_); rng_ = gsl_rng_alloc(type);}
 	  void set_number(unsigned int number);
	  void set_time(double time) {cur_time_ = time;}
 	  void set_time_step(double step) {step_ = step;}
 	  void set_resample_number(unsigned int number);
 	  void set_resample_threshold(double threshold) {resample_threshold_ = threshold;}
	  void set_data_source(const char* filename) {if(data_source_) data_source_.close(); data_source_.open(filename);}
	  void close_data_source() {data_source_.close();}
	  void readData();
 	  double computeESS(double time) const;
 	  void initializeConState();
 	  void initializeDisState();
 	  template<typename DerivedCon, typename DerivedDis>
 	  double integrate(double time, const Eigen::ArrayBase<DerivedCon>& con_state,
 			  	  	   const Eigen::ArrayBase<DerivedDis>& dis_state,
 			  	  	   std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&,
 			  	  			   	   	   	   	   	 const Eigen::ArrayBase<DerivedDis>&)> integrand) const;
 	  template <typename Filter, typename ResampleType>
 	  void iterate(Filter&, const ResampleType&);
 	  void normalizeWeight() const {std::tr1::get<0>(*particles_) /= std::tr1::get<0>(*particles_).sum();}
 	  void propagateCon() const;
 	  void propagateDis() const;
 	  template <typename CurCon, typename PreCon, typename PreDis>
 	  inline double tranPdfCon(const Eigen::ArrayBase<CurCon>&, const Eigen::ArrayBase<PreCon>&,
 			            const Eigen::ArrayBase<PreDis>&) const;
 	  template <typename CurDis, typename PreDis, typename PreCon>
 	  inline double tranProbDis(const Eigen::ArrayBase<CurDis>&, const Eigen::ArrayBase<PreDis>&,
 			  	  	  	 const Eigen::ArrayBase<PreCon>&) const;
 	  template <typename CurOb, typename CurCon, typename CurDis>
   	  inline double likelihood(const Eigen::ArrayBase<CurOb>&, const Eigen::ArrayBase<CurCon>&, const Eigen::ArrayBase<CurDis>&) const;
   	  template <typename ResampleType>
 	  void resample(const ResampleType&) const;

   protected:
 	  gsl_rng* rng_;
 	  boost::shared_ptr<Model> model_;
 	  unsigned int number_;
 	  double cur_time_;
  	  double step_;
 	  StorageMode storage_mode_;
 	  double resample_threshold_;
 	  unsigned int resample_number_;
 	  boost::scoped_ptr<Eigen::ArrayXd> resample_weight_;
 	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_count_;
 	  boost::scoped_ptr<Eigen::Array<unsigned int, Eigen::Dynamic, 1> >resample_index_;
 	  boost::shared_ptr<Particles> particles_;
 	  boost::shared_ptr<std::list<Particles> > particles_list_;
 	  std::ifstream data_source_;
	  boost::scoped_ptr<ObVars> observed_data_;
   };

  template <typename Model, typename Trait>
  ParticleSystem<Model, HK, Trait> :: ParticleSystem()
  : model_(NULL), rng_(NULL), number_(0), cur_time_(0.0), step_(1.0), storage_mode_(kNotStored), resample_threshold_(0),
    resample_number_(number_), resample_weight_(NULL), resample_count_(NULL), resample_index_(NULL),
    particles_(NULL), particles_list_(NULL), data_source_(NULL), observed_data_(NULL) {}

  template <typename Model, typename Trait>
  ParticleSystem<Model, HK, Trait>
  :: ParticleSystem(const boost::scoped_ptr<Model>& model, double step,
 		  	  	    unsigned int number, StorageMode storage_mode)
  : model_(model), number_(number), cur_time_(0.0), step_(step), storage_mode_(storage_mode),
    resample_threshold_(0.5 * number_), resample_number_(number_),
    resample_weight_(new Eigen::ArrayXd(number_)),
    resample_count_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),
    resample_index_(new Eigen::Array<unsigned int, Eigen::Dynamic, 1>(number_)),
    particles_(new Particles()), particles_list_(new std::list<Particles>()), data_source_(NULL), observed_data_(NULL) {
	  rng_ = gsl_rng_alloc(gsl_rng_ranlux389);
	  std::tr1::get<0>(*particles_).resize(number_);
 	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
 	  std::tr1::get<2>(*particles_).resize(Trait::dim_q(), number_);
   }

  template <typename Model, typename Trait>
  ParticleSystem<Model, HK, Trait> :: ~ParticleSystem() {
 	  gsl_rng_free(rng_);
   }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait> :: set_number(unsigned int number) {
 	  number_ = number;
 	  resample_weight_ -> resize(number_);
 	  resample_count_ -> resize(number_);
 	  resample_index_ -> resize(number_);
 	  std::tr1::get<0>(*particles_).resize(number_);
 	  std::tr1::get<1>(*particles_).resize(Trait::dim_x(), number_);
 	  std::tr1::get<2>(*particles_).resize(Trait::dim_q(), number_);
 	  return;
   }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait> :: set_resample_number(unsigned int number) {
 	  resample_number_ = number;
   }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait> :: initializeConState() {
	  cur_time_ = 0;
	  observed_data_.reset(new typename Trait::ObVars);
 	  for(size_t k = 0; k != number_; ++k) {
 		  std::tr1::get<1>(*particles_).col(k) = model_ -> xInitialize(rng_);
 	  }
 	  std::tr1::get<0>(*particles_) = Eigen::ArrayXd::Constant(number_, 1.0 / static_cast<double>(number_));
 	  return;
   }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait> :: initializeDisState() {
 	  cur_time_ = 0;
	  if(!observed_data_)
		  observed_data_.reset(new typename Trait::ObVars);
 	  for(size_t k = 0; k != number_; ++k) {
 		  std::tr1::get<2>(*particles_).col(k) = model_ -> qInitialize(rng_);
 	  }
 	  std::tr1::get<0>(*particles_) = Eigen::ArrayXd::Constant(number_, 1.0 / static_cast<double>(number_));
 	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait> :: readData() {
	  for(int i = 0; i < Trait::dim_y(); ++i)
		  data_source_ >> (*observed_data_)(i);
  }


  template <typename Model, typename Trait>
  double ParticleSystem<Model, HK, Trait> :: computeESS(double time) const {
   	  if (time == cur_time_)
   		  return std::tr1::get<0>(*particles_).sum() * std::tr1::get<0>(*particles_).sum()
   				  / std::tr1::get<0>(*particles_).matrix().squaredNorm();
 	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
 	  for(int t = 0; t != static_cast<int>(time / step_); ++t)
 		  ++current;
 	  return std::tr1::get<0>(*current).sum() * std::tr1::get<0>(*current).sum()
 			  / std::tr1::get<0>(*current).matrix().squaredNorm();
   }

  template <typename Model, typename Trait>
  template <typename DerivedCon, typename DerivedDis>
  double ParticleSystem<Model, HK, Trait> :: integrate(
 		                         double time, const Eigen::ArrayBase<DerivedCon>& con_state,
 		                         const Eigen::ArrayBase<DerivedDis>& dis_state,
 		                         std::tr1::function<double(double, const Eigen::ArrayBase<DerivedCon>&,
 		  	 		  	  	  	 const Eigen::ArrayBase<DerivedDis>&)> integrand) const {
 	  double integral = 0.0;
 	  if (time == cur_time_) {
 		  for (size_t k = 0; k != number_; ++k) {
 			  integral += (std::tr1::get<0>(*particles_))(k) * integrand(time,
 					  std::tr1::get<1>(*particles_).col(k), std::tr1::get<2>(*particles_).col(k));
 		  }
 		  return integral;
 	  }
 	  typename std::list<Particles>::iterator current = (*particles_list_).begin();
 	  for(size_t t = 0; t != static_cast<int>(time / step_); ++t)
 		  ++current;
 	  for (size_t k = 0; k != number_; ++k) {
 		  integral += (std::tr1::get<0>(*current))(k) * integrand(time,
 		  					  std::tr1::get<1>(*current).col(k), std::tr1::get<2>(*current).col(k));
 	  }
 	  return integral;
 }

  template <typename Model, typename Trait>
  template <typename Filter, typename ResampleType>
  void ParticleSystem<Model, HK, Trait>
   :: iterate(Filter& filter, const ResampleType& resample_type) {
 	  cur_time_ += step_;
 	  readData();
 	  if (storage_mode_)
 		  (*particles_list_).push_back(*particles_);
	  filter.doIterate(*this, resample_type);
 	  return;
   }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait>
  :: propagateCon() const {
 	  for(size_t k = 0; k != number_; ++k) {
 		  std::tr1::get<1>(*particles_).col(k) = model_ -> xPropagate(cur_time_,
 				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<1>(*particles_).col(k),
 				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<2>(*particles_).col(k), rng_);
 	  }
 	  return;
  }

  template <typename Model, typename Trait>
  void ParticleSystem<Model, HK, Trait>
  :: propagateDis() const {
 	  for(size_t k = 0; k != number_; ++k) {
 		  std::tr1::get<2>(*particles_).col(k) = model_ -> qPropagate(cur_time_,
 				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<1>(*particles_).col(k),
 				  	  	  	  	  	  	  	  	  	  	  	 std::tr1::get<2>(*particles_).col(k), rng_);
 	  }
 	  return;
  }

  template <typename Model, typename Trait>
  template <typename CurCon, typename PreCon, typename PreDis>
  double ParticleSystem<Model, HK, Trait> :: tranPdfCon(const Eigen::ArrayBase<CurCon>& cur_con_state,
 		  	  	  	  	  	  	  	  	  	  	  	  	const Eigen::ArrayBase<PreCon>& pre_con_state,
 		  	  	  	  	  	  	  	  	  	  	  	  	const Eigen::ArrayBase<PreDis>& pre_dis_state) const {
 	  return model_ -> xTranPdf(cur_time_, cur_con_state, pre_con_state, pre_dis_state);
  }

  template <typename Model, typename Trait>
  template <typename CurDis, typename PreDis, typename PreCon>
  double ParticleSystem<Model, HK, Trait>
  :: tranProbDis(const Eigen::ArrayBase<CurDis>& cur_dis_state,
 		  	   	 const Eigen::ArrayBase<PreDis>& pre_dis_state,
 		  	   	 const Eigen::ArrayBase<PreCon>& pre_con_state) const {
 	  return model_ -> qTranProb(cur_time_, cur_dis_state, pre_dis_state, pre_con_state);
  }

  template <typename Model, typename Trait>
  template <typename CurOb, typename CurCon, typename CurDis>
  double ParticleSystem<Model, HK, Trait>
  :: likelihood(const Eigen::ArrayBase<CurOb>& data, const Eigen::ArrayBase<CurCon>& cur_con_state,
 		  	    const Eigen::ArrayBase<CurDis>& cur_dis_state) const {
 	  return model_ -> likelihood(cur_time_, data, cur_con_state, cur_dis_state);
  }

  template <typename Model, typename Trait>
  template <typename ResampleType>
  void ParticleSystem<Model, HK, Trait>
  :: resample(const ResampleType& type) const {
 	  type.resample(rng_, number_, *resample_weight_, resample_number_, *resample_count_, *resample_index_);
 	  for (size_t i = 0, j = 0; i != number_; ++i) {
 		  if ((*resample_count_)[i] == 0) {
 			  while ( (*resample_count_)[j] <= 1 )
 				  ++j;
 			  std::tr1::get<1>(*particles_).col(i) = std::tr1::get<1>(*particles_).col(j);
 			  std::tr1::get<2>(*particles_).col(i) = std::tr1::get<2>(*particles_).col(j);
 			  --(*resample_count_)[j];
 		  }
 	  }
  }

}

#endif /* DEL_PARTICLE_SYSTEM_HPP_ */
