/*
 * del_model.hpp
 *
 *  Created on: Sep 28, 2012
 *      Author: erli
 */

#ifndef DEL_MODEL_HPP_
#define DEL_MODEL_HPP_

/*
 * the common property of virtual base class and CRTP base is defining same interface for the
 * purpose of polymorphism, for virtual inheritance the the type to be determined is
 * the pointer to virtual base class and the actual object on which virtual function
 * is called can be determined at runtime though pointer to base class, while for CRTP
 * the type of object on which function will be called must be determined at compilation
 * since they only share the interface inherited from template base class but don't share
 * the same base class
 * besides, function template with CRTP template base class as an argument can imitate pointer
 * to virtual base class to perform different function calls based on the object being passed
 * just like object being pointed to in the case of virutal inheritance, the difference being
 * that the type of object being pointed can be determined at runtime because its static type
 * is pointer to virtual base class which has been determined while the type of object passed
 * to template function must be determined at compile time
 * by all by,  with dynamic polymorphism, calls to actual methods are resolved at runtime
 * while static polymorphism is type polymorphism that occurs at compile time
 *
 */

#include <gsl/gsl_rng.h>
#include <Eigen/Dense>
#include "del_model_trait.h"

namespace del {

  class ModelAccessor {
  private:
	  template<typename Model, typename Trait> friend class ModelBase;

      template<typename Model>
      static typename ModelTrait<Model>::ConVars xInitialize(const Model& model, gsl_rng* rng) {
    	  return model.xInit(rng);
      }
      template<typename Model>
      static typename ModelTrait<Model>::DisVars qInitialize(const Model& model, gsl_rng* rng) {
    	  return model.qInit(rng);
      }
      template<typename Model>
      static typename ModelTrait<Model>::Params paramInitialize(const Model& model, gsl_rng* rng) {
    	  return model.paramInit(rng);
      }

      //HU
      template<typename Model, typename PreX, typename CurQ, typename Pa>
      static typename ModelTrait<Model>::ConVars xPropagate(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  	  const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.conModel(t, pre_x, cur_q, param, rng);
      }
      //overload for CK
      template<typename Model, typename PreX>
      static typename ModelTrait<Model>::ConVars xPropagate(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<PreX>& pre_x, gsl_rng* rng) {
    	  return model.conModel(t, pre_x, rng);
      }
      //overload for either CU or HK since they have same parameter list
      //for a derived model, it is at most one of CU or HK, so no conflict will occur
      template<typename Model, typename PreX, typename Pa>
      static typename ModelTrait<Model>::ConVars xPropagate(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  	  const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.conModel(t, pre_x, param, rng);
      }
      /*
      //overload for HK
      template<typename Model, typename PreX, typename CurQ>
      static ModelTrait<Model>::ConVars xPropagate(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  	  const Eigen::ArrayBase<CurQ>& cur_q, gsl_rng* rng) {
    	  return model.conModel(t, pre_x, cur_q, rng);
      }
      */

      //HU
      template<typename Model, typename PreX, typename PreQ, typename Pa>
      static typename ModelTrait<Model>::DisVars qPropagate(const Model& model, double t,
    		  	  	  	      const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<PreQ>& pre_q,
    		  	  	  	      const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.disModel(t, pre_x, pre_q, param, rng);
      }
      //overload for HK
      template<typename Model, typename PreX, typename PreQ>
      static typename ModelTrait<Model>::DisVars qPropagate(const Model& model, double t,
    		  	  	  	      const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<PreQ>& pre_q,
    		  	  	  	      gsl_rng* rng) {
    	  return model.disModel(t, pre_x, pre_q, rng);
      }

      //HU
      template<typename Model, typename CurX, typename CurQ, typename Pa>
      static typename ModelTrait<Model>::ObVars outPut(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  	  const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.obModel(t, cur_x, cur_q, param, rng);
      }
      //CK
      template<typename Model, typename CurX>
      static typename ModelTrait<Model>::ObVars outPut(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<CurX>& cur_x, gsl_rng* rng) {
    	  return model.obModel(t, cur_x, rng);
      }
      //CU or HK
      template<typename Model, typename CurX, typename Pa>
      static typename ModelTrait<Model>::ObVars outPut(const Model& model, double t,
    		  	  	  	  	  const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	  	  const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.obModel(t, cur_x, param, rng);
      }

      //HU
      template<typename Model, typename CurX, typename PreX, typename CurQ, typename Pa>
      static double xTranPdf(const Model& model, double t, const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	     const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	     const Eigen::ArrayBase<Pa>& param) {
    	  return model.conPdf(t, cur_x, pre_x, cur_q, param);
      }
      //CK
      template<typename Model, typename CurX, typename PreX>
      static double xTranPdf(const Model& model, double t, const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	     const Eigen::ArrayBase<PreX>& pre_x) {
    	  return model.conPdf(t, cur_x, pre_x);
      }
      //CU or HK
      template<typename Model, typename CurX, typename PreX, typename Pa>
      static double xTranPdf(const Model& model, double t, const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	     const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<Pa>& param) {
    	  return model.conPdf(t, cur_x, pre_x, param);
      }

      //HU
      template<typename Model, typename CurQ, typename PreQ, typename PreX, typename Pa>
      static double qTranProb(const Model& model, double t, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  	  const Eigen::ArrayBase<PreQ>& pre_q, const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  	  const Eigen::ArrayBase<Pa>& param) {
    	  return model.disProb(t, cur_q, pre_q, pre_x, param);
      }
      //HK
      template<typename Model, typename CurQ, typename PreQ, typename PreX>
      static double qTranProb(const Model& model, double t, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  	  const Eigen::ArrayBase<PreQ>& pre_q, const Eigen::ArrayBase<PreX>& pre_x) {
    	  return model.disProb(t, cur_q, pre_q, pre_x);
      }

      //HU
      template<typename Model, typename CurY, typename CurX, typename CurQ, typename Pa>
      static double likelihood(const Model& model, double t, const Eigen::ArrayBase<CurY>& cur_y,
    		  	  	  	  	   const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  	   const Eigen::ArrayBase<Pa>& param) {
    	  return model.obProb(t, cur_y, cur_x, cur_q, param);
      }
      //CK
      template<typename Model, typename CurY, typename CurX>
      static double likelihood(const Model& model, double t, const Eigen::ArrayBase<CurY>& cur_y,
    		  	  	  	  	   const Eigen::ArrayBase<CurX>& cur_x) {
    	  return model.obProb(t, cur_y, cur_x);
      }
      //CU or HK
      template<typename Model, typename CurY, typename CurX, typename Pa>
      static double likelihood(const Model& model, double t, const Eigen::ArrayBase<CurY>& cur_y,
    		  	  	  	  	   const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<Pa>& param) {
    	  return model.obProb(t, cur_y, cur_x, param);
      }

      //HU
      template<typename Model, typename PreX, typename CurQ, typename CurY, typename Pa>
      static typename ModelTrait<Model>::ConVars proposal(const Model& model, double t,
    		  	  	  	  	   const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  	   const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.proposalDraw(t, pre_x, cur_q, cur_y, param, rng);
      }
      //CK
      template<typename Model, typename PreX, typename CurY>
      static typename ModelTrait<Model>::ConVars proposal(const Model& model, double t,
    		  	  	  	  	   const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  	   const Eigen::ArrayBase<CurY>& cur_y, gsl_rng* rng) {
    	  return model.proposalDraw(t, pre_x, cur_y, rng);
      }
      //CU or HK, if it's HK, put cur_q in the same place as param for CU
      template<typename Model, typename PreX, typename CurY, typename Pa>
      static typename ModelTrait<Model>::ConVars proposal(const Model& model, double t,
    		  	  	  	  	   const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  	   const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) {
    	  return model.proposalDraw(t, pre_x, cur_y, param, rng);
      }

      //HU
      template<typename Model, typename CurX, typename PreX, typename CurQ, typename CurY, typename Pa>
      static double proposalPdf(const Model& model, double t, const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	  const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<CurQ>& cur_q,
    		  	  	  	  const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<Pa>& param) {
    	  return model.propDrawPdf(t, cur_x, pre_x, cur_q, cur_y, param);
      }
      //CK
      template<typename Model, typename CurX, typename PreX, typename CurY>
      static double proposalPdf(const Model& model, double t, const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	  const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  const Eigen::ArrayBase<CurY>& cur_y) {
    	  return model.propDrawPdf(t, cur_x, pre_x, cur_y);
      }
      //CU or HK
      template<typename Model, typename CurX, typename PreX, typename CurY, typename Pa>
      static double proposalPdf(const Model& model, double t, const Eigen::ArrayBase<CurX>& cur_x,
    		  	  	  	  const Eigen::ArrayBase<PreX>& pre_x,
    		  	  	  	  const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<Pa>& param) {
    	  return model.propDrawPdf(t, cur_x, pre_x, cur_y, param);
      }
  };



  template <typename Model, typename Trait = ModelTrait<Model> >
  class ModelBase {
  public:
	  typedef typename Trait::ConVars ConVars;
	  typedef typename Trait::DisVars DisVars;
	  typedef typename Trait::ObVars ObVars;
	  typedef typename Trait::Params Params;

	  Model& derived() const {
	      return static_cast<Model&>(*this); }
	  const Model& derivedConst() const {
		  return static_cast<const Model&>(*this); }

	  ConVars xInitialize(gsl_rng* rng) const {return ModelAccessor::xInitialize(this -> derivedConst(), rng);}
	  DisVars qInitialize(gsl_rng* rng) const {return ModelAccessor::qInitialize(this -> derivedConst(), rng);}
	  Params paramInitialize(gsl_rng* rng) const {return ModelAccessor::paramInitialize(this -> derivedConst(), rng);}

	  //HU
	  template <typename PreX, typename CurQ, typename Pa>
	  ConVars xPropagate(double t, const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<CurQ>& cur_q,
	   	   	   	   	   	 const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::xPropagate(this -> derivedConst(), t, pre_x, cur_q, param, rng);}
	  //CK
	  template <typename PreX>
	  ConVars xPropagate(double t, const Eigen::ArrayBase<PreX>& pre_x, gsl_rng* rng) const {
		  return ModelAccessor::xPropagate(this -> derivedConst(), t, pre_x, rng);}
	  //CU or HK
	  template <typename PreX, typename Pa>
	  ConVars xPropagate(double t, const Eigen::ArrayBase<PreX>& pre_x,
	   	   	   	   	   	 const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::xPropagate(this -> derivedConst(), t, pre_x, param, rng);}

	  //HU
	  template <typename PreX, typename PreQ, typename Pa>
	  DisVars qPropagate(double t, const Eigen::ArrayBase<PreX>& pre_x,
	  	  	  	            const Eigen::ArrayBase<PreQ>& pre_q, const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::qPropagate(this -> derivedConst(), t, pre_x, pre_q, param, rng);}
	  //HK
	  template <typename PreX, typename PreQ>
	  DisVars qPropagate(double t, const Eigen::ArrayBase<PreX>& pre_x,
	  	  	  	            const Eigen::ArrayBase<PreQ>& pre_q, gsl_rng* rng) const {
		  return ModelAccessor::qPropagate(this -> derivedConst(), t, pre_x, pre_q, rng);}

	  ///output is used to generate simulated data
	  //HU
	  template <typename CurX, typename CurQ, typename Pa>
	  ObVars outPut(double t, const Eigen::ArrayBase<CurX>& cur_x,
			  	   const Eigen::ArrayBase<CurQ>& cur_q, const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::outPut(this -> derivedConst(), t, cur_x, cur_q, param, rng);}
	  //CK
	  template <typename CurX>
	  ObVars outPut(double t, const Eigen::ArrayBase<CurX>& cur_x, gsl_rng* rng) const {
		  return ModelAccessor::outPut(this -> derivedConst(), t, cur_x, rng);}
	  //CU or HK
	  template <typename CurX, typename Pa>
	  ObVars outPut(double t, const Eigen::ArrayBase<CurX>& cur_x,
			  	    const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::outPut(this -> derivedConst(), t, cur_x, param, rng);}

	  //HU
	  template <typename CurX, typename PreX, typename CurQ, typename Pa>
	  double xTranPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  const Eigen::ArrayBase<CurQ>& cur_q, const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::xTranPdf(this -> derivedConst(), t, cur_x, pre_x, cur_q, param);}
	  //CK
	  template <typename CurX, typename PreX>
	  double xTranPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x) const {
		  return ModelAccessor::xTranPdf(this -> derivedConst(), t, cur_x, pre_x);}
	  //CU or HK
	  template <typename CurX, typename PreX, typename Pa>
	  double xTranPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::xTranPdf(this -> derivedConst(), t, cur_x, pre_x, param);}

	  //HU
	  template <typename CurQ, typename PreQ, typename PreX, typename Pa>
	  double qTranProb(double t, const Eigen::ArrayBase<CurQ>& cur_q, const Eigen::ArrayBase<PreQ>& pre_q,
			  	  	   const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::qTranProb(this -> derivedConst(), t, cur_q, pre_q, pre_x, param);}
	  //HK
	  template <typename CurQ, typename PreQ, typename PreX>
	  double qTranProb(double t, const Eigen::ArrayBase<CurQ>& cur_q, const Eigen::ArrayBase<PreQ>& pre_q,
			  	  	   const Eigen::ArrayBase<PreX>& pre_x) const {
		  return ModelAccessor::qTranProb(this -> derivedConst(), t, cur_q, pre_q, pre_x);}

	  //HU
	  template <typename CurY, typename CurX, typename CurQ, typename Pa>
	  double likelihood(double t, const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<CurX>& cur_x,
			  	  	    const Eigen::ArrayBase<CurQ>& cur_q, const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::likelihood(this -> derivedConst(), t, cur_y, cur_x, cur_q, param);}
	  //CK
	  template <typename CurY, typename CurX>
	  double likelihood(double t, const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<CurX>& cur_x) const {
		  return ModelAccessor::likelihood(this -> derivedConst(), t, cur_y, cur_x);}
	  //CU or HK
	  template <typename CurY, typename CurX, typename CurQ, typename Pa>
	  double likelihood(double t, const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<CurX>& cur_x,
			  	  	    const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::likelihood(this -> derivedConst(), t, cur_y, cur_x, param);}

	  //HU
	  template <typename PreX, typename CurQ, typename CurY, typename Pa>
	  ConVars proposal(double t, const Eigen::ArrayBase<PreX>& pre_x, const Eigen::ArrayBase<CurQ>& cur_q,
			  	  	  const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::proposal(this -> derivedConst(), t, pre_x, cur_q, cur_y, param, rng);}
	  //CK
	  template <typename PreX, typename CurY>
	  ConVars proposal(double t, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  const Eigen::ArrayBase<CurY>& cur_y, gsl_rng* rng) const {
		  return ModelAccessor::proposal(this -> derivedConst(), t, pre_x, cur_y, rng);}
	  //CU or HK
	  template <typename PreX, typename CurY, typename Pa>
	  ConVars proposal(double t, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  const Eigen::ArrayBase<CurY>& cur_y, const Eigen::ArrayBase<Pa>& param, gsl_rng* rng) const {
		  return ModelAccessor::proposal(this -> derivedConst(), t, pre_x, cur_y, param, rng);}

	  //HU
	  template <typename CurX, typename PreX, typename CurQ, typename CurY, typename Pa>
	  double proposalPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  	 const Eigen::ArrayBase<CurQ>& cur_q, const Eigen::ArrayBase<CurY>& cur_y,
			  	  	  	 const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::proposalPdf(this -> derivedConst(), t, cur_x, pre_x, cur_q, cur_y, param);}
	  //CK
	  template <typename CurX, typename PreX, typename CurY>
	  double proposalPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  	 const Eigen::ArrayBase<CurY>& cur_y) const {
		  return ModelAccessor::proposalPdf(this -> derivedConst(), t, cur_x, pre_x, cur_y);}
	  //CU or HK
	  template <typename CurX, typename PreX, typename CurY, typename Pa>
	  double proposalPdf(double t, const Eigen::ArrayBase<CurX>& cur_x, const Eigen::ArrayBase<PreX>& pre_x,
			  	  	  	 const Eigen::ArrayBase<CurY>& cur_y,
			  	  	  	 const Eigen::ArrayBase<Pa>& param) const {
		  return ModelAccessor::proposalPdf(this -> derivedConst(), t, cur_x, pre_x, cur_y, param);}

  };



}

#endif /* DEL_MODEL_HPP_ */

