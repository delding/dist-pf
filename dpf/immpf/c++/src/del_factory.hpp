/*
 * del_factory.hpp
 *
 *  Created on: Jan 19, 2013
 *      Author: erli
 */

#ifndef DEL_FACTORY_HPP_
#define DEL_FACTORY_HPP_

#include <map>
#include <functional>
#include <string>
#include "del_filter.hpp"
#include "del_resample.hpp"

/*
 *python is dynamically typed language, so no need to employ virtual base class, cause it is itself already dynamical
 *c++ is a statically typed language, so need to use virtual function to realize dynamical polymophism
 *and since CRTP is static polymophism it can not realize dynamical functions
 *I have to employ virtual function and have all filters share common base class in order to dynamically create objects, in which case
 *their common base class pointer is actually statically determined
 *if I want to dynamically create different type of objects, that is filters are template specification they don't share a common base neither the same type
 *I have to achieve in python
 */
namespace del {

/*

  class FilterFactory {
  public:
      static FilterFactory& instance();
      void registerFilter(std::string, const std::function<>& );
      ResampleBase& createResample(std::string, 
      ~ResampleFactory();

  private:
      std::map<std::string, > resample_creators_;
      ResampleFactory() {}
      ResampleFactory(const ResampleFactory&) {}
      ResampleFactory& operator=(const ResampleFactory&) {return *this;}

  };
  
  class ResampleFactory {
  public:
      static ResampleFactory& instance();
      void registerResample(std::string, tr1::function<>() );
      ResampleBase& createResample(std::string, 
      ~ResampleFactory();
      
  private:
      std::map<std::string, > resample_creators_;
      ResampleFactory() {}
      ResampleFactory(const ResampleFactory&) {}
      ResampleFactory& operator=(const ResampleFactory&) {return *this;}
      
  };
  
  
  class ModelFactory {
  public:
      static ResampleFactory& instance();
      void registerResample(std::string, tr1::function<>() );
      ResampleBase& createResample(std::string, 
      ~ResampleFactory();


  private:
      std::map<std::string, > resample_creators_;
      ResampleFactory() {}
      ResampleFactory(const ResampleFactory&) {}
      ResampleFactory& operator=(const ResampleFactory&) {return *this;}
      
  };

*/
}



#endif /* DEL_FACTORY_HPP_ */
