#ifndef __BINDING_HPP__
#define __BINDING_HPP__

#include "Interpolate.hh"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
template<typename T>
using array_type = cai::cuda_array_t<T>;
#else
template<typename T>
using array_type = py::array_t<T>;
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define CubicSplineWrap CubicSplineWrapGPU
#else
#define CubicSplineWrap CubicSplineWrapCPU
#endif


class CubicSplineWrap {
  public:
    CubicSpline *spline;
    CubicSplineWrap(array_type<double> x0_, array_type<double> y0_, array_type<double> c1_, array_type<double> c2_, array_type<double> c3_, int ninterps_, int length_, int spline_type_)
    {

        double *_x0 = return_pointer_and_check_length(x0_, "x0", length_, ninterps_);
        double *_y0 = return_pointer_and_check_length(y0_, "y0", length_, ninterps_);
        double *_c1 = return_pointer_and_check_length(c1_, "c1", length_, ninterps_);
        double *_c2 = return_pointer_and_check_length(c2_, "c2", length_, ninterps_);
        double *_c3 = return_pointer_and_check_length(c3_, "c3", length_, ninterps_);

        spline = new CubicSpline(_x0, _y0, _c1, _c2, _c3, ninterps_, length_, spline_type_);
    };
    ~CubicSplineWrap(){
        delete spline;
    };
    void eval_wrap(array_type<double>y_new, array_type<double>x_new, array_type<int>spline_index, int N);
    template<typename T>
    static T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
        
#else
        py::buffer_info buf1 = input1.request();

        if (buf1.size != N * multiplier)
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

};

#endif // __BINDING_HPP__

