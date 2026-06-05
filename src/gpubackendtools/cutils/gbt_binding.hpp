#ifndef __BINDING_HPP__
#define __BINDING_HPP__

#include "Interpolate.hh"
#include <string>
#include <iostream>
// Phase 3M (2026-06-04): sprint-wide pybind11 -> nanobind migration.
// The CUDA-array story is dramatically simpler now -- nanobind ships
// `nb::ndarray<T, nb::device::cuda>` with first-class
// `__cuda_array_interface__` + DLPack support, so the bespoke
// pybind11_cuda_array_interface.hpp caster has been retired from the
// active include chain (the file is kept on disk during the
// transition so any straggling downstream still on pybind11 can build,
// but no sprint binding source includes it after Phase 3M).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// `nb::class_<T>::def_*` need these specializations available.
#include <nanobind/stl/string.h>

namespace nb = nanobind;

// `array_type<T>` is the sprint-wide numpy/CuPy-array typedef every
// binding TU consumes (LAT, GBGPU, BBHx, lisa-on-gpu, and GBT itself).
// Mirrors the pre-Phase-3M pybind11 version: CPU builds bind against
// host numpy arrays, GPU builds bind against device CuPy arrays.
//
// Notes for migrating users:
// - Use `arr.data()` (typed T*) instead of `(T*)arr.request().ptr`.
// - Use `arr.size()` instead of `arr.request().size`.
// - Use `arr.shape(i)` (size_t) instead of `arr.request().shape[i]`.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
template<typename T>
using array_type = nb::ndarray<T, nb::device::cuda>;
#else
template<typename T>
using array_type = nb::ndarray<T, nb::device::cpu>;
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
    void eval_wrap_func(array_type<double>y_new, array_type<double>x_new, array_type<int>spline_index, int N);
    template<typename T>
    static T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
        // nanobind's `nb::ndarray<T, ...>` exposes `.size()` (total elements)
        // and `.data()` (typed T* into the underlying numpy/CuPy buffer)
        // uniformly across CPU/GPU device tags. No buffer_info dance.
        if (input1.size() != static_cast<size_t>(N) * static_cast<size_t>(multiplier))
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(static_cast<size_t>(N) * static_cast<size_t>(multiplier)) + ". It's length is " + std::to_string(input1.size()) + ".";
            throw std::invalid_argument(err_out);
        }
        return input1.data();
    };

};

#endif // __BINDING_HPP__
