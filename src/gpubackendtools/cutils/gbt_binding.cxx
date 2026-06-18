#include "Interpolate.hh"
#include <string>
#include <iostream>
// Phase 3M (2026-06-04): pybind11 -> nanobind migration. The bespoke
// pybind11_cuda_array_interface.hpp caster is no longer included --
// `nb::ndarray<T, nb::device::cuda>` understands `__cuda_array_interface__`
// natively (and DLPack), so the hand-rolled caster is obsolete.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "gbt_binding.hpp"

namespace nb = nanobind;


void CubicSplineWrap::eval_wrap_func(array_type<double>y_new, array_type<double>x_new, array_type<int>spline_index, int N)
{
    eval_wrap(
        spline,
        return_pointer_and_check_length(y_new, "y_new", N, 1),
        return_pointer_and_check_length(x_new, "x_new", N, 1),
        return_pointer_and_check_length(spline_index, "spline_index", N, 1),
        N
    );
}


void QuinticSplineWrap::eval_wrap_func(array_type<double>y_new, array_type<double>x_new, array_type<int>spline_index, int N)
{
    eval_quintic_wrap(
        spline,
        CubicSplineWrap::return_pointer_and_check_length(y_new, "y_new", N, 1),
        CubicSplineWrap::return_pointer_and_check_length(x_new, "x_new", N, 1),
        CubicSplineWrap::return_pointer_and_check_length(spline_index, "spline_index", N, 1),
        N
    );
}


void check_spline(CubicSpline *spline)
{
    printf("%e\n", spline->x0[0]);
}

void interpolate_wrap(array_type<double>x, array_type<double>propArrays,
                 array_type<double>B, array_type<double>upper_diag, array_type<double>diag, array_type<double>lower_diag,
                 int length, int ninterps)
{
    interpolate(
        CubicSplineWrap::return_pointer_and_check_length(x, "x", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(propArrays, "propArrays", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(B, "B", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(upper_diag, "upper_diag", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(diag, "diag", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(lower_diag, "lower_diag", length, ninterps),
        length,
        ninterps
    );
}

void interpolate_quintic_wrap(array_type<double> x, array_type<double> y,
                 array_type<double> c1, array_type<double> c2, array_type<double> c3,
                 array_type<double> c4, array_type<double> c5,
                 int length, int ninterps, int chunk, int uniform)
{
    interpolate_quintic(
        CubicSplineWrap::return_pointer_and_check_length(x, "x", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(y, "y", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(c1, "c1", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(c2, "c2", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(c3, "c3", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(c4, "c4", length, ninterps),
        CubicSplineWrap::return_pointer_and_check_length(c5, "c5", length, ninterps),
        length,
        ninterps,
        chunk,
        uniform
    );
}

#if !defined(__CUDA_COMPILATION__) && !defined(__CUDACC__)
// (CPU-only) Single-spline fit via Thomas algorithm.
void fit_cubic_spline_thomas_wrap(array_type<double> x, array_type<double> y,
                                  array_type<double> c1, array_type<double> c2, array_type<double> c3,
                                  array_type<double> B,
                                  int length, int spline_type)
{
    fit_cubic_spline_thomas_run(
        CubicSplineWrap::return_pointer_and_check_length(x, "x", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(y, "y", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(c1, "c1", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(c2, "c2", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(c3, "c3", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(B, "B", length, 1),
        length, spline_type
    );
}
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
// (GPU-only) Single-spline fit via Parallel Cyclic Reduction.
void fit_cubic_spline_pcr_wrap(array_type<double> x, array_type<double> y,
                               array_type<double> c1, array_type<double> c2, array_type<double> c3,
                               array_type<double> B,
                               int length, int spline_type)
{
    fit_cubic_spline_pcr_run(
        CubicSplineWrap::return_pointer_and_check_length(x, "x", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(y, "y", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(c1, "c1", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(c2, "c2", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(c3, "c3", length, 1),
        CubicSplineWrap::return_pointer_and_check_length(B, "B", length, 1),
        length, spline_type
    );
}
#endif


std::string get_module_path_gbt() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    nb::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("interp") must match the name used in NB_MODULE
    nb::object module = nb::module_::import_("interp");

    // Access the __file__ attribute and cast it to a C++ string
    try {
        std::string path = nb::cast<std::string>(module.attr("__file__"));
        return path;
    } catch (const nb::python_error& e) {
        // Handle the error if __file__ attribute is missing (e.g., if module is a namespace package)
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}

// NB_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void spline_part(nb::module_ &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<CubicSplineWrap>(m, "CubicSplineWrapGPU")
#else
    nb::class_<CubicSplineWrap>(m, "CubicSplineWrapCPU")
#endif

    // Bind the constructor
    .def(nb::init<array_type<double>, array_type<double>, array_type<double>, array_type<double>, array_type<double>, int, int, int>(),
         nb::arg("x0"), nb::arg("y0"), nb::arg("c1"), nb::arg("c2"), nb::arg("c3"), nb::arg("ninterps"), nb::arg("length"), nb::arg("spline_type"))
    // Bind member functions
    .def("eval_wrap", &CubicSplineWrap::eval_wrap_func, "Evaluate splines.")
    // You can also expose public data members directly using def_rw
    .def_rw("spline", &CubicSplineWrap::spline)
    // .def("get_link_ind", &CubicSplineWrap::get_link_ind, "Get link index.")
    ;


// Phase 3M (2026-06-04): the raw-pointer init that pybind11 silently
// accepted -- `.def(nb::init<double *, double *, double *, double *,
// double *, int, int, int>(), ...)` -- is rejected by nanobind because
// `int -> double *` is a narrowing conversion. It was never usable from
// Python anyway (you can't pass raw `double *` from Python). The class
// is still registered so existing isinstance / cross-module casts on
// `CubicSpline{CPU,GPU}` keep working; Python code that needs to
// construct one goes through `CubicSplineWrap{CPU,GPU}`.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<CubicSpline>(m, "CubicSplineGPU");
#else
    nb::class_<CubicSpline>(m, "CubicSplineCPU");
#endif

    // ---- Quintic spline wrapper (5 coefficient arrays) ----
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<QuinticSplineWrap>(m, "QuinticSplineWrapGPU")
#else
    nb::class_<QuinticSplineWrap>(m, "QuinticSplineWrapCPU")
#endif
    .def(nb::init<array_type<double>, array_type<double>, array_type<double>, array_type<double>, array_type<double>, array_type<double>, array_type<double>, int, int, int>(),
         nb::arg("x0"), nb::arg("y0"), nb::arg("c1"), nb::arg("c2"), nb::arg("c3"), nb::arg("c4"), nb::arg("c5"), nb::arg("ninterps"), nb::arg("length"), nb::arg("spline_type"))
    .def("eval_wrap", &QuinticSplineWrap::eval_wrap_func, "Evaluate quintic splines.")
    .def_rw("spline", &QuinticSplineWrap::spline)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<QuinticSpline>(m, "QuinticSplineGPU");
#else
    nb::class_<QuinticSpline>(m, "QuinticSplineCPU");
#endif
}



NB_MODULE(interp, m) {
    m.doc() = "Cubic Spline C++ plug-in"; // Optional module docstring

    m.attr("CUBIC_SPLINE_LINEAR_SPACING") = CUBIC_SPLINE_LINEAR_SPACING;
    m.attr("CUBIC_SPLINE_LOG10_SPACING") = CUBIC_SPLINE_LOG10_SPACING;
    m.attr("CUBIC_SPLINE_GENERAL_SPACING") = CUBIC_SPLINE_GENERAL_SPACING;

    // Call initialization functions from other files
    spline_part(m);
    m.def("check_spline", &check_spline, "Make sure that we can insert spline properly.");
    m.def("get_module_path_cpp", &get_module_path_gbt, "Returns the file path of the module");
    m.def("interpolate_wrap", &interpolate_wrap, "Interpolate arrays.");
    m.def("interpolate_quintic_wrap", &interpolate_quintic_wrap,
          nb::arg("x"), nb::arg("y"), nb::arg("c1"), nb::arg("c2"), nb::arg("c3"),
          nb::arg("c4"), nb::arg("c5"), nb::arg("length"), nb::arg("ninterps"),
          nb::arg("chunk") = 0, nb::arg("uniform") = 0,
          "Quintic (k=5) interpolation: fill c1..c5 from (x, y). chunk=0 -> auto; "
          "uniform=1 enables the Toeplitz factorization cache.");
#if !defined(__CUDA_COMPILATION__) && !defined(__CUDACC__)
    m.def("fit_cubic_spline_thomas", &fit_cubic_spline_thomas_wrap,
          "(CPU-only) Fit a single cubic spline via the Thomas algorithm (in place; fills c1, c2, c3).");
#endif
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    m.def("fit_cubic_spline_pcr", &fit_cubic_spline_pcr_wrap,
          "(GPU-only) Fit a single cubic spline via Parallel Cyclic Reduction (in place; fills c1, c2, c3).");
#endif
    // Optionally, get the path during module initialization and store it
    // This can cause an AttributeError if not handled carefully, as m.attr("__file__")
    // might not be fully set during the initial call if the module is loaded in
    // a specific way (e.g., via pythonw or as a namespace package).
    try {
        std::string path_at_init = nb::cast<std::string>(m.attr("__file__"));
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = nb::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (nb::python_error &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}
