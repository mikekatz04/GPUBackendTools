#include "Interpolate.hh"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gbt_binding.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;


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


std::string get_module_path_gbt() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    py::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("interp") must match the name used in PYBIND11_MODULE
    py::object module = py::module::import("interp");

    // Access the __file__ attribute and cast it to a C++ string
    try {
        std::string path = module.attr("__file__").cast<std::string>();
        return path;
    } catch (const py::error_already_set& e) {
        // Handle the error if __file__ attribute is missing (e.g., if module is a namespace package)
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}

// PYBIND11_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void spline_part(py::module &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<CubicSplineWrap>(m, "CubicSplineWrapGPU")
#else
    py::class_<CubicSplineWrap>(m, "CubicSplineWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<array_type<double>, array_type<double>, array_type<double>, array_type<double>, array_type<double>, int, int, int>(), 
         py::arg("x0"), py::arg("y0"), py::arg("c1"), py::arg("c2"), py::arg("c3"), py::arg("ninterps"), py::arg("length"), py::arg("spline_type"))
    // Bind member functions
    .def("eval_wrap", &CubicSplineWrap::eval_wrap_func, "Evaluate splines.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("spline", &CubicSplineWrap::spline)
    // .def("get_link_ind", &CubicSplineWrap::get_link_ind, "Get link index.")
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<CubicSpline>(m, "CubicSplineGPU")
#else
    py::class_<CubicSpline>(m, "CubicSplineCPU")
#endif

    // Bind the constructor
    .def(py::init<double *, double *, double *, double *, double *, int, int, int>(),
         py::arg("x0"), py::arg("y0"), py::arg("c1"), py::arg("c2"), py::arg("c3"), py::arg("ninterps"), py::arg("length"), py::arg("spline_type"))

    ;
}



PYBIND11_MODULE(interp, m) {
    m.doc() = "Cubic Spline C++ plug-in"; // Optional module docstring

    m.attr("CUBIC_SPLINE_LINEAR_SPACING") = CUBIC_SPLINE_LINEAR_SPACING;
    m.attr("CUBIC_SPLINE_LOG10_SPACING") = CUBIC_SPLINE_LOG10_SPACING;
    m.attr("CUBIC_SPLINE_GENERAL_SPACING") = CUBIC_SPLINE_GENERAL_SPACING;
    
    // Call initialization functions from other files
    spline_part(m);
    m.def("check_spline", &check_spline, "Make sure that we can insert spline properly.");
    m.def("get_module_path_cpp", &get_module_path_gbt, "Returns the file path of the module");
    m.def("interpolate_wrap", &interpolate_wrap, "Interpolate arrays.");
    // Optionally, get the path during module initialization and store it
    // This can cause an AttributeError if not handled carefully, as m.attr("__file__")
    // might not be fully set during the initial call if the module is loaded in
    // a specific way (e.g., via pythonw or as a namespace package).
    try {
        std::string path_at_init = m.attr("__file__").cast<std::string>();
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = py::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (py::error_already_set &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}

