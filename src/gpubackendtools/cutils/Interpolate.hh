#ifndef __INTERPOLATE_HH__
#define __INTERPOLATE_HH__

#include "gbt_global.h"
void interpolate(double* x, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int ninterps);

class CubicSplineSegment{
   public:
    double x0;
    double y0;
    double c1;
    double c2;
    double c3;
    int spline_type;

    CUDA_CALLABLE_MEMBER
    CubicSplineSegment(double x0_, double y0_, double c1_, double c2_, double c3_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        spline_type = spline_type_;
    };
    CUDA_CALLABLE_MEMBER
    double eval(double x_new)
    {
        double dx = x_new - x0;
        double out = y0 + c1 * dx + c2 * dx * dx + c3 * dx * dx * dx;
        return out;
    };
    CUDA_CALLABLE_MEMBER
    double eval_single_derivative(double x_new)
    {
        double dx = x_new - x0;
        double out = c1+ c2 * dx + c3 * dx * dx;
        return out;
    };
    CUDA_CALLABLE_MEMBER
    double eval_double_derivative(double x_new)
    {
        double dx = x_new - x0;
        double out = c2 + c3 * dx;
        return out;
    };
    CUDA_CALLABLE_MEMBER
    double eval_triple_derivative(double x_new)
    {
        double out = c3;
        return out;
    };
};


class CubicSpline{
public:
    double *x0;
    double *y0;
    double *c1;
    double *c2;
    double *c3;
    int ninterps;
    int length;
    int spline_type;

    CUDA_CALLABLE_MEMBER
    CubicSpline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        ninterps = ninterps_;
        length = length_;
        spline_type = spline_type_;
        // printf("spline_type: %d\n", spline_type);
        // printf("CHECKING:: x0[0]: %.12e; x0[length - 1]: %.12e\n", x0[0], x0[length - 1]);
    };

    CUDA_CALLABLE_MEMBER
    void check_indexing(int spline_index, int index)
    {
        if (spline_index >= ninterps)
        {
#ifdef __CUDACC__
            printf("spline_index too high. (%d > %d)\n", spline_index, ninterps);
#else
            std::string error_str = "spline_index too high. (" + std::to_string(spline_index) + ">" + std::to_string(ninterps) + ")"; 
            throw std::invalid_argument(error_str);
#endif
        }
        if (index >= length)
        {
#ifdef __CUDACC__
            printf("index too high. (%d > %d)\n", index, length);
#else
            std::string error_str = "index too high. (" + std::to_string(index) + ">" + std::to_string(length) + ")"; 
            throw std::invalid_argument(error_str);
#endif // __CUDACC__
        }
    };

    CUDA_CALLABLE_MEMBER
    double get_x0_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return x0[spline_index * length + index];
    };

    CUDA_CALLABLE_MEMBER
    double get_y0_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return y0[spline_index * length + index];
    };

    CUDA_CALLABLE_MEMBER
    double get_c1_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return c1[spline_index * length + index];
    };

    CUDA_CALLABLE_MEMBER
    double get_c2_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return c2[spline_index * length + index];
    };

    CUDA_CALLABLE_MEMBER
    double get_c3_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return c3[spline_index * length + index];
    };

    CUDA_CALLABLE_MEMBER
    ~CubicSpline(){};
    CUDA_CALLABLE_MEMBER
    int get_window(double x_new, int spline_index);
    CUDA_CALLABLE_MEMBER
    CubicSplineSegment get_cublic_spline_segment(double x_new, int spline_index);
    CUDA_CALLABLE_MEMBER
    double eval_single(double x_new, int spline_index);
    CUDA_CALLABLE_MEMBER
    void eval(double *y_new, double *x_new, int *spline_index, int N);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
    CUDA_CALLABLE_MEMBER
    int binary_search(double *array, int nmin, int nmax, double x);
    CUDA_CALLABLE_MEMBER
    int even_sampled_search(double *array, int nmin, int nmax, double x);
};

#endif // __INTERPOLATE_HH__
