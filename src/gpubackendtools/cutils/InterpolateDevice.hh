#ifndef __INTERPOLATE_DEVICE_HH__
#define __INTERPOLATE_DEVICE_HH__

// Header-only device-side cubic-spline evaluators. Downstream `.cu` files
// (LISAanalysistools, GBGPU, BBHx, FEW) `#include "InterpolateDevice.hh"`
// to evaluate a CubicSpline / CubicSplineSegment built upstream without
// linking against GBT's `Interpolate.cu` translation unit. Build/solve
// (LAPACKE tridiagonal, PCR, host launchers) stays in `Interpolate.hh`
// + `Interpolate.cu`.

#include "gbt_global.h"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define CubicSpline CubicSplineGPU
#define QuinticSpline QuinticSplineGPU
#else
#define CubicSpline CubicSplineCPU
#define QuinticSpline QuinticSplineCPU
#endif

#define CUBIC_SPLINE_LINEAR_SPACING 1
#define CUBIC_SPLINE_LOG10_SPACING 2
#define CUBIC_SPLINE_GENERAL_SPACING 3

#if !defined(__CUDACC__)
#include <string>
#include <stdexcept>
#endif

class CubicSplineSegment{
   public:
    double x0;
    double y0;
    double c1;
    double c2;
    double c3;
    int spline_type;

    CUDA_DEVICE
    CubicSplineSegment(double x0_, double y0_, double c1_, double c2_, double c3_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        spline_type = spline_type_;
    };
    CUDA_DEVICE
    double eval(double x_new)
    {
        double dx = x_new - x0;
        double out = y0 + c1 * dx + c2 * dx * dx + c3 * dx * dx * dx;
        return out;
    };
    CUDA_DEVICE
    double eval_single_derivative(double x_new)
    {
        double dx = x_new - x0;
        double out = c1+ c2 * dx + c3 * dx * dx;
        return out;
    };
    CUDA_DEVICE
    double eval_double_derivative(double x_new)
    {
        double dx = x_new - x0;
        double out = c2 + c3 * dx;
        return out;
    };
    CUDA_DEVICE
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
    };

    CUDA_DEVICE
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
#endif
        }
    };

    CUDA_DEVICE
    double get_x0_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return x0[spline_index * length + index];
    };

    CUDA_DEVICE
    double get_y0_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return y0[spline_index * length + index];
    };

    CUDA_DEVICE
    double get_c1_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return c1[spline_index * length + index];
    };

    CUDA_DEVICE
    double get_c2_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return c2[spline_index * length + index];
    };

    CUDA_DEVICE
    double get_c3_val(int spline_index, int index)
    {
        check_indexing(spline_index, index);
        return c3[spline_index * length + index];
    };

    CUDA_DEVICE
    ~CubicSpline(){};

    CUDA_DEVICE
    int even_sampled_search(double *array, int nmin, int nmax, double x)
    {
        // Index relative to the grid origin: grids need not start at 0.
        double dx = array[nmin + 1] - array[nmin];
        return (int)floor((x - array[nmin]) / dx);
    }

    // Recursive binary search. Returns the nearest smaller neighbour of x in
    // array[nmin, nmax], else -1.
    CUDA_DEVICE
    int binary_search(double *array, int nmin, int nmax, double x)
    {
        if (x == array[nmin]) return nmin;

        int next;
        if (nmax > nmin)
        {
            int mid = nmin + (nmax - nmin) / 2;

            // Skip over duplicates.
            next = mid;
            while (array[mid] == array[next]) next++;

            if (x > array[mid] && x < array[next])
                return mid;

            if (array[mid] >= x)
                return binary_search(array, nmin, mid, x);

            return binary_search(array, next, nmax, x);
        }

        return -1;
    }

    CUDA_DEVICE
    int get_window(double x_new, int spline_index)
    {
        int window = 0;
        if (spline_type == CUBIC_SPLINE_LINEAR_SPACING)
        {
            // Subtract the grid origin: uniform grids need not start at 0.
            window = int((x_new - x0[spline_index * length + 0]) / (x0[spline_index * length + 1] - x0[spline_index * length + 0]));
        }
        else if (spline_type == CUBIC_SPLINE_LOG10_SPACING)
        {
            // Same origin correction in log10 space (origin need not be 1).
            window = int((log10(x_new) - log10(x0[spline_index * length + 0])) / (log10(x0[spline_index * length + 1]) - log10(x0[spline_index * length + 0])));
        }
        else if (spline_type == CUBIC_SPLINE_GENERAL_SPACING)
        {
            window = binary_search(&x0[spline_index * length], 0, length, x_new);
        }
        else
        {
#ifdef __CUDACC__
            // printf("BAD cubic spline type. (%d)\n", spline_type);
#else
            std::string error_str = "BAD cubic spline type. (" + std::to_string(spline_type) + ")";
            throw std::invalid_argument(error_str);
#endif
        }

        if ((window < 0) || (window >= length))
        {
#ifdef __CUDACC__
            if (window < 0) window = 0;
            if (window >= length) window = length - 1;
#else
            std::string error_str = "Outside spline." + std::to_string(window) + " " + std::to_string(length) + " " + std::to_string(x_new) + " [" + std::to_string(x0[spline_index * length + 0]) + ", " + std::to_string(x0[spline_index * length + length - 1]) + "]";
            throw std::invalid_argument(error_str);
#endif
        }

        return window;
    }

    CUDA_DEVICE
    CubicSplineSegment get_cublic_spline_segment(double x_new, int spline_index)
    {
        int window = get_window(x_new, spline_index);
        if (window == -2)
        {
#ifdef __CUDACC__
#else
            throw std::invalid_argument("BAD.");
#endif
        }
        int _index = spline_index * length + window;
        CubicSplineSegment segment(x0[_index], y0[_index], c1[_index], c2[_index], c3[_index], spline_type);
        return segment;
    }

    CUDA_DEVICE
    double eval_single(double x_new, int spline_index)
    {
        CubicSplineSegment segment = get_cublic_spline_segment(x_new, spline_index);
        return segment.eval(x_new);
    }

    CUDA_DEVICE
    void eval(double *y_new, double *x_new, int *spline_index, int N)
    {
#ifdef __CUDACC__
        int start1 = threadIdx.x + blockIdx.x * blockDim.x;
        int diff1 = gridDim.x * blockDim.x;
#else
        int start1 = 0;
        int diff1 = 1;
#endif
        for (int i = start1; i < N; i += diff1)
        {
            y_new[i] = eval_single(x_new[i], spline_index[i]);
        }
    }

    CUDA_DEVICE
    void dealloc(){};
};


// ===========================================================================
// Quintic (degree-5) spline evaluators. Mirror the cubic ones but carry five
// power-basis coefficients c1..c5 per segment (about the left node x0). The
// coefficients are produced by the quintic build/solve in Interpolate.cu and
// reproduce scipy.interpolate.make_interp_spline(x, y, k=5).
// ===========================================================================
class QuinticSplineSegment{
   public:
    double x0;
    double y0;
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    int spline_type;

    CUDA_DEVICE
    QuinticSplineSegment(double x0_, double y0_, double c1_, double c2_, double c3_, double c4_, double c5_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        c4 = c4_;
        c5 = c5_;
        spline_type = spline_type_;
    };
    CUDA_DEVICE
    double eval(double x_new)
    {
        double dx = x_new - x0;
        // Horner: y0 + dx*(c1 + dx*(c2 + dx*(c3 + dx*(c4 + dx*c5))))
        return y0 + dx * (c1 + dx * (c2 + dx * (c3 + dx * (c4 + dx * c5))));
    };
    CUDA_DEVICE
    double eval_single_derivative(double x_new)
    {
        double dx = x_new - x0;
        return c1 + dx * (2.0 * c2 + dx * (3.0 * c3 + dx * (4.0 * c4 + dx * 5.0 * c5)));
    };
    CUDA_DEVICE
    double eval_double_derivative(double x_new)
    {
        double dx = x_new - x0;
        return 2.0 * c2 + dx * (6.0 * c3 + dx * (12.0 * c4 + dx * 20.0 * c5));
    };
    CUDA_DEVICE
    double eval_triple_derivative(double x_new)
    {
        double dx = x_new - x0;
        return 6.0 * c3 + dx * (24.0 * c4 + dx * 60.0 * c5);
    };
    CUDA_DEVICE
    double eval_quad_derivative(double x_new)
    {
        double dx = x_new - x0;
        return 24.0 * c4 + dx * 120.0 * c5;
    };
    CUDA_DEVICE
    double eval_quint_derivative(double x_new)
    {
        return 120.0 * c5;
    };
};


class QuinticSpline{
public:
    double *x0;
    double *y0;
    double *c1;
    double *c2;
    double *c3;
    double *c4;
    double *c5;
    int ninterps;
    int length;
    int spline_type;

    CUDA_CALLABLE_MEMBER
    QuinticSpline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double *c4_, double *c5_, double ninterps_, int length_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        c4 = c4_;
        c5 = c5_;
        ninterps = ninterps_;
        length = length_;
        spline_type = spline_type_;
    };

    CUDA_DEVICE
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
#endif
        }
    };

    CUDA_DEVICE
    ~QuinticSpline(){};

    CUDA_DEVICE
    int even_sampled_search(double *array, int nmin, int nmax, double x)
    {
        double dx = array[nmin + 1] - array[nmin];
        return (int)floor((x - array[nmin]) / dx);
    }

    // Recursive binary search. Returns the nearest smaller neighbour of x in
    // array[nmin, nmax], else -1.
    CUDA_DEVICE
    int binary_search(double *array, int nmin, int nmax, double x)
    {
        if (x == array[nmin]) return nmin;

        int next;
        if (nmax > nmin)
        {
            int mid = nmin + (nmax - nmin) / 2;

            next = mid;
            while (array[mid] == array[next]) next++;

            if (x > array[mid] && x < array[next])
                return mid;

            if (array[mid] >= x)
                return binary_search(array, nmin, mid, x);

            return binary_search(array, next, nmax, x);
        }

        return -1;
    }

    CUDA_DEVICE
    int get_window(double x_new, int spline_index)
    {
        int window = 0;
        if (spline_type == CUBIC_SPLINE_LINEAR_SPACING)
        {
            window = int((x_new - x0[spline_index * length + 0]) / (x0[spline_index * length + 1] - x0[spline_index * length + 0]));
        }
        else if (spline_type == CUBIC_SPLINE_LOG10_SPACING)
        {
            window = int((log10(x_new) - log10(x0[spline_index * length + 0])) / (log10(x0[spline_index * length + 1]) - log10(x0[spline_index * length + 0])));
        }
        else if (spline_type == CUBIC_SPLINE_GENERAL_SPACING)
        {
            window = binary_search(&x0[spline_index * length], 0, length, x_new);
        }
        else
        {
#ifdef __CUDACC__
#else
            std::string error_str = "BAD quintic spline type. (" + std::to_string(spline_type) + ")";
            throw std::invalid_argument(error_str);
#endif
        }

        if ((window < 0) || (window >= length))
        {
#ifdef __CUDACC__
            if (window < 0) window = 0;
            if (window >= length) window = length - 1;
#else
            std::string error_str = "Outside spline." + std::to_string(window) + " " + std::to_string(length) + " " + std::to_string(x_new) + " [" + std::to_string(x0[spline_index * length + 0]) + ", " + std::to_string(x0[spline_index * length + length - 1]) + "]";
            throw std::invalid_argument(error_str);
#endif
        }

        return window;
    }

    CUDA_DEVICE
    QuinticSplineSegment get_quintic_spline_segment(double x_new, int spline_index)
    {
        int window = get_window(x_new, spline_index);
        int _index = spline_index * length + window;
        QuinticSplineSegment segment(x0[_index], y0[_index], c1[_index], c2[_index], c3[_index], c4[_index], c5[_index], spline_type);
        return segment;
    }

    CUDA_DEVICE
    double eval_single(double x_new, int spline_index)
    {
        QuinticSplineSegment segment = get_quintic_spline_segment(x_new, spline_index);
        return segment.eval(x_new);
    }

    CUDA_DEVICE
    void eval(double *y_new, double *x_new, int *spline_index, int N)
    {
#ifdef __CUDACC__
        int start1 = threadIdx.x + blockIdx.x * blockDim.x;
        int diff1 = gridDim.x * blockDim.x;
#else
        int start1 = 0;
        int diff1 = 1;
#endif
        for (int i = start1; i < N; i += diff1)
        {
            y_new[i] = eval_single(x_new[i], spline_index[i]);
        }
    }

    CUDA_DEVICE
    void dealloc(){};
};

#endif // __INTERPOLATE_DEVICE_HH__
