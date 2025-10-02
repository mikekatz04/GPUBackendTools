#ifndef __INTERPOLATE_HH__
#define __INTERPOLATE_HH__

void interpolate(double* x, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int ninterps);

#endif // __INTERPOLATE_HH__
