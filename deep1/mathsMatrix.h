#pragma once

#include "matrix.h"

Matrix* multiply(Matrix* m1, Matrix* m2);
Matrix* add(Matrix* m1, Matrix* m2);
Matrix* subtract(Matrix* m1, Matrix* m2);
Matrix* beforeSoftmax(Matrix* m1);
Matrix* dot(Matrix* m1, Matrix* m2);
Matrix* apply(double (*func)(double), Matrix* m);
Matrix* scale(double n, Matrix* m);
Matrix* addScalar(double n, Matrix* m);
Matrix* transpose(Matrix* m);
double mse(Matrix* m1, Matrix* m2);
Matrix* mse_prime(Matrix* m1, Matrix* m2);