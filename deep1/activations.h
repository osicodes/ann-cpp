#pragma once

#include "matrix.h"

double sigmoid(double input);
double sigmoid_prime(double input);
double relu(double input);
double relu_prime(double input);
double tan_h(double input);
double tan_h_prime(double input);
Matrix* sigmoidPrime(Matrix* m);
Matrix* softmax(Matrix* m);
Matrix* softmax_prime(int sz, Matrix* m);