#pragma once

#include "matrix.h"

double sigmoid(double input);
double relu(double input);
double tan_h(double input);
Matrix* sigmoidPrime(Matrix* m);
Matrix* softmax(Matrix* m);