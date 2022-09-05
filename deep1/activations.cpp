#include "activations.h"

#include <cmath>
#include <algorithm>
#include "mathsMatrix.h"

double sigmoid(double input) {
	return 1.0 / (1 + exp(-1 * input));
}


double relu(double input) {
	return std::max(0.0,input);
}

double tan_h(double input) {
	return tanh(input);
}

double tan_hPrime(double input) {
	return 1 - pow(input, 2);
}

Matrix* sigmoidPrime(Matrix* m) {
	Matrix* ones = matrix_create(m->rows, m->cols);
	matrix_fill(ones, 1);
	Matrix* subtracted = subtract(ones, m);
	Matrix* multiplied = multiply(m, subtracted);
	matrix_free(ones);
	matrix_free(subtracted);
	return multiplied;
}

Matrix* softmax(Matrix* m) {
	double total = 0;
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			total += exp(m->entries[i][j]);
		}
	}
	Matrix* mat = matrix_create(m->rows, m->cols);
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			mat->entries[i][j] = exp(m->entries[i][j]) / total;
		}
	}
	return mat;
}