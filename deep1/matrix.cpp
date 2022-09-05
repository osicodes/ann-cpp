#include <iostream>
#include "matrix.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace std;

#define MAXCHAR 100

Matrix* matrix_create(int row, int col) {
	Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
	matrix->rows = row;
	matrix->cols = col;
	matrix->entries = (double**)malloc(row * sizeof(double*)); //new double* [row];
	for (int i = 0; i < row; i++) {
		matrix->entries[i] = (double*)malloc(col * sizeof(double));
	}
	return matrix;
}

void matrix_fill(Matrix* m, int n) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->entries[i][j] = n;
		}
	}
}

void matrix_myfill(Matrix* m) {
	int a;
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			cin >> a;
			m->entries[i][j] = a;
		}
	}
}

void matrix_free(Matrix* m) {
	for (int i = 0; i < m->rows; i++) {
		free(m->entries[i]);
	}
	free(m);
	m = NULL;
}

void matrix_print(Matrix* m) {
	printf("Rows: %d Columns: %d\n", m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%1.5f ", m->entries[i][j]);
		}
		printf("\n");
	}
}

Matrix* matrix_copy(Matrix* m) {
	Matrix* mat = matrix_create(m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = m->entries[i][j];
		}
	}
	return mat;
}

void matrix_save(Matrix* m, const char* file_string) {
	FILE* file = fopen(file_string, "w");
	fprintf(file, "%d\n", m->rows);
	fprintf(file, "%d\n", m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			fprintf(file, "%.6f\n", m->entries[i][j]);
		}
	}
	printf("Successfully saved matrix to %s\n", file_string);
	fclose(file);
}

Matrix* matrix_load(const char* file_string) {
	FILE* file = fopen(file_string, "r");
	char entry[MAXCHAR];
	fgets(entry, MAXCHAR, file);
	int rows = atoi(entry);
	fgets(entry, MAXCHAR, file);
	int cols = atoi(entry);
	Matrix* m = matrix_create(rows, cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			fgets(entry, MAXCHAR, file);
			m->entries[i][j] = strtod(entry, NULL);
		}
	}
	printf("Sucessfully loaded matrix from %s\n", file_string);
	fclose(file);
	return m;
}

double uniform_distribution(double low, double high) {
	double difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}


void my_matrix_randomize(Matrix* m, int in, int cu, const char* activefunc) {
	/*
	in -->  Number of nodes/neurons in input or previous layer (or number of input neurons coming into this current layer)
	cu -->  Number of nodes/neurons in current layer (not input or previous layer)
	size_l = cu;
	size_l_minus_1 = in;

	LeCun  Initialization (Good for SELU activation functions )
	----------------------
	   Min: -1 / sqrt(in)
	   Max: 1 / sqrt(in)
	---------------------


	Xavier normalized initialization  (Good for other activation functions{Linear, sigmoid/logistics, tanh, softmax, },
													 but not very good for RELU)
	 sqrt(1 / ave)
	 ave = (in + cu) / 2
	----------------------
	   Min: -sqrt(2 / (size_l_minus_1 + size_l))
	   Max: sqrt(2 / (size_l_minus_1 + size_l))
	---------------------


	Xavier uniform initialization
	----------------------
	   Min: -(sqrt(6.0) / sqrt(size_l_minus_1 + size_l))
	   Max: (sqrt(6.0) / sqrt(size_l_minus_1 + msize_l))
	---------------------


	He Initialization (Good for RELU activation functions )
	----------------------
	   Min: -sqrt(2/in)
	   Max: sqrt(2/in))
	---------------------

	For  Hyperbolic tangent activation function, tanh(x)
	standard deviation (v) = 1 / sqrt(ave)
	ave = (in + cu) / 2

	For  Sigmoid/logistics activation function, 1 / (1 + exp(-x))
	standard deviation (v) = 3.6 / sqrt(in)

	For  RELU activation function, max(0,x)
	standard deviation (v) = sqrt(2/in)

	References
	-------------------------------------
	http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
	https://arxiv.org/pdf/1704.08863.pdf   page 3.
	https://arxiv.org/abs/1502.01852
	https://bit.ly/3ehISpn
	http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
	https://www.youtube.com/watch?v=tYFO434Lpm0
	*/
	int size_l = cu;
	int size_l_minus_1 = in;

	if (activefunc == "relu")
	{
		double min = -sqrt(2) / sqrt(size_l_minus_1);
		double max = sqrt(2) / sqrt(size_l_minus_1);
		for (int i = 0; i < m->rows; i++) {
			for (int j = 0; j < m->cols; j++) {
				m->entries[i][j] = uniform_distribution(min, max);
			}
		}
	}
	else if (activefunc == "softmax" || "tanh")
	{
		double min = -sqrt(2) / sqrt(size_l_minus_1 + size_l);
		double max = sqrt(2) / sqrt(size_l_minus_1 + size_l);
		for (int i = 0; i < m->rows; i++) {
			for (int j = 0; j < m->cols; j++) {
				m->entries[i][j] = uniform_distribution(min, max);
			}
		}
	}


	//double min = -(sqrt(6.0) / sqrt(size_l_minus_1 + size_l));
	//double max = (sqrt(6.0) / sqrt(size_l_minus_1 + size_l));
}


void matrix_randomize(Matrix* m, int n) {
	// Pulling from a random distribution of 
	// Min: -1 / sqrt(n)
	// Max: 1 / sqrt(n)
	double min = -1.0 / sqrt(n);
	double max = 1.0 / sqrt(n);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->entries[i][j] = uniform_distribution(min, max);
		}
	}
}

int matrix_argmax(Matrix* m) {
	// Expects a Mx1 matrix
	double max_score = 0;
	int max_idx = 0;
	for (int i = 0; i < m->rows; i++) {
		if (m->entries[i][0] > max_score) {
			max_score = m->entries[i][0];
			max_idx = i;
		}
	}
	return max_idx;
}

Matrix* matrix_flatten(Matrix* m, int axis) {
	// Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
	Matrix* mat;
	if (axis == 0) {
		mat = matrix_create(m->rows * m->cols, 1);
	}
	else if (axis == 1) {
		mat = matrix_create(1, m->rows * m->cols);
	}
	else {
		printf("Argument to matrix_flatten must be 0 or 1");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			if (axis == 0) mat->entries[i * m->cols + j][0] = m->entries[i][j];
			else if (axis == 1) mat->entries[0][i * m->cols + j] = m->entries[i][j];
		}
	}
	return mat;
}