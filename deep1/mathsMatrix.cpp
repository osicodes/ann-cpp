#include "mathsMatrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

int check_dimensions(Matrix* m1, Matrix* m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

Matrix* multiply(Matrix* m1, Matrix* m2) {
	if (check_dimensions(m1, m2)) {
		Matrix* m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
			}
		}
		return m;
	}
	else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* add(Matrix* m1, Matrix* m2) {
	if (check_dimensions(m1, m2)) {
		Matrix* m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
			}
		}
		return m;
	}
	else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract(Matrix* m1, Matrix* m2) {
	if (check_dimensions(m1, m2)) {
		Matrix* m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
			}
		}
		return m;
	}
	else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* beforeSoftmax(Matrix* m1) {
	double maxnum = matrix_argmax(m1);
	
	Matrix* m = matrix_create(m1->rows, m1->cols);
	for (int i = 0; i < m1->rows; i++) {
		for (int j = 0; j < m1->cols; j++) {
			m->entries[i][j] = m1->entries[i][j] - maxnum;
				
		}
	}
	return m;
}

Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	return mat;
}

Matrix* identity(int num)
{
	int row, col;
	Matrix* m = matrix_create(num, num);
	for (row = 0; row < num; row++)
	{
		for (col = 0; col < num; col++)
		{
			// Checking if row is equal to column
			if (row == col)
				m->entries[row][col] = 1.0;
			else
				m->entries[row][col] = 0.0;
		}
	}
	return m;
}

Matrix* dot(Matrix* m1, Matrix* m2) {
	if (m1->cols == m2->rows) {
		Matrix* m = matrix_create(m1->rows, m2->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
			}
		}
		return m;
	}
	else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] *= n;
		}
	}
	return mat;
}

Matrix* addScalar(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] += n;
		}
	}
	return mat;
}

Matrix* transpose(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = m->entries[i][j];
		}
	}
	return mat;
}

// Mean Square Error
double mse(Matrix* m1, Matrix* m2) {
	if (check_dimensions(m1, m2)) { //true_y - predicted_y 
		double error = 0.0;
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				error += pow(m1->entries[i][j] - m2->entries[i][j],2);
			}
		}
		return error / m1->rows;
	}
	else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

// Mean Square error prime  ---  d(cost func)/d(activated output)
Matrix* mse_prime(Matrix* m1, Matrix* m2) {
	Matrix* output = subtract(m1, m2); //true_y - predicted_y  
	double sc = 2.0 / m1->rows;
	return scale(sc, output);
}

// Cross Enthropy Error
double cross_enthropy(Matrix* m1, Matrix* m2) {
	if (check_dimensions(m1, m2)) { //true_y * log(predicted_y)
		double error = 0.0;
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				error += m1->entries[i][j] * log(m2->entries[i][j]);
			}
		}
		return -1 * error;
	}
	else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* cross_enthropy_prime(Matrix* m1, Matrix* m2) {
	Matrix* output = subtract(m2, m1); //predicted_y  - true_y 
	return output;
}
