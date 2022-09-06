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
	double maxnum = m1->entries[0][0];

	for (int i = 1; i < m1->rows; i++) {
		for (int j = 0; j < m1->cols; j++) {
			if (maxnum < m1->entries[i][j])
				maxnum = m1->entries[i][j];
		}
	}
	
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
	if (check_dimensions(m1, m2)) {
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
