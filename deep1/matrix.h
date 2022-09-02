#pragma once

typedef struct Matrix {
    double** entries;
    int rows;
    int cols;
};


Matrix* matrix_create(int row, int col);
void matrix_fill(Matrix* m, int n);
void matrix_myfill(Matrix* m);
void matrix_free(Matrix* m);
void matrix_print(Matrix* m);
Matrix* matrix_copy(Matrix* m);
void matrix_save(Matrix* m, const char* file_string);
Matrix* matrix_load(const char* file_string);
void my_matrix_randomize(Matrix* m, int in, int cu, const char* activefunc);
void matrix_randomize(Matrix* m, int n);
int matrix_argmax(Matrix* m);
Matrix* matrix_flatten(Matrix* m, int axis);
