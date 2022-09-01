#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include "imag.h"
//#include "neural/activations.h"
//#include "neural/nn.h"
#include "matrix.h"
#include "mathsMatrix.h"
using namespace std;

int main() {
	srand(time(NULL));

    /*int num1 = 70;
    double num2 = 256.783;
    char ch = 'A';
    Matrix* mt;
    Matrix* pt;
    Matrix* st;

    cout << num1 << endl;    // print integer
    cout << num2 << endl;    // print double
    cout << "character: " << ch << endl;

    mt = matrix_create(4, 4);
    matrix_fill(mt, 6);
    matrix_print(mt);

    pt = matrix_create(4, 4);
    matrix_fill(pt,2);
    matrix_print(pt);

    cout << "Enter values yourself" << endl;
    st = matrix_create(4, 4);
    matrix_myfill(st);
    matrix_print(st);

    matrix_print(transpose(st));


    matrix_print(multiply(mt, pt));
    matrix_print(dot(mt, pt));*/

    const char* str = "Solution Items/mnist_test.csv";
    FILE* pFile;
    char mystring[785];

    pFile = fopen(str, "r");
    if (pFile == NULL) perror("Error opening file");
    else {
        if (fgets(mystring, 785, pFile) != NULL)
            puts(mystring);
        fclose(pFile);
    }
    //Img** img = csv_to_imgs(str, 2);
	return 0;
}

