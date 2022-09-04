#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include "imag.h"
#include "activation.h"
#include "nn.h"
#include "matrix.h"
#include "mathsMatrix.h"
    using namespace std;

    double uniform_distribution(double, double);

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

        const char* str = "C:\\Users\\ozuem\\Documents\\AI\\ANN with CPP\\mnist_test.csv\\mnist_test.csv";
        /*FILE* pFile;
        char mystring[785];

        pFile = fopen(str, "r");
        if (pFile == NULL) perror("Error opening file");
        else {
            if (fgets(mystring, 785, pFile) != NULL)
                puts(mystring);
            fclose(pFile);
        }*/

        Img** img = csv_to_imgs(str, 2);
        //img_print(img[1]);

        //matrix_print(img[1]->img_data);

        NeuralNetwork* net = network_create(784, 300, 10, 0.01);

        Img* cur_img = img[0];
        cout << "label: " << cur_img->label << endl;
        Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
        Matrix* output = matrix_create(10, 1);
        output->entries[cur_img->label][0] = 1; // Setting the result
        matrix_print(output);


        Matrix* compare = network_train(net, img_data, output);
        cout << endl;
        cout << "output to compare: " << endl;
        matrix_print(compare);

        Matrix* result = network_predict(net, img_data);
        cout << endl;
        cout << "predicted output: " << endl;
        matrix_print(result);

        //network_save(net, "one_ex");

        return 0;
    }
