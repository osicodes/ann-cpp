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
#include "layers.h"
    using namespace std;

    double uniform_distribution(double, double);

    int main() {
        srand(time(NULL));

        Matrix* ins = matrix_create(2, 1);
        //matrix_fill(ins, 2);
        ins->entries[0][0] = 10;
        ins->entries[1][0] = 5;
        Dense denl = Dense(1, 2);
        Matrix*  ss = denl.forward(ins);

        Matrix* tru = matrix_create(1, 1);
        tru->entries[0][0] = 1;
        Matrix* grad = mse_prime(tru, ss);


        cout << endl;
        cout << "weighted sum: " << endl;
        matrix_print(ss);
        cout << endl;
        cout << "weight before: " << endl;
        matrix_print(denl.weight);
        cout << endl;
        cout << endl;
        cout << "bias before: " << endl;
        matrix_print(denl.bias);
        cout << endl;
        cout << endl;

        Matrix* bk = denl.backward(grad, 0.005);
        cout << endl;
        cout << "weight after: " << endl;
        matrix_print(denl.weight);
        cout << endl;
        cout << endl;
        cout << "bias after: " << endl;
        matrix_print(denl.bias);
        cout << endl;
        cout << endl;

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


        cout << endl;
        cout << "softmax compare: " << endl;
        matrix_print(beforeSoftmax(compare));

        Matrix* result = network_predict(net, img_data);
        cout << endl;
        cout << "predicted output: " << endl;
        matrix_print(result); 


        cout << endl;
        cout << "mse: " << endl;
        cout << mse(compare, result);


        cout << endl;
        cout << "mse_prime: " << endl;
        matrix_print(mse_prime(compare,result));

        return 0;
    }
