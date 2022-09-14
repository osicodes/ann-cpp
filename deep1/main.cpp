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

        /*Matrix* ins = matrix_create(2, 1);
        //matrix_fill(ins, 2);
        ins->entries[0][0] = 10;
        ins->entries[1][0] = 5;
        Dense denl = Dense(1, 2, "he");
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
        cout << endl;*/

        

        const char* train = "C:\\Users\\ozuem\\Documents\\AI\\ANN with CPP\\mnist_train.csv\\mnist_train.csv";
        const char* test = "C:\\Users\\ozuem\\Documents\\AI\\ANN with CPP\\mnist_test.csv\\mnist_test.csv";
        /*FILE* pFile;
        char mystring[785];

        pFile = fopen(str, "r");
        if (pFile == NULL) perror("Error opening file");
        else {
            if (fgets(mystring, 785, pFile) != NULL)
                puts(mystring);
            fclose(pFile);
        }*/

        NeuralNetwork* net = network_create(784, 300, 10, 0.001);

        Img** img = csv_to_imgs(train, 5000);
        network_train_batch_imgs(net, img, 4000);

        /*Img** img = csv_to_imgs(str, 2);
        Img* cur_img = img[0];
        cout << "label: " << cur_img->label << endl;
        Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector

        Matrix* output = matrix_create(10, 1);
        column_output(output, cur_img->label);
        matrix_print(output);


        double error = network_train(net, img_data, output);
        cout << endl;
        //cout << "error: " << error << endl;*/

        cout << endl;
        cout << endl;
        cout << endl;
        //Matrix* result = network_predict_img(net, img[480]);
        Img** imag = csv_to_imgs(train, 500);
        double acc = network_predict_imgs(net, imag, 100);
        cout << endl;
        cout << "accuracy: " << acc << endl;
        cout << endl;


        /*Matrix* mat = matrix_create(10, 1);
        matrix_fill(mat, 1);
        cout << endl;
        Matrix* tmp = softmax(mat);
        cout << "tmp: " << endl;
        matrix_print(tmp);
        cout << endl;
        cout << endl;
        cout << "deriv softmax: " << endl;
        matrix_print(dot(softmax_prime(mat->rows, tmp),mat));*/
        

        return 0;
    }
