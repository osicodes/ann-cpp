#pragma once
#include "matrix.h"
#include "mathsMatrix.h"
#include "activations.h"

typedef struct Dense {
    Matrix* weight;
    Matrix* bias;
    Matrix* backwardInput;
    Dense(int in, int out, const char* initializationfunc) {
        weight = matrix_create(out, in);
        bias = matrix_create(out, 1);
        backwardInput = matrix_create(in, 1);

        my_matrix_randomize(weight, weight->rows, weight->cols, initializationfunc);
        my_matrix_randomize(bias, bias->rows, 1, initializationfunc);
    }

    Matrix* forward(Matrix* input)
    {
        backwardInput = input;
        return add(dot(weight, input),bias);
    }

    Matrix* backward(Matrix* output_gradient,double learning_rate)
    {
        Matrix* weights_gradient = dot(output_gradient, transpose(backwardInput));
        Matrix* input_gradient = dot(transpose(weight), output_gradient);
        weight = subtract(weight,scale(learning_rate,weights_gradient));
        bias = subtract(bias,scale(learning_rate,output_gradient));
        matrix_free(backwardInput);
        matrix_free(weights_gradient);
        return input_gradient;
    }
};


typedef struct Sigmoid {
    Matrix* backwardInput;

    Matrix* forward(Matrix* input)
    {
        backwardInput = matrix_create(input->rows, input->cols);
        backwardInput = input;
        return apply(sigmoid, input);
    }

    Matrix* backward(Matrix* output_gradient, double learning_rate)
    {
        return multiply(output_gradient, apply(sigmoid_prime, backwardInput));
    }
};

typedef struct Relu {
    Matrix* backwardInput;

    Matrix* forward(Matrix* input)
    {
        backwardInput = matrix_create(input->rows, input->cols);
        backwardInput = input;
        return apply(relu, input);
    }

    Matrix* backward(Matrix* output_gradient, double learning_rate)
    {
        return multiply(output_gradient, apply(relu_prime, backwardInput));
    }
};


typedef struct Tanh {
    Matrix* backwardInput;

    Matrix* forward(Matrix* input)
    {
        backwardInput = matrix_create(input->rows, input->cols);
        backwardInput = input;
        return apply(tan_h, input);
    }

    Matrix* backward(Matrix* output_gradient, double learning_rate)
    {
        return multiply(output_gradient, apply(tan_h_prime, backwardInput));
    }
};

typedef struct Softmax {
    Matrix* backwardInput;

    Matrix* forward(Matrix* input)
    {
        backwardInput = matrix_create(input->rows, input->cols);
        backwardInput = input;
        return softmax(input);
    }

    Matrix* backward(Matrix* output_gradient, double learning_rate)
    {
        Matrix* tmp = softmax(backwardInput);
        return dot(softmax_prime(backwardInput->rows, tmp), output_gradient);
    }
};