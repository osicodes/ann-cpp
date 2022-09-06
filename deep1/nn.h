#pragma once

#include "matrix.h"
#include "imag.h"
#include "layers.h"


typedef struct NeuralNetwork {
	int input;
	int hidden;
	int output;
	double learning_rate;
	Matrix* hidden_weights;
	Matrix* hidden_bias;
	Matrix* output_weights;
	Matrix* output_bias;
	Dense hidden_layer;
	Dense output_layer;
} ;

NeuralNetwork* network_create(int input, int hidden, int output, double lr);
double network_train(NeuralNetwork* net, Matrix* input_data, Matrix* output_data);
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size);
Matrix* network_predict_img(NeuralNetwork* net, Img* img);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data);
void network_save(NeuralNetwork* net, const char* file_string);
NeuralNetwork* network_load(char* file_string);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);