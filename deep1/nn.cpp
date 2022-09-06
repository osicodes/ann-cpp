
#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "mathsMatrix.h"
#include "activations.h"
#include "layers.h"

#define MAXCHAR 1000

// 784, 300, 10, 0.01
NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
	NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	/*
	net->input =  input;
	net->hidden = hidden;
	net->output = output;
	net->learning_rate = lr;
	Matrix* hidden_layer = matrix_create(hidden, input);
	Matrix* output_layer = matrix_create(output, hidden);
	//matrix_randomize(hidden_layer, hidden);
	//matrix_randomize(output_layer, output);
	my_matrix_randomize(hidden_layer, input, hidden, "relu");
	my_matrix_randomize(output_layer, hidden, output, "softmax");
	net->hidden_weights = hidden_layer;
	net->output_weights = output_layer;
	*/

	net->input = input;
	net->hidden = hidden;
	net->output = output;
	net->learning_rate = lr;
	Dense hidden_layer = Dense(input, hidden, "he");
	Dense output_layer = Dense(hidden, output,"he");
	net->hidden_weights = hidden_layer.weight;
	net->hidden_bias = hidden_layer.bias;

	net->output_weights = output_layer.weight;
	net->output_bias = output_layer.bias;

	net->learning_rate = lr;

	net->hidden_layer = hidden_layer;
	net->output_layer = output_layer;
	return net;
}

double network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
	/*
	// Feed forward
	Matrix* hidden_inputs = dot(net->hidden_weights, input);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);

	//----was just added
	Matrix* compare = matrix_create(10, 1);
	compare = softmax(beforeSoftmax(final_outputs));
	//---------------

	// Find errors
	Matrix* output_errors = subtract(output, final_outputs);
	//output_errors = multiply(output_errors, output_errors);
	Matrix* hidden_errors = dot(transpose(net->output_weights), output_errors);

	// Backpropogate
	// output_weights = add(
	//		 output_weights, 
	//     scale(
	// 			  net->lr, 
	//			  dot(
	// 		 			multiply(
	// 						output_errors, 
	//				  	sigmoidPrime(final_outputs)
	//					), 
	//					transpose(hidden_outputs)
	// 				)
	//		 )
	// )
	Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
	Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
	Matrix* transposed_mat = transpose(hidden_outputs);
	Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
	Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
	Matrix* added_mat = add(net->output_weights, scaled_mat);
	matrix_free(net->output_weights); // Free the old weights before replacing
	net->output_weights = added_mat;

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// hidden_weights = add(
	// 	 net->hidden_weights,
	// 	 scale (
	//			net->learning_rate
	//    	dot (
	//				multiply(
	//					hidden_errors,
	//					sigmoidPrime(hidden_outputs)	
	//				)
	//				transpose(inputs)
	//      )
	// 	 )
	// )
	// Reusing variables after freeing memory
	sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
	multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
	transposed_mat = transpose(input);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);
	matrix_free(net->hidden_weights); // Free the old hidden_weights before replacement
	net->hidden_weights = added_mat;

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);
	// Free matrices
	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);
	matrix_free(output_errors);
	matrix_free(hidden_errors);
	return compare;
	*/


	// Feed forward
	//Matrix* hidden_inputs = dot(net->hidden_weights, input);
	//Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	//Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	//Matrix* final_outputs = apply(sigmoid, final_inputs);

	Matrix* forward_inputs = net->hidden_layer.forward(input);
	Tanh reluhid;
	forward_inputs = reluhid.forward(forward_inputs);
	forward_inputs = net->output_layer.forward(forward_inputs);
	Tanh reluout;
	Matrix* final_outputs = reluout.forward(forward_inputs);

	

	/*//----was just added
	Matrix* compare = matrix_create(10, 1);
	compare = softmax(beforeSoftmax(final_outputs));
	//---------------*/

	//------Find errors and derivative of error-------
	double error = mse(output, final_outputs);
	//printf("\n error : %1.5f\n", error);

	//Matrix* output_errors = subtract(output, final_outputs);
	//Matrix* hidden_errors = dot(transpose(net->output_weights), output_errors);

	Matrix* grad_output_errors = mse_prime(output, final_outputs);


	grad_output_errors = reluout.backward(grad_output_errors, net->learning_rate);
	grad_output_errors = net->output_layer.backward(grad_output_errors, net->learning_rate);
	grad_output_errors = reluhid.backward(grad_output_errors, net->learning_rate);
	grad_output_errors = net->hidden_layer.backward(grad_output_errors, net->learning_rate);

	matrix_free(net->output_weights); // Free the old weights before replacing
	net->output_weights = net->output_layer.weight;

	matrix_free(net->hidden_weights); // Free the old hidden_weights before replacement
	net->hidden_weights = net->hidden_layer.weight;

	// Free matrices
	matrix_free(final_outputs);
	return error;

}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
	for (int i = 0; i < batch_size; i++) {
		if (i % 100 == 0) printf("Img No. %d\n", i);
		Img* cur_img = imgs[i];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		//output->entries[cur_img->label][0] = 1; ;// 
		column_output(output, cur_img->label);// Setting the result
		double er = network_train(net, img_data, output);
		printf("\n err : %1.5f\n", er);
		matrix_free(output);
		//matrix_free(img_data);
	}
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
	Matrix* img_data = matrix_flatten(img->img_data, 0);
	printf("\ntest label : %d\n", img->label);
	Matrix* res = network_predict(net, img_data);
	matrix_free(img_data);
	return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
	int n_correct = 0;
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i]);
		printf("predicted output: %d\n", matrix_argmax(prediction));
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
	/*Matrix* hidden_inputs = dot(net->hidden_weights, input_data);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);
	//Matrix* result = softmax(final_outputs);
	Matrix* result = softmax(beforeSoftmax(final_outputs));  //Use this when using relu activation
	return result;*/


	Matrix* forward_inputs = net->hidden_layer.forward(input_data);
	Tanh reluhid;
	forward_inputs = reluhid.forward(forward_inputs);
	forward_inputs = net->output_layer.forward(forward_inputs);
	Tanh reluout;
	Matrix* final_outputs = reluout.forward(forward_inputs);
	Matrix* result = softmax(final_outputs);
	//Matrix* result = softmax(beforeSoftmax(final_outputs));  //Use this when using relu activation
	return result; 
}

void network_save(NeuralNetwork* net, const char* file_string) {
	_mkdir(file_string);
	// Write the descriptor file
	_chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->input);
	fprintf(descriptor, "%d\n", net->hidden);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hidden_weights, "hidden");
	matrix_save(net->output_weights, "output");
	printf("Successfully written to '%s'\n", file_string);
	chdir("-"); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	_chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->input = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hidden = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hidden_weights = matrix_load("hidden");
	net->output_weights = matrix_load("output");
	printf("Successfully loaded network from '%s'\n", file_string);
	_chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork* net) {
	matrix_free(net->hidden_weights);
	matrix_free(net->output_weights);
	free(net);
	net = NULL;
}