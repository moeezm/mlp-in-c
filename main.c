#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <math.h>

// nn constants 
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// L is number of layers
// ETA is learning rate
#define L 3
#define ETA 3.0

#define BATCH_SIZE 10
#define EPOCHS 30

#define TRAINING_SIZE 60000
#define TESTING_SIZE 10000

//int layer_sizes[L] = {INPUT_SIZE, 30, OUTPUT_SIZE};
int layer_sizes[L] = {INPUT_SIZE, 30, OUTPUT_SIZE};

struct Datum {
	double input[INPUT_SIZE];
	double output[OUTPUT_SIZE];
};

struct Datum training_data[TRAINING_SIZE];
struct Datum testing_data[TESTING_SIZE];

const int DATA_HEADER_SIZE = 16;
const int LABEL_HEADER_SIZE = 8;

const char train_data_filename[] = "mnist_data/train-images.idx3-ubyte";
const char train_label_filename[] = "mnist_data/train-labels.idx1-ubyte";
const char test_data_filename[] = "mnist_data/t10k-images.idx3-ubyte";
const char test_label_filename[] = "mnist_data/t10k-labels.idx1-ubyte";

// random float between -1 and 1
double randf() {
	double res = (double)(rand()) / RAND_MAX;
	return res;
}

double gen_normal() {
	double u1 = randf();
	double u2 = randf();
	
	double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	return z0;
}

void print_picture(const double *arr) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			printf("%s ", ((arr[i*28 + j] > 0) ? "█" : "░"));
		}
		printf("\n");
	}
}

// one function since training and testing are in the same format
// n is number of images
void read_file(struct Datum *arr, const char *data_filename, const char *label_filename, int n) {
	// no buffer :(	
	FILE *data, *labels;
	data = fopen(data_filename, "rb");
	labels = fopen(label_filename, "rb");

	for (int i = 0; i < DATA_HEADER_SIZE; i++) fgetc(data);
	for (int i = 0; i < LABEL_HEADER_SIZE; i++) fgetc(labels);
	unsigned char vec[INPUT_SIZE];
	unsigned char c;
	for (int i = 0; i < n; i++) {
		fread(vec, 1, INPUT_SIZE, data);
		c = fgetc(labels);
		for (int j = 0; j < INPUT_SIZE; j++) {
			if (j == (int)c) arr[i].output[(int)c] = 1.0; 
			arr[i].input[j] = (double)vec[j] / 255;
		}
	}
	fclose(data);
	fclose(labels);
}
	
void init_data() {
	read_file(training_data, train_data_filename, train_label_filename, TRAINING_SIZE);
	read_file(testing_data, test_data_filename, test_label_filename, TESTING_SIZE);
}
// calculates AB and puts it in C
// (mxp)(pxn) = (mxn)
// A, B, C are contiguous 1D arrays of m*p, p*n, m*n size resp., and will be used as 2D arrays
void matmul(int m, int p, int n, const double *A, const double *B, double *C) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i*n + j] = 0;
			for (int k = 0; k < p; k++) {
				C[i*n + j] += A[i*p + k] * B[k*n + j];
			}
		}
	}
}

// answer is correct output vector for the input
double cost(double *answer, double *activation) {
	return 0.0;
}

// F is activation function, we'll use sigmoid
double F(double x) {
	return 1.0 / (1.0 + exp(-x));
}
// f is derivative of activation function
double f(double x) {
	double y = F(x);
	return y*(1.0 - y);
}

// remember L is the number of layers

// each element of weights is a pointer to a 2D array (the weights matrix)
double *weights[L];

// each element of biases is a pointer to a 1D array (the bias vector)
double *biases[L];

// randomly initialize weights and biases
void init_weights_and_biases() {
	for (int i = 1; i < L; i++) {
		weights[i] = malloc(layer_sizes[i] * layer_sizes[i-1] * sizeof(double));
		biases[i] = malloc(layer_sizes[i] * sizeof(double));
		for (int j = 0; j < layer_sizes[i]; j++) {
			biases[i][j] = gen_normal();
			for (int k = 0; k < layer_sizes[i-1]; k++) {
				weights[i][j*layer_sizes[i-1] + k] = gen_normal();
			}
		}
	}
}

void clean_weights_and_biases() {
	for (int i = 0; i < L; i++) {
		free(weights[i]);
		free(biases[i]);
	}
}

// compute the output on neural network of input vector x, store in res 
// if training is true, it'll store all of the "affine outputs" from each layer, i.e., the thing that goes into the activation function in outputs
void feedforward(const double *x, double *res, bool training, double *outputs[L]) {
	double *tmp = malloc(layer_sizes[0] * sizeof(double));
	for (int i = 0; i < layer_sizes[0]; i++) {
		tmp[i] = x[i];
		if (training) outputs[0][i] = x[i];
	}
	double *y = NULL;
	for (int i = 1; i < L; i++) {
		if (!training) {
			free(y);
			y = malloc(layer_sizes[i] * sizeof(double));
		}
		else y = outputs[i];
		matmul(layer_sizes[i], layer_sizes[i-1], 1, weights[i], tmp, y);
		for (int j = 0; j < layer_sizes[i]; j++) {
			y[j] += biases[i][j];
		}
		free(tmp);
		tmp = malloc(layer_sizes[i] * sizeof(double));
		for (int j = 0; j < layer_sizes[i]; j++) {
			tmp[j] = F(y[j]);
		}
	}
	// at the end, tmp holds the final result so copy into res
	for (int i = 0; i < layer_sizes[L-1]; i++) {
		res[i] = tmp[i];
	}
	free(tmp);
	if (!training) free(y);
}

// implements the backpropagation algorithm
// answer is the correct output for the input vector x
// takes in array of affine outputs from each layer, same as what feedforward outputs
// (weight/bias)_derivs store (del C)/(del weight/bias). they're the same shape as weights/biases 
// they accumulate in the above to make calculating average gradient easier
// same shape as weights and biases, resp.
void calc_gradients(double *answer, double *outputs[L], double *weight_derivs[L], double *bias_derivs[L]) {
	double *error = NULL;
	for (int i = L-1; i >= 1; i--) {
		double *error2 = malloc(layer_sizes[i] * sizeof(double));
		if (i == L-1) {
			for (int j = 0; j < layer_sizes[L-1]; j++) {
				error2[j] = F(outputs[L-1][j]) - answer[j];
			}
		}
		else {
			matmul(1, layer_sizes[i+1], layer_sizes[i], error, weights[i+1], error2);
		}
		for (int j = 0; j < layer_sizes[i]; j++) {
			error2[j] *= f(outputs[i][j]);
		}
		// compute derivatives w.r.t weights/biases
		for (int j = 0; j < layer_sizes[i]; j++) {
			for (int k = 0; k < layer_sizes[i-1]; k++) {
				weight_derivs[i][j*layer_sizes[i-1] + k] += error2[j] * ((i > 1) ? F(outputs[i-1][k]) : outputs[i-1][k]);
			}
			bias_derivs[i][j] += error2[j];
		}
		free(error);
		error = error2;
	}
	free(error);
}

// returns most probable digit for x
int inference(double *x) {
	double *res = malloc(layer_sizes[L-1] * sizeof(double));
	feedforward(x, res, false, NULL);
	int idx = 0;
	double best = 0;
	for (int i = 0; i < layer_sizes[L-1]; i++) {
		if (res[i] > best) {
			best = res[i];
			idx = i;
		}
	}
	return idx;
}

// evaluate on testing data
// returns number of correct responses
int test() {
	int actual_cnts[10];
	int cnts[10];
	memset(actual_cnts, 0, sizeof(cnts));
	memset(cnts, 0, sizeof(cnts));
	int correct = 0;
	double loss = 0;
	for (int i = 0; i < TESTING_SIZE; i++) {
		int actual_ans = 0;
		double best = 0;
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			if (testing_data[i].output[j] > best) {
				best = testing_data[i].output[j];
				actual_ans = j;
			}
		}
		double *outvec = malloc(layer_sizes[L-1] * sizeof(double));
		feedforward(testing_data[i].input, outvec, false, NULL);
		int res = 0;
		best = 0;
		for (int j= 0; j < layer_sizes[L-1]; j++) {
			double diff = (outvec[j] - testing_data[i].output[j]);
			loss += (diff * diff);
			if (outvec[j] > best) {
				best = outvec[j];
				res = j;
			}
		}
		correct += (res == actual_ans);
		actual_cnts[actual_ans]++;
		cnts[res]++;
	}
	printf("Loss: %f\n", loss/TESTING_SIZE);
	printf("Actual distribution: ");
	for (int i = 0; i < 10; i++) printf("%d ", actual_cnts[i]);
	printf("\nOutput distribution: ");
	for (int i = 0; i < 10; i++) printf("%d ", cnts[i]);
	printf("\n");
	return correct;
}

// implement stochastic gradient descent for one epoch
void epoch() {
	// randomly shuffle training data
	for (int i = TRAINING_SIZE-1; i >= 1; i--) {
		int j = rand() % i;
		struct Datum tmp = training_data[i];
		training_data[i] = training_data[j];
		training_data[j] = tmp;
	}

	// initialize stuff
	double *outputs[L];
	double *weight_derivs[L];
	double *bias_derivs[L];
	for (int i = 0; i < L; i++) {
		outputs[i] = malloc(layer_sizes[i] * sizeof(double));
		if (i > 0) bias_derivs[i] = malloc(layer_sizes[i] * sizeof(double));
		if (i > 0) weight_derivs[i] = malloc(layer_sizes[i] * layer_sizes[i-1] * sizeof(double));
	}
	// do batches
	double *res = malloc(layer_sizes[L-1] * sizeof(double));
	for (int i = 0; i < TRAINING_SIZE; i += BATCH_SIZE) {
		// zero out
		for (int j = 0; j < L; j++) {
			memset(outputs[j], 0, layer_sizes[j] * sizeof(double));
			if (j > 0) memset(bias_derivs[j], 0, layer_sizes[j] * sizeof(double));
			if (j > 0) memset(weight_derivs[j], 0, layer_sizes[j] * layer_sizes[j-1] * sizeof(double));
		}
		for (int j = i; j < TRAINING_SIZE && j < i + BATCH_SIZE; j++) {
			feedforward(training_data[j].input, res, true, outputs);
			calc_gradients(training_data[j].output, outputs, weight_derivs, bias_derivs);

		}
		double factor = (ETA / BATCH_SIZE);
		for (int j = 1; j < L; j++) {
			for (int k = 0; k < layer_sizes[j]; k++) {
				biases[j][k] -= (factor * bias_derivs[j][k]);
			}
			for (int k = 0; k < layer_sizes[j] * layer_sizes[j-1]; k++) {
				weights[j][k] -= (factor * weight_derivs[j][k]);
			}
		}
		//for (int j = 1; j < L; j++) {
		//	for (int k = 0; k < layer_sizes[j] * layer_sizes[j-1]; k++) {
		//		printf("%f ", weight_derivs[j][k]);
		//	}
		//	printf("\n\n\n\n");
		//}
	}
	const double EPS = 1e-10;
	bool zero = true;
	for (int i = 1; i < L; i++) {
		for (int j = 0; j < layer_sizes[i] * layer_sizes[i-1]; j++) {
			if (weight_derivs[i][j] > EPS) zero = false;
		}
		for (int j = 0; j < layer_sizes[i]; j++) {
			if (bias_derivs[i][j] > EPS) zero = false;
		}
	}
	if (zero) {
		printf("ZERO!!!!\n");
	}
	free(res);
	for (int i = 0; i < L; i++) {
		free(outputs[i]);
		if (i > 0) free(bias_derivs[i]);
		if (i > 0) free(weight_derivs[i]);
	}
}

void train() {
	for (int i = 1; i <= EPOCHS; i++) {
		printf("Epoch %d/%d: ", i, EPOCHS);
		epoch();
		//test(); 
		int correct = test();
		printf("%d/%d \n", correct, TESTING_SIZE);
	}
}

int main() {
	srand(time(NULL));

	double A[6] = {6, 4, 9, 6, 3, 3};
	double B[8] = {6, 1, 7, 7, 3, 1, 9, 1};
	double C[12];
	matmul(3, 2, 4, A, B, C);
	for (int i = 0; i < 12; i++) printf("%f ", C[i]);
	printf("\n");

	init_data();
	printf("Data initialized\n");

	init_weights_and_biases();
	printf("Weights and biases initialized\n");

	int correct = test();
	printf("Original score: %d/%d\n", correct, TESTING_SIZE);

	train();

	clean_weights_and_biases();
	return 0;
}
