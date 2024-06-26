#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

// ==== CONSTANTS ====
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

// number of layers
#define L 3
// learning rate
#define ETA 0.25
#define BATCH_SIZE 10
#define EPOCHS 40
// regularization constant
#define LAMBDA 0.001
#define DROPOUT_PROB 0.0

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

// ==== DEFINITIONS ====
struct Datum {
	double input[INPUT_SIZE];
	double output[OUTPUT_SIZE];
};

// ==== GLOBAL VARIABLES ====
bool DROPOUT = false;

const int layer_sizes[L] = {INPUT_SIZE, 50, OUTPUT_SIZE};
double *weights[L];
double *biases[L];
double *errors[L];
double *affine_outputs[L];
double *activations[L];
double *weight_derivs[L];
double *bias_derivs[L];
bool *dropout[L];

struct Datum train_data[TRAIN_SIZE];
struct Datum test_data[TEST_SIZE];

// ==== RANDOM DISTRIBUTIONS ====
// uniform [0, 1] distribution
double randu() {
	return (double)(rand())/RAND_MAX;
}

// normal distribution with mean 0 and std dev 1
double randn() {
	double u1 = randu();
	double u2 = randu();
	double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
	return z0;
}

// ==== COST AND ACTIVATION ====
// let's use cross-entropy with L1 regularization
double cost(const double *truth, const double *activation) {
	double ans = 0;
	double diff;
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		//assert(activation[i] >= 0 && activation[i] <= 1);
		ans += (truth[i]*log(activation[i]) + (1-truth[i])*log(1-activation[i]));
	}
	ans *= -1;
	// add L1 norm of weights
	for (int i = 1; i < L; i++) {
		for (int j = 0; j < layer_sizes[i]; j++) {
			ans += LAMBDA * abs(weights[i][j]);
		}
	}
	return ans;
}

// derivative of cost w.r.t one activation
double cost_deriv(double truth, double activation) {
	return (activation - truth)/(activation*(1-activation));
}

int sgn(double x) {
	if (x < 0) return -1;
	else if (x == 0) return 0;
	else return 1;
}

// f is activation function, we'll use sigmoid
double f(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// fp is derivative of f
double fp(double x) {
	double y = f(x);
	return y*(1-y);
}

// ==== DATA ====
const char train_image_filename[] = "mnist_data/train-images.idx3-ubyte";
const char train_label_filename[] = "mnist_data/train-labels.idx1-ubyte";
const char test_image_filename[] = "mnist_data/t10k-images.idx3-ubyte";
const char test_label_filename[] = "mnist_data/t10k-labels.idx1-ubyte";

const int IMAGE_HEADER_SIZE = 16;
const int LABEL_HEADER_SIZE = 8;

// process data + label file for either training or test
// load n images + labels from resp. files into arr
void read_file(struct Datum *arr, const char image_filename[], const char label_filename[], int n) {
	// see http://yann.lecun.com/exdb/mnist/ for file format description
	FILE *images = fopen(image_filename, "rb");
	FILE *labels = fopen(label_filename, "rb");
	unsigned char c;
	for (int i = 0; i < IMAGE_HEADER_SIZE; i++) fgetc(images);
	for (int i = 0; i < LABEL_HEADER_SIZE; i++) fgetc(labels);
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			c = fgetc(images);
			arr[i].input[j] = (double)c / 255;
		}
		c = fgetc(labels);
		arr[i].output[c] = 1.0;
	}
}

void load_data() {
	read_file(train_data, train_image_filename, train_label_filename, TRAIN_SIZE);
	read_file(test_data, test_image_filename, test_label_filename, TEST_SIZE);
}

// ==== LIN ALG OPS ====
// performs AB and stores in C
// shapes: (mxp)(pxn) = (mxn)
// A, B, and C are 1d arrays that will be treated as 2d arrays
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

// ==== MEMORY MANAGEMENT ====
void initialize_vars() {
	for (int i = 0; i < L; i++) {
		if (i > 0) {
			weights[i] = malloc(layer_sizes[i] * layer_sizes[i-1] * sizeof(double));
			weight_derivs[i] = malloc(layer_sizes[i] * layer_sizes[i-1] * sizeof(double));
		}
		else {
			weights[0] = NULL;
			weight_derivs[0] = NULL;
		}
		biases[i] = malloc(layer_sizes[i] * sizeof(double));
		errors[i] = malloc(layer_sizes[i] * sizeof(double));
		affine_outputs[i] = malloc(layer_sizes[i] * sizeof(double));
		activations[i] = malloc(layer_sizes[i] * sizeof(double));
		bias_derivs[i] = malloc(layer_sizes[i] * sizeof(double));
		dropout[i] = malloc(layer_sizes[i] * sizeof(bool));
		for (int j = 0; j < layer_sizes[i]; j++) {
			dropout[i][j] = false;
			biases[i][j] = randn();
			for (int k = 0; k < layer_sizes[i-1]; k++) {
				weights[i][j*layer_sizes[i-1] + k] = randn();
			}
		}
	}
}

void clean_memory() {
	for (int i = 0; i < L; i++) {
		free(weights[i]);
		free(biases[i]);
		free(errors[i]);
		free(affine_outputs[i]);
		free(activations[i]);
		free(weight_derivs[i]);
		free(bias_derivs[i]);
		free(dropout[i]);
	}
}

// ==== APPLICATION ====
void feedforward(double *x) {
	for (int i = 0; i < INPUT_SIZE; i++) {
		activations[0][i] = x[i];
		affine_outputs[0][i] = x[i];
	}
	for (int i = 1; i < L; i++) {
		matmul(layer_sizes[i], layer_sizes[i-1], 1, weights[i], activations[i-1], affine_outputs[i]);
		for (int j = 0; j < layer_sizes[i]; j++) {
			if (!DROPOUT) affine_outputs[i][j] *= (1 - DROPOUT_PROB);
			affine_outputs[i][j] += biases[i][j];
			activations[i][j] = f(affine_outputs[i][j]);
			if (DROPOUT && dropout[i][j]) activations[i][j] = 0;
		}
	}
}

// extract most probable digit
int extract(double *arr) {
	int ans = 0;
	double best = 0;
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		if (arr[i] > best) {
			best = arr[i];
			ans = i;
		}
	}
	return ans;
}

int inference(double *x) {
	feedforward(x);
	return extract(activations[L-1]);
}

// ==== LEARNING ====
void backprop(double *truth) {
	for (int i = L-1; i >= 1; i--) {
		if (i == L-1) {
			for (int j = 0; j < OUTPUT_SIZE; j++) {
				errors[L-1][j] = cost_deriv(truth[j], activations[L-1][j]);
			}
		}
		else {
			matmul(1, layer_sizes[i+1], layer_sizes[i], errors[i+1], weights[i+1], errors[i]);
		}
		for (int j = 0; j < layer_sizes[i]; j++) {
			errors[i][j] *= fp(affine_outputs[i][j]);
			if (DROPOUT && dropout[i][j]) errors[i][j] = 0;
			bias_derivs[i][j] += errors[i][j];
			for (int k = 0; k < layer_sizes[i-1]; k++) {
				weight_derivs[i][j * layer_sizes[i-1] + k] += errors[i][j] * activations[i-1][k];
			}
		}
	}
}

void epoch() {
	// shuffle training data
	for (int i = TRAIN_SIZE - 1; i >= 1; i--) {
		int j = rand() % i;
		struct Datum tmp = train_data[i];
		train_data[i] = train_data[j];
		train_data[j] = tmp;
	}

	DROPOUT = true;
	// do batches
	for (int i = 0; i < TRAIN_SIZE; i += BATCH_SIZE) {
		for (int j = 0; j < L; j++) {
			if (j > 0) memset(weight_derivs[j], 0, layer_sizes[j] * layer_sizes[j-1] * sizeof(double));
			memset(bias_derivs[j], 0, layer_sizes[j] * sizeof(double));
		}

		for (int i = 1; i < L-1; i++) {
			for (int j = 0; j < layer_sizes[i]; j++) {
				dropout[i][j] = (randu() < DROPOUT_PROB);
			}
		}

		for (int j = i; j < TRAIN_SIZE && j < i + BATCH_SIZE; j++) {
			feedforward(train_data[j].input);
			backprop(train_data[j].output);
		}
		double factor = ETA / BATCH_SIZE;
		for (int j = 1; j < L; j++) {
			for (int k = 0; k < layer_sizes[j]; k++) {
				biases[j][k] -= factor * bias_derivs[j][k];
				for (int l = 0; l < layer_sizes[j-1]; l++) {
					weights[j][k*layer_sizes[j-1] + l] -= factor * weight_derivs[j][k*layer_sizes[j-1] + l] + (LAMBDA * factor) * sgn(weights[j][k*layer_sizes[j-1] + l]);
				}
			}
		}
	}
	DROPOUT = false;
}

// ==== TRAIN AND TEST ====
void test(struct Datum dataset[], int sz) {
	double loss = 0;
	int correct = 0;
	for (int i = 0; i < TEST_SIZE; i++) {
		feedforward(dataset[i].input);
		correct += (extract(activations[L-1]) == extract(dataset[i].output));
		loss += cost(dataset[i].output, activations[L-1])/sz;
	}
	//loss /= TEST_SIZE;
	printf("CORRECT: %d/%d || LOSS: %f", correct, sz, loss);
}

void train() {
	for (int i = 1; i <= EPOCHS; i++) {
		printf("EPOCH %d/%d || ", i, EPOCHS);
		printf("TRAIN ");
		epoch();
		test(train_data, TRAIN_SIZE);
		printf(" || TEST ");
		test(test_data, TEST_SIZE);
		double l1norm = 0;
		for (int l = 1; l < L; l++) {
			for (int i = 0; i < layer_sizes[l]; i++) {
				for (int j = 0; j < layer_sizes[l-1]; j++) {
					l1norm += abs(weights[l][i*layer_sizes[l-1] + j]);
				}
			}
		}
		printf(" || L1 NORM OF WEIGHTS %f", l1norm);
		printf("\n");
	}
}

int main() {
	printf("%f\n", log(0));
	srand(0);
	initialize_vars();
	printf("Global variables initialized\n");
	load_data();
	printf("Data loaded\n");
	printf("Initial run (on train data) || ");
	test(train_data, TRAIN_SIZE);
	printf("\n");
	train();
	clean_memory();
}
