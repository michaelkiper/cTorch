#ifndef TENSOR_UTILS_H_
#define TENSOR_UTILS_H_
#include <stdio.h>
#include "tensor.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <errno.h>

typedef unsigned int init;
#define UNIFORM 0
#define GAUSSIAN 1

#define INIT_TYPE UNIFORM // here is where you can change the initialization type

typedef unsigned int kaiming_fan_mode;
#define IN 0
#define OUT 1

#define KAIMING_FAN_MODE IN // here is where you can define the kaiming fan mode

typedef unsigned int init_type;
#define KAIMING 0
#define XAVIER 1

int ndims(n_dims N_dims) {
    // returns the number of dimensions of the tensor
    // don't think it's the best idea to make this an enumeration
    if (N_dims & TENSOR_1D) { return 1; }
    if (N_dims & TENSOR_2D) { return 2; }
    if (N_dims & TENSOR_3D) { return 3; } 
    if (N_dims & TENSOR_4D) { return 4; }
    printf("Invalid tensor dimension\n");
    exit(1);
}

Tensor* copy_data(Tensor *input) {
    int size[] = {input->size[0], input->size[1]}; // QUESTION: why the hell does this result in a segfault if I don't include this?
    Tensor* output = tensor(input->size, ndims(input->dim));
    int num_elements = 1;
    for (int i = 0; i < ARRAYSIZE(input->size); i++) {
        num_elements *= input->size[i];
    }
    
    memcpy((*output).data, input->data, num_elements * sizeof(float));

    return output;
}

void free_tensor(Tensor* input) {
    free(input->data);
    free(input->grad_op);
    free(input->size);
    input->data = NULL;
    input->grad_op = NULL;
    input->size = NULL;
    free(input);
}

Tensor* handle_error(const char *message, int errorno) {
    fprintf(stderr, "%s\n", message);
    errno = errorno;
    return NULL;
}

void _print_1D(float* tensor, int size[]) {
    printf("\n\t\t");
    for (int i = 0; i < size[0]; i++) {
        printf("%f ", tensor[i]);
    }
}
void _print_2D(float** tensor, int size[]) {
    printf("\n\t\t");
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            printf("%f ", tensor[i][j]);
        }
        printf("\n\t\t");
    }
}
void _print_3D(float*** tensor, int size[]) {
    // 3d looks fine printed. In groups of 2D matrices
    printf("\n\t\t");
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            for (int k = 0; k < size[2]; k++) {
                printf("%f ", tensor[i][j][k]);
            }
            printf("\n\t\t");
        }
        printf("\n\t\t");
    }
}
void _print_4D(float**** tensor, int size[]) {
    // NOTE: Need to figure out how to best print a 4D tensor
    printf("\n\t\t");
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            for (int k = 0; k < size[2]; k++) {
                for (int l = 0; l < size[3]; l++) {
                    printf("%f ", tensor[i][j][k][l]);
                }
                printf("\n\t\t");
            }
            printf("\n\t\t");
        }
        printf("\n\t\t");
    }
}

void print_tensor(Tensor *tensor) {
    printf("Tensor: ");
    printf("\n\tSize: \n\t\t");
    for (int i = 0; i < ndims(tensor->dim); i++) {
        printf("%d ", tensor->size[i]);
    }
    
    printf("\n\tData: ");
    switch (tensor->dim) {
        case TENSOR_1D:
            _print_1D((float*)tensor->data, tensor->size);
            break;
        case TENSOR_2D:
            _print_2D((float**)tensor->data, tensor->size);
            break;
        case TENSOR_3D:
            _print_3D((float***)tensor->data, tensor->size);
            break;
        case TENSOR_4D:
            _print_4D((float****)tensor->data, tensor->size);
            break;
        default:
            printf("Unsupported tensor dimension\n");
            exit(1);
    }

    // printf("\n\tGrad: \n\t\t%lf\n", tensor->grad);
}


/*
INITIALIZATION FUNCTIONS
*/
void initer(Tensor *t, float (*setter)(Tensor*, float)) {
    switch (t->dim) {
        case TENSOR_1D:
            for (int i = 0; i < t->size[0]; i++) {
                ((float*)t->data)[i] = setter(t, ((float*)t->data)[i]);
            }
            break;
        case TENSOR_2D:
            for (int i = 0; i < t->size[0]; i++) {
                for (int j = 0; j < t->size[1]; j++) {
                    ((float**)t->data)[i][j] = setter(t, ((float**)t->data)[i][j]);
                }
            }
            break;
        case TENSOR_3D:
            for (int i = 0; i < t->size[0]; i++) {
                for (int j = 0; j < t->size[1]; j++) {
                    for (int k = 0; k < t->size[2]; k++) {
                        ((float***)t->data)[i][j][k] = setter(t, ((float***)t->data)[i][j][k]);
                    }
                }
            }
            break;
        case TENSOR_4D:
            for (int i = 0; i < t->size[0]; i++) {
                for (int j = 0; j < t->size[1]; j++) {
                    for (int k = 0; k < t->size[2]; k++) {
                        for (int l = 0; l < t->size[3]; l++) {
                            ((float****)t->data)[i][j][k][l] = setter(t, ((float****)t->data)[i][j][k][l]);
                        }
                    }
                }
            }
            break;
    }
}

float random_uniform() {
    // returns a uniform random number between 0 and 1
    // To get a range of [low, high], do: `low + random_uniform() * (high - low)`
    return ((float) rand () / RAND_MAX);
}

float xavier_sampler(Tensor *t, float _) {
    /* 
    this will perform the xavier initialization 
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    */
    float gain = 1.0;
    int n_in, n_out;
    switch (t->dim) {
        case TENSOR_1D: {
            n_in = (t->size)[0];
            n_out = 1;
            break;
        }
        case TENSOR_2D: {
            n_in = (t->size)[0];   // Number of input units (rows)
            n_out = (t->size)[1];  // Number of output units (columns)
            break;
        }
        // case TENSOR_3D: {
        //     n_in = (t->size)[0];   // Number of input units (rows)
        //     n_out = 1;  // Number of output units (columns)
        //     break;
        // }
        // case TENSOR_4D: {
        //     n_in = (t->size)[0];   // Number of input units (rows)
        //     n_out = 1;  // Number of output units (columns)
        //     break;
        // }
        default: {
            printf("Unsupported tensor dimension\n");
            exit(1);
        }
    }
    float alpha = gain * sqrt(6.0 / (n_in + n_out));

    //NOTE: need to hadndle gaussian sampling as well from `INIT_TYPE`
    return (-alpha) + random_uniform()*(alpha*2);
}
void xavier_uniform_init(Tensor *t) {
    initer(t, xavier_sampler);
}

float one_sampler(Tensor *t, float _) {
    return 1.0;
}
void one_init(Tensor *t) {
    initer(t, one_sampler);
}


void custom_init(Tensor *t, float data[]) {
    initer(t, NULL);
}


float kaiming_sampler(Tensor *t, float _) {
    /* 
    this will perform the kaiming initialization 
    https://arxiv.org/pdf/1502.01852.pdf

    PyTorch Notes:
    • Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
    • Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    • Recommended to use only with 'relu' or 'leaky_relu'
    */
    float gain = 1.0;
    int fan;
    switch (t->dim) {
        case TENSOR_1D: {
            if (KAIMING_FAN_MODE == IN) {
                fan = (t->size)[0];
            } else {
                fan = 1;
            }
            break;
        }
        case TENSOR_2D: {
            if (KAIMING_FAN_MODE == IN) {
                fan = (t->size)[0];
            } else {
                fan = (t->size)[1];
            }
            break;
        }
        // case TENSOR_3D: {
        //     fan = (t->size)[0];
        //     break;
        // }
        // case TENSOR_4D: {
        //     fan = (t->size)[0];
        //     break;
        // }
        default: {
            printf("Unsupported tensor dimension\n");
            exit(1);
        }
    }
    float alpha = gain * sqrt(3.0 / fan);

    //NOTE: need to hadndle gaussian sampling as well from `INIT_TYPE`
    return (-alpha) + random_uniform()*(alpha*2);

}
void kaiming_uniform_init(Tensor *t) {
    initer(t, kaiming_sampler);
}

// TODO: have kaiming_uniform_init have a function in it that takes in parameter and then returns a function

#endif
