#ifndef TENSOR_H_
#define TENSOR_H_
#include <stdio.h>
#include <stdlib.h>
#include "grad.c"
#include "utils.h"

typedef unsigned int n_dims;
#define TENSOR_1D (1 << 0)
#define TENSOR_2D (1 << 1)
#define TENSOR_3D (1 << 2)
#define TENSOR_4D (1 << 3)

static int DIM_MAP[] = {
    [1] = TENSOR_1D,
    [2] = TENSOR_2D,
    [3] = TENSOR_3D,
    [4] = TENSOR_4D,
};


void* init_tensor_data(int size[], n_dims N_dims) {
    if (N_dims & TENSOR_1D) {
        float* tensor = (float*)malloc(size[0] * sizeof(float));
        return tensor;
    } else if (N_dims & TENSOR_2D) {
        float** tensor = (float**)malloc(size[0] * sizeof(float*));
        for (int i = 0; i < size[0]; i++) {
            tensor[i] = (float*)malloc(size[1] * sizeof(float));
        }
        return tensor;
    } else if (N_dims & TENSOR_3D) {
        float*** tensor = (float***)malloc(size[0] * sizeof(float**));
        for (int i = 0; i < size[0]; i++) {
            tensor[i] = (float**)malloc(size[1] * sizeof(float*));
            for (int j = 0; j < size[1]; j++) {
                tensor[i][j] = (float*)malloc(size[2] * sizeof(float));
            }
        }
        return tensor;
    } else if (N_dims & TENSOR_4D) {
        float**** tensor = (float****)malloc(size[0] * sizeof(float***));
        for (int i = 0; i < size[0]; i++) {
            tensor[i] = (float***)malloc(size[1] * sizeof(float**));
            for (int j = 0; j < size[1]; j++) {
                tensor[i][j] = (float**)malloc(size[2] * sizeof(float*));
                for (int k = 0; k < size[2]; k++) {
                    tensor[i][j][k] = (float*)malloc(size[3] * sizeof(float));
                }
            }
        }
        return tensor;
    }
    else {
        printf("Invalid tensor dimension\n");
        exit(1);
    }
}

typedef struct {
    float* data;
    int* size;
    n_dims dim;
    double grad;
    GradMethod* gra_op;
} Tensor;

Tensor tensor(int size[], int dims) {
    n_dims N_dims = DIM_MAP[dims];
    Tensor *tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->size = size;
    tensor->dim = N_dims;
    tensor->data = init_tensor_data(size, N_dims);
    tensor->grad = 0.0;

    return *tensor;
}
#endif
