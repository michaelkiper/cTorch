#ifndef TENSOR_C
#define TENSOR_C
#include <stdio.h>
#include <stdlib.h>
#include "grad.c"
#include "utils.h"
#include "tensor.h"


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


Tensor* tensor(int size[], int dims) {
    n_dims N_dims = DIM_MAP[dims];
    
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        printf("Memory allocation failed for tensor.\n");
        exit(EXIT_FAILURE);
    }

    tensor->size = (int*)malloc(dims * sizeof(int));
    if (tensor->size == NULL) {
        printf("Memory allocation failed for tensor size.\n");
        free(tensor);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dims; i++) {
        tensor->size[i] = size[i];
    }

    tensor->dim = N_dims;
    tensor->data = init_tensor_data(size, N_dims);
    tensor->grad = NULL;
    tensor->gra_op = NULL;
    // tensor->gra_op = &(GradMethod){
    //     NULL,
    //     &(Tensor){
    //         (float) 1.0,
    //         {1},
    //         TENSOR_1D,
    //         NULL
    //     }
    // }; 
    return tensor;
}

#endif
