#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#include <stdlib.h>

#define TENSOR_1D (1 << 0)
#define TENSOR_2D (1 << 1)
#define TENSOR_3D (1 << 2)
#define TENSOR_4D (1 << 3)

typedef unsigned int n_dims;
typedef struct Tensor Tensor;

typedef struct {
    void* (*method)(Tensor*[], Tensor*); 
    Tensor** ctx_tensors; // this hold the parent Tensors that create this Tensor
    int num_ctx_tensors;
} GradMethod;


typedef struct {
    n_dims dims;
    int* size;
} Size;

struct Tensor {
    float* data;
    Size* size;
    GradMethod* grad_op; 
    Tensor* grad;
};

void init_tensor_data(Tensor *tensor);

Tensor* tensor(Size *size);

Size* size(int size[]);

int ndims(n_dims N_dims);

#endif