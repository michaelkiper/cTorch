#ifndef TENSOR_H
#define TENSOR_H

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

typedef struct Tensor Tensor;

typedef struct {
    Tensor* (*method)(Tensor*[]); 
    Tensor** ctx_tensors; // this hold the parent Tensors that create this Tensor
    int num_ctx_tensors;
} GradMethod;

struct Tensor {
    float* data;
    int* size;
    n_dims dim;
    GradMethod* gra_op; 
    Tensor* grad;
};


#endif