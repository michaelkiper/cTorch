#ifndef NN_H
#define NN_H
#include "tensor.h"
// #include "tensor_op.c"
#include "tensor_utils.h"

typedef struct Linear Linear;
struct Linear {
    Tensor* weight;
    Tensor* bias;
    // Tensor* (*forward)(Linear*, Tensor*);
};


Linear* linear(Size* s); 

void ReLU(Tensor *input);
void GELU(Tensor *input);
void Tanh(Tensor *input);

void print_linear(Linear* l);

/*
TODO: other activation functions:
• Sigmoid
• Parametric ReLU
• Swish
• SwiGLU
• Softplus
*/
#endif