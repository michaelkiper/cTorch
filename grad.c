/*
PyTorch's Autograd Rules For Non-Differentiable Functions:
https://pytorch.org/docs/stable/notes/autograd.html

If the function is differentiable and thus a gradient exists at the current point, use it.
If the function is convex (at least locally), use the sub-gradient of minimum norm (it is the steepest descent direction).
If the function is concave (at least locally), use the super-gradient of minimum norm 
    (consider -f(x) and apply the previous point).
If the function is defined, define the gradient at the current point by continuity 
    (note that inf is possible here, for example for sqrt(0)). If multiple values are possible, pick one arbitrarily.
If the function is not defined (sqrt(-1), log(-1) or most functions when the input is NaN, for example) 
    then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). 
    Most functions will use NaN as the gradient, but for performance reasons, some functions will use other values (log(-1), for example).
If the function is not a deterministic mapping (i.e. it is not a mathematical function), 
    it will be marked as non-differentiable. 
    This will make it error out in the backward if used on tensors that require grad outside of a no_grad environment.
*/

#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

// typedef struct {
//     union { // just make it a little easier on us down the road
//         float* oneD;
//         float** twoD;
//         float*** threeD;
//         float**** fourD;
//     } data;
//     n_dims dim;
// } Context;

// Tensor* _init_root_tensor () {
//     Tensor* t0 = (Tensor*)malloc(sizeof(Tensor));
//     if (t0 == NULL) {
//         printf("Memory allocation failed for tensor.\n");
//         exit(EXIT_FAILURE);
//     }
//     t0->data = (float*)malloc(sizeof(float));
//     if (t0->data == NULL) {
//         printf("Memory allocation failed for tensor: data.\n");
//         free (t0);
//         exit(EXIT_FAILURE);
//     }
//     t0->data[0] = 1.0f;

//     t0->size = (int*)malloc(sizeof(int));
//     if (t0->data == NULL) {
//         printf("Memory allocation failed for tensor: size.\n");
//         free (t0->data);
//         free (t0);
//         exit(EXIT_FAILURE);
//     }
//     t0->size[0] = 1;

//     t0->dim = TENSOR_1D;
//     t0->gra_op = NULL;
//     return t0;
// }

Tensor* backwards(Tensor* t, Tensor* wrt) {
    // Tensor* set[2];
    // set[0] = t;
    // set[1] = _init_root_tensor();
    
    return t->gra_op->method(t->gra_op->ctx_tensors);
}

Tensor* multiply_backwards(Tensor* ctx[2]) {
    // if ((*ctx).gra_op != NULL && (*(*ctx).gra_op).ctx != NULL) {
    //     return (*(*ctx).gra_op).method((*(*ctx).gra_op).ctx);
    // }

    Tensor* A = ctx[0];
    Tensor* B = ctx[1];
}

Tensor* add_backwards(Tensor* ctx) {
    // if ((*ctx).gra_op != NULL && (*(*ctx).gra_op).ctx != NULL) {
    //     Tensor* t = _init_root_tensor();
    //     t->data[0] = 0.0f;
    //     return t;
    // }
}

void transpose_backwards() {

}