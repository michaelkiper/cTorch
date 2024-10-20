#include "NN.h"
#include <math.h>
#include <errno.h>

void print_linear(Linear* l) {
    printf("********* Linear Neuron *********\n");
    printf("*** Weights: ***\n");
    print_tensor(l->weight);
    printf("*** Bias: ***\n");
    print_tensor(l->bias);
}


// Tensor* linear_forward(Linear* layer, Tensor *input) {
//     // if (input->size->dims != TENSOR_1D) {
//     //     return handle_error("The input tensor must be 1D", EINVAL);
//     // }

//     Tensor* output = matmul(input, layer->weight);
//     output = add(layer->bias, output);
//     return output;
// }

// For now, the Neaural Network will only support the 1D case
Linear* linear(Size* s) {
    if (s->dims != TENSOR_2D && s->dims != TENSOR_1D) {
        printf("Currently only handles 1D and 2D Linear Neurons\n");
        return NULL;
    }
    Size* bias_size = size(
        (int[]){1, (s->dims == TENSOR_2D ? s->size[1] : 1)}
        // (int[]){s->dims == TENSOR_2D ? s->size[0]: 1, (s->dims == TENSOR_2D ? s->size[1] : s->size[0])}
    );
    Linear *linear = (Linear*)malloc(sizeof(Linear));
    linear->weight = tensor(s);
    linear->bias = tensor(bias_size);

    one_init(linear->bias); // one init the bias

    switch (INIT_METHOD) {
        case (KAIMING): {
            kaiming_uniform_init(linear->weight);
        }
        case (XAVIER): {
            xavier_uniform_init(linear->weight);
        } 
    }
    return linear;
}

static void non_linearity(Tensor *t, void func(float*)){
    int* indices = (int*)malloc(sizeof(t->size->size));
    InitContext* i = (InitContext*)malloc(sizeof(InitContext));
    while (!i->done) {
        float* v = initer(t, i);
        func(v);
        incrementer(i, t->size);
    }
    free(indices);
    free(i);
}

static void _relu_func(float* val) { if ((*val) < 0) { *val = 0.0; } }
void ReLU(Tensor *input) { non_linearity(input, _relu_func); }

static void _gelu_func(float* val) {
    // performs the GELU approximate function
    *val = 0.5 * (*val) * (1 + tanh(sqrt(2 / M_PI) * ((*val) + 0.044715 * pow((*val), 3))));
}
void GELU(Tensor *input) { non_linearity(input, _gelu_func); }

static void _tanh(float* val) { *val = tanh(*val); }
void Tanh(Tensor *input) { non_linearity(input, _tanh); }

/*
TODO: other activation functions:
• Sigmoid
• Parametric ReLU
• Swish
• SwiGLU
• Softplus
*/
