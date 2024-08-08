#include "tensor.c"
#include "tensor_op.c"
#include "tensor_utils.h"
#include <math.h>
#include <errno.h>

typedef struct Linear Linear;
struct Linear {
    Tensor* weight;
    Tensor* bias;
    Tensor* (*forward)(Linear*, Tensor*);
};

Tensor* linear_forward_1d(Linear* layer, Tensor *input) {
    if (input->dim != TENSOR_1D) {
        return handle_error("The input tensor must be 1D", EINVAL);
    }

    Tensor* output = multiply(input, layer->weight);
    output = add(layer->bias, output);
    return output;
}

// For now, the Neaural Network will only support the 1D case
Linear* linear_1d(int proj_dim[2], int bias_dim[0], init_type _init_type) {
    // `proj_dim` should be of shape: {input_dim, output_dim}
    if (proj_dim[1] != bias_dim[0]) {
        printf("The input and output dimensions of the weight and bias must be the same\n");
        exit(1);
    }
    Linear *linear = (Linear*)malloc(sizeof(Linear));
    linear->weight = tensor(proj_dim, 2);
    linear->bias = tensor(bias_dim, 1);

    one_init(linear->bias); // one init the bias
    switch (_init_type) {
        case (KAIMING): {
            kaiming_uniform_init(linear->weight);
        }
        case (XAVIER): {
            xavier_uniform_init(linear->weight);
        }
    }
    linear->forward = linear_forward_1d;

    return linear;
}


Tensor* non_linearity(Tensor *input, float (*nl)(Tensor*, float)){
    Tensor* output = copy_data(input);
    initer(output, nl);
    return output;
}

float _relu_func(Tensor *t, float val) {
    if (val > 0) { return val; }
    return 0;
}
Tensor* ReLU(Tensor *input) {
    return non_linearity(input, _relu_func);
}

float _gelu_func(Tensor *t, float val) {
    // performs the GELU approximate function
    return 0.5 * val * (1 + tanh(sqrt(2 / M_PI) * (val + 0.044715 * pow(val, 3))));
}
Tensor* GELU(Tensor *input) {
    return non_linearity(input, _gelu_func);
}

float _tanh(Tensor *t, float val) {
    return tanh(val);
}
Tensor* Tanh(Tensor *input) {
    return non_linearity(input, _tanh);
}

/*
TODO: other activation functions:
• Sigmoid
• Parametric ReLU
• Swish
• SwiGLU
• Softplus
*/
