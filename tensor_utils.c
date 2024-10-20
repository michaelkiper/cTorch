#include "tensor_utils.h"

int get_size(int size[], n_dims N_dims) {
    // returns the number of elements in the tensor
    int num_elements = 1;
    for (int i = 0; i < ndims(N_dims); i++) {
        num_elements *= size[i];
    }
    return num_elements;
}


// Tensor* copy_data(Tensor *input) {
//     int size[] = {input->size[0], input->size[1]}; // QUESTION: why the hell does this result in a segfault if I don't include this?
//     Tensor* output = tensor(input->size, ndims(input->dim));
//     int num_elements = 1;
//     for (int i = 0; i < ARRAYSIZE(input->size); i++) {
//         num_elements *= input->size[i];
//     }
    
//     memcpy((*output).data, input->data, num_elements * sizeof(float));

//     return output;
// }

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

// TODO: I could refactor all of this to make it simpler
static void _print_1D(float* tensor, int size[]) {
    printf("\n\t\t");
    for (int i = 0; i < size[0]; i++) {
        printf("%f ", tensor[i]);
    }
}
static void _print_2D(float** tensor, int size[]) {
    printf("\n\t\t");
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            printf("%f ", tensor[i][j]);
        }
        printf("\n\t\t");
    }
}
static void _print_3D(float*** tensor, int size[]) {
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
static void _print_4D(float**** tensor, int size[]) {
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
    for (int i = 0; i < ndims(tensor->size->dims); i++) {
        printf("%d, ", tensor->size->size[i]);
    }
    
    printf("\n\tData: ");
    switch (tensor->size->dims) {
        case TENSOR_1D:
            _print_1D((float*)tensor->data, tensor->size->size);
            break;
        case TENSOR_2D:
            _print_2D((float**)tensor->data, tensor->size->size);
            break;
        case TENSOR_3D:
            _print_3D((float***)tensor->data, tensor->size->size);
            break;
        case TENSOR_4D:
            _print_4D((float****)tensor->data, tensor->size->size);
            break;
        default:
            printf("Unsupported tensor dimension\n");
            exit(1);
    }
    printf("\n");

    // printf("\n\tGrad: \n\t\t%lf\n", tensor->grad);
}


/*
INITIALIZATION FUNCTIONS
*/

// float custom_sampler(Tensor *t, float _, int index[], SamplerContext *ctx) {
//     int linear_index = 0;
//     int multiplier = 1;

//     // Calculate the linear index in the flat data array based on the Tensor's dimensions
//     for (int i = t->dim - 1; i >= 0; i--) {
//         linear_index += index[i] * multiplier;
//         multiplier *= t->size[i];
//     }

//     return ctx->data[linear_index];
// }

// void set_tensor(Tensor *t, float data[]) {
//     SamplerContext ctx = {data};
//     initer(t, custom_sampler, &ctx);
// }


// void set_tensor_2(Tensor *t, float data[]) {
//     // free(t->data);
//     int n = get_size(t->size, t->dim);
//     print_tensor(t);
//     // t->data = (float*)malloc(n * sizeof(float));
//     for (int i = 0; i < n; i++) {
//         t->data[i] = data[i];
//         printf("%f\n", data[i]);
//         printf("%f\n", (t->data)[i]);
//     }
//     print_tensor(t);
//     // printf("%d\n", n);
// }

// *****************************************************************************

void incrementer(InitContext *i, Size *s) {
    for (int idx = ndims(s->dims)-1; idx >= 0; idx--) {
        i->indices[idx]++;
        if (i->indices[idx] >= s->size[idx]) {
            i->indices[idx] = 0;
            if (idx == 0) { i->done = true; }
        } else {
            break;
        }
    }
}

float* initer(Tensor *t, InitContext *ctx) {
    switch (t->size->dims) {
        case TENSOR_1D:
            return &t->data[ctx->indices[0]];
        case TENSOR_2D:
            return &((float**) t->data)[ctx->indices[0]][ctx->indices[1]];
        case TENSOR_3D:
            return &((float***) t->data)[ctx->indices[0]][ctx->indices[1]][ctx->indices[2]];
        case TENSOR_4D:
            return &((float****) t->data)[ctx->indices[0]][ctx->indices[1]][ctx->indices[2]][ctx->indices[3]];
        default:
            printf("Unsupported tensor dimension\n");
            exit(1);
    }
}

void agnostic_setter_helper(Tensor *t, float func(Tensor*)) {
    /*
    This function is to set tensor values with an index agnostic function.
    Useful for things such as Kaiming-He/Xavier/One inits
    */
    int* indices = (int*)malloc(sizeof(t->size->size));
    InitContext* i = (InitContext*)malloc(sizeof(InitContext));
    while (!i->done) {
        float* v = initer(t, i);
        *v = func(t);
        incrementer(i, t->size);
    }
    free(indices);
    free(i);
}

static float _one_setter(Tensor *t) { return 1.0; }

void one_init(Tensor *t) {
    agnostic_setter_helper(t, _one_setter);
}

static float random_uniform() {
    // returns a uniform random number between 0 and 1
    // To get a range of [low, high], do: `low + random_uniform() * (high - low)`
    return ((float) rand () / RAND_MAX);
}

static float _xavier_uniform_setter(Tensor *t) {
    /* 
    this will perform the xavier initialization 
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    NOTE: need to hadndle gaussian sampling as well from `INIT_TYPE`
    */
    
    float gain = 1.0;
    int n_in, n_out;
    switch (t->size->dims) {
        case TENSOR_1D: {
            n_in = (t->size->size)[0];
            n_out = 1;
            break;
        }
        case TENSOR_2D: {
            n_in = (t->size->size)[0];   // Number of input units (rows)
            n_out = (t->size->size)[1];  // Number of output units (columns)
            break;
        }
        // case TENSOR_3D: {
        // }
        // case TENSOR_4D: {
        // }
        default: {
            printf("Unsupported tensor dimension\n");
            exit(1);
        }
    }

    float alpha = gain * sqrt(6.0 / (n_in + n_out));
    return (-alpha) + random_uniform()*(alpha*2);
}

void xavier_uniform_init(Tensor *t) {
    agnostic_setter_helper(t, _xavier_uniform_setter);
}

static float _kaiming_uniform_setter(Tensor *t) {
    /* 
    this will perform the kaiming initialization 
    https://arxiv.org/pdf/1502.01852.pdf

    PyTorch Notes:
    • Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
    • Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    • Recommended to use only with 'relu' or 'leaky_relu'

    NOTE: need to hadndle gaussian sampling as well from `INIT_TYPE`
    */
    float gain = 1.0;
    int fan;
    switch (t->size->dims) {
        case TENSOR_1D: {
            if (KAIMING_FAN_MODE == IN) {
                fan = (t->size->size)[0];
            } else {
                fan = 1;
            }
            break;
        }
        case TENSOR_2D: {
            if (KAIMING_FAN_MODE == IN) {
                fan = (t->size->size)[0];
            } else {
                fan = (t->size->size)[1];
            }
            break;
        }
        // case TENSOR_3D: {
        // }
        // case TENSOR_4D: {
        // }
        default: {
            printf("Unsupported tensor dimension\n");
            exit(1);
        }
    }

    float alpha = gain * sqrt(3.0 / fan);
    return (-alpha) + random_uniform()*(alpha*2);
}

void kaiming_uniform_init(Tensor *t) {
    agnostic_setter_helper(t, _kaiming_uniform_setter);
}
