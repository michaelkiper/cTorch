#include <stdio.h>
#include <stdlib.h>
#include "tensor.c"
#include "tensor_utils.h"
#include "utils.h"
#include "NN.c"
#include <assert.h>
#include <stdbool.h>


void test_linear_forward_error() {
    assert(errno!=EINVAL);
    Tensor* t = tensor((int[]){3, 2, 3}, 3);

    Linear* l = linear((int[]){3, 4}, (int[]){3}, KAIMING);

    Tensor* output = linear_forward(l, t);
    assert(errno==EINVAL);
}


void test_linear_forward(bool debug) {
    Tensor* t = tensor((int[]){4}, 1);
    kaiming_uniform_init(t);
    if (debug) {
        printf("Input:\n");
        print_tensor(t);
    }

    Linear* l = linear((int[]){4, 3}, (int[]){3}, KAIMING);
    if (debug) {
        printf("\n==================================================\n");
        printf("Weight:\n");
        print_tensor((*l).weight);

        printf("\nBias\n");
        print_tensor((*l).bias);
    }

    Tensor* output = linear_forward(l, t);
    if (debug) {
        printf("\n==================================================\n");
        printf("\nOutput:\n");
        print_tensor(output);
    }
}

int main (int argc, char *argv[]) {
    // test_linear_forward_error();
    test_linear_forward(true);
}