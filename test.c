#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "grad.c"
#include "tensor_utils.h"
#include "utils.h"
#include "NN.c"
#include <assert.h>
#include <stdbool.h>


void test_linear_forward_error() {
    assert(errno!=EINVAL);
    Tensor* t = tensor((int[]){3, 2, 3}, 3);

    Linear* l = linear_1d((int[]){3, 4}, (int[]){3}, KAIMING);

    Tensor* output = (*l).forward(l, t);
    assert(errno==EINVAL);

    free(t);
    free(l);
    free(output);
}


void test_linear_forward(bool debug) {
    Tensor* t = tensor((int[]){4}, 1);
    kaiming_uniform_init(t);
    if (debug) {
        printf("Input:\n");
        print_tensor(t);
    }

    Linear* l = linear_1d((int[]){4, 3}, (int[]){3}, KAIMING);
    if (debug) {
        printf("\n==================================================\n");
        printf("Weight:\n");
        print_tensor((*l).weight);

        printf("\nBias\n");
        print_tensor((*l).bias);
    }

    Tensor* output = (*l).forward(l, t);
    if (debug) {
        printf("\n==================================================\n");
        printf("\nOutput:\n");
        print_tensor(output);
    }

    free(t);
    free(l);
    free(output);
}

void test_matmul_backwards() {
    Tensor* t1 = tensor((int[]){2, 3}, 2);
    Tensor* t2 = tensor((int[]){3, 4}, 2);
    custom_init(t1, (float[]){
        1.0f, 2.0f, 3.0f, 4.0f, 
        1.0f, 2.0f, 3.0f, 4.0f, 
        1.0f, 2.0f, 3.0f, 4.0f
    });
    custom_init(t2, (float[]){
        2.0f, 4.0f,  6.0f, 
        0.5f, 1.0f, 1.5f
    });

    // Tensor* output = matmul(t1, t2);
    // printf("MATMUL:\n");
    // print_tensor(output);

    printf("Input Data:\n");
    print_tensor(t1);
    printf("\n==================================================\n");
    print_tensor(t2);
    printf("\n==================================================\n");

    Tensor* grad = tensor((int[]){2, 4}, 2);
    one_init(grad);
    printf("MATMUL BACKWARDS:\n");
    matmul_backwards((Tensor* []){t1, t2}, grad);

    free(t1);
    free(t2);
    // free(output);
    free(grad);
}

int main (int argc, char *argv[]) {
    // test_linear_forward_error();
    // test_linear_forward(true);
    test_matmul_backwards();
}