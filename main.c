#include <stdio.h>
#include <stdlib.h>
#include "tensor.c"
#include "tensor_utils.h"
#include "utils.h"
#include "NN.c"

#define ARRAYSIZE_2(a) (sizeof(a) / sizeof(a[0]))

int main (int argc, char *argv[]) {
    // xavier_uniform_init(&t1);
    // print_tensor(&t1);

    // int size2[] = {3, 2};
    // Tensor *t2 = tensor(size2);
    // one_init(t2);
    // print_tensor(t2);

    // int size3[] = {3, 2};
    // Tensor *t3 = tensor(size3);
    // kaiming_uniform_init(t3);
    // print_tensor(t3);

    Linear* l = linear_1d((int[]){3, 4}, (int[]){3}, KAIMING);
    printf("\n\nWeight\n");
    print_tensor((*l).weight);

    printf("\n\nBias\n");
    print_tensor((*l).bias);
    Tensor* o = ReLU((*l).weight);
    printf("\n\n");
    print_tensor(o);

    (*l).forward(l, o);
}


void test_multiply_2d(Tensor *m1, Tensor *m2) {
    Tensor* output = multiply(m1, m2);
    print_tensor(output);
}
