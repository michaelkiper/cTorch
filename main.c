#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "tensor_utils.h"
// #include "utils.h"
// #include "NN.c"

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


    Size *s = size((int[]){3, 4});
    Tensor *t = tensor(s);
    print_tensor(t);
    // for (int i = 0; i < s->size[0]; i++) {
    //     for (int j = 0; j < s->size[1]; j++) {
    //         ((float**) t->data)[i][j] = 1;
    //     }
    // }
    // print_tensor(t);
    one_init(t);
    print_tensor(t);


    // Linear* l = linear_1d((int[]){3, 4}, (int[]){3}, KAIMING);
    // printf("\n\nWeight\n");
    // print_tensor((*l).weight);

    // printf("\n\nBias\n");
    // print_tensor((*l).bias);
    // Tensor* o = ReLU((*l).weight);
    // printf("\n\n");
    // print_tensor(o);

    // (*l).forward(l, o);

    // free(l);
    // free(o);
}


// void test_matmul_2d(Tensor *m1, Tensor *m2) {
//     Tensor* output = matmul(m1, m2);
//     print_tensor(output);
//     free(output);
// }
