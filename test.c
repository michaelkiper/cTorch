#include "tensor.c"
#include "tensor_op.c"
#include "tensor_utils.h"

void test_multiply_2d(Tensor *m1, Tensor *m2) {
    Tensor output = multiply(m1, m2);
    print_tensor(&output);
}