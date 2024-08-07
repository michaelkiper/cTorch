/*
This file will contain the generic matric operations.
Sepcifically, the plan is to make the following operations:
• Add
• Multiply
• Transpose
• Softmax
*/
#include "tensor.c"


Tensor multiply2d(Tensor *m1, Tensor *m2) {
    int m1_rows = m1->size[0];
    int m1_cols = m1->size[1];
    int m2_rows = m2->size[0];
    int m2_cols = m2->size[1];

    if (m1_cols != m2_rows) {
        printf("The number of columns in the first matrix must be equal to the number of rows in the second matrix\n");
        exit(1);
    }

    Tensor output = tensor((int[]){m1_rows, m2_cols}, 2);

    for (int i = 0; i < m1_rows; i++) {
        for (int j = 0; j < m2_cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < m1_cols; k++) {
                sum += m1->data[i * m1_cols + k] * m2->data[k * m2_cols + j];
            }
            output.data[i * m2_cols + j] = sum;
        }
    }

    // TODO: Need to update the gradient info here

    return output;
}

Tensor multiply(Tensor *m1, Tensor *m2) {
    switch (m1->dim) {
        case TENSOR_2D: {
            return multiply2d(m1, m2);
        }
        default: {
            printf("Unsupported tensor dimension\n");
            exit(1);
        }
    }
}


Tensor add(Tensor *a1,Tensor *a2) {}