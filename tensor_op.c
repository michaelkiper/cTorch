/*
This file will contain the generic matric operations.
Sepcifically, the plan is to make the following operations:
• Add
• Multiply
• Transpose
• Softmax
*/
#include "tensor.c"
#include "tensor_utils.h"
#include <stdio.h>
#include <stdlib.h>


Tensor* multiply2d(Tensor *m1, Tensor *m2) {
    // m1 needs to be a 1xN matrix
    // m2 needs to be a NxM matrix
    // The output will be a 1xM matrix

    int m1_rows = 1;
    int m1_cols = m1->size[0];
    int m2_rows = m2->size[0];
    int m2_cols = m2->size[1];

    printf("m1_rows: %d\n", m1_rows);
    printf("m1_cols: %d\n", m1_cols);
    printf("m2_rows: %d\n", m2_rows);
    printf("m2_cols: %d\n", m2_cols);

    if (m1_cols != m2_rows) {
        printf("The number of columns in the first matrix must be equal to the number of rows in the second matrix\n");
        exit(EXIT_FAILURE);
    }

    // Tensor* output = malloc(sizeof(Tensor));

    Tensor output = tensor((int[]){ m2_cols}, 1);

    for (int i = 0; i < m1_rows; i++) {
        for (int j = 0; j < m2_cols; j++) {
            output.data[i * m2_cols + j] = 0;
            for (int k = 0; k < m1_cols; k++) {
                output.data[i * m2_cols + j] += m1->data[i * m1_cols + k] * m2->data[k * m2_cols + j];
            }
        }
    }

    // TODO: Need to update the gradient info here

    printf("Inter Output\n");
    print_tensor(&output);

    return &output;
}

typedef struct {
    int valid;
    Tensor* (*func)(Tensor*, Tensor*);
} optionMap;

Tensor multiply(Tensor *m1, Tensor *m2) {
    Tensor* output;

    optionMap vr[4][4] = {0};
    vr[1][2].valid = 1;
    vr[1][2].func = multiply2d;

    // 1D Case
    if (vr[m1->dim][m2->dim].valid && (vr[m1->dim][m2->dim].func != NULL)) {
        output = vr[m1->dim][m2->dim].func(m1, m2);
    } 
    else {
        printf("%d | %d\n", m1->dim, m2->dim);
        printf("func: multiply: Unsupported tensor dimension\n");
        exit(EXIT_FAILURE);
    }

    // switch (m1->dim, m2->dim) {
    //     case (TENSOR_1D, TENSOR_2D): { 
    //         output = multiply2d(m1, m2);
    //         break;
    //     }
    //     default: {
    //         printf("%d | %d\n", m1->dim, m2->dim);
    //         printf("func: multiply: Unsupported tensor dimension\n");
    //         exit(EXIT_FAILURE);
    //     }
    // }

    printf("Inter Output 2\n");
    print_tensor(output);

    return *output;
}


Tensor add(Tensor *a1,Tensor *a2) {}