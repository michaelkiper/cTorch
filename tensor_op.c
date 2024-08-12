/*
This file will contain the generic matrix operations.
Sepcifically, the plan is to make the following operations:
• Add
• Multiply
• Transpose
• Softmax
*/
#include "tensor.h"
#include "tensor_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>



Tensor* matmul2d(Tensor *m1, Tensor *m2) {
    // m1 needs to be a 1xN matrix
    // m2 needs to be a NxM matrix
    // The output will be a 1xM matrix

    int m1_rows = 1;
    int m1_cols = m1->size[0];
    int m2_rows = m2->size[0];
    int m2_cols = m2->size[1];

    if (m1_cols != m2_rows) {
        printf("The number of columns in the first matrix must be equal to the number of rows in the second matrix\n");
        exit(EXIT_FAILURE);
    }

    Tensor* output = tensor((int[]){ m2_cols}, 1);

    for (int j = 0; j < m2_cols; j++) { // M2 Cols (also M1 Cols)
        ((*output).data)[j] = 0;
        for (int k = 0; k < m2_rows; k++) { // M2 Rows
            int m2_index = k * m2_cols + j;
            ((*output).data)[j] += m1->data[j] * ((float**) m2->data)[k][j]; 
        }
    }

    // TODO: Need to update the gradient info here

    return output;
}



Tensor* add1d(Tensor *a1,Tensor *a2) {
    assert(a1->dim==TENSOR_1D && a2->dim==TENSOR_1D);

    if (a1->size[0] != a2->size[0]) {
        printf("func: add1d: Got mismatching sizes: (%d,) and (%d,)\n", a1->size[0], a2->size[0]);
        exit(EXIT_FAILURE);
    }

    Tensor* output = tensor((int[]){a1->size[0]}, 1);

    for (int i = 0; i < a1->size[0]; i++) {
        ((float*)output->data)[i] = ((float*)a1->data)[i] + ((float*)a2->data)[i];
    }

    // TODO: Need to update the gradient info here

    return output;
}



typedef struct {
    int valid;
    Tensor* (*matmul)(Tensor*, Tensor*);
    Tensor* (*add)(Tensor*, Tensor*);
} operationMap;

static operationMap OP_MAPPINGS[4][4] = {
    [1][1] = {1, NULL, add1d},
    [1][2] = {1, matmul2d, NULL},
};

Tensor* matmul(Tensor *m1, Tensor *m2) {
    Tensor* output;

    if (OP_MAPPINGS[m1->dim][m2->dim].valid && (OP_MAPPINGS[m1->dim][m2->dim].matmul != NULL)) {
        output = OP_MAPPINGS[m1->dim][m2->dim].matmul(m1, m2);
    } 
    else {
        printf("%d | %d\n", m1->dim, m2->dim);
        printf("func: matmul: Unsupported tensor dimension\n");
        exit(EXIT_FAILURE);
    }

    // switch (m1->dim, m2->dim) {
    //     case (TENSOR_1D, TENSOR_2D): { 
    //         output = matmul2d(m1, m2);
    //         break;
    //     }
    //     default: {
    //         printf("%d | %d\n", m1->dim, m2->dim);
    //         printf("func: matmul: Unsupported tensor dimension\n");
    //         exit(EXIT_FAILURE);
    //     }
    // }

    return output;
}


Tensor* add(Tensor *a1,Tensor *a2) {
    Tensor* output;

    if (OP_MAPPINGS[a1->dim][a2->dim].valid && (OP_MAPPINGS[a1->dim][a2->dim].add != NULL)) {
        output = OP_MAPPINGS[a1->dim][a2->dim].add(a1, a2);
    } 
    else {
        printf("%d | %d\n", a1->dim, a2->dim);
        printf("func: add: Unsupported tensor dimension\n");
        exit(EXIT_FAILURE);
    }

    return output;
}

