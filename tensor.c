#include "tensor.h"
#include "utils.h"

static int DIM_MAP[] = {
    [1] = TENSOR_1D,
    [2] = TENSOR_2D,
    [3] = TENSOR_3D,
    [4] = TENSOR_4D,
};

// God this is  the uglist code I've ever seen in my entire life - makes me want to vomit
void init_tensor_data(Tensor *tensor) {
    // fuck it - imma not make this 1 contiguous chunck
    // if (tensor == NULL || tensor->size == NULL || tensor->size->size == NULL) {
    //     printf("Invalid tensor or size.\n");
    //     exit(1);
    // }

    // int num_elements = 1;
    // for (int i = 0; i < ndims(tensor->size->dims); i++) {
    //     num_elements *= tensor->size->size[i];
    // }

    // tensor->data = (float*)malloc(num_elements * sizeof(float));
    // if (tensor->data == NULL) {
    //     printf("Memory allocation failed.\n");
    //     exit(1);
    // }

    if (tensor->size->dims & TENSOR_1D) {
        tensor->data = (float*)malloc(tensor->size->size[0] * sizeof(float));
    
    } else if (tensor->size->dims & TENSOR_2D) {
        float** tensor_data = (float**)malloc(tensor->size->size[0] * sizeof(float*));
        for (int i = 0; i < tensor->size->size[0]; i++) {
            tensor_data[i] = (float*)malloc(tensor->size->size[1] * sizeof(float));
        }
        tensor->data = (float*) tensor_data;
    
    } else if (tensor->size->dims & TENSOR_3D) {
        float*** tensor_data = (float***)malloc(tensor->size->size[0] * sizeof(float**));
        for (int i = 0; i < tensor->size->size[0]; i++) {
            tensor_data[i] = (float**)malloc(tensor->size->size[1] * sizeof(float*));
            for (int j = 0; j < tensor->size->size[1]; j++) {
                tensor_data[i][j] = (float*)malloc(tensor->size->size[2] * sizeof(float));
            }
        }
        tensor->data = (float*) tensor_data;
    
    } else if (tensor->size->dims & TENSOR_4D) {
        float**** tensor_data = (float****)malloc(tensor->size->size[0] * sizeof(float***));
        for (int i = 0; i < tensor->size->size[0]; i++) {
            tensor_data[i] = (float***)malloc(tensor->size->size[1] * sizeof(float**));
            for (int j = 0; j < tensor->size->size[1]; j++) {
                tensor_data[i][j] = (float**)malloc(tensor->size->size[2] * sizeof(float*));
                for (int k = 0; k < tensor->size->size[2]; k++) {
                    tensor_data[i][j][k] = (float*)malloc(tensor->size->size[3] * sizeof(float));
                }
            }
        }
        tensor->data = (float*) tensor_data;
    }
    else {
        printf("Invalid tensor dimension\n");
        exit(1);
    }
}


Tensor* tensor(Size *size) {
    
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        printf("Memory allocation failed for tensor.\n");
        exit(EXIT_FAILURE);
    }

    tensor->size = size;
    tensor->grad = NULL;
    tensor->grad_op = NULL;
    // tensor->grad_op = &(GradMethod){
    //     NULL,
    //     &(Tensor){
    //         (float) 1.0,
    //         {1},
    //         TENSOR_1D,
    //         NULL
    //     }
    // }; 
    init_tensor_data(tensor);
    return tensor;
}

Size* size(int *size, n_dims d) {
    Size* tensor_size = (Size*)malloc(sizeof(Size));
    if (tensor_size == NULL) {
        printf("Memory allocation failed for tensor size.\n");
        exit(EXIT_FAILURE);
    }
    tensor_size->dims = DIM_MAP[d];
    tensor_size->size = size;
    return tensor_size;
}

int ndims(n_dims N_dims) {
    // returns the number of dimensions of the tensor
    // don't think it's the best idea to make this an enumeration
    if (N_dims & TENSOR_1D) { return 1; }
    if (N_dims & TENSOR_2D) { return 2; }
    if (N_dims & TENSOR_3D) { return 3; } 
    if (N_dims & TENSOR_4D) { return 4; }
    printf("Invalid tensor dimension\n");
    exit(1);
}
