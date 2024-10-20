#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H
#include <stdio.h>
#include "tensor.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>

typedef unsigned int init;
#define UNIFORM 0
#define GAUSSIAN 1

#define INIT_TYPE UNIFORM // here is where you can change the initialization type

typedef unsigned int kaiming_fan_mode;
#define IN 0
#define OUT 1

#define KAIMING_FAN_MODE IN // here is where you can define the kaiming fan mode

typedef unsigned int init_type;
#define KAIMING 0
#define XAVIER 1

#define INIT_METHOD KAIMING


// returns the number of elements in the tensor
int get_size(int size[], n_dims N_dims);


void free_tensor(Tensor* input);

Tensor* handle_error(const char *message, int errorno);

void print_tensor(Tensor *tensor);


/*
INITIALIZATION FUNCTIONS
*/
typedef struct {
    bool done;
    int indices[];
} InitContext;

void incrementer(InitContext *i, Size *s);

float* initer(Tensor *t, InitContext *ctx);

void one_init(Tensor *t);

void xavier_uniform_init(Tensor *t);

void kaiming_uniform_init(Tensor *t);

#endif
