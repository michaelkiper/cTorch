typedef struct {
    float* (*method)(void*); //pointer to a function that takes a void pointer and returns a pointer to a float
} GradMethod;
