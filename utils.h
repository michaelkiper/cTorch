#ifndef UTILS_H
#define UTILS_H

#define ARRAYSIZE(a) (sizeof(a) / sizeof((a)[0]))

#define PREFIX(var) new_##var

#define GENERIC_ADD(type)                 \
    type add_##type(type (a), type (b)) { \
        return (a) + (b);                 \
    }


#define COUNT(a) do {                        \
    int c = 0;                               \
    for (int i = 0; i < ARRAYSIZE(a); i++) { \
        c++;                                 \
    }                                        \
} while(0)

#define PRINT_LOOP(iterations, ...) do {   \
    for (int i = 0; i < iterations; i++) { \
        prinft(__VA_ARGS__);               \
    }                                      \
} while(0)

#endif
