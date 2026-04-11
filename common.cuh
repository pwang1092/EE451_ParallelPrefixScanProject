#ifndef COMMON_CUH
#define COMMON_CUH

#ifndef D
#define D 16  // SSM hidden state dimension — compile with -DD=64, -DD=256, etc.
#endif

#define BLOCK_SIZE 256

struct Element {
    float A[D][D];
    float b[D];
};

// Identity element: A = I (identity matrix), b = 0
__device__ inline Element identity() {
    Element e;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            e.A[i][j] = (i == j) ? 1.0f : 0.0f;
        }
        e.b[i] = 0.0f;
    }
    return e;
}

// Associative combine: (A2, b2) ⊕ (A1, b1) = (A2·A1, A2·b1 + b2)
__device__ inline Element combine(Element left, Element right) {
    Element result;

    // result.A = right.A * left.A
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += right.A[i][k] * left.A[k][j];
            }
            result.A[i][j] = sum;
        }
    }

    // result.b = right.A * left.b + right.b
    for (int i = 0; i < D; i++) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += right.A[i][k] * left.b[k];
        }
        result.b[i] = sum + right.b[i];
    }

    return result;
}

#endif // COMMON_CUH
