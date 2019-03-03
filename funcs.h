#pragma once

//__forceinline__ __device__ 
inline double clamp(double val, double lo, double hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

inline void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

inline void crossProduct(double v1[3], double v2[3], double newV[3]) {
    newV[0] = v1[1] * v2[2] - v1[2] * v2[1];
    newV[1] = v1[2] * v2[0] - v1[0] * v2[2];
    newV[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline double dotProduct(double v1[3], double v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

inline void normalize(double v[3]) {
    double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
}
