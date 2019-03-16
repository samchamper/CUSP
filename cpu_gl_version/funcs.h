#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

inline double clamp(double val, double lo, double hi) {
    return val < lo ? lo : val > hi ? hi : val;
}

inline void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

inline void crossProductv3(double v1[3], double v2[3], double newV[3]) {
    newV[0] = v1[1] * v2[2] - v1[2] * v2[1];
    newV[1] = v1[2] * v2[0] - v1[0] * v2[2];
    newV[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline double dotProductv3(double v1[3], double v2[3]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

inline void normalizev3(double v[3]) {
    double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
}

inline void normalizev2(double v[2]) {
    double norm = sqrt(v[0] * v[0] + v[1] * v[1]);
    v[0] /= norm;
    v[1] /= norm;
}
