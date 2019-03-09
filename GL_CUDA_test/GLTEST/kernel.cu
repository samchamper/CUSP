// This file is an amalgamation of the default CUDA 10.1 template project along with opengl calls
// from the CUDA example library as well as starter code from our 441 class projects.
// I didn't write most of it - just glued it all together so I have a template for a project that
// both uses CUDA for calculations as well as draws stuff using OpenGL.

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <GL\glew.h>
#include <GL\glut.h>
#include <GL\freeglut.h>
#include <iostream>

#include "gl_funcs.h"

using namespace std;

const unsigned int window_width = 800;
const unsigned int window_height = 800;

bool initGL(int *argc, char **argv);

const int arraySize = 5;
float frame_no = 0;
int disp_frame = 0;
int c[arraySize] = { 0 };





void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Set view matrix based on values set by mouse.
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    glColor3ub(250, 128, 114);
    glPushMatrix();
    glTranslatef(0, 0, 0);
    glScalef(1.0, 1.0, 1.0);
    DrawSphere();
    glPopMatrix();

    glColor3ub(0, 200, 210);
    glPushMatrix();
    glTranslatef(2, 2, 2);
    glScalef(.2 * c[disp_frame], .5, .5);
    DrawSphere();
    glPopMatrix();

    for (size_t i = 0; i < arraySize; i++)
    {
        glColor3ub(0, 0, 255);
        glPushMatrix();
        glTranslatef(c[i], c[i], c[i]);
        glScalef(.5, .5, .5);
        DrawSphere();
        glPopMatrix();
    }

    glEnd();
    glutSwapBuffers();
    frame_no += 0.1;
    if (arraySize - frame_no < 0.00001)
        frame_no = 0;
    disp_frame = (int) frame_no;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
    // Just testing to see if thrust vectors work.
    thrust::host_vector<int> H(4);
    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
    thrust::device_vector<int> D(4);
    // initialize individual elements
    D[0] = 14;
    D[1] = 20;
    D[2] = 38;
    D[3] = 46;
    int sum = thrust::reduce(D.begin(), D.end());
    cout << sum << " " << 14 + 20 + 38 + 46 << endl;

    // Now on to the rest:
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // OpenGL stuff:
    if(initGL(&argc, argv) == false)
        return 1;
    glutDisplayFunc(display);

    glutMainLoop();
    printf("PROGRAM FINISHED.");
    return 0;
}

bool initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("TEST WINDOW");

    if (!isGLVersionSupported(2, 0)) {
        fprintf(stderr, "ERROR: OpenGL extension support is missing.");
        fflush(stderr);
        return false;
    }
    set_lighting();
    glEnable(GL_DEPTH_TEST);

    // Set viewport
    glViewport(0, 0, window_width, window_height);

    // Set projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 1, 256);

    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutMouseFunc(mouse);

    return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
