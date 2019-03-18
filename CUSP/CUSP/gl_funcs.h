#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/glew.h>
#endif
#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#ifdef __linux__
#include <GL/glx.h>
#endif /* __linux__ */
#endif
#include <iostream>
#include <sstream>
#include <cassert>
#include <vector>
#include <GL\glut.h>
#include <GL\freeglut.h>

#define REFRESH_DELAY     10  // Number of ms between frames.
const unsigned int window_width = 1000;
const unsigned int window_height = 1000;

/////////////////////////////////////////////////////////////////////////////
// Functions sourced from NVIDIA CUDA examples.
/////////////////////////////////////////////////////////////////////////////
// Mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Version check
static int isGLVersionSupported(unsigned reqMajor, unsigned reqMinor)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "glewInit() failed!" << std::endl;
        return 0;
    }
#endif
    std::string version((const char *)glGetString(GL_VERSION));
    std::stringstream stream(version);
    unsigned major, minor;
    char dot;

    stream >> major >> dot >> minor;

    assert(dot == '.');
    return major > reqMajor || (major == reqMajor && minor >= reqMinor);
}

// Keyboard events handler
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (27):
#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
    }
}

//! Mouse event handlers
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

// Timer function.
void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

/////////////////////////////////////////////////////////////////////////////
// End functions sourced from NVIDIA CUDA examples.
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Funcs from 441 University of Oregon: CIS 441: Graphics, with Hank Childs.
/////////////////////////////////////////////////////////////////////////////
class Triangle
{
public:
    double         X[3];
    double         Y[3];
    double         Z[3];
};

std::vector<Triangle> SplitTriangle(std::vector<Triangle> &list)
{
    std::vector<Triangle> output(4 * list.size());
    for (unsigned int i = 0; i < list.size(); i++)
    {
        double mid1[3], mid2[3], mid3[3];
        mid1[0] = (list[i].X[0] + list[i].X[1]) / 2;
        mid1[1] = (list[i].Y[0] + list[i].Y[1]) / 2;
        mid1[2] = (list[i].Z[0] + list[i].Z[1]) / 2;
        mid2[0] = (list[i].X[1] + list[i].X[2]) / 2;
        mid2[1] = (list[i].Y[1] + list[i].Y[2]) / 2;
        mid2[2] = (list[i].Z[1] + list[i].Z[2]) / 2;
        mid3[0] = (list[i].X[0] + list[i].X[2]) / 2;
        mid3[1] = (list[i].Y[0] + list[i].Y[2]) / 2;
        mid3[2] = (list[i].Z[0] + list[i].Z[2]) / 2;
        output[4 * i + 0].X[0] = list[i].X[0];
        output[4 * i + 0].Y[0] = list[i].Y[0];
        output[4 * i + 0].Z[0] = list[i].Z[0];
        output[4 * i + 0].X[1] = mid1[0];
        output[4 * i + 0].Y[1] = mid1[1];
        output[4 * i + 0].Z[1] = mid1[2];
        output[4 * i + 0].X[2] = mid3[0];
        output[4 * i + 0].Y[2] = mid3[1];
        output[4 * i + 0].Z[2] = mid3[2];
        output[4 * i + 1].X[0] = list[i].X[1];
        output[4 * i + 1].Y[0] = list[i].Y[1];
        output[4 * i + 1].Z[0] = list[i].Z[1];
        output[4 * i + 1].X[1] = mid2[0];
        output[4 * i + 1].Y[1] = mid2[1];
        output[4 * i + 1].Z[1] = mid2[2];
        output[4 * i + 1].X[2] = mid1[0];
        output[4 * i + 1].Y[2] = mid1[1];
        output[4 * i + 1].Z[2] = mid1[2];
        output[4 * i + 2].X[0] = list[i].X[2];
        output[4 * i + 2].Y[0] = list[i].Y[2];
        output[4 * i + 2].Z[0] = list[i].Z[2];
        output[4 * i + 2].X[1] = mid3[0];
        output[4 * i + 2].Y[1] = mid3[1];
        output[4 * i + 2].Z[1] = mid3[2];
        output[4 * i + 2].X[2] = mid2[0];
        output[4 * i + 2].Y[2] = mid2[1];
        output[4 * i + 2].Z[2] = mid2[2];
        output[4 * i + 3].X[0] = mid1[0];
        output[4 * i + 3].Y[0] = mid1[1];
        output[4 * i + 3].Z[0] = mid1[2];
        output[4 * i + 3].X[1] = mid2[0];
        output[4 * i + 3].Y[1] = mid2[1];
        output[4 * i + 3].Z[1] = mid2[2];
        output[4 * i + 3].X[2] = mid3[0];
        output[4 * i + 3].Y[2] = mid3[1];
        output[4 * i + 3].Z[2] = mid3[2];
    }
    return output;
}

void DrawSphere()
{
    int recursionLevel = 2;
    Triangle t;
    t.X[0] = 1;
    t.Y[0] = 0;
    t.Z[0] = 0;
    t.X[1] = 0;
    t.Y[1] = 1;
    t.Z[1] = 0;
    t.X[2] = 0;
    t.Y[2] = 0;
    t.Z[2] = 1;
    std::vector<Triangle> list;
    list.push_back(t);
    for (int r = 0; r < recursionLevel; r++)
    {
        list = SplitTriangle(list);
    }

    // really draw `
    for (int octent = 0; octent < 8; octent++)
    {
        glPushMatrix();
        glRotatef(90 * (octent % 4), 1, 0, 0);
        if (octent >= 4)
            glRotatef(180, 0, 0, 1);
        glBegin(GL_TRIANGLES);
        for (unsigned int i = 0; i < list.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double ptMag = sqrt(list[i].X[j] * list[i].X[j] +
                    list[i].Y[j] * list[i].Y[j] +
                    list[i].Z[j] * list[i].Z[j]);
                glNormal3f(list[i].X[j] / ptMag, list[i].Y[j] / ptMag, list[i].Z[j] / ptMag);
                glVertex3f(list[i].X[j] / ptMag, list[i].Y[j] / ptMag, list[i].Z[j] / ptMag);
            }
        }
        glEnd();
        glPopMatrix();
    }
}

void set_lighting() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat diffuse0[4] = { 0.6, 0.6, 0.6, 1 };
    GLfloat ambient0[4] = { 0.2, 0.2, 0.2, 1 };
    GLfloat specular0[4] = { 0.0, 0.0, 0.0, 1 };
    GLfloat pos0[4] = { 0, .707, 0.707, 0 };
    glLightfv(GL_LIGHT0, GL_POSITION, pos0);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);
    glEnable(GL_LIGHT1);
    GLfloat pos1[4] = { .707, -.707, 0 };
    glLightfv(GL_LIGHT1, GL_POSITION, pos1);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient0);
    glLightfv(GL_LIGHT1, GL_SPECULAR, specular0);
    glDisable(GL_LIGHT2);
    glDisable(GL_LIGHT3);
    glDisable(GL_LIGHT5);
    glDisable(GL_LIGHT6);
    glDisable(GL_LIGHT7);
    glEnable(GL_COLOR_MATERIAL);
}
/////////////////////////////////////////////////////////////////////////////
// End functions from CIS 441.
/////////////////////////////////////////////////////////////////////////////

// Other functions:
bool initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUSP Population Sim: Generation 0");

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
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.01, 256);

    // Set interaction functions.
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutMouseFunc(mouse);
    return true;
}

void draw_cube(float n) {
    glBegin(GL_LINES);
    glColor3ub(100, 100, 100);
    glVertex3f(n, n, n);
    glVertex3f(n, -n, n);
    glVertex3f(n, n, n);
    glVertex3f(-n, n, n);
    glVertex3f(n, n, n);
    glVertex3f(n, n, -n);
    glVertex3f(-n, -n, -n);
    glVertex3f(-n, n, -n);
    glVertex3f(-n, -n, -n);
    glVertex3f(n, -n, -n);
    glVertex3f(-n, -n, -n);
    glVertex3f(-n, -n, n);
    glVertex3f(n, n, -n);
    glVertex3f(-n, n, -n);
    glVertex3f(n, n, -n);
    glVertex3f(n, -n, -n);
    glVertex3f(n, -n, n);
    glVertex3f(-n, -n, n);
    glVertex3f(n, -n, n);
    glVertex3f(n, -n, -n);
    glVertex3f(-n, n, n);
    glVertex3f(-n, -n, n);
    glVertex3f(-n, n, n);
    glVertex3f(-n, n, -n);
    glEnd();
}
