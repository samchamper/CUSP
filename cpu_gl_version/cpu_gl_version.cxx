// Author: Sam Champer

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <execution>
#include <string>
#include <fstream>

#include "funcs.h"
#include "gl_funcs.h"

#ifndef M_PI
#define M_PI					    3.14159265358979323846264338327950288
#endif
#define EXPECTED_CAPACITY		    50000
#define INTERACTION_DISTANCE		0.05
#define IND_SPEED					0.02
#define NUMBER_OF_GENERATION_STEPS	10
#define FECUNDITY                   0.15
//#define TORROIDAL_BOUNDARY    // Change from default reprising bounary.
//#define GAUSSIAN_INTERACTION  // Default is flat interaction.
#define TEXT_OUTPUT
//#define SMOOTH_OUTPUT

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::string;

const unsigned int window_width = 800;
const unsigned int window_height = 800;

bool initGL(int *argc, char **argv);
void display();

// Random number stuff.
static std::random_device randomDevice;
static std::mt19937 number(randomDevice());
std::uniform_real_distribution<float> aRandom(0.0, 1.0);
std::normal_distribution<float> aNormal(0.0, 0.2);

// Precalculate a constant above which an individual is considered to be facing unhealthy competition.
#ifdef GAUSSIAN_INTERACTION
float COMPETITION_THRESHOLD = EXPECTED_CAPACITY * M_PI *  INTERACTION_DISTANCE * INTERACTION_DISTANCE * 2.2;
#else
float COMPETITION_THRESHOLD = EXPECTED_CAPACITY * M_PI *  INTERACTION_DISTANCE * INTERACTION_DISTANCE * 1.1;
#endif // GAUSSIAN_INTERACTION

class World;

class Individual {
public:
    int id;
    int hp = 1000;
    int age = 0;
    // x velocity, y velocity, position, interaction force.
    float xv = 0.0, yv = 0.0, x, y, force;
    World *indworld;
    inline void move();
};

inline void Individual::move() {
    // Initialize variables for delta x and y, store old x and y.
    float dx = x;
    float dy = y;
    x += IND_SPEED * xv;
    y += IND_SPEED * yv;
    x += (2 * aRandom(number) - 1) * IND_SPEED * 0.5;
    y += (2 * aRandom(number) - 1) * IND_SPEED * 0.5;
#ifdef TORROIDAL_BOUNDARY
    dx = x - dx;
    dy = y - dy;
    float denom = sqrt(dx * dx + dy * dy);
    if (denom != 0) {
        xv = dx / denom;
        yv = dy / denom;
    }
    if (x > 1)
        x -= 1;
    if (y > 1)
        y -= 1;
    if (x < 0)
        x += 1;
    if (y < 0)
        y += 1;
#else
    if (x > 1)
        x = 2 - x;
    if (x < 0)
        x = 0 - x;
    if (y > 1)
        y = 2 - y;
    if (y < 0)
        y = 0 - y;
    dx = x - dx;
    dy = y - dy;
    float denom = sqrt(dx * dx + dy * dy);
    if (denom != 0) {
        xv = dx / denom;
        yv = dy / denom;
    }
#endif
}

class World {
public:
    World(int populationSize);
    int popSize;
    vector<Individual> pop;
    vector<vector <float> > forceMatrix;
    void worldStep();
    int next_individual_id;
};

World::World(int populationSize) {
    this->popSize = populationSize / 4;
    this->next_individual_id = 0;
    pop.resize(popSize);
    for (int i = 0; i < popSize; ++i) {
        pop[i].x = clamp(aNormal(number) + 0.5, 0, 1);
        pop[i].y = clamp(aNormal(number) + 0.5, 0, 1);
        pop[i].move();
        pop[i].age = aRandom(number) * 10;
        pop[i].indworld = this;
        pop[i].id = next_individual_id;
        next_individual_id++;
    }
}

void World::worldStep() {
    cout << "    POP SIZE:" << popSize << endl;
    forceMatrix.resize(popSize);
    for (int i = 0; i < popSize; ++i)
        forceMatrix[i].resize(popSize);

    // Population moves, ages.
    for (int i = 0; i < popSize; ++i) {
        pop[i].move();
        // Reduce hp by a flat amount per time step.
        pop[i].hp -= 2 * pop[i].age;
        // Increment age.
        pop[i].age++;
    }
    // Calculate competion:
    for (int i = 0; i < popSize; ++i) {
        // Calculate the interaction strength between this individual and all others.
        // This is the time intensive part.
#ifdef GAUSSIAN_INTERACTION
        forceMatrix[i][i] = 1;
        for (int j = i + 1; j < popSize; ++j) {
            float dx = pop[i].x - pop[j].x;
            float dy = pop[i].y - pop[j].y;
            if (abs(dx) > INTERACTION_DISTANCE * 4 || abs(dy) > INTERACTION_DISTANCE * 4) {
                forceMatrix[i][j] = 0;
                forceMatrix[j][i] = 0;
            }
            else {
                float dist = sqrt(dx * dx + dy * dy);
                // Interaction strength as a function of distance on a normal curve.
                if (dist < INTERACTION_DISTANCE * 4) {
                    forceMatrix[i][j] = exp(-(dist * dist) / (2 * INTERACTION_DISTANCE * INTERACTION_DISTANCE));
                    forceMatrix[j][i] = forceMatrix[i][j];
                }
                else {
                    forceMatrix[i][j] = 0;
                    forceMatrix[j][i] = 0;
                }
            }
#else
        for (int j = 0; j < popSize; ++j) {
            float dx = pop[i].x - pop[j].x;
            float dy = pop[i].y - pop[j].y;
            if (abs(dx) > INTERACTION_DISTANCE || abs(dy) > INTERACTION_DISTANCE)
                forceMatrix[i][j] = 0;
            else {
                float dist = sqrt(dx * dx + dy * dy);
                if (dist < INTERACTION_DISTANCE)
                    forceMatrix[i][j] = 1;
                else
                    forceMatrix[i][j] = 0;
            }
#endif
        }
    }
    // Apply competition:
    for (int i = 0; i < popSize; ++i) {
        // Subtracting 1 from force to factor out individual competing with itself.
        pop[i].force = std::reduce(std::execution::par, forceMatrix[i].begin(), forceMatrix[i].end()) - 1;
        if (pop[i].force > COMPETITION_THRESHOLD)
            pop[i].hp -= 300;
    }
    // Prune the dead from the population:
    for (int i = 0; i < popSize; ++i) {
        // Compact the vector, leaving the dead individuals at the end:
        auto write = pop.begin();
        for (auto read = pop.begin(), end = pop.end(); read != end; ++read) {
            if (read->hp > 0) {
                if (read != write) {
                    *write = std::move(*read);
                }
                ++write;
            }
        }
        // Remove dead individuals.
        pop.erase(write, pop.end());
    }
    // Generate offspring, dependent on density interaction force:
    popSize = pop.size();
    vector <Individual> new_inds;
    for (int i = 0; i < popSize; ++i) {
        if (pop[i].age > 3 && COMPETITION_THRESHOLD > pop[i].force && pop[i].force > COMPETITION_THRESHOLD * 0.4 && aRandom(number) < FECUNDITY) {
            // Generate an offspring for population with the right number of neighbors.
            Individual new_ind;
            new_ind.indworld = this;
            new_ind.id = next_individual_id;
            next_individual_id++;
            new_ind.x = pop[i].x;
            new_ind.y = pop[i].y;
            new_ind.move();
            new_inds.push_back(new_ind);
        }
    }
    pop.reserve(popSize + new_inds.size());
    pop.insert(pop.end(), new_inds.begin(), new_inds.end());
    popSize = pop.size();
}

struct coords {
    // x, y, and color coordinates for drawing an individual.
    float x = -1, y = -1;
    int r = 0, g = 0, b = 0;
};
vector<vector <coords> > data;
string program_output = "";

#ifdef TEXT_OUTPUT
void writeData(World world, int index) {
    std::ostringstream strStr;
    strStr << "G" << endl;  // Seperator token for generations.
    for (int i = 0; i < world.popSize; i++) {
        // The data to export to text is individual position and color.
        // Squash data a bit and convert to hex in order to save space in the text file.
        // This may seem unimportant, but many thousands of individuals running for hundreds 
        // of generations can result in a text file that is dozens of MB, and this conversion saves a lot of IO time.
        int x = world.pop[i].x * 4095 + 0.5;
        int y = world.pop[i].y * 4095 + 0.5;
        int r = 0, g = 0, b = 0;
        (world.pop[i].age < 3) ? b = 255 : g = clamp((float)world.pop[i].hp / 1000 * 255 * 1.1 - 25.5, 0, 255);
        r = clamp((1000 - (float)world.pop[i].hp) / 1000 * 255 * 1.1 - 25.5, 0, 255);
        char ind_data[16];
        sprintf(ind_data, "%03X %03X %02X %02X %02X", x, y, r, g, b);
        strStr << ind_data << endl;
    }
    program_output = program_output + strStr.str();
}
#else
#ifdef SMOOTH_OUTPUT
// A much slower version of the function that places the same individual at the same index for each generation timestep.
// This is done in a horrible way - it's just a rapidly thrown together function to test smoothing of the animation.
void writeData(World world, int index) {
    // data[index] gets a list of coords for each individual in that generation.
    data[index].resize(world.next_individual_id - 1); // Need to have a spot in the vector for each possible individual, even the dead ones.
    for (int i = 0; i < world.next_individual_id - 1; i++) {  // Placeholders.
        coords current;
        data[index][i] = current;
    }
    for (int vectorIdx = 0; vectorIdx < world.next_individual_id - 1; vectorIdx++) {
        for (int i = 0; i < world.popSize; i++) {
            if (world.pop[i].id == vectorIdx) {
                data[index][vectorIdx].x = world.pop[i].x - 0.5;
                data[index][vectorIdx].y = world.pop[i].y - 0.5;
                (world.pop[i].age < 3) ? data[index][vectorIdx].b = 255 : data[index][vectorIdx].g = clamp((float) world.pop[i].hp / 1000 * 255 * 1.1 - 25.5, 0, 255);
                data[index][vectorIdx].r = clamp((1000 - (float) world.pop[i].hp) / 1000 * 255 * 1.1 - 25.5, 0, 255);
            }
        }
    }
}
#else
void writeData(World world, int index) {
    // data[index] gets a list of coords for each individual in that generation.
    data[index].resize(world.popSize);
    for (int i = 0; i < world.popSize; i++) {
        coords current;
        current.x = world.pop[i].x - 0.5;
        current.y = world.pop[i].y - 0.5;
        (world.pop[i].age < 3) ? current.b = 255 : current.g = clamp((float) world.pop[i].hp / 1000 * 255 * 1.1 - 25.5, 0, 255);
        current.r = clamp((1000 - (float) world.pop[i].hp) / 1000 * 255 * 1.1 - 25.5, 0, 255);
        data[index][i] = current;
    }
}
#endif
#endif

int main(int argc, char** argv) {
    auto start = high_resolution_clock::now();
#ifndef TEXT_OUTPUT
    data.resize(NUMBER_OF_GENERATION_STEPS + 1);
#endif
    // Setup world:
    World world(EXPECTED_CAPACITY);
    writeData(world, 0);
    // Main program loop:
    for (int i = 0; i < NUMBER_OF_GENERATION_STEPS; ++i) {
        cout << "Generation " << i + 1 << ": " << endl;
        world.worldStep();
        writeData(world, i + 1);
    }
    auto stop = high_resolution_clock::now();
    double dur = duration_cast<microseconds>(stop - start).count();
    cout << "Total calculation time: " << dur / 1000000 << endl ;

#ifdef TEXT_OUTPUT
    // Output result to text file.
    std::ofstream f;
    f.open("pop_vis_data");
    f << program_output;
    f.close();
    cout << "Program output written to 'pop_vis_data'" << endl;
#else
    // OpenGL stuff:
    if (initGL(&argc, argv) == false)
        return 1;
    glutDisplayFunc(display);

    glutMainLoop();
#endif
    cerr << "Program finished!" << endl << "Press enter to quit." << endl;
    std::cin.get();
    return EXIT_SUCCESS;
}

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

int disp_frame = 0;
int next_frame = 1;
int subframe = 0;
const int subframes_per_frame = 10;
float indSize = clamp(25.0 / EXPECTED_CAPACITY, 0.0025, 0.012);


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    char title[42];
    sprintf(title, "CUSP Population Sim: Generation %d", disp_frame);
    glutSetWindowTitle(title);
    // Set view matrix based on values set by mouse.
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // Draw the population!
    glColor3ub(250, 128, 114);
    int numInds = data[disp_frame].size();
    for (int i = 0; i < numInds; i++) {
        if (data[disp_frame][i].x >= - 0.51) {  // Don't draw placeholders.
            float x = data[disp_frame][i].x;
            float y = data[disp_frame][i].y;
            int r = data[disp_frame][i].r;
            int g = data[disp_frame][i].g;
            int b = data[disp_frame][i].b;

#ifdef SMOOTH_OUTPUT
            if (next_frame != 0 && data[next_frame][i].x >= -0.51) {
                float dx = data[next_frame][i].x - x;
                float dy = data[next_frame][i].y - y;
#ifdef TORROIDAL_BOUNDARY
                // Don't do in between frames if the indivdual just wrapped around the world.
                if (abs(dx) > IND_SPEED * 2)
                    dx = 0;
                if (abs(dy) > IND_SPEED * 2)
                    dy = 0;
#endif
                x = x + dx * subframe / subframes_per_frame;
                y = y + dy * subframe / subframes_per_frame;
                if (b == 0) {
                    r = r + (data[next_frame][i].r - r) * subframe / subframes_per_frame;
                    g = g + (data[next_frame][i].g - g) * subframe / subframes_per_frame;
                }
            }
#endif
            glColor3ub(r, g, b);
            glPushMatrix();
            glTranslatef(x, 0, y);
            glScalef(indSize, indSize, indSize);
            DrawSphere();
            glPopMatrix();
        }
    }
    glEnd();
    glutSwapBuffers();

    subframe = (subframe + 1) % subframes_per_frame;
    if (subframe == 0) {
        disp_frame = (disp_frame + 1) % NUMBER_OF_GENERATION_STEPS;
        next_frame = (next_frame + 1) % NUMBER_OF_GENERATION_STEPS;
    }
}
