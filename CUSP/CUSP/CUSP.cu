// Author: Sam Champer

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "funcs.h"
#include "gl_funcs.h"
#include "cuda_kernals.cuh"

// Paramaters defining the world and individuals.
#define EXPECTED_CAPACITY		    25000
#define INTERACTION_DISTANCE		0.05
#define IND_SPEED					0.02
#define NUMBER_OF_GENERATION_STEPS	10
#define FECUNDITY                   0.15

// Parameters defining simulation configuration.
//#define TORROIDAL_BOUNDARY    // Change from default reprising bounary.
//#define GAUSSIAN_INTERACTION  // Default is flat interaction.

// Parameters defining output type. Default is openGL output.
#define TEXT_OUTPUT
#define SMOOTH_OUTPUT           // Render frames between generations if using openGL output.

// Random number stuff.
static std::random_device randomDevice;
static std::mt19937 number(randomDevice());
std::uniform_real_distribution<float> aRandom(0.0, 1.0);
std::normal_distribution<float> aNormal(0.0, 0.2);

// Constant for threshold above which individual is considered to be facing unhealthy competition.
#ifdef GAUSSIAN_INTERACTION
float COMPETITION_THRESHOLD = EXPECTED_CAPACITY * M_PI *  INTERACTION_DISTANCE * INTERACTION_DISTANCE * 2.2;
#else
float COMPETITION_THRESHOLD = EXPECTED_CAPACITY * M_PI *  INTERACTION_DISTANCE * INTERACTION_DISTANCE * 1.1;
#endif // GAUSSIAN_INTERACTION

// Globals for openGL animation.
int disp_frame = 0;
int next_frame = 1;
int subframe = 0;
const int subframes_per_frame = 10;
float indSize = clamp(25.0 / EXPECTED_CAPACITY, 0.0025, 0.012);

using std::cout;
using std::endl;

class World;

class Individual {
public:
    int id;
    int hp = 1000;
    char age = 0;
    // x velocity, y velocity, position, interaction force.
    float xv = 0.0, yv = 0.0, x, y, force;
    World *indworld;
};

class World {
public:
    World(int populationSize);
    int popSize;
    std::vector<Individual> pop;
    thrust::host_vector<thrust::host_vector<float> > forceMatrix;
    void worldStep();
    int next_individual_id;
    inline void move(Individual ind);
};

inline void World::move(Individual ind) {
    // Initialize variables for delta x and y, store old x and y.
    float dx = ind.x;
    float dy = ind.y;
    ind.x += IND_SPEED * ind.xv;
    ind.y += IND_SPEED * ind.yv;
    ind.x += (2 * aRandom(number) - 1) * IND_SPEED * 0.5;
    ind.y += (2 * aRandom(number) - 1) * IND_SPEED * 0.5;
#ifdef TORROIDAL_BOUNDARY
    dx = ind.x - dx;  // New val - old val.
    dy = ind.y - dy;
    // Normalize the velocity.
    float denom = sqrt(dx * dx + dy * dy);
    if (denom != 0) {
        ind.xv = dx / denom;
        ind.yv = dy / denom;
    }
    // Individual positions wrap around world.
    if (ind.x > 1)
        ind.x -= 1;
    if (ind.y > 1)
        ind.y -= 1;
    if (ind.x < 0)
        ind.x += 1;
    if (ind.y < 0)
        ind.y += 1;
#else
    // Bounce individuals off the walls, mirror their velocity.
    if (ind.x > 1) {
        ind.x = 2 - ind.x;
        ind.xv = -ind.xv;
    }
    if (ind.x < 0) {
        ind.x = 0 - ind.x;
        ind.xv = -ind.xv;
    }
    if (ind.y > 1) {
        ind.y = 2 - ind.y;
        ind.yv = -ind.yv;
    }
    if (ind.y < 0) {
        ind.y = 0 - ind.y;
        ind.yv = -ind.yv;
    }
    else {
        dx = ind.x - dx;  // New val - old val.
        dy = ind.y - dy;
        // Normalize the velocity.
        float denom = sqrt(dx * dx + dy * dy);
        if (denom != 0) {
            ind.xv = dx / denom;
            ind.yv = dy / denom;
        }
    }
#endif
}

World::World(int populationSize) {
    this->popSize = populationSize / 4;
    this->next_individual_id = 0;
    pop.resize(popSize);
    for (int i = 0; i < popSize; ++i) {
        pop[i].x = clamp(aNormal(number) + 0.5, 0, 1);
        pop[i].y = clamp(aNormal(number) + 0.5, 0, 1);
        move(pop[i]);
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
        move(pop[i]);
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
        pop[i].force = -1;
        for (int j = 0; j < popSize; j++)
            pop[i].force += forceMatrix[i][j];
        //std::accumulate(forceMatrix[i].begin(), forceMatrix[i].end(), 0) - 1;
        //pop[i].force = std::reduce(std::execution::par, forceMatrix[i].begin(), forceMatrix[i].end()) - 1;
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
    thrust::host_vector<Individual> new_inds;
    for (int i = 0; i < popSize; ++i) {
        if (pop[i].age > 3 && COMPETITION_THRESHOLD > pop[i].force && pop[i].force > COMPETITION_THRESHOLD * 0.4 && aRandom(number) < FECUNDITY) {
            // Generate an offspring for population with the right number of neighbors.
            Individual new_ind;
            new_ind.indworld = this;
            new_ind.id = next_individual_id;
            next_individual_id++;
            new_ind.x = pop[i].x;
            new_ind.y = pop[i].y;
            move(new_ind);
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
std::vector<std::vector <coords> > data;

#ifdef TEXT_OUTPUT
/*
__global__ void collect_pop_data_strings(Individual* pop, char* ind_data_string) {
    int x = pop[i].x * 4095 + 0.5;
    int y = pop[i].y * 4095 + 0.5;
    int r = 0, g = 0, b = 0;
    (pop[i].age < 3) ? b = 255 : g = clamp((float)pop[i].hp / 1000 * 255 * 1.1 - 25.5, 0, 255);
    r = clamp((1000 - (float)pop[i].hp) / 1000 * 255 * 1.1 - 25.5, 0, 255);
    sprintf(ind_data_string, "%03X %03X %02X %02X %02X", x, y, r, g, b);
}*/

std::string program_output = "";
void writeData(World world, int index) {
    std::ostringstream strStr;
    strStr << "G" << endl;  // Seperator token for generations.

    // char *data = (char*) malloc(sizeof(char) * 16 * world.popSize);

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
                (world.pop[i].age < 3) ? data[index][vectorIdx].b = 255 : data[index][vectorIdx].g = clamp((float)world.pop[i].hp / 1000 * 255 * 1.1 - 25.5, 0, 255);
                data[index][vectorIdx].r = clamp((1000 - (float)world.pop[i].hp) / 1000 * 255 * 1.1 - 25.5, 0, 255);
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
        (world.pop[i].age < 3) ? current.b = 255 : current.g = clamp((float)world.pop[i].hp / 1000 * 255 * 1.1 - 25.5, 0, 255);
        current.r = clamp((1000 - (float)world.pop[i].hp) / 1000 * 255 * 1.1 - 25.5, 0, 255);
        data[index][i] = current;
    }
}
#endif
#endif

void display() {
    // Display function called to animate the openGL window.
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
        if (data[disp_frame][i].x >= -0.51) {  // Don't draw placeholders.
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

    // Update the frame. Loop the animation if at the end.
    subframe = (subframe + 1) % subframes_per_frame;
    if (subframe == 0) {
        disp_frame = (disp_frame + 1) % NUMBER_OF_GENERATION_STEPS;
        next_frame = (next_frame + 1) % NUMBER_OF_GENERATION_STEPS;
    }
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
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
    auto stop = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    cout << "Total calculation time: " << dur / 1000000 << endl;

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
    std::cerr << "Program finished!" << endl << "Press enter to quit." << endl;
    std::cin.get();
    return EXIT_SUCCESS;
}
