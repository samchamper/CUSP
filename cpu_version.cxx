// Author: Sam Champer

#include <iostream>
#include <math.h>
#include <ctime>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>
#include <execution>

#include "funcs.h"
#include "vtk_funcs.h"
#include "Screen.h"

#ifndef M_PI
#define M_PI					    3.14159265358979323846264338327950288
#endif
#define EXPECTED_CAPACITY		    10000
#define INTERACTION_DISTANCE		0.05
#define IND_SPEED					0.02
#define NUMBER_OF_GENERATION_STEPS	50
#define FECUNDITY                   0.1
//#define TORROIDAL_BOUNDARY
//#define GAUSSIAN_INTERACTION

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Random number stuff.
static std::random_device randomDevice;
//static std::mt19937 number(randomDevice());
static std::mt19937 number(1);
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
    void worldDraw(Screen screen, vtkImageData *img, int frame_num);
};

World::World(int populationSize) {
    this->popSize = populationSize / 4;
    pop.resize(popSize);
    for (int i = 0; i < popSize; ++i) {
        pop[i].x = clamp(aNormal(number) + 0.5, 0, 1);
        pop[i].y = clamp(aNormal(number) + 0.5, 0, 1);
        pop[i].move();
        pop[i].age = 8;
        pop[i].indworld = this;
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

void World::worldDraw(Screen screen, vtkImageData *img, int frame) {
    // Draw all individuals on the screen with a dot size inversely proportional to the number of individuals in existence.
    if (popSize < 10000) {
        for (int i = 0; i < popSize; ++i) {
            double r = 0, g = 0, b = 0;
            (pop[i].age < 3) ? b = 1.0 : g = clamp((double)pop[i].hp / 1000 * 1.1 - 0.1, 0, 1);
            r = clamp((1000 - (double)pop[i].hp) / 1000 * 1.1 - 0.1, 0, 1);
            for (int w = -2; w <= 2; ++w)
                for (int h = -2; h <= 2; ++h)
                    screen.setPixel(pop[i].x * screen.width + w, pop[i].y * screen.height + h, r, g, b);
            // To make more circular looking marks.
            screen.setPixel(pop[i].x * screen.width + 3, pop[i].y * screen.height, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 3, pop[i].y * screen.height - 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 3, pop[i].y * screen.height + 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 3, pop[i].y * screen.height, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 3, pop[i].y * screen.height - 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 3, pop[i].y * screen.height + 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width, pop[i].y * screen.height + 3, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 1, pop[i].y * screen.height + 3, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 1, pop[i].y * screen.height + 3, r, g, b);
            screen.setPixel(pop[i].x * screen.width, pop[i].y * screen.height - 3, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 1, pop[i].y * screen.height - 3, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 1, pop[i].y * screen.height - 3, r, g, b);
        }
    }
    else if (popSize < 20000) {
        for (int i = 0; i < popSize; ++i) {
            double r = 0, g = 0, b = 0;
            (pop[i].age < 3) ? b = 1.0 : g = clamp((double)pop[i].hp / 1000 * 1.1 - 0.1, 0, 1);
            r = clamp((1000 - (double)pop[i].hp) / 1000 * 1.1 - 0.1, 0, 1);
            for (int w = -1; w <= 1; ++w)
                for (int h = -1; h <= 1; ++h)
                    screen.setPixel(pop[i].x * screen.width + w, pop[i].y * screen.height + h, r, g, b);
            // To make more circular looking marks.
            screen.setPixel(pop[i].x * screen.width + 2, pop[i].y * screen.height, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 2, pop[i].y * screen.height - 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 2, pop[i].y * screen.height + 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 2, pop[i].y * screen.height, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 2, pop[i].y * screen.height - 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 2, pop[i].y * screen.height + 1, r, g, b);
            screen.setPixel(pop[i].x * screen.width, pop[i].y * screen.height + 2, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 1, pop[i].y * screen.height + 2, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 1, pop[i].y * screen.height + 2, r, g, b);
            screen.setPixel(pop[i].x * screen.width, pop[i].y * screen.height - 2, r, g, b);
            screen.setPixel(pop[i].x * screen.width - 1, pop[i].y * screen.height - 2, r, g, b);
            screen.setPixel(pop[i].x * screen.width + 1, pop[i].y * screen.height - 2, r, g, b);
        }
    }
    else if (popSize < 40000) {
        for (int i = 0; i < popSize; ++i) {
            double r = 0, g = 0, b = 0;
            (pop[i].age < 3) ? b = 1.0 : g = clamp((double)pop[i].hp / 1000 * 1.1 - 0.1, 0, 1);
            r = clamp((1000 - (double)pop[i].hp) / 1000 * 1.1 - 0.1, 0, 1);
            for (int w = -1; w <= 1; ++w)
                for (int h = -1; h <= 1; ++h)
                    screen.setPixel(pop[i].x * screen.width + w, pop[i].y * screen.height + h, r, g, b);
        }
    }
    else {
        for (int i = 0; i < popSize; ++i) {
            double r = 0, g = 0, b = 0;
            (pop[i].age < 3) ? b = 1.0 : g = clamp((double)pop[i].hp / 1000 * 1.1 - 0.1, 0, 1);
            r = clamp((1000 - (double)pop[i].hp) / 1000 * 1.1 - 0.1, 0, 1);
            for (int w = 0; w <= 1; ++w)
                for (int h = 0; h <= 1; ++h)
                    screen.setPixel(pop[i].x * screen.width + w, pop[i].y * screen.height + h, r, g, b);
        }
    }
    char str[16];
    sprintf(str, "frame%03d", frame);
    WriteImage(img, str);
}

int main() {
    // VTK/Screen Setup:
    int imageHeight = 1080, imageWidth = 1080;
    vtkImageData *image = NewImage(imageHeight, imageWidth);
    unsigned char *buffer = (unsigned char *)image->GetScalarPointer(0, 0, 0);
    Screen screen(imageHeight, imageWidth);
    screen.buffer = buffer;

    auto start = high_resolution_clock::now();
    // Setup world:
    World world(EXPECTED_CAPACITY);
    screen.clear();
    world.worldDraw(screen, image, 0);
    // Main program loop:
    for (int i = 0; i < NUMBER_OF_GENERATION_STEPS; ++i) {
        screen.clear();
        cout << "Generation " << i + 1 << ": " << endl;
        world.worldStep();
        world.worldDraw(screen, image, i + 1);
    }
    auto stop = high_resolution_clock::now();
    double dur = duration_cast<microseconds>(stop - start).count();
    cout << "Total run time: " << dur / 1000000 << endl ;
    cerr << "Program finished!" << endl << "Press enter to quit." << endl;
    cin.get();
    return EXIT_SUCCESS;
}
