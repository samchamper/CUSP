#include <iostream>
#include <math.h>
#include <ctime>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>

#include "funcs.h"
#include "vtk_funcs.h"
#include "Screen.h"


#define START_POPULATION_SIZE 1000
#define COMPETITION_THRESHOLD START_POPULATION_SIZE/50
#define NUMBER_OF_GENERATION_STEPS 50
#define INTERACTION_DISTANCE 0.05
#define IND_SPEED 2   // Speed as a percentage of the world.
//#define TORROIDAL_BOUNDARY
#define GAUSSIAN_INTERACTION

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Distribution for gaussian interaction.
std::normal_distribution<double> normal(1.0, INTERACTION_DISTANCE);

class World;

class Individual {
    public:
        int hp = 1000;
        int age;
        double x, y;
        World *indworld;
        vector<double> forceVector;
        double force;
        inline void move();
        inline void calcInteractionForce(int index);
};

class World {
    public:
        World(int popsize);
        vector<Individual> pop;
        void worldStep();
        void worldDraw(Screen screen);
};

World::World(int popSize) {
    pop.resize(popSize);
    for (int i = 0; i < popSize; ++i) {
        pop[i].x = (double) rand() / static_cast<double>(RAND_MAX);
        pop[i].y = (double) rand() / static_cast<double>(RAND_MAX);
        pop[i].age = 10;
        pop[i].indworld = this;
    }
}

void World::worldStep() {
    int popSize = pop.size();
    // Population moves, ages.
    for (int i = 0; i < popSize; ++i) {
        pop[i].forceVector.resize(popSize);
        pop[i].move();
        // Reduce hp by a flat amount per time step.
        pop[i].hp -= 20;
        // Increment age.
        pop[i].age++;
    }
    // Calculate competion:
    for (int i = 0; i < popSize; ++i)
        pop[i].calcInteractionForce(i);
    // Apply competition:
    for (int i = 0; i < popSize; ++i) {
        pop[i].force = std::accumulate(pop[i].forceVector.begin(), pop[i].forceVector.end(), 0);
        if (pop[i].force > COMPETITION_THRESHOLD)
            pop[i].hp -= 300;
    }
    // Prune the dead from the population:
/*
    for (int i = 0; i < popSize; ++i) {
        // TODO::::::::Implement:
        auto write = pop.begin();
        for (auto read = write, end = pop.end(); read != end; ++read) {
            if (read.hp > 0) {
                if (read != write) {
                    *write = std::move(*read);
                }
                ++write;
            }
        }
        pop.erase(w, pop.end()); // Remove empty spots.
    }*/
    // TODO: Gen offspring, dependent on density interaction force.
}

void World::worldDraw(Screen screen) {
    int popSize = pop.size();
    for (int i = 0; i < popSize; ++i) {
        if (pop[i].hp > 0) {
            double r = clamp((1000 - (double) pop[i].hp) / 1000 * 1.1 - 0.1, 0, 1);
            double g = clamp((double) pop[i].hp / 1000 * 1.1 - 0.1, 0, 1);
            for (int w = -2; w <= 2; ++w)
                for (int h = -2; h <= 2; ++h)
                    screen.setPixel(pop[i].x * screen.width + w, pop[i].y * screen.height + h, r, g, 0);
            // To make more circular looking marks.
            screen.setPixel(pop[i].x * screen.width + 3, pop[i].y * screen.height, r, g, 0);
            screen.setPixel(pop[i].x * screen.width + 3, pop[i].y * screen.height - 1, r, g, 0);
            screen.setPixel(pop[i].x * screen.width + 3, pop[i].y * screen.height + 1, r, g, 0);
            screen.setPixel(pop[i].x * screen.width - 3, pop[i].y * screen.height, r, g, 0);
            screen.setPixel(pop[i].x * screen.width - 3, pop[i].y * screen.height - 1, r, g, 0);
            screen.setPixel(pop[i].x * screen.width - 3, pop[i].y * screen.height + 1, r, g, 0);
            screen.setPixel(pop[i].x * screen.width, pop[i].y * screen.height + 3, r, g, 0);
            screen.setPixel(pop[i].x * screen.width - 1, pop[i].y * screen.height + 3, r, g, 0);
            screen.setPixel(pop[i].x * screen.width + 1, pop[i].y * screen.height + 3, r, g, 0);
            screen.setPixel(pop[i].x * screen.width, pop[i].y * screen.height - 3, r, g, 0);
            screen.setPixel(pop[i].x * screen.width - 1, pop[i].y * screen.height - 3, r, g, 0);
            screen.setPixel(pop[i].x * screen.width + 1, pop[i].y * screen.height - 3, r, g, 0);
        }
    }
}

inline void Individual::calcInteractionForce(int index){
    /* Calculate the interaction strength between
    * this individual and all others.
    * This is the time intensive part. */
    int popSize = indworld->pop.size();
    forceVector.resize(popSize);
    force = 0;
    forceVector[index] = 0;
    for (int i = index + 1; i < popSize; ++i) {
        double dist = sqrt((x - indworld->pop[i].x) * (x - indworld->pop[i].x) +
                           (y - indworld->pop[i].y) * (y - indworld->pop[i].y));
        #ifdef GAUSSIAN_INTERACTION
            // TODO: IMPLEMENT GAUSSIAN_INTERACTION INTERACTION.
            // Interaction strength as a function of distance with a normal curve distribution.
            if (dist < INTERACTION_DISTANCE * 4) {
                forceVector[i] = normal(dist);
                indworld->pop[i].forceVector[index] = normal(dist);
            }
            else {
                forceVector[i] = 0;
                indworld->pop[i].forceVector[index] = 0;
            }
        #else
            if (dist < INTERACTION_DISTANCE) {
                forceVector[i] = 1;
                indworld->pop[i].forceVector[index] = 1;
            }
            else {
                forceVector[i] = 0;
                indworld->pop[i].forceVector[index] = 0;
            }
        #endif
    }
}

inline void Individual::move() {
    // Toroidal boundary condition.
    #ifdef TORROIDAL_BOUNDARY
        x = fmod(x, (((double) rand() / static_cast<double>(RAND_MAX)) - 0.5) / 50 * IND_SPEED);
        y = fmod(y, (((double) rand() / static_cast<double>(RAND_MAX)) - 0.5) / 50 * IND_SPEED);
    #else
        x += (((double) rand() / static_cast<double>(RAND_MAX)) - 0.5) / 50 * IND_SPEED;
        if (x > 1)
            x = 2 - x;
        if (x < 0)
            x = 0 - x;
        y += (((double) rand() / static_cast<double>(RAND_MAX)) - 0.5) / 50 * IND_SPEED;
        if (y > 1)
            y = 2 - y;
        if (y < 0)
            y = 0 - y;
    #endif
}

int main() {
    // VTK Setup:
    int imageHeight = 1080, imageWidth = 1080;
    vtkImageData *image = NewImage(imageHeight, imageWidth);
    unsigned char *buffer = (unsigned char *) image->GetScalarPointer(0, 0, 0);
    Screen screen(imageHeight, imageWidth);
    screen.buffer = buffer;

    // Program specific:
    srand(time(NULL));
    int starting_population = START_POPULATION_SIZE;
    int num_steps = NUMBER_OF_GENERATION_STEPS;
    auto startZero = high_resolution_clock::now();
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    double dur;
    World world(starting_population);
    for (int i = 0; i < num_steps; ++i) {
        start = high_resolution_clock::now();
        screen.clear();
        world.worldStep();
        world.worldDraw(screen);
        char str [16];
        sprintf(str, "frame%03d" , i);
        WriteImage(image, str);
        cerr << "Done writing " << str << "." << endl;

        stop = high_resolution_clock::now();
        dur = duration_cast<microseconds>(stop - start).count();
        cout << "Step time: " << dur/1000000 << endl;
    }
    cerr << "Program finished!" << endl;
    dur = duration_cast<microseconds>(stop - startZero).count();
    cout << "Total run time: " << dur/1000000 << endl;
    return EXIT_SUCCESS;
}
