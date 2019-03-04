/* Author: Sam Chamoer 
 * TODO: Switch to std::reduce
 * TODO: switch to random generator
 * TODO: prune dead.
 * TODO: Offspring.
 */

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
	#define M_PI					3.14159265358979323846  
#endif

#define START_POPULATION_SIZE		1000
#define INTERACTION_DISTANCE		0.05
#define NUMBER_OF_GENERATION_STEPS	10
#define IND_SPEED					2   // Speed as a percentage of the world.
// #define TORROIDAL_BOUNDARY
//#define GAUSSIAN_INTERACTION

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// For random generation.
// TODO: Switch to this from rand.
static std::random_device __randomDevice;
static std::mt19937 __randomGen(__randomDevice());
#ifdef GAUSSIAN_INTERACTION
float COMPETITION_THRESHOLD = START_POPULATION_SIZE * M_PI *  INTERACTION_DISTANCE * INTERACTION_DISTANCE * 1.2;
#else
float COMPETITION_THRESHOLD = START_POPULATION_SIZE * M_PI *  INTERACTION_DISTANCE * INTERACTION_DISTANCE;
#endif // GAUSSIAN_INTERACTION


class World;

class Individual {
    public:
        int hp = 1000;
        int age;
        float x, y;
        World *indworld;
        inline void move();
};

class World {
    public:
        World(int populationSize);
		int popSize;
        vector<Individual> pop;
        void worldStep();
        void worldDraw(Screen screen);
		vector<vector <float> > forceMatrix;
};

World::World(int populationSize) {
	this->popSize = populationSize;
    pop.resize(popSize);
    for (int i = 0; i < popSize; ++i) {
        pop[i].x = (float) rand() / static_cast<float>(RAND_MAX);
        pop[i].y = (float) rand() / static_cast<float>(RAND_MAX);
        pop[i].age = 10;
        pop[i].indworld = this;
    }
}

void World::worldStep() {
	forceMatrix.resize(popSize, vector <float>(popSize));
    // Population moves, ages.
    for (int i = 0; i < popSize; ++i) {
        pop[i].move();
        // Reduce hp by a flat amount per time step.
        pop[i].hp -= 20;
        // Increment age.
        pop[i].age++;
    }
    // Calculate competion:
    for (int i = 0; i < popSize; ++i) {
		// Individual does not compete with itself.
		forceMatrix[i][i] = 0;
		// Calculate the interaction strength between this individual and all others.
		// This is the time intensive part.
		for (int j = i + 1; j < popSize; ++j) {
			float dist = sqrt((pop[i].x - pop[j].x) * (pop[i].x - pop[j].x) + (pop[i].y - pop[j].y) * (pop[i].y - pop[j].y));
#ifdef GAUSSIAN_INTERACTION
			// Interaction strength as a function of distance on a normal curve.
			if (dist < INTERACTION_DISTANCE * 4) {
				forceMatrix[i][j] = exp(- (dist * dist) / (2 * INTERACTION_DISTANCE * INTERACTION_DISTANCE));
				forceMatrix[j][i] = forceMatrix[i][j];
			}
			else {
				forceMatrix[i][j] = 0;
				forceMatrix[j][i] = 0;
			}
#else
			if (dist < INTERACTION_DISTANCE) {
				forceMatrix[i][j] = 1;
				forceMatrix[j][i] = 1;
			}
			else {
				forceMatrix[i][j] = 0;
				forceMatrix[j][i] = 0;
			}
#endif
		}
	}
    // Apply competition:
    for (int i = 0; i < popSize; ++i) {
		double force = std::reduce(std::execution::par, forceMatrix[i].begin(), forceMatrix[i].end());
		if (force > COMPETITION_THRESHOLD)
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

inline void Individual::move() {
    // Toroidal boundary condition.
    #ifdef TORROIDAL_BOUNDARY
        x = fmod(x, (((float) rand() / static_cast<float>(RAND_MAX)) - 0.5) / 50 * IND_SPEED);
        y = fmod(y, (((float) rand() / static_cast<float>(RAND_MAX)) - 0.5) / 50 * IND_SPEED);
    #else
        x += (((float) rand() / static_cast<float>(RAND_MAX)) - 0.5) / 50 * IND_SPEED;
        if (x > 1)
            x = 2 - x;
        if (x < 0)
            x = 0 - x;
        y += (((float) rand() / static_cast<float>(RAND_MAX)) - 0.5) / 50 * IND_SPEED;
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
    auto startZero = high_resolution_clock::now();
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    double dur;
    World world(START_POPULATION_SIZE);
    for (int i = 0; i < NUMBER_OF_GENERATION_STEPS; ++i) {
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
	cin.get();
    return EXIT_SUCCESS;
}
