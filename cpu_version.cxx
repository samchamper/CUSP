#include <iostream>
#include <vtkPNGWriter.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <math.h>
#include <ctime>
#include <stdlib.h>
#include <chrono>
#include <vector>

#define START_POPULATION_SIZE 1000
#define NUMBER_OF_GENERATION_STEPS 50
#define INTERACTION_DISTANCE 0.08
#define IND_SPEED 2   // Speed as a percentage of the world.
//#define TORROIDAL_BOUNDARY
//#define NORMAL_INTERACTION

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

class Screen {
    public:
        Screen(int width, int height);
        unsigned char       *buffer;
        int                 width, height, numPixels;
        void setPixel(int x, int y, double r, double g, double b);
        void clear();
};

vtkImageData *NewImage(int width, int height) {
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    return img;
}

void WriteImage(vtkImageData *img, const char *filename) {
    std::string full_filename = filename;
    full_filename += ".png";
    vtkPNGWriter *writer = vtkPNGWriter::New();
    writer->SetInputData(img);
    writer->SetFileName(full_filename.c_str());
    writer->Write();
    writer->Delete();
}

Screen::Screen(int width, int height) {
    /* Constructor for screen. Sets dimensions, allocates an array for the z-buffer init to -1s),
     * and sets up a transformation matrix to map to screen coords. */
    this->width = width;
    this->height = height;
    this->numPixels = width * height;
}

void Screen::setPixel(int x, int y, double r, double g, double b) {
    /* Sets the color value of a pixel with coords (x,y,z). */
    // Skip pixels that are out of bounds:
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;
    // If the z is already greater at write target, do nothing.
    int pixelIndex = x + width * y;
    // Each pixel has an r, g, and b value - write them to the image.
    int pixelStart = 3 * pixelIndex;
    buffer[pixelStart] = ceil(r * 255);
    buffer[pixelStart + 1] = ceil(g * 255);
    buffer[pixelStart + 2] = ceil(b * 255);
}

void Screen::clear() {
    int i = 0;
    for (int j = 0; j < numPixels; ++j) {
        buffer[i++] = 0;
        buffer[i++] = 0;
        buffer[i++] = 0;  // Loop unrolling is my favorite.
    }
}

class World;
class Individual {
    public:
        void step(int ind_num);
        double x, y;
        World *indworld;
        int hp = 1000;
        int age;
    private:
        inline void move();
        inline void calcInteractionForce();
        inline void forceFunction(double distance);
        double force;
};

class World {
    public:
        World(int popsize);
        void worldStep();
        vector<Individual> pop;
        void worldDraw(Screen screen);
};

World::World(int popSize) {
    pop.resize(popSize);
    for (int i = 0; i < popSize; ++i) {
        pop[i].x = (double) rand() / static_cast<double>(RAND_MAX);
        pop[i].y = (double) rand() / static_cast<double>(RAND_MAX);
        pop[i].indworld = this;
    }
}

void World::worldStep() {
    int popSize = pop.size();
    for (int i = 0; i < popSize; ++i)
        pop[i].step(i);
}

inline double clamp(double val, double lo, double hi) {
    return val < lo ? lo : val > hi ? hi : val; 
}

void World::worldDraw(Screen screen) {
    int popSize = pop.size();
    for (int i = 0; i < popSize; ++i) {
        if (pop[i].hp > 0) {
            double r = clamp((1000 - (double) pop[i].hp) / 1000 * 1.1 - 0.1, 0, 1);
            double g = clamp((double) pop[i].hp / 1000 * 1.1 - 0.1, 0, 1);
            for (int w = -2; w <= 2; ++w) {
                for (int h = -2; h <= 2; ++h) {
                    screen.setPixel(pop[i].x * screen.width + w, pop[i].y * screen.height + h, r, g, 0);
                }
            }
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

void Individual::step(int ind_num) {
    /*What happens during a step?
    What should be doable in parrallel:
        1. Calculate movement.
        2. Calculate density force.
            a. Calculate distances.
            b. Calculate force as f(distances)
        3. Calculate death.
        4. Calculate offspring.
    Serial:
        1. Prune dead from vector.
        2. Add offspring to vector. */
    age++;
    move();
    calcInteractionForce();
    hp -= force;
    // TODO: add death, offspring, dependent on density interaction force.
}

inline void Individual::calcInteractionForce(){
    /* Calculate the interaction strength between
    * this individual and all others.
    * This is the time intensive part. */
    force = 0;
    int popSize = indworld->pop.size();
    for (int i = 0; i < popSize; ++i) {
        // Not bothering with a boolean to prevent individuals from interacting with itself.
        // Checking a boolean hundreds of thousands of times takes more time than just
        // factoring the interaction out of the sum later.
        double dist = sqrt((x - indworld->pop[i].x) * (x - indworld->pop[i].x) + (y - indworld->pop[i].y) * (y - indworld->pop[i].y));
        forceFunction(dist);
    }
}

inline void Individual::forceFunction(double dist) {
    /* Use ifdefs for varying interaction types. */
#ifdef NORMAL_INTERACTION
    // TODO: IMPLEMENT NORMAL INTERACTION.
    // Interaction on a normal curve.
    if (dist < INTERACTION_DISTANCE)
        force += 1;
#else
    if (dist < INTERACTION_DISTANCE)
        force += 1;
#endif
}

inline void Individual::move() {
    // Torroidal boundary condition.
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
    int starting_population = START_POPULATION_SIZE;
    int num_steps = NUMBER_OF_GENERATION_STEPS;
    srand(time(NULL));
    auto startZero = high_resolution_clock::now();
    auto start = high_resolution_clock::now();
    World world(starting_population);
    auto stop = high_resolution_clock::now();
    double dur = duration_cast<microseconds>(stop - start).count();
    cout << "Setup time: " << dur/1000000 << endl;
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
