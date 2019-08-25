
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include "device_launch_parameters.h"
#include "openglcuda.cuh"
#include <time.h>

int numElementsRand = 10, numElementsMat = 100, numElementsBestCost = 100;
int sizeRand = numElementsMat * sizeof(int);
int sizeMat = numElementsMat * sizeof(int);
int sizeBestCost = numElementsBestCost * sizeof(int);

