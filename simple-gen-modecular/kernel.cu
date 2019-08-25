
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

//////////////////////////////////////////////////////////////////////////
// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

#include <stdio.h>
#include "ModecularSurfaceGenerator.h"
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
	loadPdbfile("E:/WORK/2019_08_24/PDB_files/1IZH.pdb");
	calcuate_model();
	opengl_run(argc, argv);


    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
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
*/

#include "openglcuda.cuh"

//////////////////////////////////////////////////////////////////////////
#include "CommonPara.h"
#include "ParsePDB.h"
#include "ModecularSurface.h"
#include <time.h>




//////////////////////////////////////////////////////////////////////////
#define WIDTH	800
#define HEIGHT	600
int screen_width = WIDTH;
int screen_height = HEIGHT;
enum ButtonState {
	UP = 1,
	DOWN
};

// State of mouse wheel
enum WheelState {
	WHEEL_UP = 3,
	WHEEL_DOWN = 4
};

ParsePDB pp;
ModecularSurface msf;


static int g_x = 0;
static int g_y = 0;
static int g_z = -80;

static float g_xDiff = 0.0f;
static float g_yDiff = 0.0f;

// mouse state
static int g_mouseState = UP;
// the minimized x/y/z coordinate of all grid
static double g_gridMinXcoord = 0.0f;
static double g_gridMinYcoord = 0.0f;
static double g_gridMinZcoord = 0.0f;
// the maximized x/y/z coordinate of all grid
static double g_gridMaxXcoord = 0.0f;
static double g_gridMaxYcoord = 0.0f;
static double g_gridMaxZcoord = 0.0f;

static int g_numberOfGrid = 60;
double g_iosSurfaceValue = 1.4;


int g_nsurfMode = 2; //1-MC 2-VCMC
int g_ncompute_mode = 4; //1-VWS 2-SAS 3-MS 4-SES
int g_ncolor = 2;//1-pure 2-atom 3-chain
int g_ninout = 1;//1-in and out 2-out 3-in
double g_dblradius = 0.2;//probe radius
double g_dblscale = 2.00;//scale factor

// whether to show atoms
static bool g_showAtoms = false;
// whether to show points
static bool g_showPoints = false;
// whether to show bounding box
static bool g_showBoundingBox = true;
// whether to show triangles
static bool g_showTriangles = true;
// whether to show cuboids
static bool g_showCuboids = false;
// whether to show surfaces
static bool g_showSurface = false;
// whether to show light
static bool g_showLight = false;

static GLfloat g_rotate[] = { 0.0, 0.0, 0.0 }; // Rotation (X,Y,Z)

#define PI       3.14159265358979323846   // pi
//////////////////////////////////////////////////////////////////////////

void drawAtom();
void find_pdbfile(char* pdbfile) {

}
void drawSamplePoints()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	glPointSize(3);
	glColor3f(0.0f, 1.0f, 0.0f);

}

// Function that handles the drawing of bounding box
void drawCube(int cubeIndex, GLfloat minX, GLfloat maxX, GLfloat minY, GLfloat maxY, GLfloat minZ, GLfloat maxZ, GLfloat r, GLfloat g, GLfloat b)
{
	glBegin(GL_LINE_LOOP);

	//glColor3f(r, g, b);

	if (cubeIndex & (1 << 0))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, minY, minZ);	// 0

	if (cubeIndex & (1 << 1))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, minY, minZ);	// 1

	if (cubeIndex & (1 << 2))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, maxY, minZ);	// 2

	if (cubeIndex & (1 << 3))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, maxY, minZ);	// 3

	if (cubeIndex & (1 << 0))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, minY, minZ);	// 0

	if (cubeIndex & (1 << 4))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, minY, maxZ);	// 4

	if (cubeIndex & (1 << 7))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, maxY, maxZ);	// 7

	if (cubeIndex & (1 << 3))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, maxY, minZ);	// 3

	if (cubeIndex & (1 << 3))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, maxY, minZ);	// 3

	if (cubeIndex & (1 << 2))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, maxY, minZ);	// 2

	if (cubeIndex & (1 << 6))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, maxY, maxZ);	// 6

	if (cubeIndex & (1 << 7))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, maxY, maxZ);	// 7


	glVertex3f(minX, maxY, maxZ);	// 7
	if (cubeIndex & (1 << 4))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, minY, maxZ);	// 4

	if (cubeIndex & (1 << 5))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, minY, maxZ);	// 5

	if (cubeIndex & (1 << 6))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, maxY, maxZ);	// 6

	glVertex3f(maxX, maxY, maxZ);	// 6
	if (cubeIndex & (1 << 2))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, maxY, minZ);	// 2
	if (cubeIndex & (1 << 1))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, minY, minZ);	// 1

	if (cubeIndex & (1 << 5))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, minY, maxZ);	// 5

	glVertex3f(maxX, minY, maxZ);	// 5
	if (cubeIndex & (1 << 4))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, minY, maxZ);	// 4
	if (cubeIndex & (1 << 0))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(minX, minY, minZ);	// 0
	if (cubeIndex & (1 << 1))
		glColor3f(r, g, b);
	else
		glColor3f(1.0, 0.0, 0.0);
	glVertex3f(maxX, minY, minZ);	// 1

	glEnd();
}

void drawBoundingBox()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	drawCube(0, g_gridMinXcoord, g_gridMaxXcoord, g_gridMinYcoord, g_gridMaxYcoord, g_gridMinZcoord, g_gridMaxZcoord, 1.0, 0.0, 0.0);
	//drawCube(0, msf.pmin.x, msf.pmax.x, msf.pmin.y, msf.pmax.y, msf.pmin.z, msf.pmax.z, 1.0, 0.0, 0.0);

}

void getMinMaxCoordinatesOfGrid()
{
	// There are no spheres, no need to calculate.
	if (pp.numproseq <= 0)
	{
		return;
	}

	// Set AtomMinXcoord/AtomMaxXcoord to x coordinate of center of the first sphere,
	// then let it compare to other sphere in for loop, finally get the min/max 
	// x coordinate among all spheres. Same rule apply to y/z axis.
	double AtomMinXcoord, AtomMinYcoord, AtomMinZcoord;
	double AtomMaxXcoord, AtomMaxYcoord, AtomMaxZcoord;
	double AtomMaxRadius;
	const atom firstSphere = pp.proseq[0];
	AtomMinXcoord = AtomMaxXcoord = firstSphere.x;
	AtomMinYcoord = AtomMaxYcoord = firstSphere.y;
	AtomMinZcoord = AtomMaxZcoord = firstSphere.z;
	AtomMaxRadius = firstSphere.radius;

	for (int i = 0; i < pp.numproseq; ++i)
	{
		// get min x
		if (pp.proseq[i].x < AtomMinXcoord)
		{
			AtomMinXcoord = pp.proseq[i].x;
		}
		// get max x
		else if (pp.proseq[i].x > AtomMaxXcoord)
		{
			AtomMaxXcoord = pp.proseq[i].x;
		}

		// get min y
		if (pp.proseq[i].y < AtomMinYcoord)
		{
			AtomMinYcoord = pp.proseq[i].y;
		}
		// get max y
		else if (pp.proseq[i].y > AtomMaxYcoord)
		{
			AtomMaxYcoord = pp.proseq[i].y;
		}

		// get min z
		if (pp.proseq[i].z < AtomMinZcoord)
		{
			AtomMinZcoord = pp.proseq[i].z;
		}
		// get max z
		else if (pp.proseq[i].z > AtomMaxZcoord)
		{
			AtomMaxZcoord = pp.proseq[i].z;
		}

		// get max radius
		if (pp.proseq[i].radius > AtomMaxRadius)
		{
			AtomMaxRadius = pp.proseq[i].radius;
		}

		//AtomMaxRadius = g_dblradius;

	}

	// Recenter molecular
	double xOffset = (AtomMaxXcoord + AtomMinXcoord) / 2;
	double yOffset = (AtomMaxYcoord + AtomMinYcoord) / 2;
	double zOffset = (AtomMaxZcoord + AtomMinZcoord) / 2;
	for (int i = 0; i < pp.numproseq; i++)
	{
		pp.proseq[i].x -= xOffset;
		pp.proseq[i].y -= yOffset;
		pp.proseq[i].z -= zOffset;
	}
	// Apply a delta for making sure the bounding of grid wrap all the atoms.
	double delta = 0.1;
	g_gridMinXcoord = AtomMinXcoord - xOffset - AtomMaxRadius - delta;
	g_gridMaxXcoord = AtomMaxXcoord - xOffset + AtomMaxRadius + delta;
	g_gridMinYcoord = AtomMinYcoord - yOffset - AtomMaxRadius - delta;
	g_gridMaxYcoord = AtomMaxYcoord - yOffset + AtomMaxRadius + delta;
	g_gridMinZcoord = AtomMinZcoord - zOffset - AtomMaxRadius - delta;
	g_gridMaxZcoord = AtomMaxZcoord - zOffset + AtomMaxRadius + delta;
}

void drawVoxel(const faceinfo& face, float sx, float sy, float sz)
{
	//Draw the triangles that were found.  There can be up to five per cube
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glBegin(GL_TRIANGLES);
	glColor3f(1.f, 1.f, 0.f);
	GLfloat x, y, z;

	x = g_gridMinXcoord + msf.verts[face.a].x * sx;
	y = g_gridMinYcoord + msf.verts[face.a].y * sy;
	z = g_gridMinZcoord + msf.verts[face.a].z * sz;
	glVertex3f(x, y, z);
	glColor3f(1.f, 1.f, 0.f);
	x = g_gridMinXcoord + msf.verts[face.b].x * sx;
	y = g_gridMinYcoord + msf.verts[face.b].y * sy;
	z = g_gridMinZcoord + msf.verts[face.b].z * sz;
	glVertex3f(x, y, z);
	glColor3f(1.f, 1.f, 0.f);
	x = g_gridMinXcoord + msf.verts[face.c].x * sx;
	y = g_gridMinYcoord + msf.verts[face.c].y * sy;
	z = g_gridMinZcoord + msf.verts[face.c].z * sz;
	glVertex3f(x, y, z);
	//glEnd();*/
}
__global__ void cudaDrawVoxel(faceinfo* facebuf, double* sx, double* sy, double* sz)
{
	int i = threadIdx.x;
	faceinfo vx = facebuf[i];
	printf("--------%d-----", i);
	//drawVoxel(vx, *sx, *sy, *sz);	
}


void drawTriangles()
{
	auto mode = g_showSurface ? GL_FILL : GL_LINE;
	glPolygonMode(GL_FRONT_AND_BACK, mode);
	glBegin(GL_TRIANGLES);
	glColor3f(1.0, 1.0, 0.0);

	double xScale = (g_gridMaxXcoord - g_gridMinXcoord) / msf.plength;
	double yScale = (g_gridMaxYcoord - g_gridMinYcoord) / msf.pwidth;
	double zScale = (g_gridMaxZcoord - g_gridMinZcoord) / msf.pheight;
	//msf.plength, msf.pwidth, msf.pheight, msf.scalefactor
#ifndef _CPU_
	for (int i = 0; i < msf.facenumber; ++i)
	{
		faceinfo vx = msf.faces[i];
		drawVoxel(vx, xScale, yScale, zScale);
	}
#else
	double *sx = 0, *sy = 0, *sz = 0;
	cudaError_t cudaStatus;
	faceinfo *facebuf = 0;
	cudaMalloc((void **)(&sx), sizeof(double));
	cudaMalloc((void **)(&sy), sizeof(double));
	cudaMalloc((void **)(&sz), sizeof(double));
	cudaMalloc((void **)(&facebuf), sizeof(faceinfo)*msf.facenumber);
	
	cudaMemcpy(sx, &xScale, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(sy, &yScale, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(sz, &zScale, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(facebuf, &msf.faces, sizeof(faceinfo)*msf.facenumber, cudaMemcpyHostToDevice);

	cudaDrawVoxel << <1, 1 >> > (facebuf, sx, sy, sz);
	cudaFree(facebuf);
	cudaFree(sx);
	cudaFree(sx);
	cudaFree(sx);
#endif
	glEnd();
}


void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glClear(GL_COLOR_BUFFER_BIT);   // Clear the color buffer with current clearing color

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLoadIdentity();

	glTranslatef(g_x, g_y, g_z);
	glRotatef(g_rotate[0], 1.0, 0.0, 0.0);
	glRotatef(g_rotate[1], 0.0, 1.0, 0.0);
	glRotatef(g_rotate[2], 0.0, 0.0, 1.0);

	if (g_showAtoms)
		drawAtom();

	//drawAxis();
	if (g_showBoundingBox)
		drawBoundingBox();

	//if (g_showCuboids)
		//drawCuboids();

	if (g_showTriangles)
		drawTriangles();


	if (g_showPoints)
		drawSamplePoints();

	glFlush();  // Render now
}
// Reshape() function    
void Reshape(int w, int h)
{
	//adjusts the pixel rectangle for drawing to be the entire new window    
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	//matrix specifies the projection transformation    
	glMatrixMode(GL_PROJECTION);
	// load the identity of matrix by clearing it.    
	glLoadIdentity();

	gluPerspective(g_numberOfGrid, (GLfloat)w / (GLfloat)h, 1.0, 200.0);
	//matrix specifies the modelview transformation    
	glMatrixMode(GL_MODELVIEW);
	// again  load the identity of matrix    
	glLoadIdentity();
}

void drawSphare(GLfloat x, GLfloat y, GLfloat z, GLfloat radius)
{
	glColor3f(1.0f, 0.0f, 1.0f);
	GLfloat alpha, beta; // Storage for coordinates and angles        
	int gradation = 10; //Number Of trangulation Trangulation from Sphare
	float X = x, Y = y, Z = z;
	for (alpha = 0.0; alpha < PI; alpha += PI / gradation)
	{
		glBegin(GL_TRIANGLE_STRIP);
		for (beta = 0.0; beta < 2.01*PI; beta += PI / gradation)
		{
			X = x + radius * cos(beta)*sin(alpha);
			Y = y + radius * sin(beta)*sin(alpha);
			Z = z + radius * cos(alpha);
			glVertex3f(X, Y, Z);
			X = x + radius * cos(beta)*sin(alpha + PI / gradation);
			Y = y + radius * sin(beta)*sin(alpha + PI / gradation);
			Z = z + radius * cos(alpha + PI / gradation);
			glVertex3f(X, Y, Z);
		}
		glEnd();
	}
}

void drawAtom()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	for (int i = 0; i < pp.numproseq; ++i)
	{
		atom vx = pp.proseq[i];
		drawSphare(vx.x, vx.y, vx.z, vx.radius);
	}
}

void mouseCallBack(int btn, int state, int x, int y)
{
	// Mouse left button or right button presses down may lead to rotation
	if (btn == GLUT_LEFT_BUTTON || btn == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			g_mouseState = DOWN;

			// When left or right button is pressed down, store the difference of
			// current position and previous rotation.
			g_xDiff = x - g_rotate[1];
			g_yDiff = -y + g_rotate[0];
		}
		else
		{
			g_mouseState = UP;
		}
	}
	else if (btn == WHEEL_UP) // zoom in
	{
		g_z++;
		glutPostRedisplay();
	}
	else if (btn == WHEEL_DOWN) // zoom out
	{
		g_z--;
		glutPostRedisplay();
	}
}

// Handler for mouse moving event
void onMouseMove(int x, int y)
{
	// only when the mouse is pressed down do the rotation
	if (g_mouseState == DOWN)
	{
		// Use the difference to calculate the finnal rotation value along x, y axes
		// Note that when mouse moves left and right means rotation around Y axes.
		// Mouse moves up and down means rotation around X axes
		g_rotate[1] = x - g_xDiff;
		g_rotate[0] = y + g_yDiff;

		glutPostRedisplay();
	}
}

void lighting(bool enable)
{
	if (enable)
	{
		GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
		GLfloat mat_shininess[] = { 50.0 };
		GLfloat light_position[] = { 30.0, 1.0, 0.0, 1.0 };
		glClearColor(0.0, 0.0, 0.0, 0.0);
		glShadeModel(GL_SMOOTH);

		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);

		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);
	}
	else
	{
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_LIGHTING);
	}
}


void keyboardCallback(unsigned char key, int, int)
{
	switch (key)
	{
		// increase the number of grid
	case '+':
		g_numberOfGrid += 10;

		//marchingCube();
		//msf.marchingcube(g_nsurfMode);
		glutPostRedisplay();
		break;
		// decrease the number of grid
	case '-':
		if (g_numberOfGrid > 20)
		{
			g_numberOfGrid -= 10;
			//msf.marchingcube(g_nsurfMode);
			glutPostRedisplay();
		}
		else
		{
			std::cout << "Can not decrease the number of grid any more" << std::endl;
		}
		break;

		// increase the ios surface value
	case 'a':
		g_iosSurfaceValue *= 1.1;
		//msf.marchingcube(g_nsurfMode);
		glutPostRedisplay();
		break;
		// decrease the ios surface value

	case 'm':
		g_iosSurfaceValue *= 0.9;
		//msf.marchingcube(g_nsurfMode);
		glutPostRedisplay();
		break;

		// toggle display of bouding box of grid
	case 'b':
	case 'B':
		g_showBoundingBox = !g_showBoundingBox;
		glutPostRedisplay();
		break;

		// toggle display of atoms
	case 'd':
	case 'D':
		g_showAtoms = !g_showAtoms;
		glutPostRedisplay();
		break;

		// toggle display of random sample points
	case 'p':
	case 'P':
		g_showPoints = !g_showPoints;
		glutPostRedisplay();
		break;

		// toggle display of triangles
	case 't':
	case 'T':
		g_showTriangles = !g_showTriangles;
		glutPostRedisplay();
		break;

		// toggle display of cuboids
	case 'c':
	case 'C':
		g_showCuboids = !g_showCuboids;
		glutPostRedisplay();
		break;

		// toggle display of surface
	case 's':
	case 'S':
		g_showSurface = !g_showSurface;
		glutPostRedisplay();
		break;

		// toggle display of lighting
	case 'l':
	case 'L':
		g_showLight = !g_showLight;
		lighting(g_showLight);
		glutPostRedisplay();
		break;
	case 'u':
	case 'U':
		//setCUDA();
		break;
	default:
		break;
	}
}

void loadPdbfile(char* filename) {

	pp.loadpdb(filename, 2);
	pp.extractbb(0, -1, 1);

	getMinMaxCoordinatesOfGrid();
}

GLvoid initCamera()
{
	// Set up the perspective matrix.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// FOV, AspectRatio, NearClip, FarClip
	gluPerspective(60.0f, (float)(screen_width) / screen_height, 1.0f, 1000.0f);

	// Set up the camera matrices.
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
GLvoid initTexture()
{
	glEnable(GL_TEXTURE_2D);
}

GLvoid initLights()
{
	// Define each color component.
	GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat diffuse[] = { 0.7f, 0.7f, 0.7f, 1.0f };
	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

	// Set each color component.
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

	// Define and set position.
	float lightPos[4] = { 0, 0, 20, 1 };
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

	// Turn on lighting.
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
}
GLvoid initMaterial()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

GLvoid initColors()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glColor3f(0.0, 0.0, 0.0);
	glLineWidth(1.0);
	glPointSize(5.0);
}

// Sets up OpenGL state.
GLvoid initGL()
{
	// Shading method: GL_SMOOTH or GL_FLAT
	glShadeModel(GL_SMOOTH);
	// Enable depth-buffer test.
	glEnable(GL_DEPTH_TEST);

	// Set the type of depth test.
	glDepthFunc(GL_LEQUAL);

	// 0 is near, 1 is far
	glClearDepth(1.0f);

	// Set camera settings.
	initCamera();

	// Set texture settings.
	initTexture();

	// Set lighting settings.
	initLights();

	// Set material settings.
	initMaterial();

	// Set color settings.
	initColors();
}

void calcuate_model() {
	
	if (g_nsurfMode == 1)
		msf.marchingcubeorigin2(2);
	else if (g_nsurfMode == 2)
		msf.marchingcube(2);

	clock_t remcfinish = clock();
	double duration = (double)(remcfinish - remcstart) / CLOCKS_PER_SEC;
	printf("Total time %.3f seconds\n", duration);
	//additional functions below
	msf.checkEuler();
	msf.computenorm();
	printf("No. vertices %d, No. triangles %d\n", msf.vertnumber, msf.facenumber);
	msf.calcareavolume();
	printf("Total area %.3f and volume %.3f\n", msf.sarea, msf.svolume);
	printf("Distinguish inner and outer surfaces\n");
	msf.surfaceinterior();
	printf("Calculate cavity number...\n");
	msf.cavitynumbers();
	printf("Cavity number is %d\n", msf.ncav);
	printf("Calculate cavity area and volume...\n");
	msf.cavitiesareavolume();
	printf("Cavity area %.3f and volume %.3f\n", msf.carea, msf.cvolume);
	printf("Calculate inner and outer atoms\n");
	msf.atomsinout(pp.promod[0].procha[0].chainseg.init, pp.promod[0].procha[pp.promod[0].nchain].chainseg.term, pp.proseq);
	msf.laplaciansmooth(1);
	msf.computenorm();
	msf.checkinoutpropa();

#ifdef _OUT_
	printf("Output 3D model\n");

	sprintf(filename, "%s.ply", outpname);
	msf.outputply(filename, pp.proseq, g_ncolor - 1, g_ninout - 1);
#endif

}
void opengl_run(int argc, char** argv) {
	glutInit(&argc, argv);          // Initialize GLUT
	//glutInitWindowSize(width, height);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH | GLUT_ALPHA);
	//glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);

	glutInitWindowSize(screen_width, screen_height);
	//glutInitWindowPosition(800, 600);
	glutCreateWindow("Molecular Surface SES");  // Create window with the given title

	// Initialize the scene.
	initGL();

	glutReshapeFunc(Reshape);
	glutDisplayFunc(display);       // Register callback handler for window re-paint event
	glutMouseFunc(mouseCallBack);
	glutKeyboardFunc(keyboardCallback);
	glutMotionFunc(onMouseMove);
	glutMainLoop();
}
void init_cuda(int argc, char** argv) {
	int dev = findCudaDevice(argc, (const char **)argv);
	printf("cuda device = %d", dev);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("%s Global Memory Size %d GM", deviceProp.name, deviceProp.totalGlobalMem / 1024 / 1024 / 1024);
	cudaSetDevice(dev);
}
int main(int argc, char** argv)
{
	init_cuda(argc, argv);
	char pdbfile[200] = "E:/WORK/2019_08_24/PDB_files/1bk2.pdb";
	char filename[200];

	bool bcolor;
	int i;
	clock_t remcstart, remcfinish;

	loadPdbfile(pdbfile);
	calcuate_model();

	opengl_run(argc, argv);

	return 1;
}

