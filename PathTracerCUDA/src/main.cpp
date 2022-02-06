#pragma once

#include <iostream>
#include <chrono>


//#include <cuda_gl_interop.h>

#include "utility.h"
#include "image.h"
#include "integrator.h"

// Choose scene to render:
//#define SIMPLE_SCENE
#define COMPLEX_SCENE

// Choose rendering mode:
#define INTERACTIVE
//#define STATIC

// Rendered image properties
const auto aspectRatio = 16.0f / 9.0f;
const int imageWidth = 1920;
const int imageHeight = static_cast<int>(imageWidth / aspectRatio);
const int samplesPerPixel = 10;

// Threads per block
const int blockWidth = 8;
const int blockHeight = 8;

// Camera stuff
Camera** d_camera;
bool moved = false;
float lastX = imageWidth / 2.0f;
float lastY = imageHeight / 2.0f;

#define CheckCudaErrors(val) CheckCUDA( (val), #val, __FILE__, __LINE__ )


/// <summary>
/// Check CUDA error
/// </summary>
void CheckCUDA(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA ERROR = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

/// <summary>
/// Initialize single curandState
/// </summary>
/// <param name="rand_state">cuRAND state</param>
__global__ void CUrandInit(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

/// <summary>
/// Initialize curandState for each pixel (thread)
/// </summary>
/// <param name="maxX">image width</param>
/// <param name="maxY">image hight</param>
/// <param name="rand_state">cuRAND state</param>
__global__ void RenderInit(int maxX, int maxY, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY)) return;
	int pixel_index = j * maxX + i;
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

#define CURANDOM (curand_uniform(&local_rand_state))

#ifdef SIMPLE_SCENE
__global__ void initMaterialsAndCamera(Material** d_materials, Camera** d_camera, float aspectRatio, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;


		d_materials[0] = new Diffuse(Vector3(0.6, 0.6, 0.6));
		d_materials[1] = new Diffuse(Vector3(4.0, 0.0, 0.0));
		d_materials[2] = new Metal(Vector3(0.4, 0.4, 0.4), Vector3(0.5, 0.5, 0.5), 1000.0f, 0.4);
		d_materials[3] = new Specular(Vector3(0.4, 0.4, 0.4), Vector3(0.5, 0.5, 0.5), 10000.0);
		d_materials[4] = new DiffuseLight(Vector3(8.8));

		Vector3 lookfrom(0, 3, -6);
		Vector3 lookat(0, 1, 0);
		*d_camera = new Camera(lookfrom,
			lookat,
			Vector3(0, 1, 0),
			30.0,
			aspectRatio);
	}
}
#endif // SIMPLE_SCENE

#ifdef COMPLEX_SCENE
__global__ void initMaterialsAndCamera(Material** d_materials, Camera** d_camera, float aspectRatio, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		for (size_t i = 0; i < 100; i++)
		{
			float roulette = CURANDOM;
			if (roulette < 0.05f) {
				d_materials[i] = new DiffuseLight(Vector3(fmaxf(0.2f, CURANDOM), fmaxf(0.2f, CURANDOM), fmaxf(0.2f, CURANDOM)));
			}
			else if (roulette < 0.35f) {
				d_materials[i] = new Diffuse(Vector3(fmaxf(0.2f, CURANDOM), fmaxf(0.2f, CURANDOM), fmaxf(0.2f, CURANDOM)));
			}
			else if (roulette < 0.8) {
				d_materials[i] = new Specular(Vector3(fmaxf(0.2f, CURANDOM), fmaxf(0.2f, CURANDOM), fmaxf(0.2f, CURANDOM)), Vector3(CURANDOM, CURANDOM, CURANDOM), CURANDOM * 5000.0f);
			}
			else {
				d_materials[i] = new Metal(Vector3(0.1, 0.1, 0.1), Vector3(0.8, 0.8, 0.8), 1000.0f, CURANDOM);
			}
		}


		Vector3 lookfrom(15, 4, 6);
		//Vector3 lookfrom(-70, 70, -70);
		//Vector3 lookfrom(-40, 40, -40);
		Vector3 lookat(0, 0, 0);
		*d_camera = new Camera(lookfrom,
			lookat,
			Vector3(0, 1, 0),
			30.0,
			aspectRatio);
	}
}
#endif // COMPLEX_SCENE

void CreateWorld(Sphere* d_list, BVHNode* d_nodes, int nPrimitives, int nMaterials) {
	std::vector<Sphere> list;
	list.reserve(nPrimitives);
#ifdef SIMPLE_SCENE
	list.push_back(Sphere(Vector3(0, -1000, 0), 1000.0f, 0));
	list.push_back(Sphere(Vector3(0, 0.8f, 1.73), 0.8f, 1));
	list.push_back(Sphere(Vector3(-1, 0.8f, 0), 0.8f, 2));
	list.push_back(Sphere(Vector3(1, 0.8f, 0), 0.8f, 3));
	list.push_back(Sphere(Vector3(0, 2.0f, 0.6), 0.4f, 4));
#endif // SIMPLE_SCENE

#ifdef COMPLEX_SCENE
	//for (int x = -20; x < 20; x++) {
	//	for (int y = -20; y < 20; y++) {
	//		for (int z = -20; z < 20; z++)
	for (int x = -10; x < 10; x++) {
		for (int y = -10; y < 10; y++) {
			for (int z = -10; z < 10; z++)
			{
				int chosenMat = randFloat() * nMaterials;
				Vector3 center(x + randFloat(), y + randFloat(), z + randFloat());
				list.push_back(Sphere(center, 0.2, chosenMat));
			}

		}
	}
#endif // COMPLEX_SCENE

	std::vector<BVHNode> nodes;
	nodes.resize(2 * nPrimitives);

	int indexNode = 1;
	int* idPtr = &indexNode;
	nodes[0] = BVHNode(list, 0, list.size() - 1, nodes, idPtr, 0);

	CheckCudaErrors(cudaMemcpy(d_list, list.data(), list.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
	CheckCudaErrors(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
}

/// <summary>
/// Initialize environment map
/// </summary>
__global__ void initEnvMap(EnvironmentMap** d_envmap, Color3* envMapData, int width, int height) {
	*d_envmap = new EnvironmentMap(envMapData, width, height);
}


/// <summary>
/// Free materials and camera on device.
/// </summary>
__global__ void freeMaterialsAndCamera(Material** d_materials, Camera** d_camera, int nMaterials) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < nMaterials; i++) {
			delete d_materials[i];
		}
		delete* d_camera;
	}
}

/// <summary>
/// Create output framebuffer for GLFW window from color buffer
/// </summary>
__global__ void createOutputFrameBuffer(Vector3* colorBuffer, unsigned char* frameBuffer, int maxX, int maxY, int samplesPerPixel) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = maxX * j + i;

	float r = sqrt(colorBuffer[pixelIndex].x / float(samplesPerPixel));
	float g = sqrt(colorBuffer[pixelIndex].y / float(samplesPerPixel));
	float b = sqrt(colorBuffer[pixelIndex].z / float(samplesPerPixel));

	r = __saturatef(r);
	g = __saturatef(g);
	b = __saturatef(b);

	frameBuffer[j * 3 * maxX + i * 3 + 0] = static_cast<unsigned char>(255.99 * r);
	frameBuffer[j * 3 * maxX + i * 3 + 1] = static_cast<unsigned char>(255.99 * g);
	frameBuffer[j * 3 * maxX + i * 3 + 2] = static_cast<unsigned char>(255.99 * b);
}

__global__ void clearBuffer(Vector3* colorBuffer, int maxX, int maxY) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = maxX * j + i;
	colorBuffer[pixelIndex].x = 0;
	colorBuffer[pixelIndex].y = 0;
	colorBuffer[pixelIndex].z = 0;
}

void processInput(GLFWwindow* window, Camera** d_camera)
{
	float cameraSpeed = 0.1f;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		moveCameraFront << <1, 1 >> > (d_camera, cameraSpeed);
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		moveCameraBack << <1, 1 >> > (d_camera, cameraSpeed);
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		moveCameraLeft << <1, 1 >> > (d_camera, cameraSpeed);
		moved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		moveCameraRight << <1, 1 >> > (d_camera, cameraSpeed);
		moved = true;
	}
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;
	if (xoffset != 0.0f || yoffset != 0.0f) moved = true;

	float sensitivity = 0.001f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity * 100;

	mouseMovement << <1, 1 >> > (d_camera, xoffset, yoffset);
}

int main()
{

	// Initialize the glfw library
	if (!glfwInit())
		return -1;

	// Create a windowed mode window and its OpenGL context
	GLFWwindow* window = glfwCreateWindow(imageWidth, imageHeight, "RT", nullptr, nullptr);
	if (!window)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// GLEW init
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		glfwTerminate();
		return -1;
	}

	// Tell GLFW to capture our mouse
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPos(window, imageWidth / 2, imageHeight / 2);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);


	std::cout << "Rendering a " << imageWidth << "x" << imageHeight << " image ";
	std::cout << "in " << blockWidth << "x" << blockHeight << " blocks.\n";

	int nPixels = imageWidth * imageHeight;
	int fbSize = nPixels * sizeof(Vector3);

	// Allocate frame buffer
	Vector3* frameBuffer;
	CheckCudaErrors(cudaMalloc((void**)&frameBuffer, fbSize));

	// Create random states
	curandState* d_rand_state;
	CheckCudaErrors(cudaMalloc((void**)&d_rand_state, nPixels * sizeof(curandState)));
	curandState* d_rand_state2;
	CheckCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));    
	CUrandInit << <1, 1 >> > (d_rand_state2);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaDeviceSynchronize());

	// Create world
#ifdef SIMPLE_SCENE
	int nPrimitives = 5;
	int nMaterials = 5;
#endif // SIMPLE_SCENE

#ifdef COMPLEX_SCENE
	//int nPrimitives = 40 * 40 * 40;
	int nPrimitives = 20 * 20 * 20;
	int nMaterials = 100;
#endif // COMPLEX_SCENE

	// Load environment map
	Image img("../Resources/EnvironmentMaps/rotunda.hdr");
	//Image img("../Resources/EnvironmentMaps/night.hdr");
	EnvironmentMap** d_envmap;
	CheckCudaErrors(cudaMalloc((void**)&d_envmap, sizeof(EnvironmentMap*)));

	// Create image buffer for environment map on GPU
	Color3* d_image_buffer;
	CheckCudaErrors(cudaMalloc((void**)&d_image_buffer, img.width * img.height * sizeof(Color3)));
	CheckCudaErrors(cudaMemcpy(d_image_buffer, img.data, img.width * img.height * sizeof(Color3), cudaMemcpyHostToDevice));
	initEnvMap << <1, 1 >> > (d_envmap, d_image_buffer, img.width, img.height);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaDeviceSynchronize());

	// Create buffer for on device
	Material** d_materials;
	CheckCudaErrors(cudaMalloc((void**)&d_materials, nMaterials * sizeof(Material*)));

	// Create buffer for primitives on device
	Sphere* d_list;
	CheckCudaErrors(cudaMalloc((void**)&d_list, nPrimitives * sizeof(Sphere)));
	// Create buffer for BVH nodes on device
	BVHNode* d_nodes;
	CheckCudaErrors(cudaMalloc((void**)&d_nodes, 2 * nPrimitives * sizeof(BVHNode)));

	// Construct BVH and fill buffers with primitives
	CreateWorld(d_list, d_nodes, nPrimitives, nMaterials);

	// Initialize camera and materials
	CheckCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
	initMaterialsAndCamera << <1, 1 >> > (d_materials, d_camera, aspectRatio, d_rand_state2);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaDeviceSynchronize());

	// Initialize rendering
	dim3 blocks(imageWidth / blockWidth + 1, imageHeight / blockHeight + 1);
	dim3 threads(blockWidth, blockHeight);
	RenderInit << <blocks, threads >> > (imageWidth, imageHeight, d_rand_state);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaDeviceSynchronize());

	// Initialize image buffer for window
	//unsigned char* pix = new unsigned char[3 * imageWidth * imageHeight];
	//unsigned char* d_pix;
	//CheckCudaErrors(cudaMalloc((void**)&d_pix, 3 * imageWidth * imageHeight * sizeof(unsigned char)));

	GLuint bufferObj;
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 3 * sizeof(unsigned char), NULL, GL_DYNAMIC_DRAW);

	cudaGraphicsResource* resource;
	unsigned char* devPtr; // ukazatel na data PBO v CUDA, uchar4 má položky nazvané x, y, z a w
	size_t size;



#ifdef INTERACTIVE

	int sample = 1;

	while (sample <= samplesPerPixel) {

		moved = false;
		auto start = std::chrono::high_resolution_clock::now();

		Render << <blocks, threads >> > (
			frameBuffer, 
			imageWidth, 
			imageHeight, 
			1,
			d_camera,
			d_list,
			d_nodes,
			d_materials,
			d_envmap,
			d_rand_state);
		CheckCudaErrors(cudaGetLastError());
		CheckCudaErrors(cudaDeviceSynchronize());

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		std::cout << "#" << sample << " - " << duration.count() << " [ms]" << std::endl;


		cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
		cudaGraphicsMapResources(1, &resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

		// Output FB as Image
		createOutputFrameBuffer << <blocks, threads >> > (frameBuffer, devPtr, imageWidth, imageHeight, sample);
		CheckCudaErrors(cudaGetLastError());
		CheckCudaErrors(cudaDeviceSynchronize());

		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
		processInput(window, d_camera);

		sample++;

		if (moved) {
			sample = 1;
			clearBuffer << <blocks, threads >> > (frameBuffer, imageWidth, imageHeight);
			CheckCudaErrors(cudaGetLastError());
			CheckCudaErrors(cudaDeviceSynchronize());
		}

		if (glfwWindowShouldClose(window)) break;
	}
	while (!glfwWindowShouldClose(window)) {

		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
#endif // INTERACTIVE

#ifdef STATIC
	auto start = std::chrono::high_resolution_clock::now();

	Render << <blocks, threads >> > (
		frameBuffer,
		imageWidth,
		imageHeight,
		samplesPerPixel,
		d_camera,
		d_list,
		d_nodes,
		d_materials,
		d_envmap,
		d_rand_state);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaDeviceSynchronize());

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Rendering took " << duration.count() << " [ms]." << std::endl;

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	// Output FB as Image
	createOutputFrameBuffer << <blocks, threads >> > (frameBuffer, devPtr, imageWidth, imageHeight, samplesPerPixel);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaDeviceSynchronize());


	while (!glfwWindowShouldClose(window)) {
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
#endif // STATIC

	// Clean up
	cudaGraphicsUnmapResources(1, &resource, NULL);
	cudaGraphicsUnregisterResource(resource);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &bufferObj);
	glfwTerminate();
	freeMaterialsAndCamera << <1, 1 >> > (d_materials, d_camera, nMaterials);
	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaFree(d_camera));
	CheckCudaErrors(cudaFree(d_list));
	CheckCudaErrors(cudaFree(d_nodes));
	CheckCudaErrors(cudaFree(d_rand_state));
	CheckCudaErrors(cudaFree(d_rand_state2));
	CheckCudaErrors(cudaFree(frameBuffer));
	CheckCudaErrors(cudaFree(d_image_buffer));

	cudaDeviceReset();

}