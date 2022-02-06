#pragma once
#include "utility.h"
#include "vector3.h"
#include "ray.h"
#include "sphere.h"
#include "environment_map.h"
#include "material.h"
#include "bvh.h"
#include "camera.h"

/// <summary>
/// Traces rays recursively to converge to the correct aproximation of rendering equation.
/// </summary>
/// <param name="r">Ray</param>
/// <param name="list">List of hittable objects</param>
/// <param name="nodes">List of BVH nodes</param>
/// <param name="materials">List of materials</param>
/// <param name="envMap">Environment map</param>
/// <param name="localRandState">cuRAND state</param>
/// <returns>Incoming radiance</returns>
__device__ Color3 Trace(const Ray& r, Sphere* list, BVHNode* nodes, Material** materials, EnvironmentMap** envMap, curandState* localRandState) {

	Ray currRay = r;
	Color3 L(0.0f);
	Vector3 beta(1.0f);
	for (int bounces = 0; bounces < 10; ++bounces) {
		HitRecord rec;
		// Hit BVH
		if (!nodes[0].Hit(currRay, 0.001f, FLT_MAX, rec, list, nodes)) {
			// Return background color;
			return L + beta * (*envMap)->Lookup(Normalize(currRay.direction));
		}
		Color3 Le = materials[rec.matId]->Le();
		L += Le * beta;
		Vector3 albedo = materials[rec.matId]->Albedo();

		// Generate another ray
		if (curand_uniform(localRandState) < Average(albedo)) {
			Vector3 n = rec.normal;
			Vector3 wo = -Normalize(r.direction);
			Vector3 wi;
			if (materials[rec.matId]->SampleDirection(n, wo, wi, localRandState)) {
				float pdf = materials[rec.matId]->SampleProb(rec.normal, wo, wi);
				if (pdf > 0.0f) {
					Color3 f = materials[rec.matId]->BRDF(n, wo, wi);
					beta *= f * Dot(wi, n) / (pdf * Average(albedo));
					currRay = Ray(rec.point, wi);
				}
				else {
					return L;
				}
			}
			else {
				return L;
			}
		}
		else {
			return L;
		}
	}
	return L;
}


/// <summary>
/// A CUDA kernel that generates a ray for each pixel and triggers the trace function.
/// The resulting radiation is stored in the frame buffer.
/// </summary>
/// <param name="fb">Frame buffer</param>
/// <param name="maxX">Image width</param>
/// <param name="maxY">Image height</param>
/// <param name="nSamples">Number of samples</param>
/// <param name="camera">Camera</param>
/// <param name="list">List of hittable objects</param>
/// <param name="nodes">List of BVH nodes</param>
/// <param name="materials">List of materials</param>
/// <param name="envMap">Environment map</param>
/// <param name="randState">cuRAND state</param>
__global__ void Render(Vector3* fb, int maxX, int maxY, int nSamples, Camera** camera, Sphere* list, BVHNode* nodes, Material** materials, EnvironmentMap** envMap, curandState* randState) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY)) return;

	int pixelIndex = j * maxX + i;
	curandState localRandState = randState[pixelIndex];

	Color3 color(0, 0, 0);
	for (int sample = 0; sample < nSamples; ++sample) {
		float u = float(i + curand_uniform(&localRandState)) / float(maxX);
		float v = float(j + curand_uniform(&localRandState)) / float(maxY);
		Ray r = (*camera)->GenerateRay(u, v);
	    Color3 c = Trace(r, list, nodes, materials, envMap, &localRandState);

		// Clamp [0,1]
		c.x = __saturatef(c.x);
		c.y = __saturatef(c.y);
		c.z = __saturatef(c.z);
		color += c;

	}		

	randState[pixelIndex] = localRandState;

	fb[pixelIndex] += color;
}