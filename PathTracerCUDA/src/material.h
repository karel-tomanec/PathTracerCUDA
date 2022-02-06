#pragma once
#include "utility.h"
#include "ray.h"
#include "vector3.h"
#include "hittable.h"

/// <summary>
/// Base interface for materials.
/// </summary>
class Material {
public:

    /// <summary>
    /// BRDF.cos(theta) importance sampling for input normal, wo (outgoing direction).
    /// </summary>
    /// <param name="n">Normal vector</param>
    /// <param name="wo">Outgoing direction</param>
    /// <param name="wi">Ingoing direction</param>
    /// <param name="randState">cuRAND state</param>
    __device__ virtual bool SampleDirection(const Vector3& n, const Vector3& wo, Vector3& wi, curandState* randState) const = 0;


    /// <summary>
    /// Evaluate the probability given input normal, view (outgoing) 
	/// direction and incoming light direction
    /// </summary>
    /// <param name="n">Normal vector</param>
    /// <param name="wo">Outgoing direction</param>
    /// <param name="wi">Ingoing direction</param>
    /// <returns>Sample probability</returns>
    __device__ virtual float SampleProb(const Vector3& n, const Vector3& wo, const Vector3& wi) const = 0;

	/// <summary>
	/// Evaluates the BRDF given normal, wo (outgoing direction) and wi (incoming direction).
	/// </summary>
	/// <param name="n">Normal vector</param>
	/// <param name="wo">Outgoing direction</param>
	/// <param name="wi">Ingoing direction</param>
	/// <returns>BRDF</returns>
	__device__ virtual Vector3 BRDF(const Vector3& n, const Vector3& wo, const Vector3& wi) const = 0;

    /// <summary>
    /// Emission of the material.
    /// </summary>
    /// <returns>Emitted radiance</returns>
    __device__ virtual Color3 Le() const {
        return Color3(0.0f);
    }

	__device__ virtual Vector3 Albedo() const {
		return Vector3(0.0f);
	}
};

class Diffuse : public Material {
public:

    __device__ Diffuse(const Color3& Kd) : Kd(Kd) {}

	__device__ Vector3 BRDF(const Vector3& n, const Vector3& wo, const Vector3& wi) const {

		return Kd / CUDART_PI_F;
	}


    __device__ virtual bool SampleDirection(const Vector3& n, const Vector3& wo, Vector3& wi, curandState* randState) const override {
		wi = Vector3(0, 0, 0);
		const float u_m = curand_uniform(randState);
		const float v_m = curand_uniform(randState);

		const float roulette = curand_uniform(randState);

		if (roulette < Average(Kd)) {
			// diffuse sample

			const double x = sqrtf(1.0 - u_m) * cosf(2.0 * CUDART_PI_F * v_m);
			const double y = sqrtf(1.0 - u_m) * sinf(2.0 * CUDART_PI_F * v_m);
			const double z = sqrtf(u_m);

			const Vector3 k = n;
			const Vector3 w = Normalize(Vector3(2.0 * curand_uniform(randState) - 1.0, 2.0 * curand_uniform(randState) - 1.0, 2.0 * curand_uniform(randState) - 1.0));
			const Vector3 i = Normalize(Cross(n, w));
			const Vector3 j = Normalize(Cross(i, k));


			wi = i * x + j * y + k * z;

			if (Dot(n, wi) < 0) {
				return false;
			}
			return true;
		}
		else {
			// the contribution is zero
			return false;
		}
    }

    __device__ float SampleProb(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
        return Average(Kd) * Dot(n, wi) / CUDART_PI_F;
    }

	__device__ virtual Vector3 Albedo() const {
		return Kd;
	}

public:

    Vector3 Kd;
};

class Specular : public Material {
public:

	__device__ Specular(const Color3& Kd, const Color3& Ks, float shininess) : Kd(Kd), Ks(Ks), shininess(shininess){}

	__device__ Vector3 BRDF(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
		Vector3 brdf(0, 0, 0);
		float cosThetaL = Dot(n, wi);
		float cosThetaV = Dot(n, wo);

		brdf = Kd / CUDART_PI_F;

		Vector3 r = Reflect(n, wi);
		double cosPhi = Dot(wo, r);

		if (cosPhi <= 0) return brdf; 

		return brdf + Ks * ((shininess + 1.0f) / 2.0f / CUDART_PI_F * powf(cosPhi, shininess) / fmaxf(cosThetaL, cosThetaV));
	}


	__device__ virtual bool SampleDirection(const Vector3& n, const Vector3& wo, Vector3& wi, curandState* randState) const override {
		wi = Vector3(0, 0, 0);
		const float u_m = curand_uniform(randState);
		const float v_m = curand_uniform(randState);

		const float roulette = curand_uniform(randState);

		if (roulette < Average(Kd)) {
			// diffuse sample

			const float x = sqrtf(1.0 - u_m) * cosf(2.0 * CUDART_PI_F * v_m);
			const float y = sqrtf(1.0 - u_m) * sinf(2.0 * CUDART_PI_F * v_m);
			const float z = sqrtf(u_m);

			const Vector3 k = n;
			const Vector3 w = Normalize(Vector3(2.0 * curand_uniform(randState) - 1.0, 2.0 * curand_uniform(randState) - 1.0, 2.0 * curand_uniform(randState) - 1.0));
			const Vector3 i = Normalize(Cross(n, w));
			const Vector3 j = Normalize(Cross(i, k));


			wi = i * x + j * y + k * z;

			if (Dot(n, wi) < 0) {
				return false;
			}
			return true;
		}
		else if(roulette < Average(Kd) + Average(Ks)) {
			// specular sample
			const float x = sqrtf(1.0f - powf(u_m, 2.0f / (shininess + 1.0f))) * cosf(2.0f * CUDART_PI_F * v_m);
			const float y = sqrtf(1.0f - powf(u_m, 2.0f / (shininess + 1.0f))) * sinf(2.0f * CUDART_PI_F * v_m);
			const float z = powf(u_m, 1.0f / (shininess + 1.0f));

			const Vector3 k = wo;
			const Vector3 i = Normalize(Cross(wo, n));
			const Vector3 j = Cross(i, k);

			const Vector3 r = i * x + j * y + k * z;

			wi = Normalize(n * Dot(n, r) * 2.0f - r);

			if (Dot(n, wi) < 0.0f) {
				return false;
			}

			return true;
		} else {
			// the contribution is zero
			return false;
		}
	}

	// Evaluate the probability given input normal, view (outgoing) direction and incoming light direction
	__device__ float SampleProb(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
		return Average(Kd) * Dot(n, wi) / CUDART_PI_F + Average(Ks) * ((shininess + 1.0f) / (2.0f * CUDART_PI_F)) * powf((fmaxf(Dot(wo, Reflect(n, wi)), 0.0f)), shininess);
	}

	__device__ virtual Vector3 Albedo() const {
		return Kd + Ks;
	}

public:

	Vector3 Kd;
	Vector3 Ks;
	float shininess;
};

class Metal : public Material {
public:

    __device__ Metal(const Color3& Kd, const Color3& Ks, float shininess, float f) : Kd(Kd), Ks(Ks), shininess(shininess) { if (f < 1.0f) fuzz = f; else fuzz = 1.0f; }


	__device__ Vector3 BRDF(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
		Vector3 brdf(0, 0, 0);
		float cosThetaL = Dot(n, wi);
		float cosThetaV = Dot(n, wo);

		brdf = Kd / CUDART_PI_F;

		Vector3 r = Reflect(n, wi);
		double cosPhi = Dot(wo, r);

		if (cosPhi <= 0) return brdf;

		return brdf + Ks * ((shininess + 1.0f) / 2.0f / CUDART_PI_F * powf(cosPhi, shininess) / fmaxf(cosThetaL, cosThetaV));
	}

	__device__ virtual bool SampleDirection(const Vector3& n, const Vector3& wo, Vector3& wi, curandState* randState) const override {
		wi = Vector3(0, 0, 0);
		const float u_m = curand_uniform(randState);
		const float v_m = curand_uniform(randState);

		const float roulette = curand_uniform(randState);

		if (roulette < Average(Kd)) {
			// diffuse sample

			const float x = sqrtf(1.0 - u_m) * cosf(2.0 * CUDART_PI_F * v_m);
			const float y = sqrtf(1.0 - u_m) * sinf(2.0 * CUDART_PI_F * v_m);
			const float z = sqrtf(u_m);

			const Vector3 k = n;
			const Vector3 w = Normalize(Vector3(2.0 * curand_uniform(randState) - 1.0, 2.0 * curand_uniform(randState) - 1.0, 2.0 * curand_uniform(randState) - 1.0));
			const Vector3 i = Normalize(Cross(n, w));
			const Vector3 j = Normalize(Cross(i, k));


			wi = i * x + j * y + k * z;

			if (Dot(n, wi) < 0) {
				return false;
			}
			return true;
		}
		else if (roulette < Average(Kd) + Average(Ks)) {
			// specular sample
			const float x = sqrtf(1.0f - powf(u_m, 2.0f / (shininess + 1.0f))) * cosf(2.0f * CUDART_PI_F * v_m);
			const float y = sqrtf(1.0f - powf(u_m, 2.0f / (shininess + 1.0f))) * sinf(2.0f * CUDART_PI_F * v_m);
			const float z = powf(u_m, 1.0f / (shininess + 1.0f));

			const Vector3 k = wo;
			const Vector3 i = Normalize(Cross(wo, n));
			const Vector3 j = Cross(i, k);

			const Vector3 r = i * x + j * y + k * z;

			wi = Normalize(n * Dot(n, r) * 2.0f - r + fuzz * RandVecInUnitSphere(randState));

			if (Dot(n, wi) < 0) {
				return false;
			}

			return true;
		}
		else {
			// the contribution is zero
			return false;
		}
	}

	// Evaluate the probability given input normal, view (outgoing) direction and incoming light direction
	__device__ float SampleProb(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
		return Average(Kd) * Dot(n, wi) / CUDART_PI_F + Average(Ks) * ((shininess + 1.0f) / (2.0f * CUDART_PI_F)) * powf((fmaxf(Dot(wo, Reflect(n, wi)), 0.0f)), shininess);
	}

	__device__ virtual Vector3 Albedo() const {
		return Kd + Ks;
	}

public:

    Vector3 Kd;
	Vector3 Ks;
	float shininess;
    float fuzz;
};

class DiffuseLight : public Material {
public:

    __device__ DiffuseLight(const Color3& emit) : emit(emit) { }

	__device__ bool SampleDirection(const Vector3& n, const Vector3& wo, Vector3& wi, curandState* randState) const {
		return false;
	}

	__device__ float SampleProb(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
		return 0.0f;
	}

	__device__ Vector3 BRDF(const Vector3& n, const Vector3& wo, const Vector3& wi) const {
		return Vector3(0);
	}

    __device__ virtual Color3 Le() const {
        return emit;
    }

public:

    Color3 emit;
};