#pragma once
#include "utility.h"
#include "vector3.h"
#include "ray.h"

/// <summary>
/// Representation of perspective camera - simulates the way human eye and cameras sees things.
/// </summary>
class Camera {
public:
    __device__ Camera(
        Point3 lookfrom,
        Point3 lookat,
        Vector3   vup,
        float vfov,
        float aspect_ratio) {
        float thetaVFOV = DegreesToRadians(vfov);
        float h = __tanf(thetaVFOV / 2.0f);
        viewportHeight = 2.0f * h;
        viewportWidth = aspect_ratio * viewportHeight;

        cameraFront = Normalize(lookat - lookfrom);
        cameraRight = Normalize(Cross(cameraFront, vup));
        cameraUp = Cross(cameraRight, cameraFront);

        origin = lookfrom;
        horizontal = viewportWidth * cameraRight;
        vertical = viewportHeight * cameraUp;
        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f + cameraFront;

        worldUp = vup;
        theta = RadiansToDegrees(SphericalTheta(cameraFront));
        phi = RadiansToDegrees(SphericalPhi(cameraFront) + 180);
    }

    /// <summary>
    /// Generates ray from the image coordinates.
    /// </summary>
    /// <param name="s">Horizontal image coordinate</param>
    /// <param name="t">Vertical image coordinate</param>
    /// <returns>Generatedd ray</returns>
    __device__ Ray GenerateRay(float s, float t) const {
        return Ray(origin, Normalize(lowerLeftCorner + s * horizontal + t * vertical - origin));
    }

    /// <summary>
    /// Moves origin of the camera.
    /// </summary>
    /// <param name="v">Direction</param>
    __device__ inline void MoveOrigin(Vector3 v) {
        origin += v;
        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f + cameraFront;
    }

    /// <summary>
    /// Update camera vectors based on Euler angles.
    /// </summary>
    __device__ inline void UpdateCameraVectors() {
        float costheta = cosf(DegreesToRadians(theta));
        float sintheta = sinf(DegreesToRadians(theta));
        float sinphi = sinf(DegreesToRadians(phi));
        float cosphi = cosf(DegreesToRadians(phi));

        cameraFront = Normalize(SphericalDirection(sintheta, costheta, phi));

        cameraRight = Normalize(Cross(cameraFront, worldUp));
        cameraUp = Cross(cameraRight, cameraFront);
        horizontal = viewportWidth * cameraRight;
        vertical = viewportHeight * cameraUp;
        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f + cameraFront;
    }

public:
    // Origin of the camera
    Point3 origin;
    // Lower left corner of the image plane
    Point3 lowerLeftCorner;
    // Horizontal base vector of the image plane
    Vector3 horizontal;
    // Vertical base vector of the image plane
    Vector3 vertical;
    // Front direction of the camera
    Vector3 cameraFront;
    // Right direction of the camera
    Vector3 cameraRight;
    // Up direction of the camera
    Vector3 cameraUp;
    // Up direction of the world
    Vector3 worldUp;
    // Euler angles:
    float theta;
    float phi;
    // Viewport dimensions
    float viewportWidth;
    float viewportHeight;
};

// Keyboard inputs:

__global__ void moveCameraFront(Camera** d_camera, float cameraSpeed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->MoveOrigin((*d_camera)->cameraFront * cameraSpeed);
    }
}

__global__ void moveCameraBack(Camera** d_camera, float cameraSpeed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->MoveOrigin(-(*d_camera)->cameraFront * cameraSpeed);
    }
}

__global__ void moveCameraLeft(Camera** d_camera, float cameraSpeed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->MoveOrigin(-(*d_camera)->cameraRight * cameraSpeed);
    }
}

__global__ void moveCameraRight(Camera** d_camera, float cameraSpeed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->MoveOrigin((*d_camera)->cameraRight * cameraSpeed);
    }
}

// Mouse input:
__global__ void mouseMovement(Camera** d_camera, float xoffset, float yoffset) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*d_camera)->phi += xoffset;
        (*d_camera)->theta -= yoffset;
        if ((*d_camera)->theta > 179.0f) (*d_camera)->theta = 179.0f;
        if ((*d_camera)->theta < 1.0f)(*d_camera)->theta = 1.0f;
        (*d_camera)->UpdateCameraVectors();
    }
}

