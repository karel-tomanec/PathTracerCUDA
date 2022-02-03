#pragma once
#include "hittable.h"

class Sphere {
public:

    __host__ __device__ Sphere() = default;

    Sphere(Point3 center, float radius, int matId) : center(center), radius(radius), matId(matId) {
    };

    // Get intersected point between the ray and the sphere
    __device__ bool Hit(const Ray& ray, float tmin, float tmax, HitRecord& record) const;

    AABB Box() {
        return AABB(center - Vector3(radius), center + Vector3(radius));
    }

public:
    Point3 center;
    float radius;
    int matId; // Material ID
    //AABB box; // Bounding volume of the sphere
};

__device__ bool Sphere::Hit(const Ray& ray, float tmin, float tmax, HitRecord& record) const {
    Vector3 oc = ray.origin - center;
    float a = Dot(ray.direction, ray.direction);
    float b = Dot(oc, ray.direction);
    float c = Dot(oc, oc) - radius * radius;
    float D = b * b - a * c;
    if (D > 0.0f) {
        float temp = (-b - sqrtf(D)) / a;
        if (temp < tmax && temp > tmin) {
            record.t = temp;
            record.point = ray.At(record.t);
            record.normal = (record.point - center) / radius;
            record.matId = matId;
            record.hit = 1;
            return true;
        }
        temp = (-b + sqrtf(D)) / a;
        if (temp < tmax && temp > tmin) {
            record.t = temp;
            record.point = ray.At(record.t);
            record.normal = (record.point - center) / radius;
            record.matId = matId;
            record.hit = 1;
            return true;
        }
    }
    return false;
}

// Comparator functions for the boxes:

inline bool BoxCompareX(Sphere& a, Sphere& b) {
    return BoxCompare(a.Box(), b.Box(), 0);
}

inline bool BoxCompareY(Sphere& a, Sphere& b) {
    return BoxCompare(a.Box(), b.Box(), 1);
}

inline bool BoxCompareZ(Sphere& a, Sphere& b) {
    return BoxCompare(a.Box(), b.Box(), 2);
}
