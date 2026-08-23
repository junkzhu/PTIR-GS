// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __CUDACC__

#include <3dgptir/kernels/cuda/environment.cuh>
#include <3dgptir/kernels/cuda/meshLight.cuh>
#include <3dgptir/kernels/cuda/sampler.cuh>

enum LightSamplerType {
    LightSamplerType_Env    = 0,
    LightSamplerType_Point  = 1,
    LightSamplerType_Sphere = 2,
    LightSamplerType_Mesh   = 3,
};

struct LightSample {
    float3 wi;
    float3 Li;
    float pdf;
    float dist;
    unsigned int lightType;
    unsigned int lightId;
};

static __device__ __forceinline__ LightSample emptyLightSample() {
    LightSample sample;
    sample.wi        = make_float3(0.0f);
    sample.Li        = make_float3(0.0f);
    sample.pdf       = 0.0f;
    sample.dist      = 1e20f;
    sample.lightType = LightSamplerType_Env;
    sample.lightId   = 0u;
    return sample;
}

static __device__ __forceinline__ bool hasTopLevelLightAliasTable() {
    return params.numLightEntries > 0;
}

static __device__ __forceinline__ bool hasEnvironmentLight() {
    return hasNativeSGEnvironment()
        || (params.environment.data != nullptr
            && params.environment.width > 0
            && params.environment.height > 0
            && params.environment.aliasTable.numCells > 0);
}

static __device__ __forceinline__ unsigned int packedLightType(
    const unsigned int lightId) {
    if (lightId >= params.numLights) {
        return 0xFFFFFFFFu;
    }
    return static_cast<unsigned int>(params.lights[lightId][0] + 0.5f);
}

static __device__ __forceinline__ unsigned int sampleTopLevelLightEntry(Sampler& sampler) {
    if (!hasTopLevelLightAliasTable()) {
        return 0u;
    }

    const unsigned int initialEntry = min(
        static_cast<unsigned int>(sampler.next_1d() * static_cast<float>(params.numLightEntries)),
        params.numLightEntries - 1u);
    const unsigned int aliasEntry = min(
        static_cast<unsigned int>(params.lightAliasTable[1][initialEntry] + 0.5f),
        params.numLightEntries - 1u);
    return sampler.next_1d() < params.lightAliasTable[0][initialEntry]
        ? initialEntry
        : aliasEntry;
}

static __device__ __forceinline__ LightSample sampleEnvLight(
    const float3& position,
    Sampler& sampler,
    const float selectPdf,
    const unsigned int lightId) {
    LightSample sample = emptyLightSample();
    if (!hasEnvironmentLight()) {
        return sample;
    }
    float envPdf = 0.0f;
    sample.wi = hasNativeSGEnvironment()
        ? sampleNativeSGEnvironmentDirection(sampler, envPdf)
        : sampleEnvironmentAliasDirection(sampler, envPdf);
    sample.Li = getBackgroundColor(sample.wi);
    sample.pdf = selectPdf * envPdf;
    sample.dist = 1e20f;
    sample.lightType = LightSamplerType_Env;
    sample.lightId = lightId;
    return sample;
}

static __device__ __forceinline__ LightSample samplePointLight(
    const float3& position,
    const unsigned int lightId,
    const float selectPdf) {
    LightSample sample = emptyLightSample();
    if (lightId >= params.numLights || packedLightType(lightId) != LightSamplerType_Point || selectPdf <= 0.0f) {
        return sample;
    }

    const float3 lightPosition = make_float3(
        params.lights[lightId][1],
        params.lights[lightId][2],
        params.lights[lightId][3]);
    const float3 toLight = lightPosition - position;
    const float dist2 = dot(toLight, toLight);
    if (dist2 <= 1e-12f) {
        return sample;
    }

    const float dist = sqrtf(dist2);
    sample.wi = toLight / dist;
    sample.Li = make_float3(
        params.lights[lightId][5],
        params.lights[lightId][6],
        params.lights[lightId][7]) / dist2;
    sample.pdf = selectPdf;
    sample.dist = dist;
    sample.lightType = LightSamplerType_Point;
    sample.lightId = lightId;
    return sample;
}

static __device__ __forceinline__ LightSample sampleSphereLight(
    const float3& position,
    Sampler& sampler,
    const unsigned int lightId,
    const float selectPdf) {
    LightSample sample = emptyLightSample();
    if (lightId >= params.numLights || packedLightType(lightId) != LightSamplerType_Sphere) {
        return sample;
    }

    const float centerX = params.lights[lightId][1];
    const float centerY = params.lights[lightId][2];
    const float centerZ = params.lights[lightId][3];
    const float radius  = params.lights[lightId][4];
    if (radius <= 0.0f) {
        return sample;
    }

    const float u1 = sampler.next_1d();
    const float u2 = sampler.next_1d();
    const float z = 1.0f - 2.0f * u1;
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * CUDART_PI_F * u2;
    float sinPhi;
    float cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);

    const float3 normal = make_float3(r * cosPhi, r * sinPhi, z);
    const float3 center = make_float3(centerX, centerY, centerZ);
    const float3 lightPoint = center + radius * normal;
    const float3 toLight = lightPoint - position;
    const float dist2 = dot(toLight, toLight);
    if (dist2 <= 1e-12f) {
        return sample;
    }

    const float dist = sqrtf(dist2);
    const float3 wi = toLight / dist;
    const float cosLight = fabsf(dot(normal, -wi));
    if (cosLight <= 1e-6f) {
        return sample;
    }

    const float area = 4.0f * CUDART_PI_F * radius * radius;
    const float pdfArea = 1.0f / fmaxf(area, 1e-12f);
    sample.wi = wi;
    sample.Li = make_float3(
        params.lights[lightId][5],
        params.lights[lightId][6],
        params.lights[lightId][7]);
    sample.pdf = selectPdf * pdfArea * dist2 / cosLight;
    sample.dist = dist;
    sample.lightType = LightSamplerType_Sphere;
    sample.lightId = lightId;
    return sample;
}

static __device__ __forceinline__ float sphereLightPdf(
    const float3& position,
    const float3& wi,
    const unsigned int lightId) {
    if (lightId >= params.numLights || packedLightType(lightId) != LightSamplerType_Sphere) {
        return 0.0f;
    }

    const float3 center = make_float3(
        params.lights[lightId][1],
        params.lights[lightId][2],
        params.lights[lightId][3]);
    const float radius = params.lights[lightId][4];
    if (radius <= 0.0f) {
        return 0.0f;
    }

    const float3 oc = position - center;
    const float b = dot(oc, wi);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant <= 0.0f) {
        return 0.0f;
    }

    float t = -b - sqrtf(discriminant);
    if (t <= 1e-6f) {
        t = -b + sqrtf(discriminant);
    }
    if (t <= 1e-6f) {
        return 0.0f;
    }

    const float3 lightPoint = position + t * wi;
    const float3 normal = safe_normalize(lightPoint - center);
    const float cosLight = fabsf(dot(normal, -wi));
    if (cosLight <= 1e-6f) {
        return 0.0f;
    }

    const float area = 4.0f * CUDART_PI_F * radius * radius;
    return (1.0f / fmaxf(area, 1e-12f)) * (t * t) / cosLight;
}

static __device__ __forceinline__ LightSample sampleMeshLight(
    const float3& position,
    Sampler& sampler,
    const unsigned int lightId,
    const float selectPdf) {
    LightSample sample = emptyLightSample();
    if (!hasMeshLightData() || lightId >= params.numMeshLights || selectPdf <= 0.0f) {
        return sample;
    }

    const unsigned int triangleOffset = static_cast<unsigned int>(params.meshLights[lightId][0] + 0.5f);
    const unsigned int triangleCount = static_cast<unsigned int>(params.meshLights[lightId][1] + 0.5f);
    if (triangleCount == 0u || triangleOffset + triangleCount > params.numMeshLightTriangles) {
        return sample;
    }

    const unsigned int triangleId = sampleMeshTriangleEntry(sampler, triangleOffset, triangleCount);
    float3 v0;
    float3 v1;
    float3 v2;
    if (!getMeshLightTriangle(triangleId, v0, v1, v2)) {
        return sample;
    }

    float u = sampler.next_1d();
    float v = sampler.next_1d();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }

    const float3 lightPoint = v0 + u * (v1 - v0) + v * (v2 - v0);
    const float3 toLight = lightPoint - position;
    const float dist2 = dot(toLight, toLight);
    if (dist2 <= 1e-12f) {
        return sample;
    }

    const float dist = sqrtf(dist2);
    const float3 wi = toLight / dist;
    const float3 normalUnnormalized = cross(v1 - v0, v2 - v0);
    const float normalLength = length(normalUnnormalized);
    if (normalLength <= 1e-12f) {
        return sample;
    }

    const bool twoSided = params.meshLights[lightId][7] > 0.5f;
    const float cosLight = twoSided
        ? fabsf(dot(normalUnnormalized / normalLength, -wi))
        : dot(normalUnnormalized / normalLength, -wi);
    if (cosLight <= 1e-6f) {
        return sample;
    }

    const float trianglePdf = params.meshLightTriangleAliasTable[2][triangleId];
    sample.wi = wi;
    sample.Li = make_float3(
        params.meshLights[lightId][4],
        params.meshLights[lightId][5],
        params.meshLights[lightId][6]);
    sample.pdf = selectPdf * trianglePdf * dist2 / cosLight;
    sample.dist = dist;
    sample.lightType = LightSamplerType_Mesh;
    sample.lightId = lightId;
    return sample;
}

static __device__ __forceinline__ float meshLightPdf(
    const float3& position,
    const float3& wi,
    const unsigned int lightId) {
    if (!hasMeshLightData() || lightId >= params.numMeshLights) {
        return 0.0f;
    }

    const unsigned int triangleOffset = static_cast<unsigned int>(params.meshLights[lightId][0] + 0.5f);
    const unsigned int triangleCount = static_cast<unsigned int>(params.meshLights[lightId][1] + 0.5f);
    if (triangleCount == 0u || triangleOffset + triangleCount > params.numMeshLightTriangles) {
        return 0.0f;
    }

    const bool twoSided = params.meshLights[lightId][7] > 0.5f;
    float pdf = 0.0f;
    for (unsigned int triangleId = triangleOffset; triangleId < triangleOffset + triangleCount; ++triangleId) {
        float3 v0;
        float3 v1;
        float3 v2;
        if (!getMeshLightTriangle(triangleId, v0, v1, v2)) {
            continue;
        }

        float t;
        float u;
        float v;
        if (!intersectTriangle(position, wi, v0, v1, v2, t, u, v)) {
            continue;
        }

        const float3 normalUnnormalized = cross(v1 - v0, v2 - v0);
        const float normalLength = length(normalUnnormalized);
        if (normalLength <= 1e-12f) {
            continue;
        }

        const float cosLight = twoSided
            ? fabsf(dot(normalUnnormalized / normalLength, -wi))
            : dot(normalUnnormalized / normalLength, -wi);
        if (cosLight <= 1e-6f) {
            continue;
        }

        pdf += params.meshLightTriangleAliasTable[2][triangleId] * (t * t) / cosLight;
    }
    return pdf;
}

static __device__ __forceinline__ float lightEntryPdf(
    const float3& position,
    const float3& wi,
    const unsigned int lightType,
    const unsigned int lightId) {
    switch (lightType) {
    case LightSamplerType_Env:
        return hasEnvironmentLight()
            ? (hasNativeSGEnvironment()
                ? nativeSGEnvironmentPdf(wi)
                : environmentAliasPdf(wi))
            : 0.0f;
    case LightSamplerType_Sphere:
        return sphereLightPdf(position, wi, lightId);
    case LightSamplerType_Mesh:
        return meshLightPdf(position, wi, lightId);
    default:
        return 0.0f;
    }
}

static __device__ __forceinline__ LightSample sampleLight(
    const float3& position,
    Sampler& sampler) {
    if (!hasTopLevelLightAliasTable()) {
        return sampleEnvLight(position, sampler, 1.0f, 0u);
    }

    const unsigned int entry = sampleTopLevelLightEntry(sampler);
    const unsigned int lightType = static_cast<unsigned int>(params.lightAliasTable[2][entry] + 0.5f);
    const unsigned int lightId = static_cast<unsigned int>(params.lightAliasTable[3][entry] + 0.5f);
    const float selectPdf = params.lightAliasTable[4][entry];

    switch (lightType) {
    case LightSamplerType_Env:
        return sampleEnvLight(position, sampler, selectPdf, lightId);
    case LightSamplerType_Point:
        return samplePointLight(position, lightId, selectPdf);
    case LightSamplerType_Sphere:
        return sampleSphereLight(position, sampler, lightId, selectPdf);
    case LightSamplerType_Mesh:
        return sampleMeshLight(position, sampler, lightId, selectPdf);
    default:
        return emptyLightSample();
    }
}

static __device__ __forceinline__ float lightSamplerPdf(
    const float3& position,
    const float3& wi) {
    if (!hasTopLevelLightAliasTable()) {
        return hasEnvironmentLight()
            ? (hasNativeSGEnvironment()
                ? nativeSGEnvironmentPdf(wi)
                : environmentAliasPdf(wi))
            : 0.0f;
    }

    float pdf = 0.0f;
    for (unsigned int i = 0; i < params.numLightEntries; ++i) {
        const unsigned int lightType = static_cast<unsigned int>(params.lightAliasTable[2][i] + 0.5f);
        const unsigned int lightId = static_cast<unsigned int>(params.lightAliasTable[3][i] + 0.5f);
        const float selectPdf = params.lightAliasTable[4][i];
        if (selectPdf <= 0.0f) {
            continue;
        }

        pdf += selectPdf * lightEntryPdf(position, wi, lightType, lightId);
    }

    // Point lights are delta lights and do not contribute continuous-direction pdf.
    return pdf;
}

#endif // __CUDACC__
