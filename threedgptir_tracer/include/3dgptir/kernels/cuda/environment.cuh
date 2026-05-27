// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __CUDACC__

#include <3dgptir/environment.h>
#include <3dgptir/mathUtils.h>
#include <3dgptir/payLoad.h>
#include <3dgptir/kernels/cuda/sampler.cuh>
#include <math_constants.h>

static __device__ __forceinline__ float misWeight(float pdfA, float pdfB);

static __device__ __forceinline__ int environmentClampInt(int value, int lo, int hi) {
    return max(lo, min(value, hi));
}

static __device__ __forceinline__ int environmentWrapInt(int value, int size) {
    if (size <= 0) {
        return 0;
    }

    int wrapped = value % size;
    if (wrapped < 0) {
        wrapped += size;
    }
    return wrapped;
}

struct EnvironmentBilinearFootprint {
    int x0;
    int x1;
    int y0;
    int y1;
    float w00;
    float w10;
    float w01;
    float w11;
};

static __device__ __forceinline__ float4 loadEnvironmentTexel(int x, int y) {
    return params.environment.data[y * params.environment.width + x];
}

static __device__ __forceinline__ EnvironmentBilinearFootprint computeEnvironmentBilinearFootprint(float u, float v) {
    u = u - floorf(u);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    const float x = u * static_cast<float>(params.environment.width) - 0.5f;
    const float y = v * static_cast<float>(params.environment.height) - 0.5f;

    const int x0Raw = static_cast<int>(floorf(x));
    const int y0Raw = static_cast<int>(floorf(y));
    const int x1Raw = x0Raw + 1;
    const int y1Raw = y0Raw + 1;

    const float ax = x - static_cast<float>(x0Raw);
    const float ay = y - static_cast<float>(y0Raw);

    EnvironmentBilinearFootprint footprint{};
    footprint.x0  = environmentWrapInt(x0Raw, params.environment.width);
    footprint.x1  = environmentWrapInt(x1Raw, params.environment.width);
    footprint.y0  = environmentClampInt(y0Raw, 0, params.environment.height - 1);
    footprint.y1  = environmentClampInt(y1Raw, 0, params.environment.height - 1);
    footprint.w00 = (1.0f - ax) * (1.0f - ay);
    footprint.w10 = ax * (1.0f - ay);
    footprint.w01 = (1.0f - ax) * ay;
    footprint.w11 = ax * ay;
    return footprint;
}

static __device__ __forceinline__ float4 sampleEnvironmentBilinear(const EnvironmentBilinearFootprint& footprint) {
    const float4 c00 = loadEnvironmentTexel(footprint.x0, footprint.y0);
    const float4 c10 = loadEnvironmentTexel(footprint.x1, footprint.y0);
    const float4 c01 = loadEnvironmentTexel(footprint.x0, footprint.y1);
    const float4 c11 = loadEnvironmentTexel(footprint.x1, footprint.y1);

    return make_float4(
        footprint.w00 * c00.x + footprint.w10 * c10.x + footprint.w01 * c01.x + footprint.w11 * c11.x,
        footprint.w00 * c00.y + footprint.w10 * c10.y + footprint.w01 * c01.y + footprint.w11 * c11.y,
        footprint.w00 * c00.z + footprint.w10 * c10.z + footprint.w01 * c01.z + footprint.w11 * c11.z,
        footprint.w00 * c00.w + footprint.w10 * c10.w + footprint.w01 * c01.w + footprint.w11 * c11.w);
}

template <typename PipelineParams>
static __device__ __forceinline__ void accumulateEnvironmentTexelGrad(
    PipelineParams& pipelineParams,
    const int x,
    const int y,
    const float weight,
    const float3& colorGrad) {
    atomicAdd(&pipelineParams.environmentGrad[y][x][0], weight * colorGrad.x);
    atomicAdd(&pipelineParams.environmentGrad[y][x][1], weight * colorGrad.y);
    atomicAdd(&pipelineParams.environmentGrad[y][x][2], weight * colorGrad.z);
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 sampleEnvironmentBilinearBwd(
    const EnvironmentBilinearFootprint& footprint,
    const float3& colorGrad,
    PipelineParams& pipelineParams) {
    const float4 env = sampleEnvironmentBilinear(footprint);

    accumulateEnvironmentTexelGrad(pipelineParams, footprint.x0, footprint.y0, footprint.w00, colorGrad);
    accumulateEnvironmentTexelGrad(pipelineParams, footprint.x1, footprint.y0, footprint.w10, colorGrad);
    accumulateEnvironmentTexelGrad(pipelineParams, footprint.x0, footprint.y1, footprint.w01, colorGrad);
    accumulateEnvironmentTexelGrad(pipelineParams, footprint.x1, footprint.y1, footprint.w11, colorGrad);

    return make_float3(env.x, env.y, env.z);
}

static __device__ __forceinline__ float3 rotateEnvironmentDirection(const float3& rayDir) {
    const float rotZ = params.environment.offset.x * 2.0f * CUDART_PI_F + 0.5f * CUDART_PI_F;
    const float rotX = params.environment.offset.y * 2.0f * CUDART_PI_F;

    const float3 rotatedDir = make_float3(
        rayDir.x * cosf(rotZ) - rayDir.y * sinf(rotZ),
        rayDir.x * sinf(rotZ) + rayDir.y * cosf(rotZ),
        rayDir.z);

    return make_float3(
        rotatedDir.x,
        rotatedDir.y * cosf(rotX) - rotatedDir.z * sinf(rotX),
        rotatedDir.y * sinf(rotX) + rotatedDir.z * cosf(rotX));
}

static __device__ __forceinline__ float3 inverseRotateEnvironmentDirection(const float3& envDir) {
    const float rotZ = params.environment.offset.x * 2.0f * CUDART_PI_F + 0.5f * CUDART_PI_F;
    const float rotX = params.environment.offset.y * 2.0f * CUDART_PI_F;
    float sinZ;
    float cosZ;
    sincosf(rotZ, &sinZ, &cosZ);
    float sinX;
    float cosX;
    sincosf(rotX, &sinX, &cosX);

    const float3 xRotatedDir = make_float3(
        envDir.x,
        envDir.y * cosX + envDir.z * sinX,
        -envDir.y * sinX + envDir.z * cosX);

    return make_float3(
        xRotatedDir.x * cosZ + xRotatedDir.y * sinZ,
        -xRotatedDir.x * sinZ + xRotatedDir.y * cosZ,
        xRotatedDir.z);
}

static __device__ __forceinline__ float3 environmentEquirectUVToDirection(float u, float v) {
    u = u - floorf(u);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    const float theta = u * 2.0f * CUDART_PI_F - CUDART_PI_F;
    const float phi   = (v - 0.5f) * CUDART_PI_F;
    float sinTheta;
    float cosTheta;
    sincosf(theta, &sinTheta, &cosTheta);
    float sinPhi;
    float cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);

    return make_float3(sinTheta * cosPhi, cosTheta * cosPhi, -sinPhi);
}

static __device__ __forceinline__ float3 environmentCubemapFaceUVToDirection(int face, float u, float v) {
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    const float s = 2.0f * u - 1.0f;
    const float t = 2.0f * v - 1.0f;
    float3 dir;
    switch (face) {
    case 0:
        dir = make_float3(1.0f, -t, -s);
        break;
    case 1:
        dir = make_float3(-1.0f, -t, s);
        break;
    case 2:
        dir = make_float3(s, 1.0f, t);
        break;
    case 3:
        dir = make_float3(s, -1.0f, -t);
        break;
    case 4:
        dir = make_float3(s, -t, 1.0f);
        break;
    default:
        dir = make_float3(-s, -t, -1.0f);
        break;
    }
    return safe_normalize(dir);
}

static __device__ __forceinline__ float3 sampleEnvironmentAliasDirection(
    Sampler& sampler,
    float& pdf) {
    const EnvAliasTable& aliasTable = params.environment.aliasTable;

    const int initialCell = min(static_cast<int>(sampler.next_1d() * aliasTable.numCells), aliasTable.numCells - 1);
    const int aliasCell   = environmentClampInt(static_cast<int>(aliasTable.alias[initialCell] + 0.5f), 0, aliasTable.numCells - 1);
    const int cell        = sampler.next_1d() < aliasTable.prob[initialCell] ? initialCell : aliasCell;
    const int x           = cell % aliasTable.width;
    const int y           = cell / aliasTable.width;

    float3 envDirection;
    if (params.environment.type == EnvironmentType_Cube) {
        const int faceSize = aliasTable.width;
        const int face     = environmentClampInt(y / faceSize, 0, 5);
        const int faceY    = y - face * faceSize;
        envDirection = environmentCubemapFaceUVToDirection(
            face,
            (static_cast<float>(x) + sampler.next_1d()) / static_cast<float>(faceSize),
            (static_cast<float>(faceY) + sampler.next_1d()) / static_cast<float>(faceSize));
    } else {
        envDirection = environmentEquirectUVToDirection(
            (static_cast<float>(x) + sampler.next_1d()) / static_cast<float>(aliasTable.width),
            (static_cast<float>(y) + sampler.next_1d()) / static_cast<float>(aliasTable.height));
    }

    pdf = aliasTable.pdf[cell];
    return inverseRotateEnvironmentDirection(envDirection);
}

static __device__ __forceinline__ void dirToCubemapFaceUV(const float3& dir, int& face, float& u, float& v) {
    const float absX = fabsf(dir.x);
    const float absY = fabsf(dir.y);
    const float absZ = fabsf(dir.z);

    float ma, sc, tc;
    if (absX >= absY && absX >= absZ) {
        ma = absX;
        if (dir.x > 0.0f) {
            face = 0;
            sc   = -dir.z;
            tc   = -dir.y;
        } else {
            face = 1;
            sc   = dir.z;
            tc   = -dir.y;
        }
    } else if (absY >= absX && absY >= absZ) {
        ma = absY;
        if (dir.y > 0.0f) {
            face = 2;
            sc   = dir.x;
            tc   = dir.z;
        } else {
            face = 3;
            sc   = dir.x;
            tc   = -dir.z;
        }
    } else {
        ma = absZ;
        if (dir.z > 0.0f) {
            face = 4;
            sc   = dir.x;
            tc   = -dir.y;
        } else {
            face = 5;
            sc   = -dir.x;
            tc   = -dir.y;
        }
    }

    const float invMa = 1.0f / fmaxf(ma, 1e-12f);
    u = 0.5f * (sc * invMa + 1.0f);
    v = 0.5f * (tc * invMa + 1.0f);
}

static __device__ __forceinline__ float environmentAliasPdf(const float3& rayDir) {
    const EnvAliasTable& aliasTable = params.environment.aliasTable;
    const float3 dir = rotateEnvironmentDirection(rayDir);

    int x;
    int y;
    if (params.environment.type == EnvironmentType_Cube) {
        int face;
        float u, v;
        dirToCubemapFaceUV(dir, face, u, v);
        const int faceSize = aliasTable.width;
        x = environmentClampInt(static_cast<int>(u * static_cast<float>(faceSize)), 0, faceSize - 1);
        const int faceY = environmentClampInt(static_cast<int>(v * static_cast<float>(faceSize)), 0, faceSize - 1);
        y = face * faceSize + faceY;
    } else {
        const float theta = atan2f(dir.x, dir.y);
        const float zcl   = fminf(1.0f, fmaxf(-1.0f, -dir.z));
        const float phi   = asinf(zcl);
        const float u     = (theta + CUDART_PI_F) * (0.5f * (1.0f / CUDART_PI_F));
        const float v     = 0.5f + phi * (1.0f / CUDART_PI_F);
        x = environmentWrapInt(static_cast<int>(u * static_cast<float>(aliasTable.width)), aliasTable.width);
        y = environmentClampInt(static_cast<int>(v * static_cast<float>(aliasTable.height)), 0, aliasTable.height - 1);
    }

    return aliasTable.pdf[y * aliasTable.width + x];
}

static __device__ __forceinline__ EnvironmentBilinearFootprint computeCubemapBilinearFootprint(
    int face, float u, float v) {
    const int faceSize = params.environment.width;

    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    const float x = u * static_cast<float>(faceSize) - 0.5f;
    const float y = v * static_cast<float>(faceSize) - 0.5f;

    const int x0Raw = static_cast<int>(floorf(x));
    const int y0Raw = static_cast<int>(floorf(y));
    const int x1Raw = x0Raw + 1;
    const int y1Raw = y0Raw + 1;

    const float ax = x - static_cast<float>(x0Raw);
    const float ay = y - static_cast<float>(y0Raw);
    const int yOffset = face * faceSize;

    EnvironmentBilinearFootprint footprint{};
    footprint.x0  = environmentClampInt(x0Raw, 0, faceSize - 1);
    footprint.x1  = environmentClampInt(x1Raw, 0, faceSize - 1);
    footprint.y0  = environmentClampInt(y0Raw + yOffset, yOffset, yOffset + faceSize - 1);
    footprint.y1  = environmentClampInt(y1Raw + yOffset, yOffset, yOffset + faceSize - 1);
    footprint.w00 = (1.0f - ax) * (1.0f - ay);
    footprint.w10 = ax * (1.0f - ay);
    footprint.w01 = (1.0f - ax) * ay;
    footprint.w11 = ax * ay;
    return footprint;
}

static __device__ __forceinline__ float3 getBackgroundColorEquirect(const float3& dir) {
    const float theta = atan2f(dir.x, dir.y);
    const float zcl   = fminf(1.0f, fmaxf(-1.0f, -dir.z));
    const float phi   = asinf(zcl);
    const float u     = (theta + CUDART_PI_F) * (0.5f * (1.0f / CUDART_PI_F));
    const float v     = 0.5f + phi * (1.0f / CUDART_PI_F);
    const float4 env  = sampleEnvironmentBilinear(computeEnvironmentBilinearFootprint(u, v));
    return make_float3(env.x, env.y, env.z);
}

static __device__ __forceinline__ float3 getBackgroundColorCubemap(const float3& dir) {
    int face;
    float u, v;
    dirToCubemapFaceUV(dir, face, u, v);
    const float4 env = sampleEnvironmentBilinear(computeCubemapBilinearFootprint(face, u, v));
    return make_float3(env.x, env.y, env.z);
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 getBackgroundColorEquirectBwd(
    const float3& dir,
    const float3& colorGrad,
    PipelineParams& pipelineParams) {
    const float theta = atan2f(dir.x, dir.y);
    const float zcl   = fminf(1.0f, fmaxf(-1.0f, -dir.z));
    const float phi   = asinf(zcl);
    const float u     = (theta + CUDART_PI_F) * (0.5f * (1.0f / CUDART_PI_F));
    const float v     = 0.5f + phi * (1.0f / CUDART_PI_F);
    return sampleEnvironmentBilinearBwd(computeEnvironmentBilinearFootprint(u, v), colorGrad, pipelineParams);
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 getBackgroundColorCubemapBwd(
    const float3& dir,
    const float3& colorGrad,
    PipelineParams& pipelineParams) {
    int face;
    float u, v;
    dirToCubemapFaceUV(dir, face, u, v);
    return sampleEnvironmentBilinearBwd(computeCubemapBilinearFootprint(face, u, v), colorGrad, pipelineParams);
}

static __device__ __forceinline__ float3 getBackgroundColor(const float3 rayDir) {
    if (params.environment.data == nullptr || params.environment.width <= 0 || params.environment.height <= 0) {
        return make_float3(0.0f);
    }

    const float3 dir = rotateEnvironmentDirection(rayDir);
    if (params.environment.type == EnvironmentType_Cube) {
        return getBackgroundColorCubemap(dir);
    }
    return getBackgroundColorEquirect(dir);
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 getBackgroundColorBwd(
    const float3 rayDir,
    const float3& colorGrad,
    PipelineParams& pipelineParams) {
    if (params.environment.data == nullptr || params.environment.width <= 0 || params.environment.height <= 0) {
        return make_float3(0.0f);
    }

    const float3 dir = rotateEnvironmentDirection(rayDir);
    if (params.environment.type == EnvironmentType_Cube) {
        return getBackgroundColorCubemapBwd(dir, colorGrad, pipelineParams);
    }
    return getBackgroundColorEquirectBwd(dir, colorGrad, pipelineParams);
}

static __device__ __forceinline__ void accumulateLightContribution(pathPayload& path) {
    path.currentRayPayload.contribution = make_float3(0.0f);
    const bool hasEnvironment = params.environment.data != nullptr && params.environment.width > 0 && params.environment.height > 0;
    const bool firstSecondaryBounce = path.numBounces == 1u;

#ifdef ENABLE_MIS
    const float brdfSideMis = firstSecondaryBounce
        ? misWeight(path.currentRayPayload.scatterPdf, path.currentRayPayload.lightPdf)
        : 1.0f;
    const float lightSideMis = firstSecondaryBounce
        ? misWeight(path.emitterRayPayload.lightPdf, path.emitterRayPayload.scatterPdf)
        : 1.0f;
    const float3 neeContribution = path.emitterRayPayload.contribution * lightSideMis;
#endif

    if (params.renderOpts == 1) {
#ifdef ENABLE_MIS
        if (firstSecondaryBounce) {
            path.accumulatedLighting += neeContribution;
        }
#endif
        if (path.numBounces > 0u && path.currentRayPayload.interaction.valid) {
            path.currentRayPayload.contribution = path.pathThroughput * path.currentRayPayload.radiance;
#ifdef ENABLE_MIS
            if (firstSecondaryBounce) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting += path.currentRayPayload.contribution;
            path.accumulatedIndirectLighting += path.currentRayPayload.contribution;
        } else if (path.currentRayPayload.valid && hasEnvironment) {
            path.pathThroughput *= path.currentRayPayload.transmittance;
            path.currentRayPayload.contribution = path.pathThroughput * getBackgroundColor(path.currentRayPayload.ray.direction);
#ifdef ENABLE_MIS
            if (firstSecondaryBounce) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting += path.currentRayPayload.contribution;
            path.accumulatedDirectLighting += path.currentRayPayload.contribution;
        }
        return;
    }

#ifdef ENABLE_MIS
    if (firstSecondaryBounce) {
        path.accumulatedLighting += neeContribution;
        path.accumulatedDirectLighting += neeContribution;
    }
#endif

    if (path.currentRayPayload.valid && hasEnvironment) {
        path.pathThroughput *= path.currentRayPayload.transmittance;

        path.currentRayPayload.contribution = path.pathThroughput * getBackgroundColor(path.currentRayPayload.ray.direction);
#ifdef ENABLE_MIS
        if (firstSecondaryBounce) {
            path.currentRayPayload.contribution *= brdfSideMis;
        }
#endif
        path.currentRayPayload.radiance += path.currentRayPayload.contribution;
        path.accumulatedLighting += path.currentRayPayload.contribution;

        if (path.numBounces < 2u) {
            path.accumulatedDirectLighting += path.currentRayPayload.contribution;
        } else {
            path.accumulatedIndirectLighting += path.currentRayPayload.contribution;
        }
    }
}

template <typename PipelineParams>
static __device__ __forceinline__ void accumulateLightContributionBwd(
    pathPayload& path,
    PipelineParams& pipelineParams) {
    path.currentRayPayload.contribution = make_float3(0.0f);
    const bool hasEnvironment = params.environment.data != nullptr && params.environment.width > 0 && params.environment.height > 0;
    const bool firstSecondaryBounce = path.numBounces == 1u;

#ifdef ENABLE_MIS
    const float brdfSideMis = firstSecondaryBounce
        ? misWeight(path.currentRayPayload.scatterPdf, path.currentRayPayload.lightPdf)
        : 1.0f;
    const float lightSideMis = firstSecondaryBounce
        ? misWeight(path.emitterRayPayload.lightPdf, path.emitterRayPayload.scatterPdf)
        : 1.0f;
#endif

    if (params.renderOpts == 1) {
#ifdef ENABLE_MIS
        if (firstSecondaryBounce) {
            const float3 neeContribution = path.emitterRayPayload.contribution * lightSideMis;
            path.accumulatedLighting -= neeContribution;
        }
#endif
        if (path.numBounces > 0u && path.currentRayPayload.interaction.valid) {
            path.currentRayPayload.contribution = path.pathThroughput * path.currentRayPayload.radiance;
#ifdef ENABLE_MIS
            if (firstSecondaryBounce) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting -= path.currentRayPayload.contribution;
        } else if (path.currentRayPayload.valid && hasEnvironment) {
            path.pathThroughput *= path.currentRayPayload.transmittance;
            float3 environmentGrad = path.accumulatedLightingGrad * path.pathThroughput;
#ifdef ENABLE_MIS
            if (firstSecondaryBounce) {
                environmentGrad *= brdfSideMis;
            }
#endif
            const float3 background = getBackgroundColorBwd(path.currentRayPayload.ray.direction, environmentGrad, pipelineParams);
            path.currentRayPayload.contribution = path.pathThroughput * background;
#ifdef ENABLE_MIS
            if (firstSecondaryBounce) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting -= path.currentRayPayload.contribution;
        }
        return;
    }

#ifdef ENABLE_MIS
    if (firstSecondaryBounce) {
        const float3 neeContribution = path.emitterRayPayload.contribution * lightSideMis;
        path.accumulatedLighting -= neeContribution;
    }
#endif

    if (path.currentRayPayload.valid && hasEnvironment) {
        path.pathThroughput *= path.currentRayPayload.transmittance;

        float3 environmentGrad = path.accumulatedLightingGrad * path.pathThroughput;
#ifdef ENABLE_MIS
        if (firstSecondaryBounce) {
            environmentGrad *= brdfSideMis;
        }
#endif
        const float3 background = getBackgroundColorBwd(path.currentRayPayload.ray.direction, environmentGrad, pipelineParams);
        path.currentRayPayload.contribution = path.pathThroughput * background;
#ifdef ENABLE_MIS
        if (firstSecondaryBounce) {
            path.currentRayPayload.contribution *= brdfSideMis;
        }
#endif
        path.currentRayPayload.radiance += path.currentRayPayload.contribution;
        path.accumulatedLighting -= path.currentRayPayload.contribution;
    }
}


#endif // __CUDACC__
