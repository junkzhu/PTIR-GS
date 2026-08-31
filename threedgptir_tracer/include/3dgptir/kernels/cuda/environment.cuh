// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __CUDACC__

#include <3dgptir/environment.h>
#include <3dgptir/mathUtils.h>
#include <3dgptir/payLoad.h>
#include <3dgptir/kernels/cuda/meshLight.cuh>
#include <3dgptir/kernels/cuda/sampler.cuh>
#include <math_constants.h>

static __device__ __forceinline__ float misWeight(float pdfA, float pdfB);
static constexpr float EnvironmentDirectionGradEps = 1e-3f;

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

struct EnvironmentRotation {
    float sinZ;
    float cosZ;
    float sinX;
    float cosX;
};

static __device__ __forceinline__ EnvironmentRotation computeEnvironmentRotation() {
    const float rotZ = params.environment.offset.x * 2.0f * CUDART_PI_F + 0.5f * CUDART_PI_F;
    const float rotX = params.environment.offset.y * 2.0f * CUDART_PI_F;

    EnvironmentRotation rotation;
    sincosf(rotZ, &rotation.sinZ, &rotation.cosZ);
    sincosf(rotX, &rotation.sinX, &rotation.cosX);
    return rotation;
}

static __device__ __forceinline__ float3 rotateEnvironmentDirectionWithRotation(
    const float3& rayDir,
    const EnvironmentRotation& rotation) {
    const float3 rotatedDir = make_float3(
        rayDir.x * rotation.cosZ - rayDir.y * rotation.sinZ,
        rayDir.x * rotation.sinZ + rayDir.y * rotation.cosZ,
        rayDir.z);

    return make_float3(
        rotatedDir.x,
        rotatedDir.y * rotation.cosX - rotatedDir.z * rotation.sinX,
        rotatedDir.y * rotation.sinX + rotatedDir.z * rotation.cosX);
}

static __device__ __forceinline__ float3 rotateEnvironmentDirection(const float3& rayDir) {
    const EnvironmentRotation rotation = computeEnvironmentRotation();
    const float3 rotatedDir = make_float3(
        rayDir.x * rotation.cosZ - rayDir.y * rotation.sinZ,
        rayDir.x * rotation.sinZ + rayDir.y * rotation.cosZ,
        rayDir.z);

    return make_float3(
        rotatedDir.x,
        rotatedDir.y * rotation.cosX - rotatedDir.z * rotation.sinX,
        rotatedDir.y * rotation.sinX + rotatedDir.z * rotation.cosX);
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

static constexpr int NativeSGLobeParameterCount = 7;
static constexpr int NativeSGSamplingParameterCount = 2;
static constexpr float NativeSGUniformThreshold = 1.0e-6f;

static __device__ __forceinline__ bool hasNativeSGEnvironment() {
    return params.sgEnvironment.lobes != nullptr
        && params.sgEnvironment.sampling != nullptr
        && params.sgEnvironment.numLobes > 0;
}

static __device__ __forceinline__ const float* nativeSGLobe(const int index) {
    return params.sgEnvironment.lobes + index * NativeSGLobeParameterCount;
}

static __device__ __forceinline__ const float* nativeSGSampling(const int index) {
    return params.sgEnvironment.sampling + index * NativeSGSamplingParameterCount;
}

static __device__ __forceinline__ float3 nativeSGAxis(const float* lobe) {
    const float3 rawAxis = make_float3(lobe[0], lobe[1], lobe[2]);
    const float lengthSquared = dot(rawAxis, rawAxis);
    if (!isfinite(lengthSquared) || lengthSquared <= 1.0e-16f) {
        return make_float3(0.0f, 1.0f, 0.0f);
    }
    return rawAxis * rsqrtf(lengthSquared);
}

static __device__ __forceinline__ float nativeSGSharpness(const float* lobe) {
    return isfinite(lobe[3]) ? fminf(fmaxf(lobe[3], 1.0e-8f), 1.0e4f) : 1.0e-8f;
}

static __device__ __forceinline__ float nativeSGNormalization(const float sharpness) {
    return 2.0f * CUDART_PI_F * (-expm1f(-2.0f * sharpness)) / sharpness;
}

static __device__ __forceinline__ float nativeSGBasis(
    const float3& envDirection,
    const float3& axis,
    const float sharpness) {
    const float cosine = fminf(1.0f, fmaxf(-1.0f, dot(axis, envDirection)));
    return expf(sharpness * (cosine - 1.0f));
}

static __device__ __forceinline__ float3 evalNativeSGEnvironmentRotated(
    const float3& envDirection) {
    float3 radiance = make_float3(0.0f);
    for (int k = 0; k < params.sgEnvironment.numLobes; ++k) {
        const float* lobe = nativeSGLobe(k);
        const float basis = nativeSGBasis(
            envDirection, nativeSGAxis(lobe), nativeSGSharpness(lobe));
        const float3 amplitude = make_float3(lobe[4], lobe[5], lobe[6]);
        radiance += basis * amplitude;
    }
    return make_float3(
        isfinite(radiance.x) ? radiance.x : 0.0f,
        isfinite(radiance.y) ? radiance.y : 0.0f,
        isfinite(radiance.z) ? radiance.z : 0.0f);
}

static __device__ __forceinline__ float nativeSGEnvironmentPdf(const float3& rayDirection) {
    if (!hasNativeSGEnvironment()) {
        return 0.0f;
    }

    const float3 envDirection = rotateEnvironmentDirection(rayDirection);
    float pdf = 0.0f;
    for (int k = 0; k < params.sgEnvironment.numLobes; ++k) {
        const float* lobe = nativeSGLobe(k);
        const float* sampling = nativeSGSampling(k);
        const float sharpness = nativeSGSharpness(lobe);
        const float normalization = isfinite(sampling[0]) && sampling[0] > 0.0f
            ? sampling[0]
            : nativeSGNormalization(sharpness);
        const float alpha = isfinite(sampling[1]) && sampling[1] > 0.0f
            ? sampling[1]
            : 0.0f;
        pdf += alpha * nativeSGBasis(envDirection, nativeSGAxis(lobe), sharpness)
            / fmaxf(normalization, 1.0e-20f);
    }
    return isfinite(pdf) && pdf > 0.0f ? pdf : 0.0f;
}

static __device__ __forceinline__ float3 sampleNativeSGEnvironmentDirection(
    Sampler& sampler,
    float& pdf) {
    const float selectSample = sampler.next_1d();
    float cumulative = 0.0f;
    int selected = params.sgEnvironment.numLobes - 1;
    for (int k = 0; k < params.sgEnvironment.numLobes; ++k) {
        cumulative += fmaxf(nativeSGSampling(k)[1], 0.0f);
        if (selectSample < cumulative) {
            selected = k;
            break;
        }
    }

    const float* lobe = nativeSGLobe(selected);
    const float3 axis = nativeSGAxis(lobe);
    const float sharpness = nativeSGSharpness(lobe);
    const float u1 = sampler.next_1d();
    const float u2 = sampler.next_1d();

    float cosTheta;
    if (sharpness < NativeSGUniformThreshold) {
        cosTheta = -1.0f + 2.0f * u1;
    } else {
        const float oneMinusExpNegativeTwoLambda = -expm1f(-2.0f * sharpness);
        const float oneMinusInverseCdf = fminf(
            (1.0f - u1) * oneMinusExpNegativeTwoLambda,
            1.0f - 1.19209290e-7f);
        cosTheta = 1.0f
            + log1pf(-oneMinusInverseCdf) / sharpness;
        cosTheta = fminf(1.0f, fmaxf(-1.0f, cosTheta));
    }

    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * CUDART_PI_F * u2;
    float sinPhi;
    float cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);

    float3 tangent;
    float3 bitangent;
    branchlessONB(axis, tangent, bitangent);
    const float3 envDirection = safe_normalize(
        tangent * (sinTheta * cosPhi)
        + bitangent * (sinTheta * sinPhi)
        + axis * cosTheta);
    const float3 rayDirection = inverseRotateEnvironmentDirection(envDirection);
    pdf = nativeSGEnvironmentPdf(rayDirection);
    return rayDirection;
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

static __device__ __forceinline__ float3 sampleBackgroundColorRotatedDirection(const float3& dir) {
    if (params.environment.type == EnvironmentType_Cube) {
        return getBackgroundColorCubemap(dir);
    }
    return getBackgroundColorEquirect(dir);
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 sampleBackgroundColorRotatedDirectionBwd(
    const float3& dir,
    const float3& colorGrad,
    PipelineParams& pipelineParams) {
    if (params.environment.type == EnvironmentType_Cube) {
        return getBackgroundColorCubemapBwd(dir, colorGrad, pipelineParams);
    }
    return getBackgroundColorEquirectBwd(dir, colorGrad, pipelineParams);
}

static __device__ __forceinline__ float3 getBackgroundRayDirectionGradFast(
    const float3& rayDir,
    const float3& background,
    const float3& colorGrad,
    const EnvironmentRotation& rotation) {
    const float3 unitRayDir = safe_normalize(rayDir);

    float3 tangent;
    float3 bitangent;
    branchlessONB(unitRayDir, tangent, bitangent);

    const float perturbScale = rsqrtf(1.0f + EnvironmentDirectionGradEps * EnvironmentDirectionGradEps);
    const float3 tangentRayDir   = (unitRayDir + EnvironmentDirectionGradEps * tangent) * perturbScale;
    const float3 bitangentRayDir = (unitRayDir + EnvironmentDirectionGradEps * bitangent) * perturbScale;

    const float3 tangentBackground =
        sampleBackgroundColorRotatedDirection(rotateEnvironmentDirectionWithRotation(tangentRayDir, rotation));
    const float3 bitangentBackground =
        sampleBackgroundColorRotatedDirection(rotateEnvironmentDirectionWithRotation(bitangentRayDir, rotation));

    const float invEps = 1.0f / EnvironmentDirectionGradEps;
    const float tangentGrad = dot((tangentBackground - background) * invEps, colorGrad);
    const float bitangentGrad = dot((bitangentBackground - background) * invEps, colorGrad);
    return tangent * tangentGrad + bitangent * bitangentGrad;
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 evalNativeSGEnvironmentRotatedBwd(
    const float3& envDirection,
    const float3& colorGrad,
    PipelineParams& pipelineParams,
    float3* envDirectionGrad = nullptr) {
    float3 radiance = make_float3(0.0f);
    float3 directionGrad = make_float3(0.0f);
    for (int k = 0; k < params.sgEnvironment.numLobes; ++k) {
        const float* lobe = nativeSGLobe(k);
        const float3 axis = nativeSGAxis(lobe);
        const float sharpness = nativeSGSharpness(lobe);
        const float cosine = fminf(1.0f, fmaxf(-1.0f, dot(axis, envDirection)));
        const float basis = expf(sharpness * (cosine - 1.0f));
        const float3 amplitude = make_float3(lobe[4], lobe[5], lobe[6]);
        const float exponentGrad = dot(colorGrad, amplitude) * basis;

        radiance += basis * amplitude;
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][0], exponentGrad * sharpness * envDirection.x);
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][1], exponentGrad * sharpness * envDirection.y);
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][2], exponentGrad * sharpness * envDirection.z);
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][3], exponentGrad * (cosine - 1.0f));
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][4], colorGrad.x * basis);
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][5], colorGrad.y * basis);
        atomicAdd(&pipelineParams.sgEnvironmentGrad[k][6], colorGrad.z * basis);
        directionGrad += exponentGrad * sharpness * axis;
    }
    if (envDirectionGrad != nullptr) {
        *envDirectionGrad += directionGrad;
    }
    return radiance;
}

static __device__ __forceinline__ float3 getBackgroundColor(const float3 rayDir) {
    if (hasNativeSGEnvironment()) {
        return evalNativeSGEnvironmentRotated(rotateEnvironmentDirection(rayDir));
    }
    if (params.environment.data == nullptr || params.environment.width <= 0 || params.environment.height <= 0) {
        return make_float3(0.0f);
    }

    const float3 dir = rotateEnvironmentDirection(rayDir);
    return sampleBackgroundColorRotatedDirection(dir);
}

template <typename PipelineParams>
static __device__ __forceinline__ float3 getBackgroundColorBwd(
    const float3 rayDir,
    const float3& colorGrad,
    PipelineParams& pipelineParams,
    float3* rayDirGrad = nullptr) {
    if (hasNativeSGEnvironment()) {
        const float3 envDirection = rotateEnvironmentDirection(rayDir);
        float3 envDirectionGrad = make_float3(0.0f);
        const float3 background = evalNativeSGEnvironmentRotatedBwd(
            envDirection, colorGrad, pipelineParams, &envDirectionGrad);
        if (rayDirGrad != nullptr) {
            *rayDirGrad += inverseRotateEnvironmentDirection(envDirectionGrad);
        }
        return background;
    }
    if (params.environment.data == nullptr || params.environment.width <= 0 || params.environment.height <= 0) {
        return make_float3(0.0f);
    }

    const EnvironmentRotation rotation = computeEnvironmentRotation();
    const float3 dir = rotateEnvironmentDirectionWithRotation(rayDir, rotation);
    const float3 background = sampleBackgroundColorRotatedDirectionBwd(dir, colorGrad, pipelineParams);
    if (rayDirGrad != nullptr) {
        *rayDirGrad += getBackgroundRayDirectionGradFast(rayDir, background, colorGrad, rotation);
    }
    return background;
}

struct VisibleLightHit {
    bool valid;
    bool isEnvironment;
    float dist;
    float3 radiance;
};

static __device__ __forceinline__ VisibleLightHit emptyVisibleLightHit() {
    VisibleLightHit hit;
    hit.valid = false;
    hit.isEnvironment = false;
    hit.dist = 1e20f;
    hit.radiance = make_float3(0.0f);
    return hit;
}

static __device__ __forceinline__ bool intersectSphereAreaLight(
    const Ray& ray,
    const unsigned int lightId,
    float& hitDistance,
    float3& radiance) {
    if (lightId >= params.numLights || static_cast<unsigned int>(params.lights[lightId][0] + 0.5f) != 2u) {
        return false;
    }

    const float3 center = make_float3(
        params.lights[lightId][1],
        params.lights[lightId][2],
        params.lights[lightId][3]);
    const float radius = params.lights[lightId][4];
    if (radius <= 0.0f) {
        return false;
    }

    const float3 oc = ray.origin - center;
    const float b = dot(oc, ray.direction);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant <= 0.0f) {
        return false;
    }

    const float sqrtDiscriminant = sqrtf(discriminant);
    float t = -b - sqrtDiscriminant;
    if (t <= 1e-5f) {
        t = -b + sqrtDiscriminant;
    }
    if (t <= 1e-5f) {
        return false;
    }

    radiance = make_float3(
        params.lights[lightId][5],
        params.lights[lightId][6],
        params.lights[lightId][7]);
    if (radiance.x <= 0.0f && radiance.y <= 0.0f && radiance.z <= 0.0f) {
        return false;
    }

    hitDistance = t;
    return true;
}

static __device__ __forceinline__ bool intersectMeshAreaLight(
    const Ray& ray,
    const unsigned int lightId,
    float& hitDistance,
    float3& radiance) {
    if (!hasMeshLightData() || lightId >= params.numMeshLights) {
        return false;
    }

    const unsigned int triangleOffset = static_cast<unsigned int>(params.meshLights[lightId][0] + 0.5f);
    const unsigned int triangleCount = static_cast<unsigned int>(params.meshLights[lightId][1] + 0.5f);
    if (triangleCount == 0u || triangleOffset + triangleCount > params.numMeshLightTriangles) {
        return false;
    }

    radiance = make_float3(
        params.meshLights[lightId][4],
        params.meshLights[lightId][5],
        params.meshLights[lightId][6]);
    if (radiance.x <= 0.0f && radiance.y <= 0.0f && radiance.z <= 0.0f) {
        return false;
    }

    const bool twoSided = params.meshLights[lightId][7] > 0.5f;
    float nearest = 1e20f;
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
        if (!intersectTriangle(ray.origin, ray.direction, v0, v1, v2, t, u, v)) {
            continue;
        }

        const float3 normalUnnormalized = cross(v1 - v0, v2 - v0);
        const float normalLength = length(normalUnnormalized);
        if (normalLength <= 1e-12f) {
            continue;
        }
        const float cosLight = twoSided
            ? fabsf(dot(normalUnnormalized / normalLength, -ray.direction))
            : dot(normalUnnormalized / normalLength, -ray.direction);
        if (cosLight <= 1e-6f) {
            continue;
        }
        nearest = fminf(nearest, t);
    }

    if (nearest >= 1e19f) {
        return false;
    }
    hitDistance = nearest;
    return true;
}

static __device__ __forceinline__ VisibleLightHit getVisibleLightHit(const Ray& ray) {
    VisibleLightHit nearest = emptyVisibleLightHit();

#ifdef ENABLE_VISUALIZE_LIGHTS
    // Area lights are only intersected when visible light/environment rendering is requested.
    for (unsigned int i = 0; i < params.numLights; ++i) {
        float hitDistance = 1e20f;
        float3 radiance = make_float3(0.0f);
        if (intersectSphereAreaLight(ray, i, hitDistance, radiance) && hitDistance < nearest.dist) {
            nearest.valid = true;
            nearest.isEnvironment = false;
            nearest.dist = hitDistance;
            nearest.radiance = radiance;
        }
    }
    // The optional triangle GAS supersedes the O(numTriangles) fallback.
    if (params.sceneMeshHandle == 0) {
        for (unsigned int i = 0; i < params.numMeshLights; ++i) {
            float hitDistance = 1e20f;
            float3 radiance = make_float3(0.0f);
            if (intersectMeshAreaLight(ray, i, hitDistance, radiance) && hitDistance < nearest.dist) {
                nearest.valid = true;
                nearest.isEnvironment = false;
                nearest.dist = hitDistance;
                nearest.radiance = radiance;
            }
        }
    }
#endif

    if (!nearest.valid && (hasNativeSGEnvironment()
        || (params.environment.data != nullptr && params.environment.width > 0 && params.environment.height > 0))) {
        nearest.valid = true;
        nearest.isEnvironment = true;
        nearest.dist = 1e20f;
        nearest.radiance = getBackgroundColor(ray.direction);
    }

    return nearest;
}

static __device__ __forceinline__ VisibleLightHit getPayloadVisibleLightHit(
    const rayPayload& payload) {
    if (!payload.sceneMeshHit) {
        return getVisibleLightHit(payload.ray);
    }
    VisibleLightHit hit;
    hit.valid = true;
    hit.isEnvironment = false;
    hit.dist = 0.0f;
    hit.radiance = payload.sceneMeshEmission;
    return hit;
}

static __device__ __forceinline__ void accumulateLightContribution(pathPayload& path) {
    path.currentRayPayload.contribution = make_float3(0.0f);
    const bool primarySurface = path.numBounces == 0u;
    const bool firstSecondaryBounce = path.numBounces == 1u;
    const bool hasNeeForPreviousSurface =
        firstSecondaryBounce || (params.enableSecondaryNee && path.numBounces > 1u);

#ifdef ENABLE_MIS
    const float brdfSideMis = hasNeeForPreviousSurface
        ? misWeight(path.currentRayPayload.scatterPdf, path.currentRayPayload.lightPdf)
        : 1.0f;
    const bool hasBrdfContinuation = path.numBounces + 1u < path.maxBounces;
    const float lightSideMis = hasBrdfContinuation
        ? misWeight(path.emitterRayPayload.lightPdf, path.emitterRayPayload.scatterPdf)
        : 1.0f;
    const float3 neeContribution = path.emitterRayPayload.contribution * lightSideMis;
#endif

    if (params.renderOpts == 1) {
#ifdef ENABLE_MIS
        if (path.currentRayPayload.interaction.valid
            && (primarySurface || params.enableSecondaryNee)) {
            path.accumulatedLighting += neeContribution;
            if (primarySurface) {
                path.accumulatedDirectLighting += neeContribution;
            } else {
                path.accumulatedIndirectLighting += neeContribution;
            }
        }
#endif
        if (path.numBounces > 0u && path.currentRayPayload.interaction.valid) {
            path.currentRayPayload.contribution = path.pathThroughput * path.currentRayPayload.radiance;
#ifdef ENABLE_MIS
            if (hasNeeForPreviousSurface) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting += path.currentRayPayload.contribution;
            path.accumulatedIndirectLighting += path.currentRayPayload.contribution;
        } else if (path.currentRayPayload.valid) {
            const VisibleLightHit visibleLight = getPayloadVisibleLightHit(path.currentRayPayload);
            if (!visibleLight.valid) {
                return;
            }
            path.pathThroughput *= path.currentRayPayload.transmittance;
            path.currentRayPayload.light += visibleLight.radiance;
            path.accumulatedLightNoBrdf += visibleLight.radiance;
            path.currentRayPayload.contribution = path.pathThroughput * path.currentRayPayload.light;
#ifdef ENABLE_MIS
            if (hasNeeForPreviousSurface) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting += path.currentRayPayload.contribution;
            path.accumulatedDirectLighting += path.currentRayPayload.contribution;
        }
        return;
    }

#ifdef ENABLE_MIS
    if (path.currentRayPayload.interaction.valid
        && (primarySurface || params.enableSecondaryNee)) {
        path.accumulatedLighting += neeContribution;
        if (primarySurface) {
            path.accumulatedDirectLighting += neeContribution;
        } else {
            path.accumulatedIndirectLighting += neeContribution;
        }
    }
#endif

    if (path.currentRayPayload.valid) {
        const VisibleLightHit visibleLight = getPayloadVisibleLightHit(path.currentRayPayload);
        if (!visibleLight.valid) {
            return;
        }
        path.pathThroughput *= path.currentRayPayload.transmittance;
        path.currentRayPayload.light += visibleLight.radiance;
        path.accumulatedLightNoBrdf += visibleLight.radiance;
        path.currentRayPayload.contribution = path.pathThroughput * path.currentRayPayload.light;
#ifdef ENABLE_MIS
        if (hasNeeForPreviousSurface) {
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
    const bool primarySurface = path.numBounces == 0u;
    const bool firstSecondaryBounce = path.numBounces == 1u;
    const bool hasNeeForPreviousSurface =
        firstSecondaryBounce || (params.enableSecondaryNee && path.numBounces > 1u);

#ifdef ENABLE_MIS
    const float brdfSideMis = hasNeeForPreviousSurface
        ? misWeight(path.currentRayPayload.scatterPdf, path.currentRayPayload.lightPdf)
        : 1.0f;
    const bool hasBrdfContinuation = path.numBounces + 1u < path.maxBounces;
    const float lightSideMis = hasBrdfContinuation
        ? misWeight(path.emitterRayPayload.lightPdf, path.emitterRayPayload.scatterPdf)
        : 1.0f;
#endif

    if (params.renderOpts == 1) {
#ifdef ENABLE_MIS
        if (path.currentRayPayload.interaction.valid
            && (primarySurface || params.enableSecondaryNee)) {
            const float3 neeContribution = path.emitterRayPayload.contribution * lightSideMis;
            path.accumulatedLighting -= neeContribution;
        }
#endif
        if (path.numBounces > 0u && path.currentRayPayload.interaction.valid) {
            path.currentRayPayload.contribution = path.pathThroughput * path.currentRayPayload.radiance;
#ifdef ENABLE_MIS
            if (hasNeeForPreviousSurface) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting -= path.currentRayPayload.contribution;
        } else if (path.currentRayPayload.valid) {
            const VisibleLightHit visibleLight = getPayloadVisibleLightHit(path.currentRayPayload);
            if (!visibleLight.valid) {
                return;
            }
            path.pathThroughput *= path.currentRayPayload.transmittance;
            float3 environmentGrad = path.accumulatedLightingGrad * path.pathThroughput;
#ifdef ENABLE_MIS
            if (hasNeeForPreviousSurface) {
                environmentGrad *= brdfSideMis;
            }
#endif
            environmentGrad += path.accumulatedLightNoBrdfGrad;
            const float3 visibleRadiance = visibleLight.isEnvironment
                ? getBackgroundColorBwd(path.currentRayPayload.ray.direction, environmentGrad, pipelineParams, &path.currentRayPayload.rayDirGrad)
                : visibleLight.radiance;
            path.currentRayPayload.contribution = path.pathThroughput * visibleRadiance;
#ifdef ENABLE_MIS
            if (hasNeeForPreviousSurface) {
                path.currentRayPayload.contribution *= brdfSideMis;
            }
#endif
            path.accumulatedLighting -= path.currentRayPayload.contribution;
        }
        return;
    }

#ifdef ENABLE_MIS
    if (path.currentRayPayload.interaction.valid
        && (primarySurface || params.enableSecondaryNee)) {
        const float3 neeContribution = path.emitterRayPayload.contribution * lightSideMis;
        path.accumulatedLighting -= neeContribution;
    }
#endif

    if (path.currentRayPayload.valid) {
        const VisibleLightHit visibleLight = getPayloadVisibleLightHit(path.currentRayPayload);
        if (!visibleLight.valid) {
            return;
        }
        path.pathThroughput *= path.currentRayPayload.transmittance;

        float3 environmentGrad = path.accumulatedLightingGrad * path.pathThroughput;
#ifdef ENABLE_MIS
        if (hasNeeForPreviousSurface) {
            environmentGrad *= brdfSideMis;
        }
#endif
        environmentGrad += path.accumulatedLightNoBrdfGrad;
        const float3 visibleRadiance = visibleLight.isEnvironment
            ? getBackgroundColorBwd(path.currentRayPayload.ray.direction, environmentGrad, pipelineParams, &path.currentRayPayload.rayDirGrad)
            : visibleLight.radiance;
        path.currentRayPayload.contribution = path.pathThroughput * visibleRadiance;
#ifdef ENABLE_MIS
        if (hasNeeForPreviousSurface) {
            path.currentRayPayload.contribution *= brdfSideMis;
        }
#endif
        path.currentRayPayload.radiance += path.currentRayPayload.contribution;
        path.accumulatedLighting -= path.currentRayPayload.contribution;
    }
}


#endif // __CUDACC__
