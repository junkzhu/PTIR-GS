// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optix.h>

#include <3dgptir/kernels/cuda/gaussianParticles.cuh>
#include <3dgptir/kernels/cuda/material.cuh>
#include <3dgptir/kernels/cuda/sampler.cuh>
#include <3dgptir/payLoad.h>
#include <3dgptir/pipelineParameters.h>

static constexpr float kMaxSelfOcclusionOffset = 1e-1f;

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const Ray& ray) {
    const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - ray.origin) / ray.direction;
    const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - ray.origin) / ray.direction;
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ uint32_t optixPrimitiveIndex() {
    return PipelineParameters::InstancePrimitive ? optixGetInstanceIndex() : (PipelineParameters::CustomPrimitive ? optixGetPrimitiveIndex() : static_cast<uint32_t>(optixGetPrimitiveIndex() / params.gPrimNumTri));
}

static __device__ __inline__ void trace(
    RayPayload& rayPayload,
    const Ray& ray,
    const float tmin,
    const float tmax) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = r30 = RayHit::InvalidParticleId;
    r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = r31 = __float_as_int(RayHit::InfiniteDistance);

    // Trace the ray against our scene hierarchy
    optixTrace(params.handle, ray.origin, ray.direction,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
               r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31);

    rayPayload[0].particleId  = r0;
    rayPayload[0].distance    = __uint_as_float(r1);
    rayPayload[1].particleId  = r2;
    rayPayload[1].distance    = __uint_as_float(r3);
    rayPayload[2].particleId  = r4;
    rayPayload[2].distance    = __uint_as_float(r5);
    rayPayload[3].particleId  = r6;
    rayPayload[3].distance    = __uint_as_float(r7);
    rayPayload[4].particleId  = r8;
    rayPayload[4].distance    = __uint_as_float(r9);
    rayPayload[5].particleId  = r10;
    rayPayload[5].distance    = __uint_as_float(r11);
    rayPayload[6].particleId  = r12;
    rayPayload[6].distance    = __uint_as_float(r13);
    rayPayload[7].particleId  = r14;
    rayPayload[7].distance    = __uint_as_float(r15);
    rayPayload[8].particleId  = r16;
    rayPayload[8].distance    = __uint_as_float(r17);
    rayPayload[9].particleId  = r18;
    rayPayload[9].distance    = __uint_as_float(r19);
    rayPayload[10].particleId = r20;
    rayPayload[10].distance   = __uint_as_float(r21);
    rayPayload[11].particleId = r22;
    rayPayload[11].distance   = __uint_as_float(r23);
    rayPayload[12].particleId = r24;
    rayPayload[12].distance   = __uint_as_float(r25);
    rayPayload[13].particleId = r26;
    rayPayload[13].distance   = __uint_as_float(r27);
    rayPayload[14].particleId = r28;
    rayPayload[14].distance   = __uint_as_float(r29);
    rayPayload[15].particleId = r30;
    rayPayload[15].distance   = __uint_as_float(r31);
}

static __device__ __inline__ void rayIntersect(
    const Ray& ray,
    rayPayload& payload,
    Sampler& sampler) {
    constexpr float epsT = 1e-9;
    const float2 minMaxT = intersectAABB(params.aabb, ray);
    RayPayload hitPayload;

    payload = rayPayload(ray, fmaxf(0.0f, minMaxT.x - epsT));
    float integratedDepth = 0.f;
    Material integratedMaterial;
    float3 integratedShadingnormal = make_float3(0.f);

    while ((payload.lastHitDistance <= minMaxT.y) && (payload.transmittance > params.minTransmittance)) {
        trace(hitPayload, payload.ray, payload.lastHitDistance + epsT, minMaxT.y + epsT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = hitPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (payload.transmittance > params.minTransmittance)) {
                const bool acceptedHit = processHit<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                    payload.ray.origin,
                    payload.ray.direction,
                    rayHit.particleId,
                    params.particleDensity,
                    params.particleMaterial,
                    params.particleRadiance,
                    params.hitMinGaussianResponse,
                    params.alphaMinThreshold,
                    params.sphDegree,
                    params.particleShadingNormal,
                    &payload.transmittance,
                    &payload.radiance,
                    &integratedDepth,
                    &payload.hitDistanceSecondMoment,
                    &integratedMaterial,
#ifdef ENABLE_NORMALS
                    &integratedShadingnormal,
                    &payload.normal
#else
                    nullptr,
                    nullptr
#endif
                );
                if (acceptedHit) {
                    payload.hit = 1;
                    const float rayOpacity = 1.0f - payload.transmittance;
                    payload.depthDistortion = fmaxf(rayOpacity * payload.hitDistanceSecondMoment - integratedDepth * integratedDepth, 0.0f);

                    // NOTE(qi): Race condition here, but as we are writing the same value, it seems it is safe.
                    params.particleVisibility[rayHit.particleId] = 1;
                }

                payload.lastHitDistance = fmaxf(payload.lastHitDistance, rayHit.distance);

#ifdef ENABLE_HIT_COUNTS
                payload.hitsCount += acceptedHit ? 1.0f : 0.f;
#endif
            }
        }
    }

    const float rayOpacity = 1.0f - payload.transmittance;
    payload.hitDistance = integratedDepth;
    payload.valid = false;
    if (payload.hit && rayOpacity > 0.5f) {
        payload.interaction = Interaction(
            payload.ray.origin,
            payload.ray.direction,
            integratedDepth,
            integratedShadingnormal,
            integratedMaterial,
            rayOpacity);
    } else {
        const float rand_u = sampler.next_1d();
        payload.valid = (rand_u < payload.transmittance);
    }
}

template <typename PipelineParams>
static __device__ __inline__ void rayIntersectBwd(
    const Ray& ray,
    const float rayOpacity,
    const float rayMaxHitDistance,
    const MaterialGrad& materialGrad,
    const PipelineParams& pipelineParams) {
    const float invRayOpacity = 1.0f / fmaxf(rayOpacity, 1e-12f);
    const Material rayMaterialGrad(
        materialGrad.dAlbedo * invRayOpacity,
        materialGrad.dRoughness * invRayOpacity,
        materialGrad.dMetallic * invRayOpacity);

    constexpr float epsT = 1e-9;
    const float2 minMaxT = intersectAABB(pipelineParams.aabb, ray);
    float startT         = fmaxf(0.0f, minMaxT.x - epsT);
    const float endT     = fminf(rayMaxHitDistance, minMaxT.y) + epsT;

    float rayTransmittance = 1.f;
    RayPayload hitPayload;

    while ((startT < endT) && (rayTransmittance > pipelineParams.minTransmittance)) {
        trace(hitPayload, ray, startT + epsT, endT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = hitPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > pipelineParams.minTransmittance)) {
                float3 particlePosition;
                float3 particleScale;
                float33 particleRotation;
                float particleDensity;
                fetchParticleDensity(
                    rayHit.particleId,
                    pipelineParams.particleDensity,
                    particlePosition,
                    particleScale,
                    particleRotation,
                    particleDensity);

                const float3 giscl   = make_float3(1.0f / particleScale.x, 1.0f / particleScale.y, 1.0f / particleScale.z);
                const float3 gposc   = ray.origin - particlePosition;
                const float3 gposcr  = gposc * particleRotation;
                const float3 gro     = giscl * gposcr;
                const float3 rayDirR = ray.direction * particleRotation;
                const float3 grdu    = giscl * rayDirR;
                const float3 grd     = safe_normalize(grdu);
                const float3 gcrod   = PipelineParameters::SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
                const float grayDist = dot(gcrod, gcrod);

                const float gres   = particleResponse<PipelineParameters::ParticleKernelDegree>(grayDist);
                const float galpha = fminf(0.99f, gres * particleDensity);
                if ((gres > pipelineParams.hitMinGaussianResponse) && (galpha > pipelineParams.alphaMinThreshold)) {
                    const float weight = galpha * rayTransmittance;
                    Material& particleMaterialGrad = pipelineParams.particleMaterialGrad[rayHit.particleId];
                    atomicAdd(&particleMaterialGrad.albedo.x, weight * rayMaterialGrad.albedo.x);
                    atomicAdd(&particleMaterialGrad.albedo.y, weight * rayMaterialGrad.albedo.y);
                    atomicAdd(&particleMaterialGrad.albedo.z, weight * rayMaterialGrad.albedo.z);
                    atomicAdd(&particleMaterialGrad.roughness, weight * rayMaterialGrad.roughness);
#ifdef ENABLE_METALLIC
                    atomicAdd(&particleMaterialGrad.metallic, weight * rayMaterialGrad.metallic);
#endif
                    rayTransmittance *= (1.0f - galpha);
                }
                startT = fmaxf(startT, rayHit.distance);
            }
        }
    }
}

static __device__ __inline__ void selfOcclusionRejection(Ray& ray) {
    ray.origin = ray.origin + kMaxSelfOcclusionOffset * ray.direction;
}

static __device__ __inline__ void sampleBrdfNextDirection(
    pathPayload& path,
    Sampler& sampler) {
    const Ray currentRay = path.currentRayPayload.ray;
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float3 nextRayDirection = currentRay.direction;
    const float3 brdf = sampled_fast_brdf(
        currentRay.direction,
        sampler,
        currentInteraction,
        nextRayDirection);

    path.pathThroughput *= brdf;
    Ray nextRay(currentInteraction.position, nextRayDirection);
    selfOcclusionRejection(nextRay);
    path.currentRayPayload = rayPayload(nextRay, 0.0f);
}


template <typename PipelineParams>
static __device__ __inline__ void sampleBrdfNextDirectionBwd(
    pathPayload& path,
    Sampler& sampler,
    const PipelineParams& pipelineParams) {
    const Ray currentRay = path.currentRayPayload.ray;
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float3 nextRayDirection = currentRay.direction;
    const FastBrdfValueGrad brdf = sampled_fast_brdf_with_grads(
        currentRay.direction,
        sampler,
        currentInteraction,
        nextRayDirection);

    const float3 dLoss_dBrdfNumerator = path.accumulatedLightingGrad * path.accumulatedLighting;
    const float3 dLoss_dBrdf = make_float3(
        brdf.value.x > FastBrdfEps ? dLoss_dBrdfNumerator.x / brdf.value.x : 0.0f,
        brdf.value.y > FastBrdfEps ? dLoss_dBrdfNumerator.y / brdf.value.y : 0.0f,
        brdf.value.z > FastBrdfEps ? dLoss_dBrdfNumerator.z / brdf.value.z : 0.0f);
    path.currentRayPayload.interaction.materialGrad.dAlbedo = dLoss_dBrdf * brdf.dBrdf_dAlbedo;
    path.currentRayPayload.interaction.materialGrad.dRoughness = dot(dLoss_dBrdf, brdf.dBrdf_dRoughness);
#ifdef ENABLE_METALLIC
    path.currentRayPayload.interaction.materialGrad.dMetallic = dot(dLoss_dBrdf, brdf.dBrdf_dMetallic);
#else
    path.currentRayPayload.interaction.materialGrad.dMetallic = 0.0f;
#endif

    rayIntersectBwd(
        currentRay,
        1.0f - path.currentRayPayload.transmittance,
        path.currentRayPayload.lastHitDistance,
        path.currentRayPayload.interaction.materialGrad,
        pipelineParams);

    path.pathThroughput *= brdf.value;
    Ray nextRay(currentInteraction.position, nextRayDirection);
    selfOcclusionRejection(nextRay);
    path.currentRayPayload = rayPayload(nextRay, 0.0f);
}

static __device__ __inline__ void writePrimaryRayOutputs(
    const uint3& idx,
    const rayPayload& payload) {
    params.rayRadiance[idx.z][idx.y][idx.x][0]    = payload.radiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = payload.radiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = payload.radiance.z;
    params.rayDensity[idx.z][idx.y][idx.x][0]     = 1 - payload.transmittance;
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = payload.hitDistance;
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = payload.lastHitDistance;
    params.rayHitDistanceSecondMoment[idx.z][idx.y][idx.x][0] = payload.hitDistanceSecondMoment;
    params.rayDepthDistortion[idx.z][idx.y][idx.x][0] = payload.depthDistortion;
#ifdef ENABLE_NORMALS
    params.rayNormal[idx.z][idx.y][idx.x][0] = payload.normal.x;
    params.rayNormal[idx.z][idx.y][idx.x][1] = payload.normal.y;
    params.rayNormal[idx.z][idx.y][idx.x][2] = payload.normal.z;
    params.rayShadingNormal[idx.z][idx.y][idx.x][0] = payload.interaction.shadingnormal.x;
    params.rayShadingNormal[idx.z][idx.y][idx.x][1] = payload.interaction.shadingnormal.y;
    params.rayShadingNormal[idx.z][idx.y][idx.x][2] = payload.interaction.shadingnormal.z;
#endif
    params.rayMaterial[idx.z][idx.y][idx.x][0] = payload.interaction.material.albedo.x;
    params.rayMaterial[idx.z][idx.y][idx.x][1] = payload.interaction.material.albedo.y;
    params.rayMaterial[idx.z][idx.y][idx.x][2] = payload.interaction.material.albedo.z;
    params.rayMaterial[idx.z][idx.y][idx.x][3] = payload.interaction.material.roughness;
    params.rayMaterial[idx.z][idx.y][idx.x][4] = payload.interaction.material.metallic;
#ifdef ENABLE_HIT_COUNTS
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = payload.hitsCount;
#endif
}

static __device__ __inline__ void writePbrOutputs(
    const uint3& idx,
    const pathPayload& payload) {
    params.rayPbr[idx.z][idx.y][idx.x][0] = payload.accumulatedLighting.x;
    params.rayPbr[idx.z][idx.y][idx.x][1] = payload.accumulatedLighting.y;
    params.rayPbr[idx.z][idx.y][idx.x][2] = payload.accumulatedLighting.z;

    params.rayPbrComponents[idx.z][idx.y][idx.x][0][0] = payload.accumulatedDirectLighting.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][1] = payload.accumulatedDirectLighting.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][2] = payload.accumulatedDirectLighting.z;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][0] = payload.accumulatedIndirectLighting.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][1] = payload.accumulatedIndirectLighting.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][2] = payload.accumulatedIndirectLighting.z;
}
