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
#include <3dgptir/kernels/cuda/lightSampler.cuh>
#include <3dgptir/kernels/cuda/material.cuh>
#include <3dgptir/kernels/cuda/sampler.cuh>
#include <3dgptir/payLoad.h>
#include <3dgptir/pipelineParameters.h>

static constexpr float kSelfOcclusionRayOriginOffset = 2e-2f;
// MAX_SELF_OCCLUSION_OFFSET is always supplied by the host through NVRTC.
// Keep the ray-origin offset subtraction in CUDA so this threshold is derived
// at compile time on the device side.
static constexpr float kMaxSelfOcclusionOffset =
    MAX_SELF_OCCLUSION_OFFSET - kSelfOcclusionRayOriginOffset;
static constexpr float kSelfOcclusionNormalDotThreshold = 0.0f;
static constexpr float kSelfOcclusionInteractionNormalDotThreshold = 0.9f;
static constexpr float kHardGeometryOpacityThreshold = 0.5f;

#ifdef ENABLE_RUSSIAN_ROULETTE
// Preserve the primary and first two secondary segments. Roulette starts at
// the third bounce while max_bounces remains the hard path-length limit.
static constexpr unsigned int kRussianRouletteStartBounce = 3u;
static constexpr float kRussianRouletteMinSurvivalProbability = 0.05f;
static constexpr float kRussianRouletteMaxSurvivalProbability = 0.95f;

static __device__ __forceinline__ bool applyRussianRoulette(
    pathPayload& path,
    Sampler& sampler) {
    if (path.numBounces < kRussianRouletteStartBounce) {
        return true;
    }

    const float throughputMax = fmaxf(
        path.pathThroughput.x,
        fmaxf(path.pathThroughput.y, path.pathThroughput.z));
    const float survivalProbability = fminf(
        kRussianRouletteMaxSurvivalProbability,
        fmaxf(kRussianRouletteMinSurvivalProbability, throughputMax));
    if (sampler.next_1d() >= survivalProbability) {
        path.active = 0u;
        return false;
    }

    // Compensate surviving paths so roulette changes variance, not expectation.
    path.pathThroughput /= survivalProbability;
    return true;
}
#endif

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];

struct SceneMeshHit {
    bool valid;
    unsigned int triangleId;
    float distance;
};

static __device__ __forceinline__ SceneMeshHit traceClosestSceneMesh(
    const Ray& ray,
    const float tmin = 0.0f,
    const float tmax = RayHit::InfiniteDistance) {
    SceneMeshHit hit;
    hit.valid = false;
    hit.triangleId = RayHit::InvalidParticleId;
    hit.distance = RayHit::InfiniteDistance;
    if (params.sceneMeshHandle == 0 || params.numSceneMeshTriangles == 0 || tmin >= tmax) {
        return hit;
    }

    unsigned int triangleId = RayHit::InvalidParticleId;
    unsigned int distance = __float_as_uint(RayHit::InfiniteDistance);
    optixTrace(
        params.sceneMeshHandle,
        ray.origin,
        ray.direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1,
        1,
        0,
        triangleId,
        distance);

    if (triangleId != RayHit::InvalidParticleId) {
        hit.valid = true;
        hit.triangleId = triangleId;
        hit.distance = __uint_as_float(distance);
    }
    return hit;
}

static __device__ __forceinline__ SceneMeshHit traceClosestSurfaceMesh(const Ray& ray) {
    float tmin = 0.0f;
    for (unsigned int i = 0; i < params.numMeshLights; ++i) {
        const SceneMeshHit hit = traceClosestSceneMesh(ray, tmin);
        unsigned int meshId = 0u;
        if (!hit.valid || !findSceneMesh(hit.triangleId, meshId)) {
            return hit;
        }
        if (isSceneMeshSurface(meshId)) {
            return hit;
        }
        tmin = hit.distance + 1e-4f;
    }
    return SceneMeshHit{false, RayHit::InvalidParticleId, RayHit::InfiniteDistance};
}

static __device__ __forceinline__ float misWeight(float pdfA, float pdfB) {
    const float a2 = pdfA * pdfA;
    const float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-6f);
}

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

static __device__ __forceinline__ bool isSelfOcclusionHit(
    const RayHit& rayHit,
    const Ray& ray,
    const float3& interactionShadingNormal,
    const float* particleShadingNormal) {
    if ((rayHit.distance > kMaxSelfOcclusionOffset) || (particleShadingNormal == nullptr)) {
        return false;
    }

    const float3 shadingNormal = make_float3(
        particleShadingNormal[rayHit.particleId * 3 + 0],
        particleShadingNormal[rayHit.particleId * 3 + 1],
        particleShadingNormal[rayHit.particleId * 3 + 2]);

    // This heuristic condition is an additional interaction shadingnormal safeguard
    // that reduces light leakage from poorly supervised GS normals.
    return dot(shadingNormal, ray.direction) > kSelfOcclusionNormalDotThreshold
        && dot(shadingNormal, interactionShadingNormal) > kSelfOcclusionInteractionNormalDotThreshold;
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false>
static __device__ __forceinline__ float shadowHitAlpha(
    const Ray& ray,
    const int32_t particleIdx) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleRotation;
    float particleDensity;
    fetchParticleDensity(
        particleIdx,
        params.particleDensity,
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
    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres = particleResponse<ParticleKernelDegree>(grayDist);
    const float alpha = fminf(0.99f, gres * particleDensity);
    return (gres > params.hitMinGaussianResponse && alpha > params.alphaMinThreshold) ? alpha : 0.0f;
}

template <bool EnableSelfOcclusionRejection = true>
static __device__ __inline__ bool traceShadowMonteCarloOccluded(
    const Ray& ray,
    const float3& interactionShadingNormal,
    Sampler& sampler,
    const float tmin = 0.0f,
    const float tmax = RayHit::InfiniteDistance) {
    // This branch is uniform for a launch.  Gaussian-only inversion has a
    // zero handle and pays no additional OptiX trace.
    if (params.sceneMeshHandle != 0
        && traceClosestSceneMesh(ray, tmin, tmax).valid) {
        return true;
    }

    constexpr float epsT = 1e-9f;
    const float2 minMaxT = intersectAABB(params.aabb, ray);
    float startT = fmaxf(tmin, minMaxT.x - epsT);
    const float endT = fminf(tmax, minMaxT.y) + epsT;
    if (startT >= endT) {
        return false;
    }

    RayPayload hitPayload;

    while (startT < endT) {
        trace(hitPayload, ray, startT + epsT, endT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

        float batchEndT = startT;
#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; ++i) {
            const RayHit rayHit = hitPayload[i];
            if (rayHit.particleId == RayHit::InvalidParticleId) {
                break;
            }
            if (rayHit.distance > endT) {
                break;
            }

            batchEndT = fmaxf(batchEndT, rayHit.distance);
            if (EnableSelfOcclusionRejection && isSelfOcclusionHit(
                    rayHit,
                    ray,
                    interactionShadingNormal,
                    params.particleShadingNormal)) {
                continue;
            }

            const float alpha = shadowHitAlpha<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                ray,
                rayHit.particleId);
            if (alpha > 0.0f && sampler.next_1d() < alpha) {
                return true;
            }
        }

        if (batchEndT <= startT) {
            break;
        }
        startT = batchEndT;
    }

    return false;
}

template <bool EnableSelfOcclusionRejection>
static __device__ __inline__ void rayIntersect(
    const Ray& ray,
    rayPayload& payload,
    Sampler& sampler,
    const float rayMaxT = RayHit::InfiniteDistance) {
    constexpr float epsT = 1e-9;
    float2 minMaxT = intersectAABB(params.aabb, ray);
    minMaxT.y = fminf(minMaxT.y, rayMaxT);
    RayPayload hitPayload;

#ifdef ENABLE_MIS
    const float scatterPdf = payload.scatterPdf;
    const float lightPdf   = payload.lightPdf;
#endif
    const float3 interactionShadingNormal = payload.interactionShadingNormal;
    payload = rayPayload(ray, fmaxf(0.0f, minMaxT.x - epsT));
    payload.interactionShadingNormal = interactionShadingNormal;
#ifdef ENABLE_MIS
    payload.scatterPdf = scatterPdf;
    payload.lightPdf   = lightPdf;
#endif
    float integratedDepth = 0.f;
    Material integratedMaterial;
    float3 integratedShadingnormal = make_float3(0.f);
    Material* integratedMaterialOut = &integratedMaterial;
    float3* integratedShadingnormalOut = nullptr;
    float3* integratedNormalOut = nullptr;
#ifdef ENABLE_NORMALS
    integratedShadingnormalOut = &integratedShadingnormal;
    integratedNormalOut = &payload.normal;
#endif
#ifdef ENABLE_DISCRETE_MODEL
    integratedMaterialOut = nullptr;
    integratedShadingnormalOut = nullptr;
    integratedNormalOut = nullptr;

    // Keep surface selection independent from the BRDF/MIS random stream.
    Sampler discreteSampler = sampler.fork(1ull);
    unsigned int selectedParticleId = Interaction::InvalidParticleId;
    ParticleHit selectedHit;
#endif

    while ((payload.lastHitDistance <= minMaxT.y) && (payload.transmittance > params.minTransmittance)) {
        trace(hitPayload, payload.ray, payload.lastHitDistance + epsT, minMaxT.y + epsT);
        if (hitPayload[0].particleId == RayHit::InvalidParticleId) {
            break;
        }

#pragma unroll
        for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++) {
            const RayHit rayHit = hitPayload[i];

            if ((rayHit.particleId != RayHit::InvalidParticleId) && (payload.transmittance > params.minTransmittance)) {
                if (EnableSelfOcclusionRejection && isSelfOcclusionHit(
                        rayHit,
                        payload.ray,
                        payload.interactionShadingNormal,
                        params.particleShadingNormal)) {
                    payload.lastHitDistance = fmaxf(payload.lastHitDistance, rayHit.distance);
                    continue;
                }

#ifdef ENABLE_DISCRETE_MODEL
                ParticleHit particleHit;
                ParticleHit* particleHitOut = &particleHit;
#else
                ParticleHit* particleHitOut = nullptr;
#endif
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
                    integratedMaterialOut,
                    integratedShadingnormalOut,
                    integratedNormalOut,
                    particleHitOut);
                if (acceptedHit) {
#ifdef ENABLE_DISCRETE_MODEL
                    const float accumulatedOpacity = 1.0f - payload.transmittance;
                    if (discreteSampler.next_1d() * accumulatedOpacity < particleHit.opacityWeight) {
                        selectedParticleId = rayHit.particleId;
                        selectedHit = particleHit;
                    }
#endif
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

    payload.hitDistance = integratedDepth;
    payload.valid = false;
#ifdef ENABLE_DISCRETE_MODEL
    const bool hasInteraction = selectedParticleId != Interaction::InvalidParticleId;
#else
    const float rayOpacity = 1.0f - payload.transmittance;
    const bool hasInteraction = payload.hit && rayOpacity > kHardGeometryOpacityThreshold;
#endif
    if (hasInteraction) {
#ifdef ENABLE_DISCRETE_MODEL
        // Keep the integrated hit-distance accumulator: the Python wrapper
        // normalizes it by aggregate ray opacity after the launch. The selected
        // hit distance is used only for the PBR interaction position.
        payload.normal = selectedHit.normal;
        payload.interaction = Interaction(
            make_float3(
                payload.ray.origin.x + payload.ray.direction.x * selectedHit.hitT,
                payload.ray.origin.y + payload.ray.direction.y * selectedHit.hitT,
                payload.ray.origin.z + payload.ray.direction.z * selectedHit.hitT),
            selectedHit.shadingnormal,
            selectedHit.material,
            true);
        payload.interaction.selectedParticleId = selectedParticleId;
#else
        payload.interaction = Interaction(
            payload.ray.origin,
            payload.ray.direction,
            integratedDepth,
            integratedShadingnormal,
            integratedMaterial,
            rayOpacity);
#endif
    } else {
        const float rand_u = sampler.next_1d();
        payload.valid = (rand_u < payload.transmittance);
    }
}

template <bool EnableSelfOcclusionRejection>
static __device__ __inline__ void rayIntersectScene(
    const Ray& ray,
    rayPayload& payload,
    Sampler& sampler) {
#ifdef ENABLE_VISUALIZE_LIGHTS
    const SceneMeshHit meshHit = traceClosestSceneMesh(ray);
#else
    const SceneMeshHit meshHit = traceClosestSurfaceMesh(ray);
#endif
    unsigned int meshId = 0u;
    const bool knownMesh = meshHit.valid && findSceneMesh(meshHit.triangleId, meshId);
    const bool surfaceMesh = knownMesh && isSceneMeshSurface(meshId);
#ifdef ENABLE_VISUALIZE_LIGHTS
    const VisibleLightHit analyticLightHit = getVisibleLightHit(ray);
    const bool meshIsNearest = knownMesh
        && (!analyticLightHit.valid || meshHit.distance < analyticLightHit.dist);
    const bool areaLightIsNearest = analyticLightHit.valid
        && !analyticLightHit.isEnvironment
        && (!knownMesh || analyticLightHit.dist <= meshHit.distance);
    const float nearestEmitterDistance = meshIsNearest
        ? meshHit.distance
        : (analyticLightHit.valid ? analyticLightHit.dist : RayHit::InfiniteDistance);
#else
    const bool meshIsNearest = surfaceMesh;
    const float nearestEmitterDistance = meshIsNearest
        ? meshHit.distance
        : RayHit::InfiniteDistance;
#endif
    rayIntersect<EnableSelfOcclusionRejection>(
        ray,
        payload,
        sampler,
        nearestEmitterDistance);
#ifdef ENABLE_VISUALIZE_LIGHTS
    payload.areaLightHit = areaLightIsNearest;
#endif
    if (meshIsNearest) {
        payload.sceneMeshHit = true;
        const float meshWeight = payload.transmittance;
        payload.hitDistance += meshWeight * meshHit.distance;
        payload.hitDistanceSecondMoment += meshWeight * meshHit.distance * meshHit.distance;
        payload.depthDistortion = fmaxf(
            payload.hitDistanceSecondMoment - payload.hitDistance * payload.hitDistance,
            0.0f);
        payload.lastHitDistance = meshHit.distance;

        if (surfaceMesh && !payload.interaction.valid) {
            float3 normal;
            if (getSceneMeshNormal(meshHit.triangleId, ray.direction, normal)) {
                payload.normal = normal;
                payload.interaction = Interaction(
                    ray.origin + ray.direction * meshHit.distance,
                    normal,
                    Material(
                        make_float3(
                            params.meshLights[meshId][8],
                            params.meshLights[meshId][9],
                            params.meshLights[meshId][10]),
                        params.meshLights[meshId][11],
                        params.meshLights[meshId][12]));
                payload.hit = 1u;
                payload.valid = false;
            }
        } else if (!surfaceMesh) {
            getMeshLightTriangleEmission(
                meshHit.triangleId,
                ray.direction,
                payload.sceneMeshEmission);
            // An emissive mesh is terminal when no Gaussian surface precedes it.
            // Accumulation applies the remaining Gaussian transmittance.
            if (!payload.interaction.valid) {
                payload.valid = true;
            }
        } else if (!payload.interaction.valid) {
            // A degenerate normal still leaves the opaque mesh as a terminal hit.
            payload.valid = true;
        }
    }
}

template <bool EnableSelfOcclusionRejection, typename PipelineParams>
static __device__ __inline__ void rayIntersectBwd(
    const Ray& ray,
    const float rayOpacity,
    const float rayMaxHitDistance,
    const MaterialGrad& materialGrad,
    const float3& shadingNormalGrad,
    const unsigned int selectedParticleId,
    const float3& interactionShadingNormal,
    const PipelineParams& pipelineParams) {
#ifdef ENABLE_DISCRETE_MODEL
    if (selectedParticleId == Interaction::InvalidParticleId) {
        return;
    }

    Material& particleMaterialGrad = pipelineParams.particleMaterialGrad[selectedParticleId];
    atomicAdd(&particleMaterialGrad.albedo.x, materialGrad.dAlbedo.x);
    atomicAdd(&particleMaterialGrad.albedo.y, materialGrad.dAlbedo.y);
    atomicAdd(&particleMaterialGrad.albedo.z, materialGrad.dAlbedo.z);
    atomicAdd(&particleMaterialGrad.roughness, materialGrad.dRoughness);
#ifdef ENABLE_METALLIC
    atomicAdd(&particleMaterialGrad.metallic, materialGrad.dMetallic);
#endif
    atomicAdd(&pipelineParams.particleShadingNormalGrad[selectedParticleId * 3 + 0], shadingNormalGrad.x);
    atomicAdd(&pipelineParams.particleShadingNormalGrad[selectedParticleId * 3 + 1], shadingNormalGrad.y);
    atomicAdd(&pipelineParams.particleShadingNormalGrad[selectedParticleId * 3 + 2], shadingNormalGrad.z);
#else
    const float invRayOpacity = 1.0f / fmaxf(rayOpacity, 1e-12f);
    const Material rayMaterialGrad(
        materialGrad.dAlbedo * invRayOpacity,
        materialGrad.dRoughness * invRayOpacity,
        materialGrad.dMetallic * invRayOpacity);
    const float3 rayShadingNormalGrad = shadingNormalGrad * invRayOpacity;

    constexpr float epsT = 1e-9;
    const float2 minMaxT = intersectAABB(pipelineParams.aabb, ray);
    float startT         = fmaxf(0.0f, minMaxT.x - epsT);
    // Replay the same full AABB traversal as rayIntersect().  Clipping to the
    // last reported hit is vulnerable to the OptiX tmax boundary excluding
    // that final hit, which silently drops its material gradient.
    const float endT     = minMaxT.y + epsT;

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
                if (EnableSelfOcclusionRejection && isSelfOcclusionHit(
                        rayHit,
                        ray,
                        interactionShadingNormal,
                        pipelineParams.particleShadingNormal)) {
                    startT = fmaxf(startT, rayHit.distance);
                    continue;
                }

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
                    atomicAdd(&pipelineParams.particleShadingNormalGrad[rayHit.particleId * 3 + 0], weight * rayShadingNormalGrad.x);
                    atomicAdd(&pipelineParams.particleShadingNormalGrad[rayHit.particleId * 3 + 1], weight * rayShadingNormalGrad.y);
                    atomicAdd(&pipelineParams.particleShadingNormalGrad[rayHit.particleId * 3 + 2], weight * rayShadingNormalGrad.z);
                    rayTransmittance *= (1.0f - galpha);
                }
                startT = fmaxf(startT, rayHit.distance);
            }
        }
    }
#endif
}

template <typename PipelineParams>
static __device__ __inline__ void PendingRayDirectionGradBwd(
    pathPayload& path,
    const PipelineParams& pipelineParams) {
    PendingRayDirectionGrad& pending = path.pendingRayDirectionGrad;
    if (!pending.valid) {
        return;
    }

    const float dRoughness = dot(path.currentRayPayload.rayDirGrad, pending.dNextDirDRoughness);
    const Ray pendingRay = pending.ray;
    const float pendingOpacity = pending.opacity;
    const float pendingMaxHitDistance = pending.maxHitDistance;
    const float3 pendingInteractionShadingNormal = pending.interactionShadingNormal;
    const unsigned int pendingNumBounces = pending.numBounces;
    const unsigned int pendingSelectedParticleId = pending.selectedParticleId;
    path.pendingRayDirectionGrad.clear();
    if (dRoughness == 0.0f) {
        return;
    }

    MaterialGrad materialGrad;
    materialGrad.dRoughness = dRoughness;
    if (pendingNumBounces > 1u) {
        rayIntersectBwd<true>(
            pendingRay,
            pendingOpacity,
            pendingMaxHitDistance,
            materialGrad,
            make_float3(0.0f),
            pendingSelectedParticleId,
            pendingInteractionShadingNormal,
            pipelineParams);
    } else {
        rayIntersectBwd<false>(
            pendingRay,
            pendingOpacity,
            pendingMaxHitDistance,
            materialGrad,
            make_float3(0.0f),
            pendingSelectedParticleId,
            pendingInteractionShadingNormal,
            pipelineParams);
    }
}

static __device__ __inline__ void sampleBrdfNextDirection(
    pathPayload& path,
    Sampler& sampler) {
    const Ray currentRay = path.currentRayPayload.ray;
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float3 nextRayDirection = currentRay.direction;
    float scatterPdf = 0.0f;
    const float3 brdfThroughput = sample_material_fast_brdf_throughput(
        currentRay.direction,
        sampler,
        currentInteraction,
        nextRayDirection,
        scatterPdf);

    const float3 nextPathThroughput = path.pathThroughput * brdfThroughput;
    if (nextPathThroughput.x == 0.0f && nextPathThroughput.y == 0.0f && nextPathThroughput.z == 0.0f) {
        path.pathThroughput = make_float3(0.0f);
        path.active = 0u;
        return;
    }

    path.pathThroughput = nextPathThroughput;
    Ray nextRay(currentInteraction.position + safe_normalize(nextRayDirection) * kSelfOcclusionRayOriginOffset, nextRayDirection);
    path.currentRayPayload = rayPayload(nextRay, 0.0f);
    path.currentRayPayload.interactionShadingNormal = currentInteraction.shadingnormal;
#ifdef ENABLE_MIS
    path.currentRayPayload.scatterPdf = scatterPdf;
    path.currentRayPayload.lightPdf = lightSamplerPdf(currentInteraction.position, nextRay.direction);
#endif
}

static __device__ __inline__ void sampleNee(
    pathPayload& path,
    Sampler& sampler) {
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float lightPdf = 0.0f;
    float scatterPdf = 0.0f;
    const LightSample lightSample = sampleLight(currentInteraction.position, sampler);
    const float3 lightDirection = lightSample.wi;
    lightPdf = lightSample.pdf;
    if (lightPdf <= 0.0f || (lightSample.Li.x == 0.0f && lightSample.Li.y == 0.0f && lightSample.Li.z == 0.0f)) {
        path.emitterRayPayload = rayPayload();
        return;
    }
    if (dot(currentInteraction.shadingnormal, lightDirection) <= 0.0f) {
        path.emitterRayPayload = rayPayload();
        return;
    }

    const float3 brdfTimesCos = eval_material_fast_brdf_light_sample(
        path.currentRayPayload.ray.direction,
        currentInteraction,
        lightDirection,
        scatterPdf);
    if (lightSample.lightType == LightSamplerType_Point) {
        scatterPdf = 0.0f;
    }
    const Ray lightRay(
        currentInteraction.position + lightDirection * kSelfOcclusionRayOriginOffset,
        lightDirection);
    const float shadowTmax = lightSample.dist < 1e19f
        ? fmaxf(0.0f, lightSample.dist - kSelfOcclusionRayOriginOffset - 1e-4f)
        : RayHit::InfiniteDistance;
    if (traceShadowMonteCarloOccluded<true>(
            lightRay,
            currentInteraction.shadingnormal,
            sampler,
            0.0f,
            shadowTmax)) {
        path.emitterRayPayload = rayPayload();
        return;
    }
    path.emitterRayPayload = rayPayload(lightRay, 0.0f);

    path.emitterRayPayload.lightPdf = lightPdf;
    path.emitterRayPayload.scatterPdf = scatterPdf;
    path.emitterRayPayload.light = lightSample.Li;
    path.emitterRayPayload.contribution = path.pathThroughput * brdfTimesCos * path.emitterRayPayload.light / fmaxf(lightPdf, 1e-6f);
}

#ifdef ENABLE_MIS
template <typename PipelineParams>
static __device__ __inline__ void accumulateNeeGradBwd(
    pathPayload& path,
    const Ray& currentRay,
    const Interaction& currentInteraction,
    PipelineParams& pipelineParams) {
    if (path.numBounces != 0u) {
        return;
    }
    if (path.emitterRayPayload.lightPdf <= 0.0f) {
        return;
    }

    float scatterPdf = 0.0f;
    const float3 lightDirection = path.emitterRayPayload.ray.direction;
    const FastBrdfValueGrad brdfTimesCos = eval_material_fast_brdf_light_sample_with_grads(
        currentRay.direction,
        currentInteraction,
        lightDirection,
        scatterPdf);

    const float lightSideMis = path.maxBounces > 1u
        ? misWeight(
              path.emitterRayPayload.lightPdf,
              path.emitterRayPayload.scatterPdf)
        : 1.0f;
    const float neeScale = lightSideMis / fmaxf(path.emitterRayPayload.lightPdf, 1e-6f);

    // Keep NEE gradients from updating the environment: although the estimator
    // is valid, it worsens the lighting-material ambiguity in prior-free inverse
    // rendering and consistently degrades the recovered material metrics.
    // const float3 environmentGrad = path.accumulatedLightingGrad * path.pathThroughput * brdfTimesCos.value * neeScale;
    // getBackgroundColorBwd(lightDirection, environmentGrad, pipelineParams);
    const float3 sampledLight = path.emitterRayPayload.light;
    const float3 dLoss_dBrdf = path.accumulatedLightingGrad * path.pathThroughput * sampledLight * neeScale;

    path.currentRayPayload.interaction.materialGrad.dAlbedo += dLoss_dBrdf * brdfTimesCos.dBrdf_dAlbedo;
    path.currentRayPayload.interaction.shadingNormalGrad += make_float3(
        dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dNormalX),
        dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dNormalY),
        dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dNormalZ));

    // NEE samples bright environment directions directly; enabling this roughness gradient can easily overfit specular highlights.
    //path.currentRayPayload.interaction.materialGrad.dRoughness += dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dRoughness);
// #ifdef ENABLE_METALLIC
//     path.currentRayPayload.interaction.materialGrad.dMetallic += dot(dLoss_dBrdf, brdfTimesCos.dBrdf_dMetallic);
// #endif
}
#endif


template <typename PipelineParams>
static __device__ __inline__ void sampleBrdfNextDirectionBwd(
    pathPayload& path,
    Sampler& sampler,
    PipelineParams& pipelineParams) {
    const Ray currentRay = path.currentRayPayload.ray;
    const Interaction currentInteraction = path.currentRayPayload.interaction;

    float3 nextRayDirection = currentRay.direction;
    float scatterPdf = 0.0f;
    const FastBrdfValueGrad brdfThroughput = sample_material_fast_brdf_throughput_with_grads(
        currentRay.direction,
        sampler,
        currentInteraction,
        nextRayDirection,
        scatterPdf);
    const float3 nextPathThroughput = path.pathThroughput * brdfThroughput.value;
    const bool validNextPath =
        nextPathThroughput.x != 0.0f ||
        nextPathThroughput.y != 0.0f ||
        nextPathThroughput.z != 0.0f;

#ifdef ENABLE_MIS
    const float lightPdf = validNextPath
        ? lightSamplerPdf(currentInteraction.position, nextRayDirection)
        : 0.0f;
#endif

    if (validNextPath) {
        const float3 dLoss_dBrdfNumerator =
            path.accumulatedLightingGrad * path.accumulatedLighting;
        const float3 dLoss_dBrdf = make_float3(
            brdfThroughput.value.x > FastBrdfEps ? dLoss_dBrdfNumerator.x / brdfThroughput.value.x : 0.0f,
            brdfThroughput.value.y > FastBrdfEps ? dLoss_dBrdfNumerator.y / brdfThroughput.value.y : 0.0f,
            brdfThroughput.value.z > FastBrdfEps ? dLoss_dBrdfNumerator.z / brdfThroughput.value.z : 0.0f);
        path.currentRayPayload.interaction.materialGrad.dAlbedo += dLoss_dBrdf * brdfThroughput.dBrdf_dAlbedo;
        path.currentRayPayload.interaction.shadingNormalGrad += make_float3(
            dot(dLoss_dBrdf, brdfThroughput.dBrdf_dNormalX),
            dot(dLoss_dBrdf, brdfThroughput.dBrdf_dNormalY),
            dot(dLoss_dBrdf, brdfThroughput.dBrdf_dNormalZ));
        path.currentRayPayload.interaction.materialGrad.dRoughness += dot(dLoss_dBrdf, brdfThroughput.dBrdf_dRoughness);
#ifdef ENABLE_METALLIC
        path.currentRayPayload.interaction.materialGrad.dMetallic += dot(dLoss_dBrdf, brdfThroughput.dBrdf_dMetallic);
#endif
    }

    if (path.numBounces > 1u) {
        rayIntersectBwd<true>(
            currentRay,
            1.0f - path.currentRayPayload.transmittance,
            path.currentRayPayload.lastHitDistance,
            path.currentRayPayload.interaction.materialGrad,
            path.currentRayPayload.interaction.shadingNormalGrad,
            path.currentRayPayload.interaction.selectedParticleId,
            path.currentRayPayload.interactionShadingNormal,
            pipelineParams);
    } else {
        rayIntersectBwd<false>(
            currentRay,
            1.0f - path.currentRayPayload.transmittance,
            path.currentRayPayload.lastHitDistance,
            path.currentRayPayload.interaction.materialGrad,
            path.currentRayPayload.interaction.shadingNormalGrad,
            path.currentRayPayload.interaction.selectedParticleId,
            path.currentRayPayload.interactionShadingNormal,
            pipelineParams);
    }

    if (!validNextPath) {
        path.pathThroughput = make_float3(0.0f);
        path.active = 0u;
        return;
    }

    path.pathThroughput = nextPathThroughput;
    Ray nextRay(currentInteraction.position + safe_normalize(nextRayDirection) * kSelfOcclusionRayOriginOffset, nextRayDirection);
    path.currentRayPayload = rayPayload(nextRay, 0.0f);
    path.currentRayPayload.interactionShadingNormal = currentInteraction.shadingnormal;
#ifdef ENABLE_MIS
    path.currentRayPayload.scatterPdf = scatterPdf;
    path.currentRayPayload.lightPdf = lightPdf;
#endif
}

static __device__ __inline__ void writePrimaryRayOutputs(
    const uint3& idx,
    const rayPayload& payload) {
    params.rayRadiance[idx.z][idx.y][idx.x][0]    = payload.radiance.x;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = payload.radiance.y;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = payload.radiance.z;
    // Opaque Mesh/Sphere emitters terminate primary rays. Gaussian
    // transmittance only describes the volume in front of them.
    params.rayDensity[idx.z][idx.y][idx.x][0]     = payload.sceneMeshHit || payload.areaLightHit
        ? 1.0f
        : 1.0f - payload.transmittance;
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
    params.rayLight[idx.z][idx.y][idx.x][0] = payload.accumulatedLightNoBrdf.x;
    params.rayLight[idx.z][idx.y][idx.x][1] = payload.accumulatedLightNoBrdf.y;
    params.rayLight[idx.z][idx.y][idx.x][2] = payload.accumulatedLightNoBrdf.z;

    params.rayPbrComponents[idx.z][idx.y][idx.x][0][0] = payload.accumulatedDirectLighting.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][1] = payload.accumulatedDirectLighting.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][0][2] = payload.accumulatedDirectLighting.z;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][0] = payload.accumulatedIndirectLighting.x;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][1] = payload.accumulatedIndirectLighting.y;
    params.rayPbrComponents[idx.z][idx.y][idx.x][1][2] = payload.accumulatedIndirectLighting.z;
}
