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

// fullStochastic forward estimator (ported from Stoch3DGS), adapted to PTIR's
// PipelineParameters and output conventions:
//   - rayDensity holds opacity (matches referenceOptix.cu: 1 - transmittance),
//     here estimated as the accepted-sample fraction.
//   - shading normal is sampled per accepted particle (SH-color-like), written to
//     rayShadingNormal; depth second moment / depth distortion are set to 0.
//   - particleVisibility is populated for SelectiveAdam (as in referenceOptix.cu).

#include <3dgrt/kernels/cuda/gaussianParticles.cuh>
#include <3dgrt/pipelineParameters.h>
// clang-format on

extern "C" {
__constant__ PipelineParameters params;
}

struct RayHit {
    unsigned int particleId;
    float distance;

    static constexpr unsigned int InvalidParticleId = 0xFFFFFFFF;
    static constexpr float InfiniteDistance         = 1e20f;
};
using RayPayload = RayHit[PipelineParameters::MaxNumHitPerTrace];

static __device__ __inline__ float2 intersectAABB(const OptixAabb& aabb, const float3& rayOri, const float3& rayDir) {
    const float3 t0   = (make_float3(aabb.minX, aabb.minY, aabb.minZ) - rayOri) / rayDir;
    const float3 t1   = (make_float3(aabb.maxX, aabb.maxY, aabb.maxZ) - rayOri) / rayDir;
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    const float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    return float2{fmaxf(0.f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z))), fminf(tmax.x, fminf(tmax.y, tmax.z))};
}

static __device__ __inline__ uint32_t optixPrimitiveIndex() {
    return PipelineParameters::InstancePrimitive ? optixGetInstanceIndex() : (PipelineParameters::CustomPrimitive ? optixGetPrimitiveIndex() : static_cast<uint32_t>(optixGetPrimitiveIndex() / params.gPrimNumTri));
}

static __device__ __inline__ void trace(
    RayPayload& rayPayload,
    const float3& rayOri,
    const float3& rayDir,
    const float tmin,
    const float tmax,
    uint32_t randomSeed) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = RayHit::InvalidParticleId;
    r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = __float_as_int(RayHit::InfiniteDistance);
    r30 = 0; // boolean flag from raygen to anyhit (unused in forward)
    r31 = randomSeed; // r31 holds the random seed for stochastic sampling

    optixTrace(params.handle, rayOri, rayDir,
               tmin,                     // Min intersection distance
               tmax,                     // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
               0, // SBT offset
               1, // SBT stride
               0, // missSBTIndex
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

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return;
    }

    float3 rayOrigin    = params.rayWorldOrigin(idx);
    float3 rayDirection = params.rayWorldDirection(idx);

#ifdef ENABLE_NORMALS
    float3 avg_normal        = make_float3(0.f);
    float3 avg_shadingnormal = make_float3(0.f);
#endif
#ifdef ENABLE_HIT_COUNTS
    float rayHitsCount = 0.f;
#endif

    float2 minMaxT       = intersectAABB(params.aabb, rayOrigin, rayDirection);
    constexpr float epsT = 1e-9;

    float rayStartT = fmaxf(0.0f, minMaxT.x - epsT);

    float avg_depth      = 0.f;
    float avg_alpha      = 0.f;
    float3 avg_radiance  = make_float3(0.f);
    float max_distance   = 0.f;
    const int spp_fwd      = 2;
    const int payload_size = 15;
    const float invCount = 1.0f / (float(payload_size) * float(spp_fwd));

#pragma unroll
    for (int spp = 0; spp < spp_fwd; spp++) {
        uint32_t randomSeed = (256 + params.frameNumber + spp);
        RayPayload rayPayload;

        trace(rayPayload, rayOrigin, rayDirection, rayStartT, minMaxT.y + epsT, randomSeed);

#pragma unroll
        for (int it = 0; it < payload_size; it++) {
            int32_t particleId = rayPayload[it].particleId;
            if (particleId == RayHit::InvalidParticleId) {
                continue; // no hit
            }
            float alpha       = 0.f;
            float3 radiance   = make_float3(0.f);
            float depth       = 0.f;
            float3 normal     = make_float3(0.f);
            float3 shadingnrm = make_float3(0.f);

            bool acceptHit = processHitStochastic<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                rayOrigin, rayDirection,
                particleId,
                params.particleDensity,
                params.particleRadiance,
                params.hitMinGaussianResponse,
                params.alphaMinThreshold,
                params.sphDegree,
                params.particleShadingNormal,
                &alpha, &radiance, &depth,
#ifdef ENABLE_NORMALS
                &shadingnrm,
                &normal
#else
                nullptr,
                nullptr
#endif
            );
            if (!acceptHit) {
                continue;
            }
            avg_depth    += depth;
            avg_alpha    += 1.0f;
            avg_radiance += radiance;
#ifdef ENABLE_NORMALS
            avg_normal        += normal;
            avg_shadingnormal += shadingnrm;
#endif
            max_distance = fmaxf(max_distance, depth);

            // SelectiveAdam visibility (as in referenceOptix.cu). Race on write of
            // the same value is benign.
            params.particleVisibility[particleId] = 1;
        }
    }

    params.rayRadiance[idx.z][idx.y][idx.x][0]    = avg_radiance.x * invCount;
    params.rayRadiance[idx.z][idx.y][idx.x][1]    = avg_radiance.y * invCount;
    params.rayRadiance[idx.z][idx.y][idx.x][2]    = avg_radiance.z * invCount;
    // PTIR convention: rayDensity = opacity (accepted-sample fraction).
    params.rayDensity[idx.z][idx.y][idx.x][0]     = avg_alpha * invCount;
    params.rayHitDistance[idx.z][idx.y][idx.x][0] = avg_depth * invCount;
    params.rayHitDistance[idx.z][idx.y][idx.x][1] = max_distance;
    // Not estimated in stochastic mode; kept in ABI, no loss consumes them.
    params.rayHitDistanceSecondMoment[idx.z][idx.y][idx.x][0] = 0.f;
    params.rayDepthDistortion[idx.z][idx.y][idx.x][0]         = 0.f;
#ifdef ENABLE_NORMALS
    params.rayNormal[idx.z][idx.y][idx.x][0] = avg_normal.x * invCount;
    params.rayNormal[idx.z][idx.y][idx.x][1] = avg_normal.y * invCount;
    params.rayNormal[idx.z][idx.y][idx.x][2] = avg_normal.z * invCount;
    params.rayShadingNormal[idx.z][idx.y][idx.x][0] = avg_shadingnormal.x * invCount;
    params.rayShadingNormal[idx.z][idx.y][idx.x][1] = avg_shadingnormal.y * invCount;
    params.rayShadingNormal[idx.z][idx.y][idx.x][2] = avg_shadingnormal.z * invCount;
#endif
#ifdef ENABLE_HIT_COUNTS
    params.rayHitsCount[idx.z][idx.y][idx.x][0] = rayHitsCount;
#endif
}

extern "C" __global__ void __intersection__is() {
    float hitDistance;
    const bool intersect = PipelineParameters::InstancePrimitive ? intersectInstanceParticle(optixGetObjectRayOrigin(),
                                                                                             optixGetObjectRayDirection(),
                                                                                             optixGetInstanceIndex(),
                                                                                             optixGetRayTmin(),
                                                                                             optixGetRayTmax(),
                                                                                             params.hitMaxParticleSquaredDistance,
                                                                                             hitDistance)
                                                                 : intersectCustomParticle(optixGetWorldRayOrigin(),
                                                                                           optixGetWorldRayDirection(),
                                                                                           optixGetPrimitiveIndex(),
                                                                                           params.particleDensity,
                                                                                           optixGetRayTmin(),
                                                                                           optixGetRayTmax(),
                                                                                           params.hitMaxParticleSquaredDistance,
                                                                                           hitDistance);
    if (intersect) {
        optixReportIntersection(hitDistance, 0);
    }
}

extern "C" __global__ void __anyhit__ah() { // stochastic reservoir, sorting removed
    RayHit hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};
    float3 rayOrigin    = optixGetWorldRayOrigin();
    float3 rayDirection = optixGetWorldRayDirection();
    unsigned int randomSeed = optixGetPayload_31();
    uint3 idx = optixGetLaunchIndex();
    unsigned int seed = ((idx.x * 73856093u) ^ (idx.y * 19349663u) ^ (idx.z * 83492791u) ^ hit.particleId ^ randomSeed * 2654435761u);
    float curr_distance;
    float rand1d;
    float density = getDensityStochastic<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
                                                        rayOrigin, rayDirection, hit.particleId,
                                                        params.particleDensity);

    #define STOCHASTIC_HIT_PAYLOAD(i_id, i_distance)                                      \
        {                                                                                 \
            curr_distance = __uint_as_float(optixGetPayload_##i_distance());              \
            seed = 1664525u * seed + 1013904223u;                                         \
            rand1d = (seed & 0x00FFFFFF) / float(0x01000000);                             \
            if (hit.distance < curr_distance && rand1d < density && density > params.alphaMinThreshold) { \
                optixSetPayload_##i_id(hit.particleId);                                    \
                optixSetPayload_##i_distance(__float_as_uint(hit.distance));              \
            }                                                                             \
        }

    STOCHASTIC_HIT_PAYLOAD(0, 1)
    STOCHASTIC_HIT_PAYLOAD(2, 3)
    STOCHASTIC_HIT_PAYLOAD(4, 5)
    STOCHASTIC_HIT_PAYLOAD(6, 7)
    STOCHASTIC_HIT_PAYLOAD(8, 9)
    STOCHASTIC_HIT_PAYLOAD(10, 11)
    STOCHASTIC_HIT_PAYLOAD(12, 13)
    STOCHASTIC_HIT_PAYLOAD(14, 15)
    STOCHASTIC_HIT_PAYLOAD(16, 17)
    STOCHASTIC_HIT_PAYLOAD(18, 19)
    STOCHASTIC_HIT_PAYLOAD(20, 21)
    STOCHASTIC_HIT_PAYLOAD(22, 23)
    STOCHASTIC_HIT_PAYLOAD(24, 25)
    STOCHASTIC_HIT_PAYLOAD(26, 27)
    STOCHASTIC_HIT_PAYLOAD(28, 29)
    optixIgnoreIntersection();
}
