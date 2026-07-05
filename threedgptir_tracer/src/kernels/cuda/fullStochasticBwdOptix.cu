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

// fullStochastic backward path tracer (stage2, PRB): identical to referenceBwdOptix.cu
// except surface intersection/replay uses stochastic single-Gaussian free-flight
// selection (rayIntersectStochastic) with the same RNG seed as the forward pass, and
// material gradients are splatted to the single sampled particle (weight 1) via
// sampleBrdfNextDirectionStochasticBwd / PendingRayDirectionGradStochasticBwd. Environment
// gradients (accumulateLightContributionBwd) are unchanged. IS/AH programs unchanged.

#include <3dgptir/pipelineParameters.h>
#include <3dgptir/kernels/cuda/sampler.cuh>
// clang-format on

extern "C" {
__constant__ PipelineBackwardParameters params;
}

#include <3dgptir/kernels/cuda/environment.cuh>
#include <3dgptir/kernels/cuda/pathTracer.cuh>

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    if ((idx.x > params.frameBounds.x) || (idx.y > params.frameBounds.y)) {
        return;
    }

    Ray ray(params.rayWorldOrigin(idx), params.rayWorldDirection(idx));
    Sampler sampler;
    sampler.initFromLaunch(idx, params.frameNumber);

    pathPayload path(1u, 0u, params.maxBounces);
    path.accumulatedLighting = make_float3(params.rayPbr[idx.z][idx.y][idx.x][0], params.rayPbr[idx.z][idx.y][idx.x][1], params.rayPbr[idx.z][idx.y][idx.x][2]);
    path.accumulatedLightingGrad = make_float3(params.rayPbrGrad[idx.z][idx.y][idx.x][0], params.rayPbrGrad[idx.z][idx.y][idx.x][1], params.rayPbrGrad[idx.z][idx.y][idx.x][2]);
    path.accumulatedLightNoBrdf = make_float3(params.rayLight[idx.z][idx.y][idx.x][0], params.rayLight[idx.z][idx.y][idx.x][1], params.rayLight[idx.z][idx.y][idx.x][2]);
    path.accumulatedLightNoBrdfGrad = make_float3(params.rayLightGrad[idx.z][idx.y][idx.x][0], params.rayLightGrad[idx.z][idx.y][idx.x][1], params.rayLightGrad[idx.z][idx.y][idx.x][2]);

    rayIntersectStochastic<false>(ray, path.currentRayPayload, sampler);

    if (path.currentRayPayload.interaction.valid) {
        path.currentRayPayload.interaction.materialGrad.dAlbedo += make_float3(params.rayMaterialGrad[idx.z][idx.y][idx.x][0], params.rayMaterialGrad[idx.z][idx.y][idx.x][1], params.rayMaterialGrad[idx.z][idx.y][idx.x][2]);
        path.currentRayPayload.interaction.materialGrad.dRoughness += params.rayMaterialGrad[idx.z][idx.y][idx.x][3];
#ifdef ENABLE_METALLIC
        path.currentRayPayload.interaction.materialGrad.dMetallic += params.rayMaterialGrad[idx.z][idx.y][idx.x][4];
#endif
    }

#ifndef ENABLE_VISUALIZE_LIGHTS
    if (!path.currentRayPayload.interaction.valid) { return; }
#endif

#ifdef ENABLE_MIS
    sampleNee(path, sampler);
#endif

    for (unsigned int depth = 0; depth < params.maxBounces && path.active; ++depth) {
        accumulateLightContributionBwd(path, params);
        PendingRayDirectionGradStochasticBwd(path, params);

        path.active &= (depth + 1u < path.maxBounces) && path.currentRayPayload.interaction.valid;
        if (!path.active) { break; }
        path.numBounces = depth + 1u;

        sampleBrdfNextDirectionStochasticBwd(path, sampler, params);
        const float throughputMax = fmaxf(path.pathThroughput.x, fmaxf(path.pathThroughput.y, path.pathThroughput.z));
        if (throughputMax < 1e-4f) { break; }
        rayIntersectStochastic<true>(path.currentRayPayload.ray, path.currentRayPayload, sampler);
    }
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

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance)                      \
    {                                                                             \
        const float distance = __uint_as_float(optixGetPayload_##i_distance());   \
        if (hit.distance < distance) {                                            \
            optixSetPayload_##i_distance(__float_as_uint(hit.distance));          \
            const uint32_t id = optixGetPayload_##i_id();                         \
            optixSetPayload_##i_id(hit.particleId);                               \
            hit.distance   = distance;                                            \
            hit.particleId = id;                                                  \
        }                                                                         \
    }

extern "C" __global__ void __anyhit__ah() {
    RayHit hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};

    if (hit.distance < __uint_as_float(optixGetPayload_31())) {
        compareAndSwapHitPayloadValue(hit, 0, 1);
        compareAndSwapHitPayloadValue(hit, 2, 3);
        compareAndSwapHitPayloadValue(hit, 4, 5);
        compareAndSwapHitPayloadValue(hit, 6, 7);
        compareAndSwapHitPayloadValue(hit, 8, 9);
        compareAndSwapHitPayloadValue(hit, 10, 11);
        compareAndSwapHitPayloadValue(hit, 12, 13);
        compareAndSwapHitPayloadValue(hit, 14, 15);
        compareAndSwapHitPayloadValue(hit, 16, 17);
        compareAndSwapHitPayloadValue(hit, 18, 19);
        compareAndSwapHitPayloadValue(hit, 20, 21);
        compareAndSwapHitPayloadValue(hit, 22, 23);
        compareAndSwapHitPayloadValue(hit, 24, 25);
        compareAndSwapHitPayloadValue(hit, 26, 27);
        compareAndSwapHitPayloadValue(hit, 28, 29);
        compareAndSwapHitPayloadValue(hit, 30, 31);

        // ignore all inserted hits, expect if the last one
        if (__uint_as_float(optixGetPayload_31()) > optixGetRayTmax()) {
            optixIgnoreIntersection();
        }
    }
}
