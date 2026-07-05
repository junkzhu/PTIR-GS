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

#include <3dgptir/interaction.h>

struct Ray {
    __device__ Ray()
        : Ray(make_float3(0.f), make_float3(0.f)) {
    }

    __device__ Ray(
        const float3& origin,
        const float3& direction) {
        this->origin    = origin;
        this->direction = direction;
    }

    float3 origin;
    float3 direction;
};

struct rayPayload {
    __device__ rayPayload()
        : rayPayload(Ray(), 0.f) {
    }

    __device__ rayPayload(
        const Ray& ray,
        const float initialLastHitDistance) {
        this->ray = ray;

        interaction             = Interaction();

        radiance                 = make_float3(0.f);
        light                    = make_float3(0.f);
        contribution             = make_float3(0.f);
        normal                   = make_float3(0.f);
        transmittance            = 1.f;
        hitDistance              = 0.f;
        hitDistanceSecondMoment  = 0.f;
        lastHitDistance          = initialLastHitDistance;
        depthDistortion          = 0.f;
        hitsCount                = 0.f;
        scatterPdf               = 0.f;
        lightPdf                 = 0.f;
        rayDirGrad               = make_float3(0.f);
        hit                      = 0;
        valid                    = false;
        interactionParticleId    = -1;
    }

    Ray ray;
    Interaction interaction;

    float3 radiance;
    float3 light;
    float3 contribution;
    float3 normal;

    float transmittance;
    float hitDistance;
    float hitDistanceSecondMoment;
    float lastHitDistance;
    float depthDistortion;
    float hitsCount;

    float scatterPdf;
    float lightPdf;
    float3 rayDirGrad;

    unsigned int hit;
    bool valid;
    int32_t interactionParticleId; ///< stochastic pipeline: sampled surface particle (-1 = none)
};

struct PendingRayDirectionGrad {
    __device__ PendingRayDirectionGrad() {
        clear();
    }

    __device__ void clear() {
        valid = false;
        ray = Ray();
        opacity = 0.f;
        maxHitDistance = 0.f;
        dNextDirDRoughness = make_float3(0.f);
        numBounces = 0u;
        interactionParticleId = -1;
    }

    __device__ void set(
        const Ray& pendingRay,
        const float pendingOpacity,
        const float pendingMaxHitDistance,
        const float3& pendingDNextDirDRoughness,
        const unsigned int pendingNumBounces,
        const int32_t pendingInteractionParticleId = -1) {
        const float gradLength2 =
            pendingDNextDirDRoughness.x * pendingDNextDirDRoughness.x +
            pendingDNextDirDRoughness.y * pendingDNextDirDRoughness.y +
            pendingDNextDirDRoughness.z * pendingDNextDirDRoughness.z;
        valid = gradLength2 > 0.0f;
        ray = pendingRay;
        opacity = pendingOpacity;
        maxHitDistance = pendingMaxHitDistance;
        dNextDirDRoughness = pendingDNextDirDRoughness;
        numBounces = pendingNumBounces;
        interactionParticleId = pendingInteractionParticleId;
    }

    bool valid;
    Ray ray;
    float opacity;
    float maxHitDistance;
    float3 dNextDirDRoughness;
    unsigned int numBounces;
    int32_t interactionParticleId; ///< stochastic pipeline: sampled surface particle of the pending surface
};

struct pathPayload {
    __device__ pathPayload()
        : pathPayload(1u, 0u, 0u) {
    }

    __device__ pathPayload(
        const unsigned int active,
        const unsigned int numBounces,
        const unsigned int maxBounces) {
        this->active     = active;
        this->numBounces = numBounces;
        this->maxBounces = maxBounces;

        accumulatedLighting         = make_float3(0.f);
        accumulatedLightingGrad     = make_float3(0.f);
        accumulatedDirectLighting   = make_float3(0.f);
        accumulatedIndirectLighting = make_float3(0.f);
        accumulatedLightNoBrdf      = make_float3(0.f);
        accumulatedLightNoBrdfGrad  = make_float3(0.f);
        pathThroughput              = make_float3(1.f);
    }

    unsigned int active;
    unsigned int numBounces;
    unsigned int maxBounces;

    rayPayload currentRayPayload;
    rayPayload emitterRayPayload;

    float3 accumulatedLighting;
    float3 accumulatedLightingGrad;
    float3 accumulatedDirectLighting;
    float3 accumulatedIndirectLighting;
    float3 accumulatedLightNoBrdf;
    float3 accumulatedLightNoBrdfGrad;
    float3 pathThroughput;

    PendingRayDirectionGrad pendingRayDirectionGrad;
};
