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
        contribution             = make_float3(0.f);
        normal                   = make_float3(0.f);
        transmittance            = 1.f;
        hitDistance              = 0.f;
        hitDistanceSecondMoment  = 0.f;
        lastHitDistance          = initialLastHitDistance;
        depthDistortion          = 0.f;
        hitsCount                = 0.f;
        hit                      = 0;
        valid                    = false;
    }

    Ray ray;
    Interaction interaction;

    float3 radiance;
    float3 contribution;
    float3 normal;

    float transmittance;
    float hitDistance;
    float hitDistanceSecondMoment;
    float lastHitDistance;
    float depthDistortion;
    float hitsCount;

    unsigned int hit;
    bool valid;
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
        pathThroughput              = make_float3(1.f);
    }

    unsigned int active;
    unsigned int numBounces;
    unsigned int maxBounces;

    rayPayload currentRayPayload;

    float3 accumulatedLighting;
    float3 accumulatedLightingGrad;
    float3 accumulatedDirectLighting;
    float3 accumulatedIndirectLighting;
    float3 pathThroughput;
};
