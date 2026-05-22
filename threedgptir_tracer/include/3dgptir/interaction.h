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

#include <3dgptir/material.h>

struct Interaction {
#ifdef __CUDACC__
    __device__ Interaction()
        : Interaction(make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f), Material(), false) {
    }

    __device__ Interaction(
        const float3& position,
        const float3& shadingnormal,
        const Material& material)
        : Interaction(position, shadingnormal, material, true) {
    }

    __device__ Interaction(
        const float3& position,
        const float3& shadingnormal,
        const Material& material,
        const bool valid) {
        this->valid         = valid;
        this->position      = position;
        this->shadingnormal = shadingnormal;
        this->material      = material;
        this->materialGrad  = MaterialGrad();
    }

    __device__ Interaction(
        const float3& rayOrigin,
        const float3& rayDirection,
        const float integratedDepth,
        const float3& integratedShadingnormal,
        const Material& integratedMaterial,
        const float opacity) {
        const float invOpacity = 1.0f / fmaxf(opacity, 1e-12f);
        const float depth      = integratedDepth * invOpacity;

        valid    = true;
        position = make_float3(
            rayOrigin.x + rayDirection.x * depth,
            rayOrigin.y + rayDirection.y * depth,
            rayOrigin.z + rayDirection.z * depth);
        shadingnormal = make_float3(
            integratedShadingnormal.x * invOpacity,
            integratedShadingnormal.y * invOpacity,
            integratedShadingnormal.z * invOpacity);
        material     = Material(integratedMaterial, invOpacity);
        materialGrad = MaterialGrad();
    }
#endif

    bool valid;
    float3 position;
    float3 shadingnormal;
    Material material;
    MaterialGrad materialGrad;
};
