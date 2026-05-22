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

struct Material {
#ifdef __CUDACC__
    __device__ Material()
        : Material(make_float3(0.f, 0.f, 0.f), 0.f, 0.f) {
    }

    __device__ Material(
        const float3& albedo,
        const float roughness,
        const float metallic) {
        this->albedo    = albedo;
        this->roughness = roughness;
        this->metallic  = metallic;
    }

    __device__ Material(
        const Material& integratedMaterial,
        const float invOpacity) {
        albedo = make_float3(
            integratedMaterial.albedo.x * invOpacity,
            integratedMaterial.albedo.y * invOpacity,
            integratedMaterial.albedo.z * invOpacity);
        roughness = integratedMaterial.roughness * invOpacity;
        metallic  = integratedMaterial.metallic * invOpacity;
    }
#endif

    float3 albedo;
    float roughness;
    float metallic;
};

struct MaterialGrad {
#ifdef __CUDACC__
    __device__ MaterialGrad()
        : MaterialGrad(make_float3(0.f, 0.f, 0.f), 0.f, 0.f) {
    }

    __device__ MaterialGrad(
        const float3& dAlbedo,
        const float dRoughness,
        const float dMetallic) {
        this->dAlbedo    = dAlbedo;
        this->dRoughness = dRoughness;
        this->dMetallic  = dMetallic;
    }
#endif

    float3 dAlbedo;
    float dRoughness;
    float dMetallic;
};
