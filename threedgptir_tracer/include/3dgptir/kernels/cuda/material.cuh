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

#ifdef __CUDACC__

#include <optix.h>

#include <3dgptir/interaction.h>
#include <3dgptir/kernels/cuda/sampler.cuh>
#include <3dgptir/mathUtils.h>

static constexpr float FastBrdfEps      = 1e-6f;
static constexpr float FastBrdfMinRough = 0.05f;
static constexpr float FastBrdfPi2      = 6.28318530717958647692f;

#ifdef FAST_BRDF_GUARD_NAN
static constexpr float FastBrdfMaxValue = 1e20f;

static __device__ __forceinline__ bool fast_brdf_is_finite(const float v) {
    return (v == v) && (fabsf(v) < FastBrdfMaxValue);
}
#endif

static __device__ __forceinline__ float fast_brdf_saturate(const float v) {
#ifdef FAST_BRDF_GUARD_NAN
    if (!fast_brdf_is_finite(v)) {
        return 0.0f;
    }
#endif
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

static __device__ __forceinline__ float3 fast_brdf_saturate(const float3& v) {
    return make_float3(fast_brdf_saturate(v.x), fast_brdf_saturate(v.y), fast_brdf_saturate(v.z));
}

static __device__ __forceinline__ float fast_brdf_clamp_roughness(const float roughness) {
#ifdef FAST_BRDF_GUARD_NAN
    if (!fast_brdf_is_finite(roughness)) {
        return 1.0f;
    }
#endif
    return fminf(fmaxf(roughness, FastBrdfMinRough), 1.0f);
}

static __device__ __forceinline__ float fast_brdf_positive_dot(const float3& a, const float3& b) {
    return fast_brdf_saturate(dot(a, b));
}

static __device__ __forceinline__ float3 fast_brdf_clamp_nonnegative(const float3& v) {
#ifdef FAST_BRDF_GUARD_NAN
    const float x = (fast_brdf_is_finite(v.x) && v.x > 0.0f) ? v.x : 0.0f;
    const float y = (fast_brdf_is_finite(v.y) && v.y > 0.0f) ? v.y : 0.0f;
    const float z = (fast_brdf_is_finite(v.z) && v.z > 0.0f) ? v.z : 0.0f;
    return make_float3(x, y, z);
#else
    return make_float3(fmaxf(v.x, 0.0f), fmaxf(v.y, 0.0f), fmaxf(v.z, 0.0f));
#endif
}

static __device__ __forceinline__ float3 fast_brdf_safe_normalize(const float3& v, const float3& fallback) {
    const float len2 = dot(v, v);
#ifdef FAST_BRDF_GUARD_NAN
    if (fast_brdf_is_finite(len2) && (len2 > FastBrdfEps)) {
        return v * rsqrtf(len2);
    }
#else
    if (len2 > FastBrdfEps) {
        return v * rsqrtf(len2);
    }
#endif
    return fallback;
}

static __device__ __forceinline__ float3 fast_brdf_lerp(const float3& a, const float3& b, const float t) {
    return a + (b - a) * t;
}

static __device__ __forceinline__ float3 compute_fast_brdf_normal_space(const float3& normal, const float3& localDir) {
    float3 tangent;
    float3 bitangent;
    branchlessONB(normal, tangent, bitangent);

    const float3 worldDir = tangent * localDir.x + bitangent * localDir.y + normal * localDir.z;
#ifdef FAST_BRDF_GUARD_NAN
    return fast_brdf_safe_normalize(worldDir, normal);
#else
    return worldDir;
#endif
}

static __device__ __forceinline__ float3 compute_fast_brdf_f0(const float3& albedo, const float metallic) {
    return fast_brdf_lerp(make_float3(0.04f), albedo, metallic);
}

static __device__ __forceinline__ float fast_brdf_effective_metallic(const float metallic) {
#ifdef ENABLE_METALLIC
    return fast_brdf_saturate(metallic);
#else
    return 0.0f;
#endif
}

static __device__ __forceinline__ float3 compute_fast_brdf_fresnel_schlick(const float cosTheta, const float3& f0) {
    const float x  = 1.0f - fast_brdf_saturate(cosTheta);
    const float x2 = x * x;
    const float x5 = x2 * x2 * x;
    return f0 + (make_float3(1.0f) - f0) * x5;
}

static __device__ __forceinline__ float3 sample_fast_brdf_diffuse_direction(
    const float3& normal,
    const float u1,
    const float u2) {
    const float xi  = fast_brdf_saturate(u1);
    const float r   = sqrtf(xi);
    const float phi = FastBrdfPi2 * fast_brdf_saturate(u2);
    float s;
    float c;
    sincosf(phi, &s, &c);

    const float z = sqrtf(fmaxf(0.0f, 1.0f - xi));
    const float3 localDir = make_float3(r * c, r * s, z);
    return compute_fast_brdf_normal_space(normal, localDir);
}

static __device__ __forceinline__ float3 sample_fast_brdf_ggx_half_vector(
    const float3& normal,
    const float roughness,
    const float u1,
    const float u2) {
    const float rough  = fast_brdf_clamp_roughness(roughness);
    const float alpha  = rough * rough;
    const float alpha2 = alpha * alpha;
    const float xi     = fast_brdf_saturate(u1);
    const float denom  = fmaxf(1.0f + (alpha2 - 1.0f) * xi, FastBrdfEps);

    const float cosTheta = sqrtf(fmaxf(0.0f, (1.0f - xi) / denom));
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi      = FastBrdfPi2 * fast_brdf_saturate(u2);
    float s;
    float c;
    sincosf(phi, &s, &c);

    const float3 localH = make_float3(sinTheta * c, sinTheta * s, cosTheta);
    return compute_fast_brdf_normal_space(normal, localH);
}

static __device__ __forceinline__ float3 sample_fast_brdf(
    const float3& wo,
    const float3& normal,
    const float3& albedo,
    const float metallic,
    const float roughness,
    const float3& rand,
    float3& nextRayDirection) {
    const float rough = fast_brdf_clamp_roughness(roughness);
    const float3 f0   = compute_fast_brdf_f0(albedo, metallic);

    float3 outFactor = make_float3(0.0f);
    float3 L         = normal;

    if (rand.z < 0.5f) {
        L = sample_fast_brdf_diffuse_direction(normal, rand.x, rand.y);
        const float3 H = fast_brdf_safe_normalize(wo + L, normal);
        const float3 F = compute_fast_brdf_fresnel_schlick(fast_brdf_positive_dot(wo, H), f0);

        const float3 diffuseColor = albedo * (1.0f - metallic);
        outFactor                 = diffuseColor * (make_float3(1.0f) - F);
    } else {
        const float3 H = sample_fast_brdf_ggx_half_vector(normal, rough, rand.x, rand.y);
        const float rawVdotH = dot(wo, H);
        L                   = 2.0f * rawVdotH * H - wo;

        const float NdotV = fast_brdf_positive_dot(normal, wo);
        const float NdotL = fast_brdf_positive_dot(normal, L);
        const float NdotH = fast_brdf_positive_dot(normal, H);
        const float VdotH = fast_brdf_positive_dot(wo, H);

        const float3 F = compute_fast_brdf_fresnel_schlick(VdotH, f0);

        const float k  = 0.5f * rough * rough;
        const float Gv = NdotV / fmaxf(NdotV * (1.0f - k) + k, FastBrdfEps);
        const float Gl = NdotL / fmaxf(NdotL * (1.0f - k) + k, FastBrdfEps);
        const float G  = Gv * Gl;

        outFactor = F * (G * VdotH / fmaxf(NdotH * NdotV, 1e-3f));
    }

    nextRayDirection = L;
    if (dot(normal, nextRayDirection) <= 0.0f) {
        nextRayDirection = normal;
    }

    return fast_brdf_clamp_nonnegative(outFactor * 2.0f);
}

struct FastBrdfValueGrad {
    float3 value;
    float3 dBrdf_dAlbedo;
    float3 dBrdf_dMetallic;
    float3 dBrdf_dRoughness;
};

static __device__ __forceinline__ float fast_brdf_nonnegative_grad_mask(const float v) {
#ifdef FAST_BRDF_GUARD_NAN
    return (fast_brdf_is_finite(v) && v > 0.0f) ? 1.0f : 0.0f;
#else
    return (v > 0.0f) ? 1.0f : 0.0f;
#endif
}

static __device__ __forceinline__ float3 fast_brdf_nonnegative_grad_mask(const float3& v) {
    return make_float3(
        fast_brdf_nonnegative_grad_mask(v.x),
        fast_brdf_nonnegative_grad_mask(v.y),
        fast_brdf_nonnegative_grad_mask(v.z));
}

static __device__ __forceinline__ FastBrdfValueGrad sample_fast_brdf_with_grads(
    const float3& wo,
    const float3& normal,
    const float3& albedo,
    const float metallic,
    const float roughness,
    const float3& rand,
    float3& nextRayDirection) {
    const float rough = fast_brdf_clamp_roughness(roughness);
    const float3 f0   = compute_fast_brdf_f0(albedo, metallic);

    float3 outFactor         = make_float3(0.0f);
    float3 dOut_dAlbedo      = make_float3(0.0f);
    float3 dOut_dRoughness   = make_float3(0.0f);
#ifdef ENABLE_METALLIC
    float3 dOut_dMetallic    = make_float3(0.0f);
#endif
    float3 L                 = normal;

    if (rand.z < 0.5f) {
        L = sample_fast_brdf_diffuse_direction(normal, rand.x, rand.y);
        const float3 H = fast_brdf_safe_normalize(wo + L, normal);

        const float cosTheta  = fast_brdf_positive_dot(wo, H);
        const float x         = 1.0f - fast_brdf_saturate(cosTheta);
        const float x2        = x * x;
        const float q         = x2 * x2 * x;
        const float oneMinusQ = 1.0f - q;

        const float3 F = f0 + (make_float3(1.0f) - f0) * q;

        const float dF_dAlbedo = oneMinusQ * metallic;
#ifdef ENABLE_METALLIC
        const float3 dF_dMetallic = (albedo - make_float3(0.04f)) * oneMinusQ;
#endif

        const float3 oneMinusF   = make_float3(1.0f) - F;
        const float oneMinusMetallic = 1.0f - metallic;
        const float3 diffuseColor = albedo * oneMinusMetallic;
        outFactor                 = diffuseColor * oneMinusF;

        dOut_dAlbedo   = oneMinusMetallic * (oneMinusF - albedo * dF_dAlbedo);
#ifdef ENABLE_METALLIC
        dOut_dMetallic = -albedo * oneMinusF - albedo * oneMinusMetallic * dF_dMetallic;
#endif
    } else {
        const float3 H = sample_fast_brdf_ggx_half_vector(normal, rough, rand.x, rand.y);
        const float rawVdotH = dot(wo, H);
        L                   = 2.0f * rawVdotH * H - wo;

        const float NdotV = fast_brdf_positive_dot(normal, wo);
        const float NdotL = fast_brdf_positive_dot(normal, L);
        const float NdotH = fast_brdf_positive_dot(normal, H);
        const float VdotH = fast_brdf_positive_dot(wo, H);

        const float x         = 1.0f - fast_brdf_saturate(VdotH);
        const float x2        = x * x;
        const float q         = x2 * x2 * x;
        const float oneMinusQ = 1.0f - q;

        const float3 F = f0 + (make_float3(1.0f) - f0) * q;

        const float dF_dAlbedo = oneMinusQ * metallic;
#ifdef ENABLE_METALLIC
        const float3 dF_dMetallic = (albedo - make_float3(0.04f)) * oneMinusQ;
#endif

        const float k  = 0.5f * rough * rough;
        const float Dv = fmaxf(NdotV * (1.0f - k) + k, FastBrdfEps);
        const float Dl = fmaxf(NdotL * (1.0f - k) + k, FastBrdfEps);
        const float Gv = NdotV / Dv;
        const float Gl = NdotL / Dl;
        const float G  = Gv * Gl;

        const float denom = fmaxf(NdotH * NdotV, 1e-3f);
        const float S     = G * VdotH / denom;

        outFactor       = F * S;
        dOut_dAlbedo    = make_float3(S * dF_dAlbedo);
#ifdef ENABLE_METALLIC
        dOut_dMetallic  = dF_dMetallic * S;
#endif

        const float dGv_dk = -NdotV * (1.0f - NdotV) / (Dv * Dv);
        const float dGl_dk = -NdotL * (1.0f - NdotL) / (Dl * Dl);
        const float dG_dk  = Gl * dGv_dk + Gv * dGl_dk;

        const float drough_dinput = (roughness > FastBrdfMinRough && roughness < 1.0f) ? 1.0f : 0.0f;
        const float dk_droughness = rough * drough_dinput;
        const float dS_dRoughness = VdotH / denom * dG_dk * dk_droughness;
        dOut_dRoughness           = F * dS_dRoughness;
    }

    nextRayDirection = L;
    if (dot(normal, nextRayDirection) <= 0.0f) {
        nextRayDirection = normal;
    }

    const float3 valueMask = fast_brdf_nonnegative_grad_mask(outFactor);

    FastBrdfValueGrad result;
    result.value                = fast_brdf_clamp_nonnegative(outFactor * 2.0f);
    result.dBrdf_dAlbedo        = dOut_dAlbedo * 2.0f * valueMask;
    result.dBrdf_dRoughness     = dOut_dRoughness * 2.0f * valueMask;
#ifdef ENABLE_METALLIC
    result.dBrdf_dMetallic      = dOut_dMetallic * 2.0f * valueMask;
#else
    result.dBrdf_dMetallic      = make_float3(0.0f);
#endif
    return result;
}

static __device__ __forceinline__ float3 sampled_fast_brdf(
    const float3& rayDirection,
    Sampler& sampler,
    const Interaction& interaction,
    float3& nextRayDirection) {
    const float3 normalFallback = make_float3(0.0f, 0.0f, 1.0f);
    const float3 normal = fast_brdf_safe_normalize(interaction.shadingnormal, normalFallback);
    const float3 wo     = fast_brdf_safe_normalize(-rayDirection, normal);

    const float3 albedo = fast_brdf_saturate(interaction.material.albedo);
    const float metallic = fast_brdf_effective_metallic(interaction.material.metallic);
    const float roughness = fast_brdf_clamp_roughness(interaction.material.roughness);

    return sample_fast_brdf(wo, normal, albedo, metallic, roughness, sampler.next_3d(), nextRayDirection);
}

static __device__ __forceinline__ FastBrdfValueGrad sampled_fast_brdf_with_grads(
    const float3& rayDirection,
    Sampler& sampler,
    const Interaction& interaction,
    float3& nextRayDirection) {
    const float3 normalFallback = make_float3(0.0f, 0.0f, 1.0f);
    const float3 normal = fast_brdf_safe_normalize(interaction.shadingnormal, normalFallback);
    const float3 wo     = fast_brdf_safe_normalize(-rayDirection, normal);

    const float3 albedo = fast_brdf_saturate(interaction.material.albedo);
    const float metallic = fast_brdf_effective_metallic(interaction.material.metallic);
    const float roughness = interaction.material.roughness;

    return sample_fast_brdf_with_grads(wo, normal, albedo, metallic, roughness, sampler.next_3d(), nextRayDirection);
}

#endif
