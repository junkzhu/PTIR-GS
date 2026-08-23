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

// No standard headers like <cstdint>: OptiX NVRTC JIT often cannot resolve them.

#if defined(__CUDACC__) || defined(__OPTIX__)
#include <vector_types.h>
#endif

// Uniform RNG for Monte Carlo / PBR in device code (PCG32).

struct Sampler {
    unsigned long long state;
    unsigned long long inc;

    static __host__ __device__ __forceinline__ unsigned long long reproducibleLaunchSeed(
        unsigned int launchX,
        unsigned int launchY,
        unsigned int launchZ,
        unsigned int frameNumber) {
        unsigned long long s = 1469598103934665603ull;
        s ^= static_cast<unsigned long long>(launchX) + 1099511628211ull * s;
        s ^= static_cast<unsigned long long>(launchY) + 1099511628211ull * s;
        s ^= static_cast<unsigned long long>(launchZ) + 1099511628211ull * s;
        s ^= static_cast<unsigned long long>(frameNumber) + 1099511628211ull * s;
        return s ? s : 1ull;
    }

    __host__ __device__ __forceinline__ void initFromLaunch(
        unsigned int launchX,
        unsigned int launchY,
        unsigned int launchZ,
        unsigned int frameNumber,
        unsigned long long streamSeq = 0) {
        init(reproducibleLaunchSeed(launchX, launchY, launchZ, frameNumber), streamSeq);
    }

#if defined(__CUDACC__) || defined(__OPTIX__)
    __host__ __device__ __forceinline__ void initFromLaunch(const uint3& launchIndex, unsigned int frameNumber, unsigned long long streamSeq = 0) {
        initFromLaunch(launchIndex.x, launchIndex.y, launchIndex.z, frameNumber, streamSeq);
    }
#endif

    __host__ __device__ __forceinline__ void init(unsigned long long seed, unsigned long long seq = 0) {
        state = 0;
        inc   = (seq << 1u) | 1ull;
        next_u32();
        state += seed;
        next_u32();
    }

    __host__ __device__ __forceinline__ unsigned int next_u32() {
        const unsigned long long oldstate = state;
        state                             = oldstate * 6364136223846793005ull + inc;
        unsigned int xorshifted           = static_cast<unsigned int>(((oldstate >> 18ull) ^ oldstate) >> 27ull);
        const unsigned int rot            = static_cast<unsigned int>(oldstate >> 59ull);
        return (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));
    }

    __host__ __device__ __forceinline__ float next_1d() {
        const unsigned int u = next_u32() & 0x00FFFFFFu;
        return static_cast<float>(u) * (1.0f / 16777216.0f);
    }

    __host__ __device__ __forceinline__ Sampler fork(unsigned long long streamOffset) const {
        Sampler forked = *this;
        forked.inc += streamOffset << 1u;
        forked.next_u32(); // Make the next value depend on the forked stream.
        return forked;
    }

#if defined(__CUDACC__) || defined(__OPTIX__)
    __host__ __device__ __forceinline__ float3 next_3d() {
        return make_float3(next_1d(), next_1d(), next_1d());
    }
#endif
};
