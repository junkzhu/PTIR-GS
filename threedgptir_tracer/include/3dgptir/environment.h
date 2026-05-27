// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optix.h>

#if defined(NVDR_TORCH) && !defined(__CUDACC__)
#include <torch/types.h>
#endif

enum EnvironmentType {
    EnvironmentType_2D   = 0,
    EnvironmentType_Cube = 1,
};

struct EnvAliasTable {
#if defined(NVDR_TORCH) && !defined(__CUDACC__)
    EnvAliasTable()
        : width(0),
          height(0),
          numCells(0),
          prob(nullptr),
          alias(nullptr),
          pdf(nullptr) {
    }

    explicit EnvAliasTable(const torch::Tensor& table);
#endif

    int width;
    int height;
    int numCells;
    const float* prob;
    const float* alias;
    const float* pdf;
};

struct Environment {
#if defined(NVDR_TORCH) && !defined(__CUDACC__)
    Environment()
        : data(nullptr),
          width(0),
          height(0),
          type(EnvironmentType_2D),
          offset{0.0f, 0.0f},
          aliasTable() {
    }

    explicit Environment(const torch::Tensor& environment);
    Environment(const torch::Tensor& environment, const torch::Tensor& aliasTable);
#endif

    const float4* data; ///< Optional environment, stored as HxWx4 float data.
    int width;
    int height;
    int type; ///< EnvironmentType_2D or EnvironmentType_Cube.
    float2 offset;
    EnvAliasTable aliasTable;
};
