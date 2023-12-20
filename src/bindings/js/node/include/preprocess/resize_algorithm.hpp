// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include "helper.hpp"

/** \brief Creates JS object to represent C++ enum class ResizeAlgorithm */
Napi::Value enumResizeAlgorithm(const Napi::CallbackInfo& info);
