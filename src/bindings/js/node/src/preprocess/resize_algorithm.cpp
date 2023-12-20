// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "preprocess/resize_algorithm.hpp"

#include <typeinfo>

#include "openvino/runtime/core.hpp"

Napi::Value enumResizeAlgorithm(const Napi::CallbackInfo& info) {
    auto enumObj = Napi::Object::New(info.Env());
    std::vector<Napi::PropertyDescriptor> pds;

    static const std::array<std::string, 3> resizeAlgorithms = {"RESIZE_LINEAR", "RESIZE_CUBIC", "RESIZE_NEAREST"};

    for (auto& algorithm : resizeAlgorithms) {
        pds.push_back(
            Napi::PropertyDescriptor::Value(algorithm, Napi::String::New(info.Env(), algorithm), napi_default));
    }

    enumObj.DefineProperties(pds);
    return enumObj;
}
