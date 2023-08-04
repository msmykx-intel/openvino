// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;
class NgramShapeInfer : public ShapeInferEmptyPads {
public:
    NgramShapeInfer(const size_t k) : m_k(k) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    size_t m_k;
};

class NgramShapeInferFactory : public ShapeInferFactory {
public:
    NgramShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

