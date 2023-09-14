// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makePooling(const ov::Output<Node>& in,
                                  const std::vector<size_t>& strides,
                                  const std::vector<size_t>& padsBegin,
                                  const std::vector<size_t>& padsEnd,
                                  const std::vector<size_t>& kernel,
                                  const op::RoundingType& roundingType,
                                  const op::PadType& padType,
                                  bool excludePad,
                                  const ov::test::utils::PoolingTypes& poolType) {
    std::shared_ptr<ov::Node> pooling;
    switch (poolType) {
    case ov::test::utils::PoolingTypes::MAX:
        pooling = std::make_shared<ov::op::v1::MaxPool>(in, strides, padsBegin, padsEnd, kernel, roundingType, padType);

        break;
    case ov::test::utils::PoolingTypes::AVG:
        pooling = std::make_shared<ov::op::v1::AvgPool>(in,
                                                        strides,
                                                        padsBegin,
                                                        padsEnd,
                                                        kernel,
                                                        excludePad,
                                                        roundingType,
                                                        padType);
        break;
    }
    return pooling;
}

std::shared_ptr<Node> makeMaxPoolingV8(const ov::Output<Node>& in,
                                       const std::vector<size_t>& strides,
                                       const std::vector<size_t>& dilation,
                                       const std::vector<size_t>& padsBegin,
                                       const std::vector<size_t>& padsEnd,
                                       const std::vector<size_t>& kernel,
                                       const op::RoundingType& roundingType,
                                       const op::PadType& padType,
                                       const ov::element::Type& indexElementType,
                                       const int64_t axis) {
    std::shared_ptr<ov::Node> pooling = std::make_shared<ov::op::v8::MaxPool>(in,
                                                                              strides,
                                                                              dilation,
                                                                              padsBegin,
                                                                              padsEnd,
                                                                              kernel,
                                                                              roundingType,
                                                                              padType,
                                                                              indexElementType,
                                                                              axis);
    return pooling;
}

}  // namespace builder
}  // namespace ngraph
