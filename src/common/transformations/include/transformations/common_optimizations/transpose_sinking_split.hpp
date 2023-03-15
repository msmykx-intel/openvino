// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingSplitBackward;
class TRANSFORMATIONS_API TransposeSinkingSplitForward;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingSplitForward transformation sinks Transpose through Split, VariadicSplit operations
 * in the forward direction.
 */
class ov::pass::TransposeSinkingSplitForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSplitForward", "0");
    TransposeSinkingSplitForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingSplitBackward transformation sinks Transpose through Split, VariadicSplit operations
 * in the backward direction.
 */
class ov::pass::TransposeSinkingSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingSplitBackward", "0");
    TransposeSinkingSplitBackward();
};
