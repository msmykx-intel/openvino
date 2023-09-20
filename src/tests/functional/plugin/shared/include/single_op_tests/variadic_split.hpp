// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/variadic_split.hpp"

namespace ov {
namespace test {
TEST_P(VariadicSplitLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
