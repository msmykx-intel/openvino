// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<InputShape> inputShape = {
    {{}, {{1, 16}}},
    {{}, {{1, 32}}},
    {{}, {{1, 1}}},
    {{}, {{1, 9}}},
    {{}, {{1, 17}}},
    {{}, {{1, 19}}},
    {{}, {{1, 49}}},
    {{}, {{1, 50}}},
    {{}, {{5, 16}}},
    {{}, {{5, 32}}},
    {{}, {{5, 1}}},
    {{}, {{5, 9}}},
    {{}, {{5, 17}}},
    {{}, {{5, 19}}},
    {{}, {{5, 49}}},
    {{}, {{5, 50}}},
    {{}, {{1, 3, 128, 128}}},
    {{}, {{1, 3, 128, 129}}},
    {{}, {{1, 3, 128, 130}}},
    {{}, {{1, 3, 128, 1}}},
    {{}, {{1, 3, 128, 9}}},
    {{}, {{1, 3, 128, 16}}},
    {{}, {{1, 3, 128, 17}}},
    {{}, {{1, 3, 128, 20}}},
    // DS
    {{-1, -1}, {{1, 16}, {1, 32}, {1, 1}, {1, 9}, {1, 17}, {1, 19}, {1, 49}, {1, 50}, {5, 16}, {1, 16}, {1, 9}}},
    {{-1, -1, -1, -1}, {{1, 3, 128, 128}, {1, 3, 128, 129}, {1, 3, 128, 130}, {1, 3, 128, 1}, {1, 3, 128, 16}, {1, 3, 128, 1}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Softmax, Softmax,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShape),
                             ::testing::Values(-1),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     Softmax::getTestCaseName);

const std::vector<std::pair<InputShape, InputShape>> inputShapesPair = {
    {{{}, {{1, 5, 16, 35}}}, {{}, {{1, 5, 16, 35}}}},
    {{{}, {{1, 5, 16,  1}}}, {{}, {{1, 5, 16, 35}}}},
    {{{}, {{1, 5, 16, 35}}}, {{}, {{1, 5,  1,  1}}}},
    {{{}, {{1, 5, 16,  1}}}, {{}, {{1, 5, 16,  1}}}},
    {{{}, {{1, 5, 16, 35}}}, {{}, {{1, 5,  1, 35}}}},
    {{{}, {{1, 5,  1, 35}}}, {{}, {{1, 5,  1, 35}}}},
    // DS
    {{{-1, -1, -1, -1}, {{1, 5, 16, 35}, {1, 5, 16,  1}, {1, 5, 16, 35}}}, {{-1, -1, -1, -1}, {{1, 5, 16, 35}, {1, 5, 16, 35}, {1, 5, 16, 35}}}},
    {{{-1, {1, 8}, {1, 16}, {1, 16}}, {{1, 3, 1, 8}, {1, 8, 16, 16}, {1, 3, 1, 8}}}, {{-1, {1, 8}, -1, {1, 8}}, {{1, 3, 2, 8}, {2, 1, 1, 1}, {1, 3, 2, 8}}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_AddSoftmax, AddSoftmax,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShapesPair),
                             ::testing::Values(-1),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     AddSoftmax::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov