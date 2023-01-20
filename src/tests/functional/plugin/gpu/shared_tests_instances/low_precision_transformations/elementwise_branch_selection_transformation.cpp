// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/elementwise_branch_selection_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
};

const std::vector<std::string> elementwiseTypes = {
    "add",
    "multiply"
};

const std::vector<LayerTestsDefinitions::ElementwiseBranchSelectionTestValues> params = {
    {
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            {}
        },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {}, // GPU doesn't returns Reorders in performance counters
        {
            {"convolution1", "U8"},
            {"convolution2", "U8"},
            {"eltwise", "U8"}
        }
    },
    {
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            {}
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {}, // GPU doesn't returns Reorders in performance counters
        {
            {"convolution1", "U8"},
            {"convolution2", "U8"},
            {"eltwise", "U8"}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ElementwiseBranchSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(elementwiseTypes)),
    ElementwiseBranchSelectionTransformation::getTestCaseName);
}  // namespace
