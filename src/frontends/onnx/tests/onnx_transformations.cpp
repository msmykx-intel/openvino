// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_control.hpp"
#include "editor.hpp"
#include "gtest/gtest.h"
#include "onnx_test_util.hpp"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::onnx_editor;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

namespace {
// Names of input and output names of nodes after a function expanding have names based on a node address.
// As a result, the names are different during each tests execution.
// It requires custom way of input/output names comparison.
// https://github.com/onnx/onnx/blob/767f752829f83dbc9bd0a364d6138890f667fc38/onnx/defs/function.cc#L23
bool after_func_expand_name_comp(std::string lhs, std::string rhs) {
    // it is equivalent (simplified) to (0x)?[0-9A-Fa-f]{8,} regex, but GCC 4.8 has limited support
    auto cut_hex_address = [](std::string& name) {
        auto is_hex_symbol = [](const char s) {
            if ((s >= 'a' && s <= 'f') || (s >= 'A' && s <= 'F') || (s >= '0' && s <= '9') ||
                (s == 'x')) {  // if begin with "0x"
                return true;
            }
            return false;
        };
        // minimum address length (32 bit platforms)
        const auto min_address = 8;
        auto cut_begin = -1;
        auto cut_length = -1;

        auto founded_hex = 0;
        for (size_t i = 0; i < name.size(); ++i) {
            if (is_hex_symbol(name[i])) {
                ++founded_hex;
                if (cut_begin == -1) {
                    cut_begin = static_cast<int>(i);
                }
                if (founded_hex >= min_address) {
                    cut_length = founded_hex;
                }
            } else if (founded_hex < min_address) {
                cut_begin = -1;
                cut_length = -1;
                founded_hex = 0;
            }
        }
        if (cut_begin > 0 && cut_length > 0) {
            return name.erase(cut_begin, cut_length);
        }
        return name;
    };
    return cut_hex_address(lhs) == cut_hex_address(rhs);
}
}  // namespace

OPENVINO_TEST(onnx_transformations, expand_function_greater_or_equal) {
    ONNXModelEditor editor{util::path_join({ov::test::utils::getExecutableDirectory(),
                                            TEST_ONNX_MODELS_DIRNAME,
                                            "transformations/greater_or_equal.onnx"})};
    editor.decode();  // onnx transformations are applied

    const auto ref_model = util::path_join({ov::test::utils::getExecutableDirectory(),
                                            TEST_ONNX_MODELS_DIRNAME,
                                            "transformations/reference/"
                                            "greater_or_equal_expanded.onnx"});

    const auto result = compare_onnx_models(editor.model_string(), ref_model, after_func_expand_name_comp);

    // After operation translation was implemented - check it doesn't apply
    EXPECT_FALSE(result.is_ok) << result.error_message;
}

// Disabled, ticket: #81976
OPENVINO_TEST(onnx_transformations, DISABLED_expand_function_softmax_crossentropy) {
    ONNXModelEditor editor{util::path_join({ov::test::utils::getExecutableDirectory(),
                                            TEST_ONNX_MODELS_DIRNAME,
                                            "transformations/softmax_crossentropy_consumed.onnx"})};
    editor.decode();  // onnx transformations are applied

    const auto ref_model = util::path_join({ov::test::utils::getExecutableDirectory(),
                                            TEST_ONNX_MODELS_DIRNAME,
                                            "transformations/reference/"
                                            "softmax_crossentropy_consumed_expanded.onnx"});

    const auto result = compare_onnx_models(editor.model_string(), ref_model, after_func_expand_name_comp);
    EXPECT_TRUE(result.is_ok) << result.error_message;
}
