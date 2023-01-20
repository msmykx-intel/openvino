// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/op_conversions/convert_batch_to_space.hpp"
#include "transformations/op_conversions/convert_space_to_batch.hpp"

bool ov::pass::ConvertOpSet2ToOpSet1::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(ConvertOpSet2ToOpSet1);
    ngraph::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);

    manager.register_pass<ngraph::pass::ConvertSpaceToBatch>();
    manager.register_pass<ngraph::pass::ConvertBatchToSpace>();

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
