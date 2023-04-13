// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_cpu {

#define CPU_REGISTER_PASS_COMMON(MANAGER, PASS, ...) \
    MANAGER.register_pass<PASS>(__VA_ARGS__);

#define CPU_DISABLE_PASS_COMMON(MANAGER, PASS) \
    MANAGER.get_pass_config()->disable<PASS>();

#define CPU_ENABLE_PASS_COMMON(MANAGER, PASS) \
    MANAGER.get_pass_config()->enable<PASS>();

#define CPU_SET_CALLBACK_COMMON(MANAGER, CALLBACK, ...) \
    MANAGER.get_pass_config()->set_callback<__VA_ARGS__>(CALLBACK);

#if defined(OPENVINO_ARCH_X86_64)

#define CPU_REGISTER_PASS_X64(MANAGER, PASS, ...) CPU_REGISTER_PASS_COMMON(MANAGER, PASS, __VA_ARGS__)
#define CPU_DISABLE_PASS_X64(MANAGER, PASS) CPU_DISABLE_PASS_COMMON(MANAGER, PASS)
#define CPU_ENABLE_PASS_X64(MANAGER, PASS) CPU_ENABLE_PASS_COMMON(MANAGER, PASS)
#define CPU_SET_CALLBACK_X64(MANAGER, CALLBACK, ...) CPU_SET_CALLBACK_COMMON(MANAGER, CALLBACK, __VA_ARGS__)

#else

#define CPU_REGISTER_PASS_X64(MANAGER, PASS, ...)
#define CPU_DISABLE_PASS_X64(MANAGER, PASS)
#define CPU_ENABLE_PASS_X64(MANAGER, PASS)
#define CPU_SET_CALLBACK_X64(MANAGER, CALLBACK, ...)

#endif

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)

#define CPU_REGISTER_PASS_ARM(MANAGER, PASS, ...) CPU_REGISTER_PASS_COMMON(MANAGER, PASS, __VA_ARGS__)
#define CPU_DISABLE_PASS_ARM(MANAGER, PASS) CPU_DISABLE_PASS_COMMON(MANAGER, PASS)
#define CPU_ENABLE_PASS_ARM(MANAGER, PASS) CPU_ENABLE_PASS_COMMON(MANAGER, PASS)
#define CPU_SET_CALLBACK_ARM(MANAGER, CALLBACK, ...) CPU_SET_CALLBACK_COMMON(MANAGER, CALLBACK, __VA_ARGS__)

#else

#define CPU_REGISTER_PASS_ARM(MANAGER, PASS, ...)
#define CPU_DISABLE_PASS_ARM(MANAGER, PASS)
#define CPU_ENABLE_PASS_ARM(MANAGER, PASS)
#define CPU_SET_CALLBACK_ARM(MANAGER, CALLBACK, ...)

#endif

}   // namespace intel_cpu
}   // namespace ov
