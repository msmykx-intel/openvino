// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include <openvino/runtime/auto/properties.hpp>

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCommon,
                         OVClassBasicPropsTestP,
                         ::testing::Values(std::make_pair("openvino_intel_cpu_plugin", "CPU")));

const std::vector<ov::AnyMap> cpu_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_properties)),
                         OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_Auto_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("YES")},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("NO")},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("YES")},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("NO")}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_Auto_properties)),
                         OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> cpu_setcore_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::enable_profiling(false)}};
const std::vector<ov::AnyMap> cpu_compileModel_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(10),
     ov::enable_profiling(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_cpuCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_setcore_properties),
                                            ::testing::ValuesIn(cpu_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_setcore_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)}};
const std::vector<ov::AnyMap> multi_compileModel_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)}};

INSTANTIATE_TEST_SUITE_P(smoke_MultiCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_setcore_properties),
                                            ::testing::ValuesIn(multi_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_setcore_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
};
const std::vector<ov::AnyMap> auto_compileModel_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)}};
INSTANTIATE_TEST_SUITE_P(smoke_AutoCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_setcore_properties),
                                            ::testing::ValuesIn(auto_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(false)},
        {ov::log::level("LOG_NONE")},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
        {ov::intel_auto::device_bind_buffer(false)},
        {ov::intel_auto::enable_startup_fallback(true)},
        {ov::device::priorities("")}
};
INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorTests, OVPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(default_properties)),
        OVPropertiesDefaultTests::getTestCaseName);

const std::vector<std::pair<ov::AnyMap, std::string>> automultiExeDeviceConfigs = {
    std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}}, "CPU")};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(automultiExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_multi_device_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{{ov::num_streams(4), ov::enable_profiling(true)}}}})}};

const std::vector<ov::AnyMap> auto_multi_incorrect_device_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsNoThrow,
                         OVSetSupportPropCompileModelWithoutConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(auto_multi_device_properties)),
                         OVSetSupportPropCompileModelWithoutConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsThrow,
                         OVSetUnsupportPropCompileModelWithoutConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(auto_multi_incorrect_device_properties)),
                         OVSetUnsupportPropCompileModelWithoutConfigTests::getTestCaseName);

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVSetEnableHyperThreadingHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVSetSchedulingCoreTypeHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiHeteroOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values("MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetConfigTest,
                         OVGetConfigTest_ThrowUnsupported,
                         ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetAvailableDevicesPropsTest,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVSetEnableCpuPinningHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVSetModelPriorityConfigTest,
                         OVSetModelPriorityConfigTest,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVSetLogLevelConfigTest, OVSetLogLevelConfigTest, ::testing::Values("MULTI", "AUTO"));

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

const std::vector<ov::AnyMap> configsDeviceProperties = {
    {ov::device::properties("CPU", ov::num_streams(3))},
    {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(3)}}})}};

const std::vector<ov::AnyMap> configsDevicePropertiesDouble = {
    {ov::device::properties("CPU", ov::num_streams(5)), ov::num_streams(3)},
    {ov::device::properties("CPU", ov::num_streams(5)),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(7)}}}),
     ov::num_streams(3)},
    {ov::device::properties("CPU", ov::num_streams(3)), ov::device::properties("CPU", ov::num_streams(5))},
    {ov::device::properties("CPU", ov::num_streams(3)),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(5)}}})},
    {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(3)}}}),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(5)}}})}};

const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
    {ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::device::priorities(CommonTestUtils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> heteroConfigsWithSecondaryProperties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::device::priorities(CommonTestUtils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO", "HETERO"),
                                            ::testing::ValuesIn(multiConfigs)));

// IE Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPUOVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU", "AUTO:CPU", "MULTI:CPU", "HETERO:CPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_HETERO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("HETERO"),
                                            ::testing::ValuesIn(heteroConfigsWithSecondaryProperties)));

// IE Class load and check network with ov::device::properties
INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckWithSecondaryPropertiesDoubleTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDevicePropertiesDouble)));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVGetConfigTest, OVGetConfigTest, ::testing::Values("CPU"));

// IE Class load and check network with ov::device::properties

}  // namespace
