// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/utils.hpp"

#include "intel_gpu/plugin/sync_infer_request.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_allocators.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <utility>

namespace {

inline std::string get_port_name(const ov::Output<const ov::Node>& port, const bool is_legacy_api) {
    std::string name;
    // TODO: Should use tensor name as the port name, but many legacy tests still use legacy name
    // plus sometimes it will get empty tensor name.
    if (!is_legacy_api) {
        name = {};
    }
    if (name.empty()) {
        bool is_input = ov::op::util::is_parameter(port.get_node());
        if (is_input) {
            name = ov::op::util::get_ie_output_name(port);
        } else {
            const auto node = port.get_node_shared_ptr();
            name = ov::op::util::get_ie_output_name(node->input_value(0));
        }
    }
    return name;
}

template <typename src_t, typename dst_t>
void convert_any_copy(const src_t* src, dst_t* dst, size_t size) {
    OPENVINO_ASSERT(src && dst, "[GPU] Src or Dst ptr is null");
    for (size_t i = 0; i < size; i++)
        dst[i] = static_cast<dst_t>(src[i]);
}

void convert_and_copy(const void* src_ptr, ov::element::Type src_et, void* dst_ptr, ov::element::Type dst_et, size_t size) {
    if (size == 0)
        return;

    if (src_et == dst_et) {
        std::memcpy(dst_ptr, src_ptr, size);
        return;
    }

    #define CASE(s_et, d_et, s_type, d_type) \
        if (src_et == s_et && dst_et == d_et) return convert_any_copy(static_cast<const s_type*>(src_ptr), static_cast<d_type*>(dst_ptr), size)

    // For unsupported inputs
    CASE(ov::element::f64, ov::element::f32, double, float);
    CASE(ov::element::i16, ov::element::f32, int16_t, float);
    CASE(ov::element::u16, ov::element::f32, uint16_t, float);
    CASE(ov::element::u64, ov::element::i32, uint64_t, int32_t);
    CASE(ov::element::i64, ov::element::i32, int64_t, int32_t);
    CASE(ov::element::u32, ov::element::i32, uint32_t, int32_t);

    // For unsupported outputs
    CASE(ov::element::f32, ov::element::f64, float, double);
    CASE(ov::element::i32, ov::element::i64, int32_t, int64_t);
    CASE(ov::element::i32, ov::element::u64, int32_t, uint64_t);
    CASE(ov::element::i32, ov::element::u32, int32_t, uint32_t);
    CASE(ov::element::f32, ov::element::i16, float, int16_t);
    CASE(ov::element::f32, ov::element::u16, float, uint16_t);

    // TODO: Need instances below?
    CASE(ov::element::u32, ov::element::i64, uint32_t, int64_t);
    CASE(ov::element::u32, ov::element::u64, uint32_t, uint64_t);

    OPENVINO_THROW("[GPU] Unsupported element types combination for copy: ", src_et, " -> ", dst_et);
}

bool is_convert_required(ov::element::Type src_et, ov::element::Type dst_et) {
    return src_et != dst_et && !(dst_et == ov::element::boolean && src_et == ov::element::u8);
}

void convert_and_copy(const cldnn::memory::ptr src, ov::ITensor const* dst, const cldnn::stream& stream) {
    auto src_et = cldnn::data_type_to_element_type(src->get_layout().data_type);
    auto dst_et = dst->get_element_type();

    size_t size = ov::shape_size(dst->get_shape());

    cldnn::mem_lock<uint8_t> src_lock(src, stream);
    std::unique_ptr<cldnn::mem_lock<uint8_t>> dst_lock = nullptr;

    const void* src_ptr = src_lock.data();
    void* dst_ptr = nullptr;

    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(dst)) {
        auto mem = remote->get_original_memory();
        dst_lock.reset(new cldnn::mem_lock<uint8_t>(mem, stream));
        dst_ptr = dst_lock->data();
    } else {
        dst_ptr = dst->data();
    }

    return convert_and_copy(src_ptr, src_et, dst_ptr, dst_et, size);
}

void convert_and_copy(const ov::ITensor* src, ov::ITensor const* dst, const cldnn::stream& stream) {
    auto src_et = src->get_element_type();
    auto dst_et = dst->get_element_type();

    size_t size = ov::shape_size(dst->get_shape());

    const void* src_ptr = nullptr;
    void* dst_ptr = nullptr;

    std::unique_ptr<cldnn::mem_lock<uint8_t>> src_lock = nullptr;
    std::unique_ptr<cldnn::mem_lock<uint8_t>> dst_lock = nullptr;

    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(src)) {
        auto mem = remote->get_original_memory();
        src_lock.reset(new cldnn::mem_lock<uint8_t>(mem, stream));
        src_ptr = src_lock->data();
    } else {
        src_ptr = src->data();
    }

    if (auto remote = dynamic_cast<const ov::intel_gpu::RemoteTensorImpl*>(dst)) {
        auto mem = remote->get_original_memory();
        dst_lock.reset(new cldnn::mem_lock<uint8_t>(mem, stream));
        dst_ptr = dst_lock->data();
    } else {
        dst_ptr = dst->data();
    }

    return convert_and_copy(src_ptr, src_et, dst_ptr, dst_et, size);
}

bool same_host_mem(cldnn::memory::cptr memory, const uint8_t* host_ptr) {
    const uint8_t* device_ptr = memory->get_allocation_type() == cldnn::allocation_type::usm_host ?
                                static_cast<uint8_t*>(memory->get_internal_params().mem) : nullptr;
    return device_ptr == host_ptr;
}

ov::Shape predict_shape(const std::string& name, const ov::Shape current_shape, ov::element::Type element_type, cldnn::ShapePredictor& shape_predictor) {
    auto et_size = cldnn::ceil_div(element_type.bitwidth(), 8);
    auto prealloc_info = shape_predictor.predict_preallocation_shape(name, current_shape, et_size, false);
    const auto& preallocation_shape = prealloc_info.second;
    auto can_preallocate_buffer = prealloc_info.first &&
                                    shape_predictor.can_preallocate(ov::shape_size(preallocation_shape) * et_size);
    if (can_preallocate_buffer) {
        return preallocation_shape;
    }

    return current_shape;
}

inline bool all_remote_buffers(const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const ov::SoPtr<ov::ITensor>& tensor) {
        if (auto remote_ptr = std::dynamic_pointer_cast<ov::intel_gpu::RemoteTensorImpl>(tensor._ptr)) {
            return !remote_ptr->is_surface();
        }
        return false;
    });
}

inline bool all_remote_surfaces(const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const ov::SoPtr<ov::ITensor>& tensor) {
        if (auto remote_ptr = std::dynamic_pointer_cast<ov::intel_gpu::RemoteTensorImpl>(tensor._ptr)) {
            return remote_ptr->is_surface();
        }
        return false;
    });
}

inline bool all_host_tensors(const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const ov::SoPtr<ov::ITensor>& tensor) {
        return std::dynamic_pointer_cast<ov::intel_gpu::RemoteTensorImpl>(tensor._ptr) == nullptr;
    });
}

}  // namespace

namespace ov {
namespace intel_gpu {

// ----------------------------------------------------------------------------------------------- //
// ---------------------------- OpenVINO API impl ------------------------------------------------ //
// ----------------------------------------------------------------------------------------------- //

SyncInferRequest::SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model)
    , m_graph(compiled_model->get_graph(0))
    , m_context(std::static_pointer_cast<RemoteContextImpl>(compiled_model->get_context_impl()))
    , m_enable_profiling(m_graph->get_config().get_property(ov::enable_profiling))
    , m_use_external_queue(m_graph->use_external_queue()) {
    bool is_legacy_api = !compiled_model->is_new_api();
    init_mappings(is_legacy_api);
    allocate_inputs();
    allocate_outputs();
    allocate_states();
}

void SyncInferRequest::infer() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::infer");
    setup_stream_graph();
    std::lock_guard<std::mutex> lk(m_graph->get_mutex());
    enqueue();
    wait();
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    OPENVINO_ASSERT(m_enable_profiling, "[GPU] Profiling data was not collected: please check that ov::enable_profiling property was set to true");
    return m_graph->get_profiling_info();
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> ret{};
    const auto& variable_states = m_graph->get_network()->get_variable_memories();
    for (const auto& pair : variable_states) {
        ret.emplace_back(std::make_shared<VariableState>(pair.first, pair.second, m_graph->get_engine()));
    }
    auto expected_states_count = m_graph->get_network()->get_variables_state_info().size();
    OPENVINO_ASSERT(expected_states_count == ret.size(), "[GPU] Mismatch of expected states count (",
                                                         expected_states_count,  ") and actual size (", ret.size(), ")");
    return ret;
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::set_tensor");
    const auto& compiled_model = std::static_pointer_cast<const CompiledModel>(get_compiled_model());
    const auto name = get_port_name(port, !compiled_model->is_new_api());
    const auto& shape = port.get_partial_shape();

    OPENVINO_ASSERT(tensor != nullptr, "[GPU] Failed to set empty tensor to port: \'", name, "\'");
    OPENVINO_ASSERT(port.get_element_type() == tensor->get_element_type(),
                    "[GPU] Mismtach tensor and port type: ", port.get_element_type(), " vs ", tensor->get_element_type());
    OPENVINO_ASSERT(shape.compatible(ov::PartialShape(tensor->get_shape())) || tensor->get_shape() == ov::Shape{0},
                    "[GPU] The tensor size is not equal to model, can't set input tensor with name: ",
                    name,
                    ", because model input (shape=",
                    shape,
                    ") and tensor (shape=",
                    tensor->get_shape(),
                    ") are incompatible");

    bool is_input = ov::op::util::is_parameter(port.get_node());

    if (is_input) {
        m_user_inputs[name] = { tensor._ptr, TensorOwner::USER };
    } else {
        m_user_outputs[name] = { tensor._ptr, TensorOwner::USER };
    }

    ov::ISyncInferRequest::set_tensor(port, tensor);
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    if (tensors.size() == 1) {
        return set_tensor(port, tensors[0]);
    }
    bool is_input = ov::op::util::is_parameter(port.get_node());
    OPENVINO_ASSERT(is_input, "[GPU] set_tensors_impl is not supported for output port");

    bool is_remote = all_remote_buffers(tensors) || all_remote_surfaces(tensors);
    bool is_host = all_host_tensors(tensors);

    OPENVINO_ASSERT(is_host || is_remote, "[GPU] Incorrect input blobs. All blobs must be of the same type");

    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("[GPU] Cannot find input tensors for port ", port);
}

ov::SoPtr<ov::ITensor> SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    bool is_input = ov::op::util::is_parameter(port.get_node());
    const auto& compiled_model = std::static_pointer_cast<const CompiledModel>(get_compiled_model());
    const auto name = get_port_name(port, !compiled_model->is_new_api());
    if (is_input) {
        OPENVINO_ASSERT(m_user_inputs.count(name) == 1, "[GPU] Input tensor with name ", name, " is not found");
        return { m_user_inputs.at(name).ptr, nullptr };
    } else {
        OPENVINO_ASSERT(m_user_outputs.count(name) == 1, "[GPU] Output tensor with name ", name, " is not found");
        return { m_user_outputs.at(name).ptr, nullptr };
    }
}

void SyncInferRequest::check_tensors() const {
    const auto& inputs = get_compiled_model()->inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        if (!is_batched_input(inputs[i]))
            check_tensor(inputs[i], get_tensor_ptr(inputs[i]));
    }
    const auto& outputs = get_compiled_model()->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        check_tensor(outputs[i], get_tensor_ptr(outputs[i]));
    }
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal pipeline stages ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void SyncInferRequest::set_task_executor(const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor) {
    m_stream_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(task_executor);
}

void SyncInferRequest::enqueue_notify() {
    m_graph->wait(Graph::Stage::EXECUTE);
    enqueue();
}

void SyncInferRequest::wait_notify() {
    wait();
    m_graph->notify(Graph::Stage::EXECUTE);
}

void SyncInferRequest::enqueue() {
    // set input and output memory from request blob maps
    // into the network object primitives
    std::vector<cldnn::event::ptr> dependencies;

    for (const auto& it : m_input_ports_map) {
        const auto& name = it.first;
        const auto& port = it.second;

        if (m_batched_tensors.count(port.get_tensor_ptr()) > 0) {
            auto events = prepare_batched_input(name, port, m_batched_tensors.at(port.get_tensor_ptr()));
            std::move(events.begin(), events.end(), std::back_inserter(dependencies));
        } else {
            auto events = prepare_input(name, port, m_user_inputs.at(name));
            std::move(events.begin(), events.end(), std::back_inserter(dependencies));
        }
    }

    for (const auto& it : m_output_ports_map) {
        const auto& name = it.first;
        const auto& port = it.second;

        auto events = prepare_output(name, port, m_user_outputs.at(name));
        std::move(events.begin(), events.end(), std::back_inserter(dependencies));
    }

    auto network = m_graph->get_network();
    network->assign_variables_memories();

    m_internal_outputs.clear();
    m_internal_outputs = network->execute(dependencies);

    // If dump layers path is set, only runs first inference.
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0 && debug_config->dump_iteration.empty()) {
        GPU_DEBUG_INFO << "Only run first inference to dump layers." << std::endl;
        exit(0);
    }
}

void SyncInferRequest::wait() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::wait");
    OPENVINO_ASSERT(!m_internal_outputs.empty(), "[GPU] Inference was not started!\n");

    // wait for completion & collect outputs as requested by the model
    // for in_order_queue, it is enough to call finish only once
    bool do_sync_per_output = (m_graph->get_network()->get_stream().get_queue_type() == QueueTypes::in_order) ? false : true;
    if (!do_sync_per_output)
        m_graph->get_network()->get_stream().finish();

    std::vector<cldnn::event::ptr> copy_events;

    for (const auto& it : m_output_ports_map) {
        const auto& name = it.first;
        const auto& port = it.second;
        cldnn::primitive_id internal_name = m_output_names_map.at(name);
        auto output_memory = m_internal_outputs.at(internal_name).get_memory(do_sync_per_output);
        auto output_layout = m_internal_outputs.at(internal_name).get_layout();

        if (output_memory) {
            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::wait::reinterpret_memory");
            OPENVINO_ASSERT(!output_memory->get_layout().data_padding, "[GPU] Unexpected padding in output buffer");
            output_memory = m_graph->get_engine().reinterpret_buffer(*output_memory, output_layout);
        }

        OPENVINO_ASSERT(m_user_outputs.count(name) > 0, "[GPU] Output ", name, " is not found in output tensors map");
        auto output_tensor_wrapper = m_user_outputs.at(name);
        auto output_tensor = output_tensor_wrapper.ptr;
        auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(output_tensor);
        bool is_remote = remote_ptr != nullptr;

        bool need_output_update = output_layout.bytes_count() == 0 || (output_memory && output_tensor->get_byte_size() != output_memory->size());
        if (need_output_update) {
            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::wait::update_output");
            auto mem_shape = output_layout.get_shape();
            // In case of old shape infer we need to shrink out tensor shape to avoid redudnant dimensions that occur due to rank extension
            // For new shape infer this shouldn't happen, thus remove that WA once we migrate to ngraph-based shape infer for all cases
            if (!m_graph->get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
                OPENVINO_ASSERT(port.get_partial_shape().is_static(), "[GPU] Unexpected dynamic shape for legacy shape inference");
                OPENVINO_ASSERT(ov::shape_size(port.get_shape()) == ov::shape_size(mem_shape), "[GPU] Unexpected elements count for output tensor");
                mem_shape = port.get_shape();
            }
            output_tensor->set_shape(mem_shape);
        }

        // mapping remote blobs not needed -
        // let the user take care of them explicitly
        if (!is_remote && output_memory) {
            auto dst_ptr = static_cast<uint8_t*>(output_tensor->data());
            bool same_mem = same_host_mem(output_memory, dst_ptr);
            if (!same_mem && output_memory->size()) {
                if (auto ev = copy_output_data(output_memory, *output_tensor)) {
                    copy_events.push_back(ev);
                }
            }
        }
    }

    if (!copy_events.empty()) {
        auto& stream = m_graph->get_network()->get_stream();
        if (stream.get_queue_type() == QueueTypes::in_order) {
            // wait only the last one
            stream.wait_for_events({copy_events.back()});
        } else {
            stream.wait_for_events(copy_events);
        }
    }

    // finally collect profiling info
    if (m_enable_profiling) {
        m_graph->update_profiling_info();
    }
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal utils --------- ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void SyncInferRequest::setup_stream_graph() {
    int stream_id = 0;
    auto& stream_graphs = std::static_pointer_cast<const CompiledModel>(get_compiled_model())->get_graphs();
    if (nullptr != m_stream_executor) {
        stream_id = m_stream_executor->get_stream_id();
        auto num_graphs = stream_graphs.size();
        stream_id = stream_id % num_graphs;
    }
    m_graph = stream_graphs[stream_id];
}

std::shared_ptr<ov::ITensor> SyncInferRequest::create_host_tensor(const ov::PartialShape& port_shape, const ov::element::Type& port_element_type) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::create_host_tensor");
    // Disable USM usage as USMHostAllocator may fail for attempt to allocate 0 bytes
    // If we add WA for such case to avoid driver call, then deallocate method will return false and Blob::setShape call will throw an exception
    bool use_usm = m_graph->get_engine().use_unified_shared_memory() && !port_shape.is_dynamic();

    auto shape = port_shape.is_static() ? port_shape.to_shape() : ov::Shape(port_shape.size(), 0);
    auto usm_allocator = USMHostAllocator(m_context);
    return use_usm ? ov::make_tensor(port_element_type, shape, usm_allocator)
                   : ov::make_tensor(port_element_type, shape);
}

std::shared_ptr<ov::ITensor> SyncInferRequest::create_device_tensor(const ov::Shape& shape, ov::element::Type element_type,
                                                                    bool need_lockable_memory, void* mem_ptr) const {
    TensorType tensor_type = TensorType::BT_EMPTY;
    if (mem_ptr) {
        tensor_type = TensorType::BT_USM_SHARED;
    } else if (m_graph->get_engine().use_unified_shared_memory()) {
        tensor_type = need_lockable_memory ? TensorType::BT_USM_HOST_INTERNAL : TensorType::BT_USM_DEVICE_INTERNAL;
    } else {
        tensor_type = TensorType::BT_BUF_INTERNAL;
    }

    // Currently, clDeviceMemAllocINTEL returns memory address allocated to other input blob if the current blob is empty
    // W/A for this issue:
    // Allocate with non-empty shape and then reinterprete with original shape
    auto shape_copy = shape;
    for (auto &i : shape_copy) {
        if (i == 0)
            i = 1;
    }

    return std::make_shared<RemoteTensorImpl>(m_context,
                                              shape_copy,
                                              element_type,
                                              tensor_type,
                                              mem_ptr);
}

std::shared_ptr<ov::ITensor> SyncInferRequest::create_shared_device_tensor(const ov::Shape& shape, ov::element::Type element_type, void* usm_host_mem) const {
    return create_device_tensor(shape, element_type, false, usm_host_mem);
}

TensorWrapper SyncInferRequest::create_or_share_device_tensor(const TensorWrapper& user_tensor_wrapper,
                                                              const std::string& name,
                                                              const ov::PartialShape& port_pshape,
                                                              ov::element::Type element_type,
                                                              bool need_lockable_mem) const {
    auto user_tensor = user_tensor_wrapper.ptr;
    auto tensor_shape = user_tensor->get_shape();
    bool is_dynamic = port_pshape.is_dynamic();
    OPENVINO_ASSERT(std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensor) == nullptr, "[GPU] Unexpected remote tensor");
    auto input_ptr = user_tensor->data();
    const auto alloc_type = m_graph->get_engine().detect_usm_allocation_type(input_ptr);
    const auto is_usm_host = alloc_type == cldnn::allocation_type::usm_host;
    bool can_share = is_usm_host && !is_convert_required(user_tensor->get_element_type(), element_type);

    if (can_share) {
        // For USM case we create host blob using custom USM host allocator
        // and then create shared device blob on top of this buffer
        return { create_shared_device_tensor(tensor_shape, element_type, input_ptr), user_tensor_wrapper.owner };
    }

    auto actual_memory_shape = tensor_shape;
    if (is_dynamic) {
        auto& shape_predictor = m_graph->get_network()->get_shape_predictor();
        actual_memory_shape = predict_shape(name, tensor_shape, element_type, shape_predictor);
    }

    return { create_device_tensor(actual_memory_shape, element_type, need_lockable_mem), TensorOwner::PLUGIN };
}

cldnn::event::ptr SyncInferRequest::copy_output_data(cldnn::memory::ptr src, const ov::ITensor& dst) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::copy_output_data");
    OPENVINO_ASSERT(src->count() <= dst.get_size(),
                    "[GPU] Unexpected elements count of dst tensor: ",
                    "expected at least ", src->count(), ", but ",
                    "only ", dst.get_size(), " got");

    const auto& layout = src->get_layout();
    auto& stream = m_graph->get_network()->get_stream();

    if (is_convert_required(cldnn::data_type_to_element_type(layout.data_type), dst.get_element_type())) {
        convert_and_copy(src, &dst, stream);
        return nullptr;
    } else {
        return src->copy_to(stream, dst.data(), false);
    }
}

void SyncInferRequest::allocate_input(const ov::Output<const ov::Node>& port, const std::string& name) {
    const auto& shape = port.get_partial_shape();
    auto element_type = port.get_element_type();

    m_user_inputs[name] = { create_host_tensor(shape, element_type), TensorOwner::PLUGIN };
    ov::ISyncInferRequest::set_tensor(port, m_user_inputs.at(name).ptr);
}

void SyncInferRequest::allocate_output(const ov::Output<const ov::Node>& port, const std::string& name) {
    const auto& shape = port.get_partial_shape();
    auto element_type = port.get_element_type();

    m_user_outputs[name] = { create_host_tensor(shape, element_type), TensorOwner::PLUGIN };
    ov::ISyncInferRequest::set_tensor(port, m_user_outputs.at(name).ptr);
}

void SyncInferRequest::allocate_inputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::allocate_inputs");

    for (const auto& it : m_input_ports_map) {
        const auto& name = it.first;
        const auto& port = it.second;
        GPU_DEBUG_LOG << "[init " << name << " input blob]" << std::endl;

        bool is_nv12_input = false;
        if (port.get_rt_info().count(ov::preprocess::TensorInfoMemoryType::get_type_info_static())) {
            std::string mem_type = port.get_rt_info().at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                                                     .as<ov::preprocess::TensorInfoMemoryType>().value;
            if (mem_type.find(ov::intel_gpu::memory_type::surface) != std::string::npos) {
                is_nv12_input = true;
            }
        }

        if (!is_nv12_input) {
            allocate_input(port, name);
        }
    }
}

void SyncInferRequest::allocate_outputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::allocate_outputs");

    // allocate outputs
    for (const auto& it : m_output_ports_map) {
        const auto& name = it.first;
        const auto& port = it.second;
        GPU_DEBUG_LOG << "[init " << name << " output blob]" << std::endl;

        allocate_output(port, name);
    }
}

void SyncInferRequest::allocate_states() {
    m_graph->get_network()->allocate_variables_memories();
}

std::vector<cldnn::event::ptr> SyncInferRequest::prepare_batched_input(const std::string& name,
                                                                       const ov::Output<const ov::Node>& port,
                                                                       const std::vector<ov::SoPtr<ov::ITensor>>& user_tensors) {
    std::vector<cldnn::event::ptr> ret_events;
    bool is_host = all_host_tensors(user_tensors);
    bool is_remote_buffer = all_remote_buffers(user_tensors);
    // Host buffers are merged to single tensor
    if (is_host || is_remote_buffer) {
        auto tmp_shape = user_tensors.at(0)->get_shape();
        auto tmp_et = user_tensors.at(0)->get_element_type();
        tmp_shape[0] = user_tensors.size();
        std::shared_ptr<ov::ITensor> merged_tensor = nullptr;
        if (is_host) {
            merged_tensor = m_context->create_host_tensor(tmp_et, tmp_shape)._ptr;
            auto ptr = static_cast<uint8_t*>(merged_tensor->data());
            ov::parallel_for(user_tensors.size(), [&](size_t i) {
                const auto& tensor = user_tensors.at(i);
                std::memcpy(ptr + i * tensor->get_byte_size(), static_cast<uint8_t*>(tensor->data()), tensor->get_byte_size());
            });
        } else {
            const auto& stream = m_graph->get_network()->get_stream();
            merged_tensor = m_context->create_tensor(tmp_et, tmp_shape, {})._ptr;
            auto merged_memory = std::dynamic_pointer_cast<RemoteTensorImpl>(merged_tensor)->get_memory();
            cldnn::mem_lock<uint8_t> dst_lock(merged_memory, stream);
            for (size_t i = 0; i < user_tensors.size(); i++) {
                auto input_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensors[i]._ptr);
                cldnn::mem_lock<uint8_t> src_lock(input_tensor->get_memory(), stream);
                std::memcpy(dst_lock.data() + i * input_tensor->get_byte_size(), src_lock.data(), input_tensor->get_byte_size());
            }
        }

        auto events = prepare_input(name, port, {merged_tensor, TensorOwner::PLUGIN});
        std::move(events.begin(), events.end(), std::back_inserter(ret_events));
    } else {
        for (size_t i = 0; i < user_tensors.size(); i++) {
            auto new_name = name + "_" + std::to_string(i);
            auto events = prepare_input(new_name, port, {user_tensors[i]._ptr, TensorOwner::USER});
            std::move(events.begin(), events.end(), std::back_inserter(ret_events));
        }
    }

    return ret_events;
}

std::vector<cldnn::event::ptr> SyncInferRequest::prepare_input(const std::string& name,
                                                               const ov::Output<const ov::Node>& port,
                                                               const TensorWrapper& user_tensor_wrapper) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::prepare_input");
    auto pshape = port.get_partial_shape();
    auto is_dynamic = pshape.is_dynamic();
    auto user_tensor = user_tensor_wrapper.ptr;
    auto element_type = user_tensor->get_element_type();
    auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensor);
    bool is_remote = remote_ptr != nullptr;

    auto network = m_graph->get_network();
    auto& engine = m_graph->get_engine();
    auto& stream = network->get_stream();

    OPENVINO_ASSERT(pshape.compatible(ov::PartialShape(user_tensor->get_shape())) || is_batched_input(port),
                    "[GPU] The input tensor size is not equal to model port shape, can't handle input tensor with name: ",
                    name,
                    ", because model input (shape=",
                    pshape,
                    ") and tensor (shape=",
                    user_tensor->get_shape(),
                    ") are incompatible");

    if (is_remote) {
        m_plugin_inputs[name] = user_tensor_wrapper;
    }

    auto device_tensor_et = convert_to_supported_device_type(element_type);
    bool convert_needed = is_convert_required(element_type, device_tensor_et);
    bool update_device_tensor = m_plugin_inputs.count(name) == 0 || (m_plugin_inputs[name].owner == TensorOwner::USER && !is_remote);

    if (update_device_tensor) {
        // If device input hasn't been created, then try to use user memory if it's usm_host, or allocate new device buffer
        m_plugin_inputs[name] = create_or_share_device_tensor(user_tensor_wrapper, name, pshape, device_tensor_et, convert_needed);
    } else if (!is_remote) {
        // Device memory has been created on previous iterations. Try to reuse whenever it's possible
        auto device_tensor_wrapper = m_plugin_inputs.at(name);
        auto device_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(device_tensor_wrapper.ptr);
        if (is_dynamic) {
            if (device_tensor->get_original_memory()->size() < user_tensor->get_byte_size()) {
                auto& shape_predictor = network->get_shape_predictor();
                auto actual_shape = predict_shape(name, user_tensor->get_shape(), device_tensor_et, shape_predictor);
                auto new_tensor = create_device_tensor(actual_shape, device_tensor_et, false);
                new_tensor->set_shape(user_tensor->get_shape());
                m_plugin_inputs[name] = { new_tensor, TensorOwner::PLUGIN };
            }
        }
    }

    auto device_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(m_plugin_inputs.at(name).ptr);
    if (is_dynamic) {
        OPENVINO_ASSERT(device_tensor->get_original_memory()->size() >= user_tensor->get_size(),
                        "[GPU] Size of input device tensor (=",
                        device_tensor->get_original_memory()->size(),
                        ") is expected to be greater or equal to user tensor (=",
                        user_tensor->get_size(),
                        ") in dynamic case for ", name);
        // tensor reshape below is expected to work w/o reallocation
        device_tensor->set_shape(user_tensor->get_shape());
    } else {
        OPENVINO_ASSERT(device_tensor->get_size() == user_tensor->get_size(),
                        "[GPU] Size of user tensor (=",
                        user_tensor->get_size(),
                        ") and device tensor (=",
                        device_tensor->get_size(),
                        ") don't match for ", name,
                        ". Those are expected to be equal in case of static shape of the port");
    }

    auto memory = device_tensor->get_memory();
    // WA to extend shape to ranks expected by legacy shape infer. Remove after full migration to new shape infer
    if (!m_graph->get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        auto new_layout = memory->get_layout();
        new_layout.set_partial_shape(m_graph->get_input_layouts().at(name).get_shape());
        memory = engine.reinterpret_buffer(*memory, new_layout);
    }

    cldnn::event::ptr ret_event = nullptr;
    if (!is_remote) {
        if (device_tensor->get_element_type() != user_tensor->get_element_type()) {
            convert_and_copy(user_tensor.get(), device_tensor.get(), stream);
        } else {
            auto src_ptr = static_cast<uint8_t*>(user_tensor->data());
            if (!same_host_mem(memory, src_ptr)) {
                ret_event = memory->copy_from(stream, src_ptr, false);
            }
        }
    }

    const cldnn::primitive_id internal_name = "parameter:" + name;
    network->set_input_data(internal_name, memory);

    if (ret_event)
        return { ret_event };
    else
        return {};
}

std::vector<cldnn::event::ptr> SyncInferRequest::prepare_output(const std::string& name,
                                                                const ov::Output<const ov::Node>& port,
                                                                const TensorWrapper& user_tensor_wrapper) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "SyncInferRequest::prepare_output");
    auto pshape = port.get_partial_shape();
    auto is_dynamic = pshape.is_dynamic();
    auto element_type = port.get_element_type();
    auto user_tensor = user_tensor_wrapper.ptr;
    auto remote_ptr = std::dynamic_pointer_cast<RemoteTensorImpl>(user_tensor);
    bool is_remote = remote_ptr != nullptr;

    if (user_tensor->get_size() > 0) {
        OPENVINO_ASSERT(pshape.compatible(ov::PartialShape(user_tensor->get_shape())),
                        "[GPU] The output tensor size is not equal to model port shape, can't handle output tensor with name: ",
                        name,
                        ", because model output (shape=",
                        pshape,
                        ") and tensor (shape=",
                        user_tensor->get_shape(),
                        ") are incompatible");
    }

    auto network = m_graph->get_network();
    auto device_tensor_et = convert_to_supported_device_type(element_type);
    bool convert_needed = is_convert_required(device_tensor_et, element_type);
    cldnn::primitive_id internal_name = m_output_names_map.at(name);
    if (is_remote && !convert_needed) {
        m_plugin_outputs[name] = user_tensor_wrapper;
    }

    if (!is_dynamic) {
        auto is_cpu_impl = network->is_cpu_impl(internal_name);
        bool has_device_buffer = m_plugin_outputs.count(name) > 0;
        bool update_device_tensor = !has_device_buffer ||
                                    (m_plugin_outputs[name].owner == TensorOwner::USER && !is_remote);
        if (update_device_tensor) {
            m_plugin_outputs[name] = create_or_share_device_tensor(user_tensor_wrapper, name, pshape, device_tensor_et, is_cpu_impl || convert_needed);
        }
    }

    // Missing output in _plugin_outputs means that the network is dynamic and outputs couldn't be pre-allocated
    if (m_plugin_outputs.find(name) == m_plugin_outputs.end())
        return {};

    auto output_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(m_plugin_outputs.at(name).ptr);
    auto output_memory = output_tensor->get_memory();
    return network->set_output_memory(internal_name, output_memory);
}

void SyncInferRequest::init_mappings(bool is_legacy_api) {
    for (const auto& in : get_inputs()) {
        auto port_name = get_port_name(in, is_legacy_api);
        m_input_ports_map[port_name] = in;
    }
    for (const auto& out : get_outputs()) {
        auto port_name = get_port_name(out, is_legacy_api);
        m_output_ports_map[port_name] = out;
        m_output_names_map[port_name] = m_graph->out_name_to_internal(port_name);
    }
}

bool SyncInferRequest::is_batched_input(const ov::Output<const ov::Node>& port) const {
    return m_batched_tensors.count(port.get_tensor_ptr()) > 0;
}

}  // namespace intel_gpu
}  // namespace ov
