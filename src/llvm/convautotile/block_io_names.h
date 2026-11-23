/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Names of the per-port block-I/O statics a kernel TU gets (aie_block_io.h):
 *   static const aiehlc_blk_desc aiehlc_desc_<X>;  static aiehlc_blk_cursor aiehlc_cur_<X>;
 * emitted by conv_desugar.cpp (kernelTUDecls) / trace_routing.cpp (routedTUDecls)
 * and referenced by the aie::load_buffer / store_buffer rewrite in
 * tilinglinalg_pipeline.cpp. All three MUST use these helpers.
 *
 * Every kernel TU of a program sees the statics of ALL its kernels, so they are
 * file-scope names keyed by port. A template __global__ with two or more
 * instances (kernel_template.cpp: instance i is named mangle(kernel, args) =
 * "<kernel>__<arg>_<arg>...") has the same port names in every instance with
 * different descriptors: there <X> = <instance>_<port>. A plain kernel and a
 * template with one instance (named like the template) keep <X> = <port>, so
 * their output is unchanged.
 ******************************************************************************/
#pragma once

#include <string>

namespace convautotile {

// kernel_template.cpp mangle(): a multi-instance template's instances carry "__".
inline bool isMultiInstanceKernelName(const std::string &kernel) { return kernel.find("__") != std::string::npos; }

inline std::string blockIoKey(const std::string &kernel, const std::string &port) {
    return isMultiInstanceKernelName(kernel) ? kernel + "_" + port : port;
}
inline std::string blockDescVar(const std::string &kernel, const std::string &port) {
    return "aiehlc_desc_" + blockIoKey(kernel, port);
}
inline std::string blockCurVar(const std::string &kernel, const std::string &port) {
    return "aiehlc_cur_" + blockIoKey(kernel, port);
}

} // namespace convautotile
