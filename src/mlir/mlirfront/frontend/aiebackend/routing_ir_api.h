/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Routing IR API (routing_ir_parse / _edit / _check) -- see routing_ir_api.cpp.
 ******************************************************************************/
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

pybind11::dict routing_ir_parse(const std::string &text);
std::string routing_ir_edit(const std::string &text, const pybind11::list &edits);
pybind11::dict routing_ir_check(const std::string &text, const std::string &aieGen, const std::string &workDir);

// Registers the three functions on the _aiebackend module.
void bindRoutingIrApi(pybind11::module_ &m);
