/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef _TYPE_dfscheblueprint_
#define _TYPE_dfscheblueprint_
include "dfscheblueprintattr.td"

//=== dfscheblueprint Types ===//

// Blueprint type - represents a schedule blueprint handle
def BlueprintType : dfscheblueprinttype<"Blueprint"> {
  let mnemonic = "blueprint";
  let summary = "Schedule blueprint handle";
}

// Config type - represents a schedule configuration
def ConfigType : dfscheblueprinttype<"Config"> {
  let mnemonic = "config";
  let summary = "Schedule configuration handle";
}

// Manifest type - represents a transfer manifest
def ManifestType : dfscheblueprinttype<"Manifest"> {
  let mnemonic = "manifest";
  let summary = "Transfer manifest handle";
}

// Data handle type
def DataHandleType : dfscheblueprinttype<"DataHandle"> {
  let mnemonic = "data_handle";
  let summary = "Handle to declared data";
}

// View handle type
def ViewHandleType : dfscheblueprinttype<"ViewHandle"> {
  let mnemonic = "view_handle";
  let summary = "Handle to a data view/partition";
}

#endif //_TYPE_dfscheblueprint_

