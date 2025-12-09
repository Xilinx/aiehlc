/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include "dfscheblueprintmanager.h"
#include "../../routing/routingmanager.h"
#include <iostream>
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_OP_CLASSES
#define GET_OP_DEFS
#include "dfscheblueprintdialect.cc.inc"
#include "dfscheblueprintenums.cc.inc"
#include "dfscheblueprintattr.cc.inc"
#include "dfscheblueprinttype.cc.inc"

#include "dfscheblueprintop.cc.inc"
#undef GET_OP_DEFS

#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

//===----------------------------------------------------------------------===//
// Custom Printers and Parsers
//===----------------------------------------------------------------------===//

// ConfigOp printer
void dfscheblueprint::ConfigOp::print(OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/true, /*printBlockTerminators=*/false);
}

// ConfigOp parser
ParseResult dfscheblueprint::ConfigOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return failure();
    
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, {}, {}))
        return failure();
    
    // Ensure the region has a block
    if (body->empty())
        body->emplaceBlock();
    
    return success();
}

// TransferManifestOp printer
void dfscheblueprint::TransferManifestOp::print(OpAsmPrinter &printer) {
    printer << " @" << getSymName() << " {";
    printer.printNewline();
    printer << "    payload_slice = " << getPayloadSlice() << ",";
    printer.printNewline();
    printer << "    packet_id = " << getPacketId() << ",";
    printer.printNewline();
    printer << "    source = " << getSource() << ",";
    printer.printNewline();
    printer << "    destinations = " << getDestinations();
    printer.printNewline();
    printer << "  }";
}

// TransferManifestOp parser
ParseResult dfscheblueprint::TransferManifestOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return failure();
    
    if (parser.parseLBrace())
        return failure();
    
    dfscheblueprint::SliceAttr payloadSlice;
    IntegerAttr packetId;
    dfscheblueprint::EndpointAttr source;
    ArrayAttr destinations;
    
    while (true) {
        StringRef attrName;
        if (parser.parseOptionalKeyword(&attrName)) {
            break;
        }
        
        if (parser.parseEqual())
            return failure();
        
        if (attrName == "payload_slice") {
            if (parser.parseAttribute(payloadSlice))
                return failure();
        } else if (attrName == "packet_id") {
            if (parser.parseAttribute(packetId))
                return failure();
        } else if (attrName == "source") {
            if (parser.parseAttribute(source))
                return failure();
        } else if (attrName == "destinations") {
            if (parser.parseAttribute(destinations))
                return failure();
        }
        
        parser.parseOptionalComma();
    }
    
    if (parser.parseRBrace())
        return failure();
    
    result.addAttribute("payload_slice", payloadSlice);
    result.addAttribute("packet_id", packetId);
    result.addAttribute("source", source);
    result.addAttribute("destinations", destinations);
    
    return success();
}

// DataSliceOp printer
// Format: %result = dfscheblueprint.data_slice @sym_name wrap %tensor_slice : tensor_type
void dfscheblueprint::DataSliceOp::print(OpAsmPrinter &printer) {
    printer << " @" << getSymName() << " wrap ";
    printer << getTensorSlice() << " : " << getTensorSlice().getType();
    printer.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{"sym_name"});
}

// DataSliceOp parser
// Format: %result = dfscheblueprint.data_slice @sym_name wrap %tensor_slice : tensor_type
ParseResult dfscheblueprint::DataSliceOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return failure();
    
    if (parser.parseKeyword("wrap"))
        return failure();
    
    OpAsmParser::UnresolvedOperand tensorOperand;
    Type tensorType;
    if (parser.parseOperand(tensorOperand) || parser.parseColonType(tensorType))
        return failure();
    
    if (parser.resolveOperand(tensorOperand, tensorType, result.operands))
        return failure();
    
    // Set the result type to match the input tensor type
    result.addTypes(tensorType);
    
    parser.parseOptionalAttrDict(result.attributes);
    
    return success();
}

// FlowConfigOp printer
void dfscheblueprint::FlowConfigOp::print(OpAsmPrinter &printer) {
    printer << " @" << getSymName() << " {";
    printer.increaseIndent();
    printer.printNewline();
    printer << "target = " << getTarget() << ",";
    printer.printNewline();
    printer << "view = " << getView() << " : " << getView().getType() << ",";
    printer.printNewline();
    printer << "distribution = \"" << getDistribution() << "\",";
    printer.printNewline();
    printer << "dma = " << getDma();
    if (getSliceSymbols()) {
        printer.printNewline();
        printer << ",slice_symbols = " << getSliceSymbols();
    }
    if (getType()) {
        printer.printNewline();
        printer << ",type = \"" << getType() << "\"";
    }
    printer.decreaseIndent();
    printer.printNewline();
    printer << "}";
    printer.printOptionalAttrDict(getOperation()->getAttrs(), 
        /*elidedAttrs=*/{"sym_name", "target", "view", "distribution", "dma", "slice_symbols", "type"});
}

// FlowConfigOp parser
ParseResult dfscheblueprint::FlowConfigOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return failure();
    
    if (parser.parseLBrace())
        return failure();
    
    SymbolRefAttr target;
    OpAsmParser::UnresolvedOperand viewOperand;
    Type viewType;
    StringAttr distribution;
    dfscheblueprint::DMAAttr dma;
    ArrayAttr sliceSymbols;
    StringAttr type;
    
    while (true) {
        OptionalParseResult res = parser.parseOptionalRBrace();
        if (res.has_value()) {
            if (failed(res.value())) return failure();
            break;
        }

        StringRef attrName;
        if (parser.parseKeyword(&attrName) || parser.parseEqual())
            return failure();
        
        if (attrName == "target") {
            if (parser.parseAttribute(target, "target", result.attributes)) return failure();
        } else if (attrName == "view") {
            if (parser.parseOperand(viewOperand) || parser.parseColonType(viewType))
                return failure();
        } else if (attrName == "distribution") {
            if (parser.parseAttribute(distribution, "distribution", result.attributes)) return failure();
        } else if (attrName == "dma") {
            if (parser.parseAttribute(dma, "dma", result.attributes)) return failure();
        } else if (attrName == "slice_symbols") {
            if (parser.parseAttribute(sliceSymbols, "slice_symbols", result.attributes)) return failure();
        } else if (attrName == "type") {
            if (parser.parseAttribute(type, "type", result.attributes)) return failure();
        } else {
             return parser.emitError(parser.getCurrentLocation(), "unknown attribute: ") << attrName;
        }
        
        parser.parseOptionalComma();
    }
    
    if (parser.resolveOperand(viewOperand, viewType, result.operands))
        return failure();
        
    parser.parseOptionalAttrDict(result.attributes);
    
    return success();
}

// FlowTransferOp printer
void dfscheblueprint::FlowTransferOp::print(OpAsmPrinter &printer) {
    printer << " @" << getSymName() << " {";
    printer.increaseIndent();
    printer.printNewline();
    printer << "type = \"" << getType() << "\",";
    printer.printNewline();
    printer << "from = " << getFrom() << ",";
    printer.printNewline();
    printer << "to = " << getTo();
    if (getOrdering()) {
        printer.printNewline();
        printer << ",ordering = \"" << getOrdering() << "\"";
    }
    printer.printNewline();
    printer << ",base_packet_id = " << getBasePacketId();
    printer.decreaseIndent();
    printer.printNewline();
    printer << "}";
    printer.printOptionalAttrDict(getOperation()->getAttrs(), 
        /*elidedAttrs=*/{"sym_name", "type", "from", "to", "ordering", "base_packet_id"});
}

// FlowTransferOp parser
ParseResult dfscheblueprint::FlowTransferOp::parse(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return failure();
    
    if (parser.parseLBrace())
        return failure();
    
    StringAttr type;
    SymbolRefAttr from;
    SymbolRefAttr to;
    StringAttr ordering;
    IntegerAttr basePacketId;
    
    while (true) {
        OptionalParseResult res = parser.parseOptionalRBrace();
        if (res.has_value()) {
            if (failed(res.value())) return failure();
            break;
        }

        StringRef attrName;
        if (parser.parseKeyword(&attrName) || parser.parseEqual())
            return failure();
        
        if (attrName == "type") {
            if (parser.parseAttribute(type, "type", result.attributes)) return failure();
        } else if (attrName == "from") {
            if (parser.parseAttribute(from, "from", result.attributes)) return failure();
        } else if (attrName == "to") {
            if (parser.parseAttribute(to, "to", result.attributes)) return failure();
        } else if (attrName == "ordering") {
            if (parser.parseAttribute(ordering, "ordering", result.attributes)) return failure();
        } else if (attrName == "base_packet_id") {
            if (parser.parseAttribute(basePacketId, "base_packet_id", result.attributes)) return failure();
        } else {
             return parser.emitError(parser.getCurrentLocation(), "unknown attribute: ") << attrName;
        }
        
        parser.parseOptionalComma();
    }
    
    parser.parseOptionalAttrDict(result.attributes);
    
    return success();
}

//===----------------------------------------------------------------------===//
// Dialect Initialization
//===----------------------------------------------------------------------===//

void dfscheblueprintdialect::initialize()  { 
    addOperations<
    #define GET_OP_LIST
    #include "dfscheblueprintop.cc.inc"
        >();
    // Add Attributes
     addAttributes<
    #define GET_ATTRDEF_LIST
    #include "dfscheblueprintattr.cc.inc"
    >();

    // Add Types
    addTypes<
    #define GET_TYPEDEF_LIST
    #include "dfscheblueprinttype.cc.inc"
    >();
}

ModuleOp dfscheblueprintmanager::ops_test(MLIRContext* ctx) {
    loaddialect(ctx);
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    
    // Set insertion point to the module body
    builder.setInsertionPointToEnd(m.getBody());
    
    createBlueprintExample(builder, ctx);
    
    // Print with custom format (not generic)
    OpPrintingFlags flags;
    flags.enableDebugInfo(false);
    flags.printGenericOpForm(false);  // Disable generic form
    m.print(llvm::outs(), flags);
    llvm::outs() << "\n";
    
    return m;
}

void dfscheblueprintmanager::loaddialect(MLIRContext* ctx) {
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<dfscheblueprint::dfscheblueprintdialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<routing::routingdialect>();
}

/**
 * Creates an example schedule blueprint as shown in the user's specification:
 * 
 * dfscheblueprint.config @tiling_and_broadcast_blueprint {
 *   dfscheblueprint.transfer_manifest @broadcast_upper_half {
 *     payload_slice = #dfscheblueprint.slice<...>,
 *     packet_id = 10 : i32,
 *     source = #dfscheblueprint.endpoint<tile=(2,0), direction="MM2S", channel=0>,
 *     destinations = [
 *       #dfscheblueprint.endpoint<tile=(2,2), direction="S2MM", channel=0>,
 *       #dfscheblueprint.endpoint<tile=(2,3), direction="S2MM", channel=0>
 *     ]
 *   }
 *   dfscheblueprint.transfer_manifest @broadcast_lower_half { ... }
 * }
 */
void dfscheblueprintmanager::createBlueprintExample(OpBuilder& builder, MLIRContext* ctx) {
    auto location = builder.getUnknownLoc();
    
    // Create the top-level dfscheblueprint.config operation
    auto configOp = builder.create<dfscheblueprint::ConfigOp>(
        location,
        builder.getStringAttr("broadcast_gather_blueprint")
    );
    
    // Set insertion point inside the config body
    Block *configBody = new Block();
    configOp.getBody().push_back(configBody);
    builder.setInsertionPointToStart(configBody);

    // ============================================================
    // 1. Physical Resources
    // ============================================================
    
    // Shim Gateway: tiles = [(0, 2)]
    auto shimTiles = builder.getArrayAttr({
        builder.getArrayAttr({builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(2)})
    });
    builder.create<dfscheblueprint::TileGroupOp>(
        location,
        builder.getStringAttr("shim_gateway"),
        shimTiles
    );

    // Compute Row: tiles = [(2, 0), (2, 1), (2, 2), (2, 3)]
    auto computeTiles = builder.getArrayAttr({
        builder.getArrayAttr({builder.getI64IntegerAttr(2), builder.getI64IntegerAttr(0)}),
        builder.getArrayAttr({builder.getI64IntegerAttr(2), builder.getI64IntegerAttr(1)}),
        builder.getArrayAttr({builder.getI64IntegerAttr(2), builder.getI64IntegerAttr(2)}),
        builder.getArrayAttr({builder.getI64IntegerAttr(2), builder.getI64IntegerAttr(3)})
    });
    builder.create<dfscheblueprint::TileGroupOp>(
        location,
        builder.getStringAttr("compute_row"),
        computeTiles
    );

    // ============================================================
    // 2. Logical Data View (using Tensor types)
    // ============================================================
    
    // Create Root Tensor (tensor<1024x1024xf32>) - logical declaration with init data
    SmallVector<int64_t> tensorShape = {1024, 1024};
    auto tensorType = RankedTensorType::get(tensorShape, builder.getF32Type());
    
    // Create arith.constant dense with init data starting from 1
    int64_t totalElements = 1024 * 1024;
    SmallVector<float> initValues;
    for (int64_t i = 1; i <= totalElements; ++i) {
        initValues.push_back(static_cast<float>(i));
    }
    auto denseAttr = DenseElementsAttr::get(tensorType, ArrayRef<float>(initValues));
    auto initConstant = builder.create<arith::ConstantOp>(location, tensorType, denseAttr);
    
    // Create DeclareDataOp with the init tensor
    auto declareDataOp = builder.create<dfscheblueprint::DeclareDataOp>(
        location,
        tensorType,  // result type
        initConstant.getResult()  // init_tensor
    );
    auto rootTensor = declareDataOp.getResult();

    // Create tensor.extract_slice (Partition 0: 256x1024)
    // Offset=[0, 0], Size=[256, 1024], Stride=[1, 1] relative to source
    SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0), builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(256), builder.getIndexAttr(1024)};
    SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1), builder.getIndexAttr(1)};
    
    auto extractSliceOp = builder.create<tensor::ExtractSliceOp>(
        location,
        rootTensor,
        offsets,
        sizes,
        strides
    );
    auto viewSplit = extractSliceOp.getResult();

    // ============================================================
    // 3. Binding & Channel Config
    // ============================================================

    // First: Create all 4 tensor.extract_slice ops as a group
    // Each slice is 64x1024 from the 256x1024 view (viewSplit)
    llvm::SmallVector<mlir::Value, 4> tensorSlices;
    for (int i = 0; i < 4; ++i) {
        // Offset: [i * 64, 0], Size: [64, 1024], Stride: [1, 1]
        SmallVector<OpFoldResult> sliceOffsets = {
            builder.getIndexAttr(i * 64),
            builder.getIndexAttr(0)
        };
        SmallVector<OpFoldResult> sliceSizes = {
            builder.getIndexAttr(64),
            builder.getIndexAttr(1024)
        };
        SmallVector<OpFoldResult> sliceStrides = {
            builder.getIndexAttr(1),
            builder.getIndexAttr(1)
        };
        
        auto tensorSlice = builder.create<tensor::ExtractSliceOp>(
            location,
            viewSplit,
            sliceOffsets,
            sliceSizes,
            sliceStrides
        );
        tensorSlices.push_back(tensorSlice.getResult());
    }
    
    // Second: Create all 4 dfscheblueprint.data_slice ops as a group to wrap the tensor slices
    llvm::SmallVector<mlir::Attribute, 4> outSliceSymbols;
    llvm::SmallVector<mlir::Value, 4> dataSliceResults;
    for (int i = 0; i < 4; ++i) {
        std::string sliceName = "out_slice_" + std::to_string(i);
        auto dataSliceOp = builder.create<dfscheblueprint::DataSliceOp>(
            location,
            tensorSlices[i].getType(), // Result type matches input tensor type
            builder.getStringAttr(sliceName),
            tensorSlices[i]
        );
        
        outSliceSymbols.push_back(mlir::SymbolRefAttr::get(ctx, sliceName));
        dataSliceResults.push_back(dataSliceOp.getResult());
    }

    // Bind Shim TX
    llvm::SmallVector<int64_t, 1> shimTxCh = {0};
    auto shimTxDMA = dfscheblueprint::DMAAttr::get(ctx, shimTxCh, dfscheblueprint::bp_direction::MM2S);
    builder.create<dfscheblueprint::FlowConfigOp>(
        location,
        builder.getStringAttr("flow_shim_tx"),
        mlir::SymbolRefAttr::get(ctx, "shim_gateway"),
        viewSplit,
        builder.getStringAttr("root"),
        shimTxDMA,
        nullptr, // slice_symbols
        builder.getStringAttr("shim") // type
    );

    // Bind Cores Input (S2MM - Receive)
    llvm::SmallVector<int64_t, 1> coresInCh = {0};
    auto coresInDMA = dfscheblueprint::DMAAttr::get(ctx, coresInCh, dfscheblueprint::bp_direction::S2MM);
    builder.create<dfscheblueprint::FlowConfigOp>(
        location,
        builder.getStringAttr("flow_cores_in"),
        mlir::SymbolRefAttr::get(ctx, "compute_row"),
        viewSplit,
        builder.getStringAttr("linear"),
        coresInDMA,
        nullptr, // slice_symbols
        builder.getStringAttr("core") // type
    );

    // Bind Cores Output (MM2S - Send)
    llvm::SmallVector<int64_t, 1> coresOutCh = {1};
    auto coresOutDMA = dfscheblueprint::DMAAttr::get(ctx, coresOutCh, dfscheblueprint::bp_direction::MM2S);
    builder.create<dfscheblueprint::FlowConfigOp>(
        location,
        builder.getStringAttr("flow_cores_out"),
        mlir::SymbolRefAttr::get(ctx, "compute_row"),
        viewSplit,
        builder.getStringAttr("linear"),
        coresOutDMA,
        builder.getArrayAttr(outSliceSymbols), // slice_symbols
        builder.getStringAttr("core") // type
    );

    // Bind Shim RX
    llvm::SmallVector<int64_t, 1> shimRxCh = {0};
    auto shimRxDMA = dfscheblueprint::DMAAttr::get(ctx, shimRxCh, dfscheblueprint::bp_direction::S2MM);
    builder.create<dfscheblueprint::FlowConfigOp>(
        location,
        builder.getStringAttr("flow_shim_rx"),
        mlir::SymbolRefAttr::get(ctx, "shim_gateway"),
        viewSplit,
        builder.getStringAttr("root"),
        shimRxDMA,
        nullptr, // slice_symbols
        builder.getStringAttr("shim") // type
    );

    // ============================================================
    // 4. Collective Transfers
    // ============================================================

    // Input Scatter
    builder.create<dfscheblueprint::FlowTransferOp>(
        location,
        builder.getStringAttr("input_scatter"),
        builder.getStringAttr("one_to_many"),
        mlir::SymbolRefAttr::get(ctx, "flow_shim_tx"),
        mlir::SymbolRefAttr::get(ctx, "flow_cores_in"),
        nullptr, // ordering
        builder.getI32IntegerAttr(10)
    );

    // Output Gather
    builder.create<dfscheblueprint::FlowTransferOp>(
        location,
        builder.getStringAttr("output_gather"),
        builder.getStringAttr("many_to_one"),
        mlir::SymbolRefAttr::get(ctx, "flow_cores_out"),
        mlir::SymbolRefAttr::get(ctx, "flow_shim_rx"),
        builder.getStringAttr("sequential"),
        builder.getI32IntegerAttr(20)
    );
}
