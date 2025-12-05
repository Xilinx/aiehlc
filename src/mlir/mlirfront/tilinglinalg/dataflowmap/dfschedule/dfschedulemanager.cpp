/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include "dfschedulemanager.h"
#include <iostream>

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_OP_CLASSES
#define GET_OP_DEFS
#include "dfscheduledialect.cc.inc"
#include "dfscheduleattr.cc.inc"
#include "dfscheduletype.cc.inc"
#include "dfscheduleenums.cc.inc"

#include "dfscheduleop.cc.inc"

#undef GET_OP_DEFS

#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

void dfscheduledialect::initialize()  { 
    addOperations<
    #define GET_OP_LIST
    #include "dfscheduleop.cc.inc"
        >();
    addAttributes<
    #define GET_ATTRDEF_LIST
    #include "dfscheduleattr.cc.inc"
    >();

    addTypes<
    #define GET_TYPEDEF_LIST
    #include "dfscheduletype.cc.inc"
    >();
}

// ===== Custom Print/Parse for Block Operations =====

// HostBlockOp - Top-level host block
::mlir::ParseResult dfschedule::HostBlockOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, {}, {}))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    return mlir::success();
}

void dfschedule::HostBlockOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

// DSKernelComputeOp - Compute kernel logic block
::mlir::ParseResult dfschedule::DSKernelComputeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    // Parse arguments like (%buf: memref<...>, %loop_count: index)
    llvm::SmallVector<mlir::OpAsmParser::Argument, 4> args;
    if (parser.parseArgumentList(args, mlir::OpAsmParser::Delimiter::Paren, true))
        return mlir::failure();
    
    // Parse optional -> compute return
    if (succeeded(parser.parseOptionalArrow())) {
        // Parse the return type
        mlir::Type returnType;
        if (parser.parseType(returnType))
            return mlir::failure();
    }
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, args))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    // Add result type
    result.addTypes(dfschedule::ComputeType::get(parser.getContext()));
    
    return mlir::success();
}

void dfschedule::DSKernelComputeOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    
    // Print block arguments inline (not as entry block args)
    auto &block = getBody().front();
    if (!block.getArguments().empty()) {
        printer << "(";
        llvm::interleaveComma(block.getArguments(), printer, [&](mlir::BlockArgument arg) {
            printer.printRegionArgument(arg, {}, /*omitType=*/false);
        });
        printer << ")";
    }
    
    printer << " -> " << getCompute().getType();
    printer << " ";
    // Print region without entry block args since we printed them inline
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

// DSKernelReceiverOp - Receiver kernel with ping-pong buffering
::mlir::ParseResult dfschedule::DSKernelReceiverOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    // Parse arguments like (%arg0: packet, %computelogic: compute, %loop_count: index)
    llvm::SmallVector<mlir::OpAsmParser::Argument, 4> args;
    if (parser.parseArgumentList(args, mlir::OpAsmParser::Delimiter::Paren, true))
        return mlir::failure();
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, args))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    return mlir::success();
}

void dfschedule::DSKernelReceiverOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    
    // Print block arguments inline
    auto &block = getBody().front();
    if (!block.getArguments().empty()) {
        printer << "(";
        llvm::interleaveComma(block.getArguments(), printer, [&](mlir::BlockArgument arg) {
            printer.printRegionArgument(arg, {}, /*omitType=*/false);
        });
        printer << ")";
    }
    
    printer << " ";
    // Print region without entry block args since we printed them inline
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

// HostConfigOp (legacy)
::mlir::ParseResult dfschedule::HostConfigOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, {}, {}))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    return mlir::success();
}

void dfschedule::HostConfigOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    printer << " ";
    printer.printRegion(getBody(), false, false);
}

// KernelConfigOp
::mlir::ParseResult dfschedule::KernelConfigOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, {}, {}))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    return mlir::success();
}
 
void dfschedule::KernelConfigOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    printer << " ";
    printer.printRegion(getBody(), false, false);
}

// HostScheduleOp
::mlir::ParseResult dfschedule::HostScheduleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, {}, {}))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    return mlir::success();
}

void dfschedule::HostScheduleOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    printer << " ";
    printer.printRegion(getBody(), false, false);
}

// KernelScheduleOp
::mlir::ParseResult dfschedule::KernelScheduleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes))
        return mlir::failure();
    
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, {}, {}))
        return mlir::failure();
    
    if (body->empty())
        body->emplaceBlock();
    
    return mlir::success();
}

void dfschedule::KernelScheduleOp::print(::mlir::OpAsmPrinter &printer) {
    printer << " @" << getSymName();
    printer << " ";
    printer.printRegion(getBody(), false, false);
}

// ConfigDmaBdOp - DMA Buffer Descriptor Configuration
::mlir::ParseResult dfschedule::ConfigDmaBdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::OpAsmParser::UnresolvedOperand bufferOperand, tileOperand, bdIdOperand;
    mlir::Type bufferType, tileType, bdIdType;
    
    // Parse: (%buffer, %tile, %bd_id)
    if (parser.parseLParen() ||
        parser.parseOperand(bufferOperand) ||
        parser.parseComma() ||
        parser.parseOperand(tileOperand) ||
        parser.parseComma() ||
        parser.parseOperand(bdIdOperand) ||
        parser.parseRParen())
        return mlir::failure();
    
    // Parse attributes: { offset = ..., len = ..., enable_packet = ..., packet_id = ..., next_bd = ... }
    if (parser.parseOptionalAttrDict(result.attributes))
        return mlir::failure();
    
    // Parse: : (type($buffer), type($tile), type($bd_id)) -> type($bd_handle)
    if (parser.parseColon() ||
        parser.parseLParen() ||
        parser.parseType(bufferType) ||
        parser.parseComma() ||
        parser.parseType(tileType) ||
        parser.parseComma() ||
        parser.parseType(bdIdType) ||
        parser.parseRParen() ||
        parser.parseArrow())
        return mlir::failure();
    
    mlir::Type bdHandleType;
    if (parser.parseType(bdHandleType))
        return mlir::failure();
    
    // Resolve operands
    if (parser.resolveOperand(bufferOperand, bufferType, result.operands) ||
        parser.resolveOperand(tileOperand, tileType, result.operands) ||
        parser.resolveOperand(bdIdOperand, bdIdType, result.operands))
        return mlir::failure();
    
    // Add result type
    result.addTypes(bdHandleType);
    
    return mlir::success();
}

void dfschedule::ConfigDmaBdOp::print(::mlir::OpAsmPrinter &printer) {
    printer << "(";
    printer << getBuffer() << ", " << getTile() << ", " << getBdId();
    printer << ") {";
    
    // Print attributes with proper indentation
    printer.increaseIndent();
    printer.printNewline();
    printer << "offset = " << getOffset() << ",";
    printer.printNewline();
    printer << "len = " << getLen() << ",";
    printer.printNewline();
    printer << "enable_packet = " << (getEnablePacket() ? "true" : "false") << ",";
    printer.printNewline();
    printer << "packet_id = " << getPacketId() << ",";
    printer.printNewline();
    printer << "next_bd = " << getNextBd();
    printer.decreaseIndent();
    printer.printNewline();
    printer << "} ";
    
    // Print types
    printer << ": (";
    printer << getBuffer().getType() << ", " << getTile().getType() << ", " << getBdId().getType();
    printer << ") -> ";
    printer << getBdHandle().getType();
}

// ConfigCreateIoOp - Create IO handle for DMA operations
::mlir::ParseResult dfschedule::ConfigCreateIoOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::OpAsmParser::UnresolvedOperand bdConfigOperand, tileOperand;
    mlir::Type bdConfigType, tileType;
    
    // Parse: (%bd_config, %tile)
    if (parser.parseLParen() ||
        parser.parseOperand(bdConfigOperand) ||
        parser.parseComma() ||
        parser.parseOperand(tileOperand) ||
        parser.parseRParen())
        return mlir::failure();
    
    // Parse attributes: { channel = ..., direction = ..., io_operation = ... }
    if (parser.parseOptionalAttrDict(result.attributes))
        return mlir::failure();
    
    // Parse: : (type($bd_config), type($tile)) -> type($io_handle)
    if (parser.parseColon() ||
        parser.parseLParen() ||
        parser.parseType(bdConfigType) ||
        parser.parseComma() ||
        parser.parseType(tileType) ||
        parser.parseRParen() ||
        parser.parseArrow())
        return mlir::failure();
    
    mlir::Type ioHandleType;
    if (parser.parseType(ioHandleType))
        return mlir::failure();
    
    // Resolve operands
    if (parser.resolveOperand(bdConfigOperand, bdConfigType, result.operands) ||
        parser.resolveOperand(tileOperand, tileType, result.operands))
        return mlir::failure();
    
    // Add result type
    result.addTypes(ioHandleType);
    
    return mlir::success();
}

void dfschedule::ConfigCreateIoOp::print(::mlir::OpAsmPrinter &printer) {
    printer << "(";
    printer << getBdConfig() << ", " << getTile();
    printer << ") {";
    
    // Print attributes with proper indentation
    printer.increaseIndent();
    printer.printNewline();
    printer << "channel = " << getChannel() << ",";
    printer.printNewline();
    printer << "direction = \"" << getDirection() << "\",";
    printer.printNewline();
    printer << "io_operation = \"" << getIoOperation() << "\"";
    printer.decreaseIndent();
    printer.printNewline();
    printer << "} ";
    
    // Print types
    printer << ": (";
    printer << getBdConfig().getType() << ", " << getTile().getType();
    printer << ") -> ";
    printer << getIoHandle().getType();
}

// LoadKernelGroupOp - Load kernel group configuration
::mlir::ParseResult dfschedule::LoadKernelGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> tileOperands;
    llvm::SmallVector<mlir::Type, 4> tileTypes;
    
    // Parse: (%tile0, %tile1, ...)
    if (parser.parseLParen())
        return mlir::failure();
    
    if (parser.parseOptionalRParen()) {
        // Parse operand list
        do {
            mlir::OpAsmParser::UnresolvedOperand operand;
            if (parser.parseOperand(operand))
                return mlir::failure();
            tileOperands.push_back(operand);
        } while (succeeded(parser.parseOptionalComma()));
        
        if (parser.parseRParen())
            return mlir::failure();
    }
    
    // Parse attributes: { callee = [...], ... }
    if (parser.parseOptionalAttrDict(result.attributes))
        return mlir::failure();
    
    // Parse: : (type($tiles)) -> type($kernel_group)
    if (parser.parseColon() ||
        parser.parseLParen())
        return mlir::failure();
    
    if (parser.parseOptionalRParen()) {
        // Parse type list
        do {
            mlir::Type type;
            if (parser.parseType(type))
                return mlir::failure();
            tileTypes.push_back(type);
        } while (succeeded(parser.parseOptionalComma()));
        
        if (parser.parseRParen())
            return mlir::failure();
    }
    
    if (parser.parseArrow())
        return mlir::failure();
    
    mlir::Type kernelGroupType;
    if (parser.parseType(kernelGroupType))
        return mlir::failure();
    
    // Resolve operands
    for (size_t i = 0; i < tileOperands.size(); ++i) {
        mlir::Type type = (i < tileTypes.size()) ? tileTypes[i] : tileTypes.back();
        if (parser.resolveOperand(tileOperands[i], type, result.operands))
            return mlir::failure();
    }
    
    // Add result type
    result.addTypes(kernelGroupType);
    
    return mlir::success();
}

void dfschedule::LoadKernelGroupOp::print(::mlir::OpAsmPrinter &printer) {
    printer << "(";
    llvm::interleaveComma(getTiles(), printer, [&](mlir::Value tile) {
        printer << tile;
    });
    printer << ") {";
    
    // Print attributes with proper indentation
    printer.increaseIndent();
    printer.printNewline();
    printer << "callee = ";
    printer.printAttribute(getCalleeAttr());
    printer << ",";
    printer.printNewline();
    printer << "distributed_compute_kernel_args = ";
    printer.printAttribute(getDistributedComputeKernelArgsAttr());
    printer << ",";
    printer.printNewline();
    printer << "distributed_args = ";
    printer.printAttribute(getDistributedArgsAttr());
    printer.decreaseIndent();
    printer.printNewline();
    printer << "} ";
    
    // Print types
    printer << ": (";
    llvm::interleaveComma(getTiles().getTypes(), printer, [&](mlir::Type type) {
        printer << type;
    });
    printer << ") -> ";
    printer << getKernelGroup().getType();
}

// DSKernelLaunchDmaLoopOp - Launch DMA S2M loop with ping-pong buffers
::mlir::ParseResult dfschedule::DSKernelLaunchDmaLoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    mlir::OpAsmParser::UnresolvedOperand pingOperand, pongOperand;
    mlir::OpAsmParser::UnresolvedOperand pingAcqLockOperand, pingRelLockOperand;
    mlir::OpAsmParser::UnresolvedOperand pongAcqLockOperand, pongRelLockOperand;
    mlir::OpAsmParser::UnresolvedOperand channelOperand;
    mlir::Type pingType, pongType;
    
    // Parse: (%ping, %pong)
    if (parser.parseLParen() ||
        parser.parseOperand(pingOperand) ||
        parser.parseComma() ||
        parser.parseOperand(pongOperand) ||
        parser.parseRParen())
        return mlir::failure();
    
    // Parse: { locks_ping = {acq = %lock, rel = %lock}, locks_pong = {acq = %lock, rel = %lock}, channelid = %channel }
    if (parser.parseLBrace() ||
        parser.parseKeyword("locks_ping") ||
        parser.parseEqual() ||
        parser.parseLBrace() ||
        parser.parseKeyword("acq") ||
        parser.parseEqual() ||
        parser.parseOperand(pingAcqLockOperand) ||
        parser.parseComma() ||
        parser.parseKeyword("rel") ||
        parser.parseEqual() ||
        parser.parseOperand(pingRelLockOperand) ||
        parser.parseRBrace() ||
        parser.parseComma() ||
        parser.parseKeyword("locks_pong") ||
        parser.parseEqual() ||
        parser.parseLBrace() ||
        parser.parseKeyword("acq") ||
        parser.parseEqual() ||
        parser.parseOperand(pongAcqLockOperand) ||
        parser.parseComma() ||
        parser.parseKeyword("rel") ||
        parser.parseEqual() ||
        parser.parseOperand(pongRelLockOperand) ||
        parser.parseRBrace() ||
        parser.parseComma() ||
        parser.parseKeyword("channelid") ||
        parser.parseEqual() ||
        parser.parseOperand(channelOperand) ||
        parser.parseRBrace())
        return mlir::failure();
    
    // Parse optional attr-dict
    if (parser.parseOptionalAttrDict(result.attributes))
        return mlir::failure();
    
    // Parse: : (type($ping), type($pong))
    if (parser.parseColon() ||
        parser.parseLParen() ||
        parser.parseType(pingType) ||
        parser.parseComma() ||
        parser.parseType(pongType) ||
        parser.parseRParen())
        return mlir::failure();
    
    // Get lock and channel types
    auto lockType = dfschedule::LockType::get(parser.getContext());
    auto channelType = dfschedule::DmaChannelType::get(parser.getContext());
    
    // Resolve operands in the correct order
    if (parser.resolveOperand(pingOperand, pingType, result.operands) ||
        parser.resolveOperand(pongOperand, pongType, result.operands) ||
        parser.resolveOperand(pingAcqLockOperand, lockType, result.operands) ||
        parser.resolveOperand(pingRelLockOperand, lockType, result.operands) ||
        parser.resolveOperand(pongAcqLockOperand, lockType, result.operands) ||
        parser.resolveOperand(pongRelLockOperand, lockType, result.operands) ||
        parser.resolveOperand(channelOperand, channelType, result.operands))
        return mlir::failure();
    
    return mlir::success();
}

void dfschedule::DSKernelLaunchDmaLoopOp::print(::mlir::OpAsmPrinter &printer) {
    printer << "(";
    printer << getPing() << ", " << getPong();
    printer << ") {";
    
    // Print with proper indentation
    printer.increaseIndent();
    printer.printNewline();
    printer << "locks_ping = {acq = " << getPingAcqLock() << ", rel = " << getPingRelLock() << "},";
    printer.printNewline();
    printer << "locks_pong = {acq = " << getPongAcqLock() << ", rel = " << getPongRelLock() << "},";
    printer.printNewline();
    printer << "channelid = " << getChannel();
    printer.decreaseIndent();
    printer.printNewline();
    printer << "} ";
    
    // Print types
    printer << ": (";
    printer << getPing().getType() << ", " << getPong().getType();
    printer << ")";
}

// ===== Manager Implementation =====

ModuleOp dfschedulemanager::ops_test(MLIRContext* ctx, int totalN) {
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    
    builder.setInsertionPointToStart(m.getBody());

    SymbolTable symTable(m);
    
    // Create the host block with full example
    createHostBlock(builder, ctx, symTable);
    
    // Create the compute kernel
    createDSKernelCompute(builder, ctx);
    
    // Create the receiver kernel
    createDSKernelReceiver(builder, ctx);

    // Print the final module
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm(false);
    flags.enableDebugInfo(false);
    m.print(llvm::errs(), flags);
    llvm::errs() << "\n";
    return m;
}

void dfschedulemanager::loaddialect(MLIRContext* ctx) {
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<dfschedule::dfscheduledialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();
}

// Create the host block matching example.ir
void dfschedulemanager::createHostBlock(OpBuilder& builder, MLIRContext* ctx, SymbolTable& symTable) {
    auto location = builder.getUnknownLoc();
    
    // Create host block: dfschedule.host @host_0 { ... }
    auto hostBlockOp = builder.create<dfschedule::HostBlockOp>(
        location,
        builder.getStringAttr("host_0")
    );
    
    Block *hostBody = new Block();
    hostBlockOp.getBody().push_back(hostBody);
    builder.setInsertionPointToStart(hostBody);
    
    // %tensor = tensor.empty() : tensor<1024x1024xf32>
    auto tensorType = mlir::RankedTensorType::get({1024, 1024}, builder.getF32Type());
    auto tensor = builder.create<mlir::tensor::EmptyOp>(location, tensorType.getShape(), builder.getF32Type());
    
    // %gmem = dfschedule.declaretensor(%tensor) : (tensor<1024x1024xf32>) -> memref<1024xf32>
    auto memrefType = mlir::MemRefType::get({1024}, builder.getF32Type());
    auto gmem = builder.create<dfschedule::DeclareTensorOp>(location, memrefType, tensor.getResult());
    
    // %shim0 = dfschedule.declaretile {col = 3, row = 0} : !dfschedule.tile
    auto tileType = dfschedule::TileType::get(ctx);
    auto shim0 = builder.create<dfschedule::DeclareTileOp>(
        location, tileType,
        builder.getI32IntegerAttr(3),  // col
        builder.getI32IntegerAttr(0)   // row
    );
    
    // Get bd_id for this tile
    auto bdIdForConfig = builder.create<dfschedule::GetBdIdOp>(location, builder.getI32Type(), shim0.getResult());
    
    // %bd_config = dfschedule.config.dma_bd(%gmem, %shim0, %bd_id) {...}
    auto bdHandleType = dfschedule::BdHandleType::get(ctx);
    auto bdConfig = builder.create<dfschedule::ConfigDmaBdOp>(
        location, bdHandleType,
        gmem.getResult(),
        shim0.getResult(),
        bdIdForConfig.getBdId(),          // bd_id from GetBdIdOp
        builder.getI32IntegerAttr(0),     // offset
        builder.getI32IntegerAttr(1024),  // len
        builder.getBoolAttr(true),        // enable_packet
        builder.getI32IntegerAttr(10),    // packet_id
        builder.getI32IntegerAttr(-1)     // next_bd
    );
    
    // %io_0 = dfschedule.config.create_io(%bd_config, %shim0) {...}
    auto ioHandleType = dfschedule::IoHandleType::get(ctx);
    auto io0 = builder.create<dfschedule::ConfigCreateIoOp>(
        location, ioHandleType,
        bdConfig.getResult(),
        shim0.getResult(),
        builder.getI32IntegerAttr(0),       // channel
        builder.getStringAttr("MM2S"),      // direction
        builder.getStringAttr("SEND")       // io_operation
    );
    
    // Launch kernel section
    // %core0 = dfschedule.declaretile {col = 0, row = 3} : !dfschedule.tile
    auto core0 = builder.create<dfschedule::DeclareTileOp>(
        location, tileType,
        builder.getI32IntegerAttr(0),  // col
        builder.getI32IntegerAttr(3)   // row
    );
    
    // %core1 = dfschedule.declaretile {col = 1, row = 3} : !dfschedule.tile
    auto core1 = builder.create<dfschedule::DeclareTileOp>(
        location, tileType,
        builder.getI32IntegerAttr(1),  // col
        builder.getI32IntegerAttr(3)   // row
    );
    
    // %kernel_packet_0 = dfschedule.packet @packet0 (%gmem) {dma_channel=0}
    auto packetType = dfschedule::PacketType::get(ctx);
    auto kernelPacket0 = builder.create<dfschedule::PacketOp>(
        location, packetType,
        builder.getStringAttr("packet0"),
        gmem.getResult(),
        builder.getI32IntegerAttr(0)  // dma_channel
    );
    
    // %kernel_packet_1 = dfschedule.packet @packet1 (%gmem) {dma_channel=0}
    auto kernelPacket1 = builder.create<dfschedule::PacketOp>(
        location, packetType,
        builder.getStringAttr("packet1"),
        gmem.getResult(),
        builder.getI32IntegerAttr(0)  // dma_channel
    );
    
    // %kernel_group = dfschedule.config.load_kernel_group(%core0, %core1) {...}
    auto kernelGroupType = dfschedule::KernelGroupType::get(ctx);
    llvm::SmallVector<Value, 2> tiles = {core0.getResult(), core1.getResult()};
    
    // Create symbol ref arrays
    llvm::SmallVector<mlir::Attribute, 1> calleeRefs = {
        mlir::FlatSymbolRefAttr::get(ctx, "dskernel_receiver")
    };
    llvm::SmallVector<mlir::Attribute, 2> computeKernelRefs = {
        mlir::FlatSymbolRefAttr::get(ctx, "compute0"),
        mlir::FlatSymbolRefAttr::get(ctx, "compute0")
    };
    llvm::SmallVector<mlir::Attribute, 2> packetRefs = {
        mlir::FlatSymbolRefAttr::get(ctx, "packet0"),
        mlir::FlatSymbolRefAttr::get(ctx, "packet1")
    };
    
    auto kernelGroup = builder.create<dfschedule::LoadKernelGroupOp>(
        location, kernelGroupType,
        tiles,
        builder.getArrayAttr(calleeRefs),
        builder.getArrayAttr(computeKernelRefs),
        builder.getArrayAttr(packetRefs)
    );
    
    // %evt_kernel_group = dfschedule.schedule.launch_kernel_group(%kernel_group) {...}
    auto eventType = dfschedule::EventType::get(ctx);
    auto evtKernelGroup = builder.create<dfschedule::LaunchKernelGroupOp>(
        location, eventType,
        kernelGroup.getResult()
    );
    
    // Schedule section
    // %bd_id = dfschedule.schedule.getbdid(%shim0) : (!dfschedule.tile) -> i32
    auto bdId = builder.create<dfschedule::GetBdIdOp>(location, builder.getI32Type(), shim0.getResult());
    
    // %evnt_io = dfschedule.schedule.start_io(%io_0, %bd_id) {...}
    auto evntIo = builder.create<dfschedule::StartIoOp>(
        location, eventType,
        io0.getResult(),
        bdId.getResult()
    );
    
    // dfschedule.schedule.wait(%evt_io, %evt_kernel_group) : (...)
    llvm::SmallVector<Value, 2> waitEvents = {evntIo.getResult(), evtKernelGroup.getResult()};
    builder.create<dfschedule::ScheduleWaitOp>(location, waitEvents);
    
    // Move insertion point back to module level
    builder.setInsertionPointAfter(hostBlockOp);
}

// Create the compute kernel: dfschedule.dskernel.compute @compute0 {...}
void dfschedulemanager::createDSKernelCompute(OpBuilder& builder, MLIRContext* ctx) {
    auto location = builder.getUnknownLoc();
    
    auto computeOp = builder.create<dfschedule::DSKernelComputeOp>(
        location,
        dfschedule::ComputeType::get(ctx),
        builder.getStringAttr("compute0")
    );
    
    // Create body block with arguments
    Block *body = new Block();
    computeOp.getBody().push_back(body);
    
    // Add arguments: %buf: memref<16x256xf32, "LOCAL">, %loop_count: index
    auto memrefType = mlir::MemRefType::get(
        {16, 256}, builder.getF32Type(),
        mlir::AffineMap(), 
        builder.getStringAttr("LOCAL")
    );
    auto bufArg = body->addArgument(memrefType, location);
    auto loopCountArg = body->addArgument(builder.getIndexType(), location);
    
    builder.setInsertionPointToStart(body);
    
    // %c0 = arith.constant 0 : index
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(location, 0);
    // %c1 = arith.constant 1 : index
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(location, 1);
    // %rows = arith.constant 16 : index
    auto rows = builder.create<mlir::arith::ConstantIndexOp>(location, 16);
    // %cols = arith.constant 256 : index
    auto cols = builder.create<mlir::arith::ConstantIndexOp>(location, 256);
    
    // Outer loop: scf.for %r = %c0 to %rows step %c1
    auto outerLoop = builder.create<mlir::scf::ForOp>(
        location, c0.getResult(), rows.getResult(), c1.getResult()
    );
    builder.setInsertionPointToStart(outerLoop.getBody());
    Value r = outerLoop.getInductionVar();
    
    // Inner loop: scf.for %c = %c0 to %cols step %c1
    auto innerLoop = builder.create<mlir::scf::ForOp>(
        location, c0.getResult(), cols.getResult(), c1.getResult()
    );
    builder.setInsertionPointToStart(innerLoop.getBody());
    Value c = innerLoop.getInductionVar();
    
    // %v = memref.load %buf[%r, %c] : memref<16x256xf32, "LOCAL">
    auto v = builder.create<mlir::memref::LoadOp>(location, bufArg, ValueRange{r, c});
    
    // %sq = arith.mulf %v, %v : f32
    auto sq = builder.create<mlir::arith::MulFOp>(location, v.getResult(), v.getResult());
    
    // memref.store %sq, %buf[%r, %c] : memref<16x256xf32, "LOCAL">
    builder.create<mlir::memref::StoreOp>(location, sq.getResult(), bufArg, ValueRange{r, c});
    
    // Move insertion point back to module level
    builder.setInsertionPointAfter(computeOp);
}

// Create the receiver kernel: dfschedule.dskernel_receiver @... {...}
void dfschedulemanager::createDSKernelReceiver(OpBuilder& builder, MLIRContext* ctx) {
    auto location = builder.getUnknownLoc();
    
    auto receiverOp = builder.create<dfschedule::DSKernelReceiverOp>(
        location,
        builder.getStringAttr("dskernel_receiver")
    );
    
    // Create body block with arguments
    Block *body = new Block();
    receiverOp.getBody().push_back(body);
    
    // Add arguments: 
    //   %arg0: !dfschedule.packet - packet containing tensor and DMA info
    //   %tile: !dfschedule.tile - tile handle for DMA configuration
    //   %computelogic: !dfschedule.compute - compute kernel to invoke
    //   %loop_count: index - number of iterations
    auto packetType = dfschedule::PacketType::get(ctx);
    auto tileType = dfschedule::TileType::get(ctx);
    auto computeType = dfschedule::ComputeType::get(ctx);
    auto arg0 = body->addArgument(packetType, location);
    auto tileArg = body->addArgument(tileType, location);
    auto computelogic = body->addArgument(computeType, location);
    auto loop_count = body->addArgument(builder.getIndexType(), location);
    
    builder.setInsertionPointToStart(body);
    
    // %input_tensor = dfschedule.gettensor(%arg0) : (!dfschedule.packet) -> tensor<16x256xf32>
    auto tensorType = mlir::RankedTensorType::get({16, 256}, builder.getF32Type());
    auto inputTensor = builder.create<dfschedule::GetTensorOp>(location, tensorType, arg0);
    
    // %channel = dfschedule.getdmachannel(%arg0) : (!dfschedule.packet) -> !dfschedule.dma_channel
    auto dmaChannelType = dfschedule::DmaChannelType::get(ctx);
    auto channel = builder.create<dfschedule::GetDmaChannelOp>(location, dmaChannelType, arg0);
    
    // Allocate ping-pong buffers in local memory
    auto localMemrefType = mlir::MemRefType::get(
        {16, 256}, builder.getF32Type(),
        mlir::AffineMap(),
        builder.getStringAttr("LOCAL")
    );
    
    // %ping = dfschedule.kernel.memalloc(%input_tensor) : (tensor<16x256xf32>) -> memref<16x256xf32, "LOCAL">
    auto ping = builder.create<dfschedule::KernelMemAllocOp>(location, localMemrefType, inputTensor.getResult());
    
    // %pong = dfschedule.kernel.memalloc(%input_tensor) : (tensor<16x256xf32>) -> memref<16x256xf32, "LOCAL">
    auto pong = builder.create<dfschedule::KernelMemAllocOp>(location, localMemrefType, inputTensor.getResult());
    
    // ===== DMA Buffer Descriptor Configuration =====
    ///*
    // Configure DMA BD for ping buffer
    auto bdHandleType = dfschedule::BdHandleType::get(ctx);
    
    // DMA configuration parameters (derived from buffer shape and local constants)
    int64_t bufferLen = 16 * 256;  // Total elements in buffer
    int64_t packetIdBase = 10;     // Base packet ID
    
    // Get bd_ids for ping and pong buffers
    auto pingBdIdOp = builder.create<dfschedule::GetBdIdOp>(location, builder.getI32Type(), tileArg);
    auto pongBdIdOp = builder.create<dfschedule::GetBdIdOp>(location, builder.getI32Type(), tileArg);
    
    // %bd_ping = dfschedule.config.dma_bd(%ping, %tile, %bd_id_ping) { ... }
    auto bdPing = builder.create<dfschedule::ConfigDmaBdOp>(
        location, bdHandleType,
        ping.getResult(),
        tileArg,
        pingBdIdOp.getBdId(),                      // bd_id from GetBdIdOp
        builder.getI32IntegerAttr(0),              // offset
        builder.getI32IntegerAttr(bufferLen),      // len
        builder.getBoolAttr(true),                 // enable_packet
        builder.getI32IntegerAttr(packetIdBase),   // packet_id
        builder.getI32IntegerAttr(1)               // next_bd (chain to pong)
    );
    
    // %bd_pong = dfschedule.config.dma_bd(%pong, %tile, %bd_id_pong) { ... }
    auto bdPong = builder.create<dfschedule::ConfigDmaBdOp>(
        location, bdHandleType,
        pong.getResult(),
        tileArg,
        pongBdIdOp.getBdId(),                      // bd_id from GetBdIdOp
        builder.getI32IntegerAttr(0),              // offset
        builder.getI32IntegerAttr(bufferLen),      // len
        builder.getBoolAttr(true),                 // enable_packet
        builder.getI32IntegerAttr(packetIdBase + 1), // packet_id
        builder.getI32IntegerAttr(0)               // next_bd (chain back to ping)
    );
    //*/
    // Initialize locks
    ///*
    auto lockType = dfschedule::LockType::get(ctx);
    
    // %l_ping_acq = dfschedule.dskernel.lock_init(0, "ping_acquire_lock") -> !dfschedule.lock
    auto l_ping_acq = builder.create<dfschedule::DSKernelLockInitOp>(
        location, lockType, 0, builder.getStringAttr("ping_acquire_lock")
    );
    
    // %l_pong_acq = dfschedule.dskernel.lock_init(0, "pong_acquire_lock") -> !dfschedule.lock
    auto l_pong_acq = builder.create<dfschedule::DSKernelLockInitOp>(
        location, lockType, 0, builder.getStringAttr("pong_acquire_lock")
    );
    
    // %l_ping_rel = dfschedule.dskernel.lock_init(1, "ping_release_lock") -> !dfschedule.lock
    auto l_ping_rel = builder.create<dfschedule::DSKernelLockInitOp>(
        location, lockType, 1, builder.getStringAttr("ping_release_lock")
    );
    
    // %l_pong_rel = dfschedule.dskernel.lock_init(0, "pong_release_lock") -> !dfschedule.lock
    auto l_pong_rel = builder.create<dfschedule::DSKernelLockInitOp>(
        location, lockType, 0, builder.getStringAttr("pong_release_lock")
    );
    
    // Launch DMA loop
    auto sharedMemrefType = mlir::MemRefType::get(
        {16, 256}, builder.getF32Type(),
        mlir::AffineMap(),
        builder.getStringAttr("SHARED")
    );
    
    builder.create<dfschedule::DSKernelLaunchDmaLoopOp>(
        location,
        ping.getResult(),
        pong.getResult(),
        l_ping_acq.getResult(),
        l_ping_rel.getResult(),
        l_pong_acq.getResult(),
        l_pong_rel.getResult(),
        channel.getResult()
    );
    
    // Compute loop with ping-pong pattern
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(location, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(location, 1);
    auto c2 = builder.create<mlir::arith::ConstantIndexOp>(location, 2);
    
    // scf.for %i = %c0 to %loop_count step %c1
    auto forLoop = builder.create<mlir::scf::ForOp>(
        location, c0.getResult(), loop_count, c1.getResult()
    );
    builder.setInsertionPointToStart(forLoop.getBody());
    Value iv = forLoop.getInductionVar();
    
    // Ping/Pong selection logic
    // %is_ping = arith.cmpi "eq", arith.remui(%i, %c2), %c0 : index
    auto rem = builder.create<mlir::arith::RemUIOp>(location, iv, c2.getResult());
    auto isPing = builder.create<mlir::arith::CmpIOp>(
        location, mlir::arith::CmpIPredicate::eq, rem.getResult(), c0.getResult()
    );
    ///*
    // Select buffer (must use localMemrefType to match ping/pong types)
    auto ifBuffer = builder.create<mlir::scf::IfOp>(
        location, localMemrefType, isPing.getResult(), true
    );
    builder.setInsertionPointToStart(&ifBuffer.getThenRegion().front());
    builder.create<mlir::scf::YieldOp>(location, ping.getResult());
    builder.setInsertionPointToStart(&ifBuffer.getElseRegion().front());
    builder.create<mlir::scf::YieldOp>(location, pong.getResult());
    builder.setInsertionPointAfter(ifBuffer);
    Value currBuf = ifBuffer.getResult(0);
    
    // Select read lock
    auto ifReadLock = builder.create<mlir::scf::IfOp>(
        location, lockType, isPing.getResult(), true
    );
    
    builder.setInsertionPointToStart(&ifReadLock.getThenRegion().front());
    builder.create<mlir::scf::YieldOp>(location, l_ping_acq.getResult());
    builder.setInsertionPointToStart(&ifReadLock.getElseRegion().front());
    builder.create<mlir::scf::YieldOp>(location, l_pong_acq.getResult());
    builder.setInsertionPointAfter(ifReadLock);
    Value currReadLock = ifReadLock.getResult(0);
    ///*
    // Select write lock
    auto ifWriteLock = builder.create<mlir::scf::IfOp>(
        location, lockType, isPing.getResult(), true
    );
    builder.setInsertionPointToStart(&ifWriteLock.getThenRegion().front());
    builder.create<mlir::scf::YieldOp>(location, l_ping_rel.getResult());
    builder.setInsertionPointToStart(&ifWriteLock.getElseRegion().front());
    builder.create<mlir::scf::YieldOp>(location, l_pong_rel.getResult());
    builder.setInsertionPointAfter(ifWriteLock);
    Value currWriteLock = ifWriteLock.getResult(0);
   
    
    // Acquire lock (wait for data)
    builder.create<dfschedule::DSKernelAcquireLockOp>(location, currReadLock, builder.getI32IntegerAttr(1));
    
    // Compute
    builder.create<dfschedule::CoreComputeOp>(location, currBuf, computelogic);
    
    // Release lock
    builder.create<dfschedule::DSKernelReleaseLockOp>(location, currWriteLock, builder.getI32IntegerAttr(1));
    
    // Move insertion point back to module level
    //*/
    builder.setInsertionPointAfter(receiverOp);
}

// Legacy functions (kept for backward compatibility)
void dfschedulemanager::createdfschedulefuncByDim(OpBuilder& builder, MLIRContext* ctx, SymbolTable& symTable) {
    createHostBlock(builder, ctx, symTable);
}

mlir::func::FuncOp dfschedulemanager::createDSKernelFunc(OpBuilder &builder, MLIRContext *ctx) {
    // This legacy function is no longer needed but kept for API compatibility
    auto location = builder.getUnknownLoc();
    auto funcType = builder.getFunctionType({builder.getI32Type()}, {});
    auto funcOp = builder.create<mlir::func::FuncOp>(location, "dskernel_legacy", funcType);
    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    builder.create<mlir::func::ReturnOp>(location);
    return funcOp;
}
