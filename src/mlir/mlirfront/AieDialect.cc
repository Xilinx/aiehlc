/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <iostream>
#include <string>
#include <algorithm>
#include <type_traits>
#include <fstream>

#include "AieDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "aiedialect.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "aietype.cc.inc"
#define GET_OP_CLASSES
#include "aieop.cc.inc"
using namespace mlir;
using namespace mlir::aie;

/*****************************************************************************/
/*
*
* Function that is called during AieADialect Constructor that registers the
* Type and Operations.
*
* @param       void
* @return	void
*
* @note	None.
*
******************************************************************************/
void AieADialect::initialize() {
	addTypes<
    		#define GET_TYPEDEF_LIST
    		#include "aietype.cc.inc"
  	>();

	addOperations< 
		#define GET_OP_LIST
		#include "aieop.cc.inc"
	>();
}


/*****************************************************************************/
/*
*
* This API traverses and reads the information from the .mlir file and generates
* the wrapper code for the .mlir by identifying loadkernel operation.
*
* This API also generates the Linker Configuration File
*
* Current design:
* The arguments for loadkernel op in mlir is:
* 	1. Name of the kernel
* 	2. Col number
* 	3. Row number
*	4. TileKernelObject [numinput, numoutput, ArrayRef<WindowType>]
*
* Change this based on requirement (still prototype)
*
* @param       file_path: std::string file with path to .mlir file.
* @return	void
*
* @note	None.
*
******************************************************************************/
void generateWrapperCode(std::string file_path)
{
	mlir::MLIRContext context;
	llvm::SourceMgr sourceMgr;
	context.getOrLoadDialect<AieADialect>();

	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
		llvm::MemoryBuffer::getFile(file_path.c_str());

	if (auto err = fileOrErr.getError()) {
		llvm::errs() << "Error when reading" << file_path << err.message() << "\n";
		return;
	}

	sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
	auto module = mlir::parseSourceFile(sourceMgr, &context);
	if (!module) {
		llvm::errs() << "Error when parsing input.mlir\n";
		return;
	}

	// Currently assuming there is one input buffer and one output buffer (prototype) for
	// all the kernels.
	// Change the logic to vecto<string> if the case for multiple inputs and result.
	struct KernelDetails{
		std::string name;
		std::vector<std::string> input_buffer_name;
		std::vector<std::string> result_buffer_name;
		std::vector<std::string> win_input_name;
		std::vector<std::string> win_result_name;
		int col;
		int row;
		int num_input;
		int num_output;
		std::vector<int> window_direction;
	};
	std::vector<KernelDetails> kernel_details;

	mlir::Operation *op = module.get();
	op->walk([&](mlir::Operation *op){
		if(op->getName().getStringRef() == "Aie.load_kernel_op") {
			auto getIntValue = [&](std::string name){
				auto val = dyn_cast<IntegerAttr>(op->getAttr(name.c_str())).getValue();
				return static_cast<int>(val.getSExtValue());
			};
			KernelDetails kdet;
			kdet.name = dyn_cast<StringAttr>(op->getAttr("name")).getValue();
			kdet.col = getIntValue("col");
			kdet.row = getIntValue("row");
			
			// assuming all the window are homogenous (Incorporate the window details)
			kdet.num_input = dyn_cast<KernelType>(op->getOperand(0).getType()).getNumInputArgs();
			kdet.num_output = dyn_cast<KernelType>(op->getOperand(0).getType()).getNumOutputArgs();

			// Currently implmenting for one input and out result.
			// Have to change the logic based on number of input/output window.
			// Have to ask hua (is it captured from the kernel source or should be
			// passed via loadkernel api).
			for(int i=0; i<kdet.num_input; i++){
				std::string _name = "buf_ip" + std::to_string(kernel_details.size()) 
					+ std::to_string(kdet.input_buffer_name.size());
				kdet.input_buffer_name.push_back(_name);
				kdet.win_input_name.push_back("window_"+ _name +"_d");
			}

			for(int i=0; i<kdet.num_output; i++){
				std::string _name = "buf_r" + std::to_string(kernel_details.size()) 
					+ std::to_string(kdet.result_buffer_name.size());
				kdet.result_buffer_name.push_back(_name);
				kdet.win_result_name.push_back("window_"+ _name +"_d");
			}

			// Now capture the which variadic arguments are in which direction;
			int num_win_arguments = kdet.num_input + kdet.num_output;
			std::vector<int> dir;
			auto window_list = dyn_cast<KernelType>(op->getOperand(0).getType()).getArguments();
			for(int i=0; i<num_win_arguments; i++){
				dir.push_back(window_list[i].getDirection());
			}

			kdet.window_direction = dir;
			kernel_details.push_back(kdet);

		}
	});

	auto WrapperGetMainFunction = [&](){
		std::string code = "int main(void){\n";
		// Init Windows
		for(auto kerneldet: kernel_details){
			for(int i=0; i<kerneldet.num_input; i++){
				code += "  " + kerneldet.win_input_name[i] + "[1];\n";
				code += "  window_init("+ kerneldet.win_input_name[i] +", 1, "+ 
				kerneldet.input_buffer_name[i] +", BUF_SZ, BUF_SZ);\n";
			}
			for(int i=0; i<kerneldet.num_output; i++){
				code += "  " + kerneldet.win_result_name[i] + "[1];\n";

				code += "  window_init("+ kerneldet.win_result_name[i] +", 1, "+ 
				kerneldet.result_buffer_name[i] +", BUF_SZ, BUF_SZ);\n";
			}
		}
		code += "\n";

		// Get ASync Windows
		for(auto kerneldet: kernel_details){
			for(int i=0; i<kerneldet.num_input; i++){
				code += "  input_window_int32 * async_input_" + kerneldet.win_input_name[i] + " = " +
					"(get_input_async_window_int32("+kerneldet.win_input_name[i]+"));\n";
			}
			for(int i=0; i<kerneldet.num_output; i++){				
				code += "  output_window_int32 * async_output_" + kerneldet.win_result_name[i] + " = " +
					"(get_output_async_window_int32("+kerneldet.win_result_name[i]+"));\n";
			}
		}
		code += "\n";

		// Call the Kernels
		for(auto kerneldet: kernel_details){
			// TODO: all the logic is currently written for 2 arguments
			// have to change this based on Variadic information captured by MLIR
			// iterate throught the window type in MLIR and determine if it is 
			// input window or output window and then call the kernel accordingly
			// in the same order.
			int cnt_input = 0;
			int cnt_output = 0;

			// Try to format with new line to make it pretty incase there are large
			// number of windows.
			code += "  CALL_KERNEL(FUNC_NAME";
			for(int i=0; i<kerneldet.window_direction.size(); i++){
				if(kerneldet.window_direction[i]){
					code += ", async_output_" + kerneldet.win_result_name[cnt_output++];
				}else{
					code += ", async_input_" + kerneldet.win_input_name[cnt_input++];
				}
			}

			code += ");\n";
		}
		code += "\n";

		code += "  chess_memory_fence();\n";
		code += "  done();\n";
		code += "  return 0;\n";
		code += "}";

		return code;
	};


	auto WrapperGetDeclarations = [&](){
		std::string code = "";
		code += "#include <adf.h>\n";
		code += "#include <adf/sync/memsycn.h>\n\n";
		code += "#define XSTRINGIFY(s) #s\n";
		code += "#define STRINGIFY(s) XSTRINGIFY(s)\n";
		code += "#include STRINGIFY(KERNEL_SRC)\n";
		code += "#define CALL_KERNEL(KERNEL_CALL, ...) KERNEL_CALL(__VA_ARGS__)\n";
		code += "#define FOR_READ  1\n";
		code += "#define FOR_WRITE 0\n";
		code += "#define BUF_SZ 256\n";
		// Based on the number of kernels we assign the buffers.
		// This detail will be captured from MLIR.
		for(auto kerneldet : kernel_details){
			for(int i=0; i<kerneldet.num_input; i++)
				code += "v4int32 " + kerneldet.input_buffer_name[i] + "[BUF_SZ];\n";
			for(int i=0; i<kerneldet.num_output; i++)
				code += "v4int32 " + kerneldet.result_buffer_name[i] + "[BUF_SZ];\n";
		}

		return code;
	};

	auto decls = WrapperGetDeclarations();
	auto main_fn = WrapperGetMainFunction();
	auto wrapper_code = decls + main_fn;
	std::ofstream wout("wrapper.cc");
	wout << wrapper_code;
	wout.close();


	// Generate Linker configuration file
	// Note: The Linker configuration generation highly depends on
	// he ADF graph used for computation.
	//
	// Some aspects that influences Linker config file. If output of kernel is input of 
	// another kernel, then the core tiles need to be adjacent and buffer can be accessed 
	// directly. Then output buffer of kernel1 and input buffer of kernel2 should be in 
	// the same direction
	// 
	// This Linker config file depends on optimal location of kernel (should be implemented by
	// kernel). The direction of buffer depends on the connection between kernels and location
	// of kernels. (These needs to be handled in future generations).
	//
	// Current implementation:
	// Assumption 1 : All the kernels are independent and communicate via stream switch.
	// So all the input and output buffers to windows are present in west direction (random)
	// Assumption 2 : The number of input and output window does not exceed the count of
	// available number of buffer in south direction.
	// Assumption 3 : The buffer Address implmented is for AIE2.
	auto LinkerConfigCode = [&](){
		std::string code = "";
		code += "_reserved DMb 0x0 0x40000\n";
		// Doubt: I have doubt regarding this reservation as it coincides with south 
		// direction buffer.
		code += "_reserved DMb 0x40000 0x10000\n";
		code += "_reserved DMb 0x80000 0x80000 // And everything else the core can't see\n";
		code += "_entry_point _main_init\n";
		code += "_symbol      _main _after _main_init\n";
		code += "_symbol      _main_init 0x0\n";

		// Ignoring the need for sync buffer (not implemented in the prototype)
		// Allocating the Stack for the Core
		code += "\n";
		code += "_stack DM_stack 0x7e020  0x400 //stack for core\n";
		code += "\n";
		
		// Buffer Allocation
		// The buffer addresses with Direction [AIE2]:
		// +-----------+-------------------+
		// | Direction | Address           |
		// +-----------+-------------------+
		// | South     | 0x40000 - 0x4ffff |
		// +-----------+-------------------+
		// | West      | 0x50000 - 0x5ffff |
		// +-----------+-------------------+
		// | North     | 0x60000 - 0x6ffff |
		// +-----------+-------------------+
		// | East      | 0x70000 - 0x7ffff |
		// +-----------+-------------------+
		for(auto kdet: kernel_details){
			int total_window = kdet.num_input + kdet.num_output;
			// TODO : change based on addressing logic
			if(total_window >= 10){
				llvm::errs() << "large number of windows for this kernel\n";
				exit(1);
			}

			int index = 0;
			for(auto input_name : kdet.input_buffer_name){
				code += input_name + " 0x5" + std::to_string(index) + "000\n";
				index++;
			}

			for(auto result_name: kdet.result_buffer_name){
				code += result_name + " 0x5" + std::to_string(index) + "000\n";
				index++;
			}
		}

		return code;
	};

	auto linker_config_code = LinkerConfigCode();
	std::ofstream lout("aieml.bcf");
	lout << linker_config_code;
	lout.close();
}

/*****************************************************************************/
/*
*
* This Api traverses all the operations in MLIR represented by the file at 
* given path and prints basic information.
*
* @param       file_path: std::string file with path to .mlir file.
* @return	void
*
* @note	None.
*
******************************************************************************/
void TraverseMLIR(std::string file_path)
{
	mlir::MLIRContext context;
	llvm::SourceMgr sourceMgr;
	context.getOrLoadDialect<AieADialect>();

	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
		llvm::MemoryBuffer::getFile(file_path.c_str());

	if (auto err = fileOrErr.getError()) {
		llvm::errs() << "Error when reading" << file_path << err.message() << "\n";
		return;
	}

	sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
	auto module = mlir::parseSourceFile(sourceMgr, &context);
	if (!module) {
		llvm::errs() << "Error when parsing input.mlir\n";
		return;
	}

	mlir::Operation *op = module.get();
	op->walk([&](mlir::Operation *op){
		llvm::outs() << "visiting op: '" << op->getName() << "' with "
			<< op->getNumOperands() << " operands and "
			<< op->getNumResults() << " results\n";
	});

	return;
}


/*****************************************************************************/
/*
*
* This Api takes tests the working of parser of LoadKernel Operation. We take
* int .mlir file using a file path, read it and convert it into in-memory 
* representation of ir and dump it.
*
* @param       file_path: std::string file with path to .mlir file.
* @return	void
*
* @note	None.
*
******************************************************************************/
void TestParserOperation(std::string file_path){
	mlir::MLIRContext context;
	llvm::SourceMgr sourceMgr;
	context.getOrLoadDialect<AieADialect>();

	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
		llvm::MemoryBuffer::getFile(file_path.c_str());

	if (auto err = fileOrErr.getError()) {
		llvm::errs() << "Error when reading" << file_path << err.message() << "\n";
		return;
	}

	sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
	auto module = mlir::parseSourceFile(sourceMgr, &context);
	if (!module) {
		llvm::errs() << "Error when parsing input.mlir\n";
		return;
	}
    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }
    auto moduleOp = mlir::dyn_cast<mlir::ModuleOp>(module.get());
    if (!moduleOp) {
		llvm::errs() << "Error when casting to ModuleOp\n";
		std::cout << "TestParserOperation fails " << file_path << ":[Error]\n";
		return;
	}

	if (failed(moduleOp.verify())) {
		llvm::errs() << "Error when verifying the module\n";
		std::cout << "TestParserOperation fails " << file_path << ":[Error]\n";
		return;
	}

	std::cout << "TestParserOperation works " << file_path << ":[Success]\n";
	return;
}


/*****************************************************************************/
/*
*
* This Api takes int two int value, row and col and (future: emits LoadKernel MLIR) 
* creates LoadKernel MLIR Operations and add it to module and dump it. This also
* checks working of printer of a operation.
*
* @param       x: row
* @param       y: col
* @param	name: name of the kernel
* @return	void
*
* @note	None.
*
******************************************************************************/
void CreateLoadKernel(int x, int y, std::string name="")
{
	mlir::MLIRContext context;
	context.getOrLoadDialect<AieADialect>();

	ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
	OpBuilder builder(&context);
	builder.setInsertionPointToEnd(module.getBody());
	
	auto dataType = mlir::IntegerType::get(&context, 32);
	auto _col = mlir::IntegerAttr::get(dataType, x);
	auto _row = mlir::IntegerAttr::get(dataType, y);

	mlir::StringAttr nameAttr = builder.getStringAttr(name.c_str());

	SmallVector<Value> arguments;
	ValueRange args(arguments);

	builder.create<LoadKernel>(builder.getUnknownLoc(), nameAttr, _col, _row, arguments);
    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }
}


/*****************************************************************************/
/*
*
* This Api takes required arguments for window type and uses CreateWindowOperation
* to emit WindowType MLIR Operation.
*
* @param       arg_dir: direction
* @param       arg_size: size
* @param       arg_pingaddr: ping address
* @param       arg_pongaddr: pong address
* @param       arg_pinglockid: ping lock id
* @param       arg_ponglockid: pong lock id
* @return	void
*
* @note	None.
*
******************************************************************************/
void CreateWindowOperation(int arg_dir, int arg_size, long arg_pingaddr, long arg_pongaddr, 
	int arg_pinglockid, int arg_ponglockid, std::string name)
{
	mlir::MLIRContext context;
	context.getOrLoadDialect<AieADialect>();

	ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
	OpBuilder builder(&context);
	builder.setInsertionPointToEnd(module.getBody());

	auto i32dataType = mlir::IntegerType::get(&context, 32);
	auto i64dataType = mlir::IntegerType::get(&context, 64);

	auto attr_direction = mlir::IntegerAttr::get(i32dataType, arg_dir);
	auto attr_size = mlir::IntegerAttr::get(i32dataType, arg_size);
	auto attr_pingaddr = mlir::IntegerAttr::get(i64dataType, arg_pingaddr);
	auto attr_pongaddr = mlir::IntegerAttr::get(i64dataType, arg_pongaddr);
	auto attr_pinglockid = mlir::IntegerAttr::get(i32dataType, arg_pinglockid);
	auto attr_ponglockid = mlir::IntegerAttr::get(i32dataType, arg_ponglockid);

	mlir::StringAttr nameAttr = builder.getStringAttr(name);

	mlir::Type typeinfo = mlir::aie::WindowType::get(&context, arg_dir, arg_size, arg_pingaddr, 
		arg_pongaddr, arg_pinglockid, arg_ponglockid, nameAttr);
	
	builder.create<CreateWindowOp>(builder.getUnknownLoc(), typeinfo, attr_direction, attr_size, 
		attr_pingaddr, attr_pongaddr, attr_pinglockid, attr_ponglockid, nameAttr);
    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }
}


/*****************************************************************************/
/*
*
* This Api just tests the functionality of KernelObject creation using custom
* type and function. In actual functionality this will just emit the KernelObjectOperation
* based on the arguments given. Right now the Windows are generated with dummy default values.
*
* @param       NumInputArgs
* @param       NumOutputArgs
* @return	void
*
* @note	None.
*
******************************************************************************/
/*
void CreateKernelObjectOperation(int NumInputArgs = 1, int NumOutputArgs = 1)
{
	mlir::MLIRContext context;
	context.getOrLoadDialect<AieADialect>();

	ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
	OpBuilder builder(&context);
	builder.setInsertionPointToEnd(module.getBody());

	auto i32dataType = mlir::IntegerType::get(&context, 32);
	auto attr_num_input_args = mlir::IntegerAttr::get(i32dataType, NumInputArgs);
	auto attr_num_output_args = mlir::IntegerAttr::get(i32dataType, NumOutputArgs);
	
	SmallVector<Value> arguments;
	SmallVector<WindowType> type_args;
	int total_args = NumInputArgs + NumOutputArgs;

	for(int i=0; i<total_args; i++){
		auto i32dataType = mlir::IntegerType::get(&context, 32);
		auto i64dataType = mlir::IntegerType::get(&context, 64);

		auto attr_direction = mlir::IntegerAttr::get(i32dataType, 1);
		auto attr_size = mlir::IntegerAttr::get(i32dataType, 0);
		auto attr_pingaddr = mlir::IntegerAttr::get(i64dataType, 1);
		auto attr_pongaddr = mlir::IntegerAttr::get(i64dataType, 1);
		auto attr_pinglockid = mlir::IntegerAttr::get(i32dataType, 0);
		auto attr_ponglockid = mlir::IntegerAttr::get(i32dataType, 1);

		mlir::StringAttr nameAttr = builder.getStringAttr("mywindow");
		mlir::StringAttr windownameAttr = builder.getStringAttr("mywindow");

		WindowType type = mlir::aie::WindowType::get(&context, 1, 0, 1, 1, 0, 1, windownameAttr);
		Value win = builder.create<CreateWindowOp>(builder.getUnknownLoc(), type, 
			attr_direction, attr_size, attr_pingaddr, attr_pongaddr, 
			attr_pinglockid, attr_ponglockid, nameAttr).getResult();
		arguments.push_back(win);
		type_args.push_back(type);
	}

	ArrayRef<Value> valueArrayRef(arguments);
	ArrayRef<WindowType> typeArrayRef(type_args);
	mlir::Type type_val = mlir::aie::KernelType::get(&context, NumInputArgs, NumOutputArgs, 
		typeArrayRef);

	builder.create<CreateKernelObjectOp>(builder.getUnknownLoc(), type_val, attr_num_input_args, 
		attr_num_output_args, valueArrayRef);
	module.dump();
}
*/

/*****************************************************************************/
/*
*
* This Api takes int two int value, row and col and string representing kernel source.
* creates CreateTileKernelObjectOperation and add it to module and dump it. This also
* checks working of printer of a operation and type.
*
* @param       x: row
* @param       y: col
* @return	void
*
* @note	None.
*
******************************************************************************/
void CreateTileKernelObjectOperation(int row, int col, std::string kernel_src)
{
	mlir::MLIRContext context;
	context.getOrLoadDialect<AieADialect>();

	ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
	OpBuilder builder(&context);
	builder.setInsertionPointToEnd(module.getBody());

	auto dataType = mlir::IntegerType::get(&context, 32);
	auto _row = mlir::IntegerAttr::get(dataType, row);
	auto _col = mlir::IntegerAttr::get(dataType, col);
        Value val_row = builder.create<ConstantInteger32Op>(UnknownLoc::get(&context), _row)
		.getResult();
        Value val_col = builder.create<ConstantInteger32Op>(UnknownLoc::get(&context), _col)
		.getResult();
	mlir::StringAttr kernelsrcAttr = builder.getStringAttr(kernel_src.c_str());

	mlir::Type type_val = mlir::aie::TileKernelType::get(&context, row, col, kernelsrcAttr);

        builder.create<CreateTileKernelObjectOp>(builder.getUnknownLoc(), type_val, val_row, val_col, 
		kernelsrcAttr);
        {
            module->print(llvm::errs());
            llvm::errs() << "\n";
        }
}


void dumpmlir() {
	mlir::MLIRContext context;
	// Load our Dialect in this MLIR Context.
	context.getOrLoadDialect<AieADialect>();
	ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
	OpBuilder builder(&context);
	builder.setInsertionPointToEnd(module.getBody());
	auto elementType = IntegerType::get(&context, 64); 

	// prepare the mlir::value for loadkernel parameter
	std::vector<double> data = {0};
	std::vector<int64_t> dims = {1};
	auto dataType = mlir::RankedTensorType::get(dims, builder.getF64Type());
	auto denseAttr = DenseElementsAttr::get(dataType, llvm::ArrayRef(data));
	auto constantOp = builder.create<ConstantOp>(UnknownLoc::get(&context), denseAttr);
	mlir::Value value = constantOp.getResult();

	builder.create<LoadKernelOp>(builder.getUnknownLoc(), value, value);
    {
        module->print(llvm::errs());
        llvm::errs() << "\n";
    }
}
