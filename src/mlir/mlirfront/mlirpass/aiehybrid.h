/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "../../common.h"
#include "../AieDialect.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace mlir;

class ExportFile {
private:
	std::string base_dir = std::string(AOUT) + "./kernelcfg/";
	void try_create_folder(std::string dirPath) {
		///*
		std::string command = "mkdir -p " + dirPath; 
		int result = system(command.c_str());           // Run the command

    if (result == 0) {
        std::cout << "Directory created." << std::endl;
    } else {
        std::cerr << "Failed to create directory." << std::endl;
    }
		//*/
		/*
		if (!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directory(dirPath);
        std::cout << "Directory created: " << dirPath << std::endl;
    } else {
        std::cout << "Directory already exists: " << dirPath << std::endl;
    }
		*/
	}
public:
	std::string kernel_name = "";
	std::string name;
	virtual std::string generate()  = 0;
	void exportfile() {
		std::string dir = base_dir + kernel_name + "/";
		auto fname = dir + name;
		std::cout << "Exporting file to: " << fname << std::endl;
		std::cout << "Directory: " << dir << std::endl;
		try_create_folder(dir);
		std::ofstream ofs(fname);
		if (ofs.is_open()) {
			ofs << generate();
			ofs.close();
		}
	}
};

class LdScript;

class Bcf: public ExportFile {
private:
	friend class LdScript;
	std::string entrypoint = "_main_init";
  	std::string symbolmain = "_main _after ";
	uint32_t entrypointoffset = 0;
	std::vector<std::pair<uint32_t, uint32_t>>  reservedDMB;
	std::vector<std::pair<std::string, uint32_t>> symbols;
	std::vector<uint32_t> stack;
public:
	Bcf() {
		name = "aieml.bcf";
	}

	std::string getfilename() {
		return name;
	}
	
	void addreservedDMB(uint32_t dmb, uint32_t len) {
		reservedDMB.push_back({dmb, len});
	}
	
	void addsymbols(std::string name, uint32_t addr) {
		symbols.push_back({name, addr});
	}
	
	void setstack(uint32_t addr, uint32_t offset) {
		stack = {addr, offset};
	}
public:
	std::string generate() override {
		std::sort(symbols.begin(), symbols.end(), [](const auto& a, const auto& b) {
			return a.second < b.second;
		});

		std::ostringstream ostr;
		ostr << "_entry_point " << entrypoint << "\n";
		ostr << "_symbol " << symbolmain << " " << entrypoint << "\n";
		ostr << "_symbol " << entrypoint << " " << entrypointoffset << "\n";
		for (auto x:symbols) {
			ostr << "_symbol " <<  x.first << " 0x" << std::hex << x.second << "\n";
		}
		ostr << "_stack DM_stack ";
		for (auto x: stack) {
			ostr << std::hex << "0x" << x << " ";
		}
		ostr << "\n";
		for (auto x:reservedDMB) {
			ostr << "_reserved DMb 0x" << std::hex << x.first << " 0x" << std::hex << x.second  << "\n";
		}
		return ostr.str();
	}
};

class LdScript : public ExportFile {
private:
	std::string entrypoint = "_main_init";
	// std::string symbolmain = "_main _after ";
	// uint32_t entrypointoffset = 0;
	// std::vector<std::pair<uint32_t, uint32_t>>  reservedDMB;
	std::vector<std::pair<std::string, uint32_t>> symbols;
	std::vector<uint32_t> stack;
	std::vector<std::string> text_items = {"*(.text._main_init)", "_ctors_start", "_init_array_start", "KEEP(SORT(*.init_array))", "_ctors_end", "_init_array_end", "_dtors_start", "_dtors_end"};
	std::vector<std::string> data_items = {};
	// std::vector<std::string> bss_items = {};
	std::string prefix = R"(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x004000
   data (!RX) : ORIGIN = 0x70404, LENGTH = 0xFBFC
}
)";

	std::string suffix = R"(
  .bss.DMb.4 : { *(.bss.DMb.4) } > data 
}

PROVIDE(_main = main);
)";

public:
	LdScript() {
		name = "main.ld.script";
	}

	LdScript(Bcf& bcf) {
		name = "main.ld.script";
		entrypoint = bcf.entrypoint;
		// symbolmain = bcf.symbolmain;
		// entrypointoffset = bcf.entrypointoffset;
		// reservedDMB = bcf.reservedDMB;
		symbols = bcf.symbols;
		stack = bcf.stack;
		kernel_name = bcf.kernel_name;
	}
	std::string getfilename() {
		return name;
	}

	// void addreservedDMB(uint32_t dmb, uint32_t len) {
	// 	reservedDMB.push_back({dmb, len});
	// }

	void addsymbols(std::string name, uint32_t addr) {
		symbols.push_back({name, addr});
	}

	void setstack(uint32_t addr, uint32_t offset) {
		stack = {addr, offset};
	}

	std::string generate() override {
		int indent = 1, spaces = 2;
		std::string indent_str(indent * spaces, ' ');
		std::string nested_indent_str(2 * indent * spaces, ' ');

		std::ostringstream ostr;
		ostr << prefix;

		ostr << "ENTRY(" << entrypoint << ")\n";
		ostr << "SECTIONS\n";
		ostr << "{\n";
		ostr << indent_str << ". = 0x0;\n";
		ostr << "\n";

		ostr << indent_str + ".text : {\n";
		for(auto x: text_items) {
			ostr << nested_indent_str << x << "= .;\n";
		}
		ostr << nested_indent_str << "*(.text)\n";
		ostr << indent_str << "} > program\n";
		ostr << "\n";

		ostr << indent_str << ".data : {\n";
		for(auto x: data_items) {
			ostr << nested_indent_str << x << " = .;\n";
		}
		ostr << nested_indent_str << "*(.data*)\n";
		ostr << nested_indent_str << "*(.rodata*)\n";
		ostr << indent_str << "} > data\n";
		ostr << "\n";

		ostr << indent_str << ". = 0x" << std::hex << stack[0] << ";\n";
		ostr << indent_str << "_sp_start_value_DM_stack = .;\n";
		ostr << indent_str << ". += 0x" << std::hex << stack[1] << "; /* stack */\n";
		ostr << indent_str << ". = 0x71000;\n";
		ostr << "\n";

		ostr << indent_str << ".bss : {\n";
		std::sort(symbols.begin(), symbols.end(), [](const auto& a, const auto& b) {
			return a.second < b.second;
		});
		for (auto x:symbols) {
			ostr << nested_indent_str << ". = 0x" << std::hex << x.second << ";\n";
			ostr << nested_indent_str << x.first << " = .;\n";
		}
		ostr << nested_indent_str << "*(.bss*)\n";
		ostr << indent_str << "} > data\n";

		ostr << suffix;

		return ostr.str();
	}
};

/* the pfx looks like following
<project name="Project" processor="me">
  <file type="lbc" name="kernel.ll" path="../../build/obj/"/>
  <issinit/>
  <option id="cpp.define" value="__AIENGINE__ __AIEARCH__=10" inherit = "1"/>
 <option id="llvm.xargs" value="-fno-jump-tables -fno-discard-value-names -mllvm -chess-collapse-struct-types-during-linking=0" inherit = "1"/>
 <option id="llvm.lang" value="Follow file extension"/>
  <option id="bridge.cfg" value="aie.bcf" path="./"/>
  <option id="cpp.include" value="&lt;XILINX_VITIS_AIETOOLS&gt;../TheHouseOfCommons ./ /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/aie_api /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/drivers/aiengine/ /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/aie_api /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/drivers/aiengine/ " inherit="1"/>
  <option id="project.dir" value="&lt;CONFIG&gt;./"/>
  <option id="project.name" value="kernel"/>
  <option id="project.type" value="exe"/>
</project>
*/

class Prx: public ExportFile {
private:
	boost::property_tree::ptree pt;
public:
	std::string getfilename() {
		return name;
	}

	void add_kernel_info(std::string ftype, std::string fname, std::string fpath) {
		boost::property_tree::ptree& filetype = pt.add("project.file", "");
		filetype.put("<xmlattr>.type", ftype);
		filetype.put("<xmlattr>.name", fname);
		filetype.put("<xmlattr>.path", fpath);
	}

	Prx(std::string pname, std::string processor) {
		name = "aieml.prx";
		pt.put("project.<xmlattr>.name", pname);
		pt.put("project.<xmlattr>.processor", processor);
		pt.add("project.issinit", "");
	}
	void setOption(std::string id, std::string value, std::string inherit, std::string path) {
		boost::property_tree::ptree& option = pt.add("project.option", "");
		if (!id.empty()) {
			option.put("<xmlattr>.id", id);
		}
		if (!value.empty()) {
			option.put("<xmlattr>.value", value);
		}
		if (!inherit.empty()) {
			option.put("<xmlattr>.inherit", inherit);
		}
		if (!path.empty()) {
			option.put("<xmlattr>.path", path);
		}
	}

	std::string generate() override{
		std::ostringstream oss;
		write_xml(oss, pt, boost::property_tree::xml_parser::xml_writer_make_settings<std::string>(' ', 4));
		auto str = oss.str();
		if (str.find("<?xml") ==0) {
			std::size_t pos = str.find('\n');
			str = str.substr(pos + 1);
		}
		std::cout << str << std::endl;
		return str;
	}

};
/*

  window_internal window_bufIP_bufIP_d[1];
  window_init(window_bufIP_bufIP_d, 1, bufIP, BUF_SZ, BUF_SZ);//need to use a ping pong buffer
  //window_init(window_buf0_buf0d, 1, buf0, LOCK_2_1_0_PRD, buf0d, LOCK_2_1_1_CNS, 8, 8);

  window_internal window_bufR_bufR_d[1];
  window_init(window_bufR_bufR_d, 1, bufR, BUF_SZ, BUF_SZ);

  input_window_int32 *input_window_i2_p1 = (get_input_async_window_int32(window_bufIP_bufIP_d));
  output_window_int32 *output_window_i2_p4 = (get_output_async_window_int32(window_bufR_bufR_d));

  int32 index = 1;
   // Kernel call: adp
  CALL_KERNEL(FUNC_NAME, input_window_i2_p1, output_window_i2_p4);
  chess_memory_fence();
  done();
  return 0;
*/
class Buffer {
public:
		Buffer(uint32_t type, std::string piName, std::string poName):
			wtype(type), pingName(piName), pongName(poName) {
			window_type = std::string((wtype == 0 ? "input_window_int32":"output_window_int32")) ;
			window_getfunc = std::string((wtype == 0 ? "get_input_async_window_int32":"get_output_async_window_int32"));

		}

		std::string getwinparamname() {
			std::string code("");
			code += pingName;
			return code;
		}

		std::string getwinpointername() {
			std::string code("");
			code += pingName+std::string("_win_ptr");
			return code;
		}


		std::string getbufdeclare() {
			std::string code;
			code = "v4int32 " + pingName +"[BUF_SZ];\n";
			code += "v4int32 " + pongName +"[BUF_SZ];\n";
			return code;
		}

		std::string getbufdefine() {
			std::string code;
			auto internal_win = getwinintername();
			code = "\twindow_internal " + internal_win + "[1];\n";
			code += "\twindow_init(" + internal_win + ",1," + pingName + "," + "BUF_SZ, BUF_SZ);\n"; 
			auto paramname = getwinpointername();
			code += "\t" + window_type + "*  " + paramname + " = " + window_getfunc + "(" + internal_win + ");\n";
			return code;
		}
private:
		uint32_t wtype;
		std::string pingName;
		std::string pongName;

		std::string window_type;
		std::string window_getfunc;

		std::string getwinintername() {
			std::string code;
			code = std::string(wtype == 0 ? "in":"out") + pingName + "_" + pongName;
			return code;
		}
};

class Wrapper : public ExportFile{
private:
		std::vector<std::string> headers;
		std::vector<std::string> macs;
		std::vector<Buffer> params;
		std::string fname;
		uint32_t lbuf_size=128;
		std::string kernel_in_param_type ="";
		std::string kernel_out_param_type ="";
public:
		Wrapper(std::string kernelname, std::string filename) {
			name = "wrapper.cc";
			kernel_name = kernelname;
			fname = filename;
		}
		std::string getfilename() {
			return name;
		}
		std::string generate() override{
			std::string code;
			code += "#include <adf.h>\n";
			for (auto x:headers) {
				code += "#include \"" + x + "\"\n";
			};
			//code += "#define XSTRINGIFY(s) #s\n";
			//code += "#define STRINGIFY(s) XSTRINGIFY(s)\n";
			//code += "#define CALL_KERNEL(KERNEL_CALL, ...) KERNEL_CALL(__VA_ARGS__)\n";
			code += "#define FOR_READ  1\n";
			code += "#define FOR_WRITE 0\n";
			code += "#define BUF_SZ " + std::to_string(lbuf_size) + "\n";
			code += "\nvolatile static int sync_buffer[8] = {0, -1};\n\n";
			code += "#include <adf/sync/mesync.h>\n\n";
			for (auto x:params) {
				code += x.getbufdeclare();
			}
			code += "#include \"../../" + kernel_name +".cc\"";
			code += "\nint main(void) {\n";
			for (auto x:params) {
				code += x.getbufdefine();
				code += "\n";
			}
			code += "\t";
			code += kernel_name;
			code += "(";
			int len = params.size();
			for (int i = 0; i < len; i++) {
				std::string kernel_param_type = this->kernel_out_param_type;
				if(i == 0)
					kernel_param_type = this->kernel_in_param_type;

				if(kernel_param_type == "" || kernel_param_type.find("input_window") != std::string::npos || kernel_param_type.find("output_window") != std::string::npos) {
					code += params[i].getwinpointername();
				} else {
					if (kernel_param_type.back() != '*') {
						code += "*";
						kernel_param_type += "*";
					}
					code += "(" + kernel_param_type + ")" + params[i].getwinpointername() + "->ptr";
				}
				code += (i < len - 1) ? "," : "";
			}
			code += ");\n";
			code += "\tchess_memory_fence();\n";
			code += "\tdone();\n";
			code += "\treturn 0;\n";
			code += "}";

			return code;
		}
		void addkernelfuncparams(std::vector<Buffer>& bufs) {
			params = bufs;
		}

		void setbufsize(uint32_t bufsize) {
			lbuf_size = bufsize;
			//TODO current just use int32 buffer
			lbuf_size = lbuf_size/sizeof(uint32_t);
			//deal with xchess logic issue that need to ask 4 time real reserved mem
			lbuf_size = lbuf_size / 4;
		}

		void set_kernel_in_param_type(std::string kernel_param_type) {
			kernel_param_type.erase(kernel_param_type.find_last_not_of(" \n\r") + 1);
			this->kernel_in_param_type = kernel_param_type;
		}

		void set_kernel_out_param_type(std::string kernel_param_type) {
			kernel_param_type.erase(kernel_param_type.find_last_not_of(" \n\r") + 1);
			this->kernel_out_param_type = kernel_param_type;
		}
};



class HybridPass : public PassWrapper<HybridPass, OperationPass<>> {
	void runOnOperation() override;
	void convertDialect(mlir::Operation* op);
	void exportTofiles(Bcf& bcf, Prx& pfx, Wrapper& wrap);
};
