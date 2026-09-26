/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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
    struct ReservedDMB {
        uint32_t addr;
        uint32_t len;
        std::string comment; // emitted as a trailing "// ..." when non-empty
    };
    std::vector<ReservedDMB> reservedDMB;
    std::vector<std::pair<std::string, uint32_t>> symbols;
	std::vector<uint32_t> stack;
public:
	Bcf() {
		name = "aieml.bcf";
	}

	std::string getfilename() {
		return name;
	}

    void addreservedDMB(uint32_t dmb, uint32_t len, const std::string &comment = "") {
        reservedDMB.push_back({dmb, len, comment});
    }

    // Carve out the tail of the core's data memory that the kernel must not be
    // given: 0x7F800..0x7FFFF backs the kernel log / trace-and-monitor buffers
    // written by the runtime, and everything from 0x80000 up is outside what
    // the core can address at all. Without these the xchesscc linker is free to
    // place a buffer or spill slot there and silently corrupt the log.
    void addlogandtmreserved() {
        addreservedDMB(0x7F800, 0x800);
        addreservedDMB(0x80000, 0x80000, "And everything else the core can't see");
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
            ostr << "_reserved DMb 0x" << std::hex << x.addr << " 0x" << std::hex << x.len;
            if (!x.comment.empty())
                ostr << " // " << x.comment;
            ostr << "\n";
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
  Buffer(uint32_t type, std::string piName, std::string poName, uint32_t pingAddr = 0, uint32_t pongAddr = 0,
         int32_t acquireLock = -1, int32_t releaseLock = -1)
      : wtype(type), pingName(piName), pongName(poName), pingAddress(pingAddr), pongAddress(pongAddr),
        acquireLockId(acquireLock), releaseLockId(releaseLock) {
      window_type = std::string((wtype == 0 ? "input_window_int32" : "output_window_int32"));
      window_getfunc = std::string((wtype == 0 ? "get_input_async_window_int32" : "get_output_async_window_int32"));
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

        std::string getPingName() const { return pingName; }
        std::string getPongName() const { return pongName; }
        uint32_t getDirection() const { return wtype; }
        uint32_t getPingAddress() const { return pingAddress; }
        uint32_t getPongAddress() const { return pongAddress; }
        int32_t getAcquireLockId() const { return acquireLockId; }
        int32_t getReleaseLockId() const { return releaseLockId; }
        // Returns true if this is ping-pong mode (pongAddress != 0)
        bool isPingPong() const { return pongAddress != 0; }
        // Returns true if lock IDs are explicitly set
        bool hasExplicitLockIds() const { return acquireLockId >= 0 && releaseLockId >= 0; }

        std::string getbufdeclare() {
			std::string code;
            // commenting because of sim crash
            // code = "v4int32 " + pingName +"[BUF_SZ];\n";
            code = "volatile int32 " + pingName + "[BUF_SZ];\n";
            // Only declare pong buffer in ping-pong mode
            if (isPingPong()) {
                // commenting because of sim crash
                // code += "v4int32 " + pongName + "[BUF_SZ];\n";
                code += "volatile int32 " + pongName + "[BUF_SZ];\n";
            }
            return code;
        }

		std::string getbufdefine() {
			std::string code;
			auto internal_win = getwinintername();
			code = "\twindow_internal " + internal_win + "[1];\n";
            if (isPingPong()) {
                // Ping-pong mode: 8-parameter window_init
                code += "\twindow_init(" + internal_win + ",1," + pingName + "," + pongName +
                        ",BUF_SZ,BUF_SZ,BUF_SZ,BUF_SZ);\n";
            } else {
                // Legacy single buffer mode: 5-parameter window_init
                code += "\twindow_init(" + internal_win + ",1," + pingName + "," + "BUF_SZ, BUF_SZ);\n";
            }
            auto paramname = getwinpointername();
            code += "\t" + window_type + "*  " + paramname + " = " + window_getfunc + "(" + internal_win + ");\n";
			return code;
		}
private:
		uint32_t wtype;
		std::string pingName;
		std::string pongName;
        uint32_t pingAddress;
        uint32_t pongAddress;
        int32_t acquireLockId;
        int32_t releaseLockId;

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
        // Streaming mode configuration
        bool streaming_mode = false;
        // Debug logging configuration
        bool enable_logging = true;
        uint32_t log_base_addr = (0x70000 + 16 * 1024 - 4 * 1024);

      public:
		Wrapper(std::string kernelname, std::string filename) {
			name = "wrapper.cc";
			kernel_name = kernelname;
			fname = filename;
		}
		std::string getfilename() {
			return name;
		}

        void enableStreaming() { streaming_mode = true; }
        void enableLogging(bool enable = true) { enable_logging = enable; }
        void setLogBaseAddr(uint32_t addr) { log_base_addr = addr; }

        // Generate debug logging code block
        std::string generateLoggingCode() {
            std::ostringstream addr_hex;
            addr_hex << std::hex << log_base_addr;

            std::string code;
            code += "// =============================================================================\n";
            code += "// Debug logging at fixed address 0x" + addr_hex.str() + "\n";
            code += "// =============================================================================\n";
            code += "#define LOG_BASE_ADDR 0x" + addr_hex.str() + "\n";
            code += "static volatile int* log_ptr = (volatile int*)LOG_BASE_ADDR;\n";
            code += "static int log_index = 0;\n\n";
            code += "// Log a value to the debug buffer (auto-incrementing address)\n";
            code += "inline void log(int value) {\n";
            code += "    log_ptr[log_index++] = value;\n";
            code += "}\n\n";
            code += "// Log a value at a specific index\n";
            code += "inline void log_at(int index, int value) {\n";
            code += "    log_ptr[index] = value;\n";
            code += "}\n\n";
            code += "// Reset log index to start\n";
            code += "inline void log_reset() {\n";
            code += "    log_index = 0;\n";
            code += "}\n\n";
            return code;
        }

        std::string generate() override{
            if (streaming_mode) {
                return generateStreamingWrapper();
            }
            return generateNormalWrapper();
        }

        std::string generateNormalWrapper() {
            std::string code;
			code += "#include <adf.h>\n";
            code += "#include <aie_api/aie.hpp>\n";
            code += "#include <aie_api/aie_adf.hpp>\n";
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

            // commenting because of sim crash
            // if (enable_logging) {
            //     code += generateLoggingCode();
            // }
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

        std::string generateStreamingWrapper() {
            std::string code;
            code += "#include <adf.h>\n";
            code += "#include <aie_api/aie.hpp>\n";
            code += "#include <aie_api/aie_adf.hpp>\n";
            for (auto x : headers) {
                code += "#include \"" + x + "\"\n";
            }
            code += "#define FOR_READ  1\n";
            code += "#define FOR_WRITE 0\n";
            code += "#define BUF_SZ " + std::to_string(lbuf_size) + "\n\n";

            code += "volatile static int sync_buffer[8] = {0, -1};\n\n";
            code += "#include <adf/sync/mesync.h>\n\n";

            // Add debug logging code
            if (enable_logging) {
                code += generateLoggingCode();
            }

            // Generate lock definitions
            // For ping-pong mode: 2 locks per parameter (ACQ and REL)
            // For single buffer mode: 1 lock per parameter
            int autoLockId = 16; // Start from lock ID 16 for auto-assignment
            for (size_t i = 0; i < params.size(); i++) {
                auto pingName = params[i].getPingName();
                if (params[i].isPingPong()) {
                    auto pongName = params[i].getPongName();
                    // Use explicit lock IDs if set, otherwise auto-increment
                    int acquireLock = params[i].hasExplicitLockIds() ? params[i].getAcquireLockId() : autoLockId++;
                    int releaseLock = params[i].hasExplicitLockIds() ? params[i].getReleaseLockId() : autoLockId++;
                    code += "#define LOCK_" + pingName + "_ACQ " + std::to_string(acquireLock) + "\n";
                    code += "#define LOCK_" + pongName + "_REL " + std::to_string(releaseLock) + "\n";
                } else {
                    // Legacy single buffer mode: one lock
                    int lock = params[i].hasExplicitLockIds() ? params[i].getAcquireLockId() : autoLockId++;
                    code += "#define LOCK_" + pingName + " " + std::to_string(lock) + "\n";
                }
            }
            code += "\n";

            // Generate buffer declarations (uses getbufdeclare which handles both modes)
            for (auto &x : params) {
                code += x.getbufdeclare();
            }
            code += "\n";

            code += "#include \"../../" + kernel_name + ".cc\"\n";
            code += "\nint main(void) {\n";
            if (enable_logging) {
                code += "\tlog(1);  // Log: entering main\n";
            }
            code += "\tsync_buffer[0] = 0; // reset end signal\n\n";

            // window_init for each parameter
            // Ping-pong mode: 8-parameter window_init with locks
            // Single buffer mode: 5-parameter window_init
            if (enable_logging) {
                code += "\tlog(2);  // Log: before window init\n";
            }
            for (size_t i = 0; i < params.size(); i++) {
                auto pingName = params[i].getPingName();
                auto baseName = params[i].getwinparamname();
                std::string wfunc =
                    (params[i].getDirection() == 0) ? "get_input_async_window_int32" : "get_output_async_window_int32";
                std::string wtypedef = (params[i].getDirection() == 0) ? "input_window_int32" : "output_window_int32";

                code += "\twindow_internal window_" + baseName + "[1];\n";

                if (params[i].isPingPong()) {
                    // Ping-pong mode: 8-parameter window_init
                    auto pongName = params[i].getPongName();
                    code += "\twindow_init(window_" + baseName + ", 1, ";
                    code += pingName + ", LOCK_" + pingName + "_ACQ, ";
                    code += pongName + ", LOCK_" + pongName + "_REL, ";
                    code += "BUF_SZ, BUF_SZ);\n";
                } else {
                    // Legacy single buffer mode: 5-parameter window_init
                    code += "\twindow_init(window_" + baseName + ", 1, ";
                    code += pingName + ", BUF_SZ, BUF_SZ);\n";
                }
                code += "\t" + wtypedef + "* " + baseName + "_ptr = " + wfunc + "(window_" + baseName + ");\n\n";
            }

            // Call kernel (window_acquire/release handled inside kernel)
            if (enable_logging) {
                code += "\tlog(3);  // Log: before kernel call\n";
            }
            code += "\t// Call kernel\n";
            code += "\t" + kernel_name + "(";
            for (size_t i = 0; i < params.size(); i++) {
                code += params[i].getwinparamname() + "_ptr";
                code += (i < params.size() - 1) ? ", " : "";
            }
            code += ");\n\n";

            if (enable_logging) {
                code += "\tlog(4);  // Log: after kernel, before done\n";
            }
            code += "\tdone();\n";
            if (enable_logging) {
                code += "\tlog(5);  // Log: after done\n";
            }
            code += "\treturn 0;\n";
            code += "}";

            return code;
        }
};



class HybridPass : public PassWrapper<HybridPass, OperationPass<>> {
	void runOnOperation() override;
	void convertDialect(mlir::Operation* op);
	void exportTofiles(Bcf& bcf, Prx& pfx, Wrapper& wrap);
};
