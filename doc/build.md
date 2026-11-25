# Building aiehlc

## Quick path (docker, tested with Vitis 2026.1 + LLVM 19.1.4)

```bash
bash script/docker.sh                 # builds aiehlc-dev image; maps /proj, /scratch, ~/.ssh, license dirs
bash script/llvm/build.sh             # optional: cmake runs it if no LLVM is found
source script/setup.sh                # BSPs + upstream aie-rt (branch xlnx_rel_v2026.2)
mkdir -p build && cd build && cmake .. && make -j$(nproc) aiehlc && cd ..
source script/aiehlc.sh --aie-version 5 \
    --runtime-source-file ./example/tileprogram/ccode/simpleconv2dkernel_port_less.cc
python3 script/test/apppaltest.py -y --netload ./aout/main.elf
```

`LLVM_INSTALL_DIR` is taken from `-DLLVM_INSTALL_DIR`, then the environment,
then `thirdparty/llvm-project/build`, then `../llvm-project/build`, then `/usr/local`. If none has LLVM+Clang+MLIR,
`cmake ..` runs `script/llvm/build.sh` itself (LLVM 19.1.4 into
`thirdparty/llvm-project/build`, about an hour; `-DAIEHLC_LLVM_SRC_DIR=<dir>` to move it,
`-DAIEHLC_AUTO_BUILD_LLVM=OFF` to get an error instead). aie-rt repo/branch: `AIE_RT_REPO`, `AIE_RT_BRANCH`.

## Dependencies

1. Install cmake 20+

2. Install antlr4(optional)

```bash
sudo apt-get install antlr4
sudo apt-get install libantlr4-runtime-dev
```

3. install boost

MacosX

```
brew upgrade
brew install boost
```

ubuntu

```
sudo apt install boost
```

4. Install curses and ninja

```bash
sudo apt-get install libncurses5-dev
sudo apt-get install ninja-build
```

5. Install LLVM, MLIR, and clang, build from LLVM 19.1.4 source or Install llvm 19.1.4 with RTTI enable. 

Build from source and enable RTTI option

<https://mlir.llvm.org/getting_started/>

When configuring your project with CMake, we recommend including the following settings to ensure compatibility with aiehlc.
The most critical setting is enabling Run-Time Type Information (RTTI) for the LLVM build:
-DLLVM_ENABLE_RTTI=ON
This is required to resolve "undefined reference to typeinfo" linker errors and to enable the use of dynamic casting mechanisms like mlir::dyn_cast.

```bash
-DLLVM_ENABLE_PROJECTS="mlir;clang" \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_ENABLE_RTTI=ON 
```

5. Install Vitis

<https://docs.amd.com/r/en-US/ug1400-vitis-embedded/Installing-the-Vitis-Software-Platform>

<p align="center">Copyright&copy; 2025 Advanced Micro Devices, Inc</p>


## Compile

### Find the llvm build or install path

for example the llvm build is

```
/Users/user/src/app/thirdparty/llvm-project/build
```

### cmake with set llvm path into the said path

default will use /usr/local/, if the LLVM path is different or need to use local llvm do following

```bash
mkdir ./build
cd ./build
cmake ../ -DLLVM_INSTALL_DIR=/Users/user/src/app/thirdparty/llvm-project/build
make -j32
ls
```
