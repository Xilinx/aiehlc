# Building aiehlc

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

5. LLVM, MLIR and clang 24 (RTTI on). aiehlc's TableGen files need LLVM 24; an older
   `mlir-tblgen` stops with `Value 'useStrictPropertiesInAssemblyFormat' unknown!`.

   Nothing to do by hand: when CMake does not get an LLVM 24+ build, `script/ensure_llvm.sh`
   looks for one (`$LLVM_INSTALL_DIR`, `$AIEHLC_LLVM_BUILD`, `$AIEHLC_LLVM_SRC/build`,
   `/scratch/staff/$USER/llvm-project/build`, `thirdparty/llvm-project/build`) and, if there is
   none, builds the pinned commit with `script/build_llvm.sh` (Release + assertions + RTTI,
   projects `mlir;clang`; about an hour on 32 cores, once). To build it ahead of time:

```bash
bash script/build_llvm.sh            # --src DIR / --jobs N / --ref REF / --update
source script/llvm_env.sh            # tcsh: source script/llvm_env.csh
```

   An LLVM you built yourself needs `-DLLVM_ENABLE_RTTI=ON` ("undefined reference to typeinfo"
   otherwise) and `-DLLVM_ENABLE_PROJECTS="mlir;clang"`.

5. Install Vitis

<https://docs.amd.com/r/en-US/ug1400-vitis-embedded/Installing-the-Vitis-Software-Platform>

<p align="center">Copyright&copy; 2025 Advanced Micro Devices, Inc</p>


## Compile

### Configure and build

```bash
mkdir ./build
cd ./build
cmake -G Ninja ..                                   # finds or builds LLVM 24 (see above)
# cmake -G Ninja .. -DLLVM_INSTALL_DIR=<llvm build> # use a specific LLVM 24+ build
# cmake -G Ninja .. -DAIEHLC_AUTO_BUILD_LLVM=OFF    # only look, never build LLVM
ninja
```
