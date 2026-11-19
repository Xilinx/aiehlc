# Source (tcsh/csh): point aiehlc's builds at the local LLVM/Clang/MLIR build.
#   source script/llvm_env.csh     ($AIEHLC_LLVM_SRC or /scratch/staff/$USER/llvm-project)
if ( ! $?AIEHLC_LLVM_SRC ) setenv AIEHLC_LLVM_SRC /scratch/staff/${USER}/llvm-project
if ( ! $?AIEHLC_LLVM_BUILD ) setenv AIEHLC_LLVM_BUILD ${AIEHLC_LLVM_SRC}/build
if ( ! -x ${AIEHLC_LLVM_BUILD}/bin/mlir-tblgen ) echo "[llvm_env] WARNING: ${AIEHLC_LLVM_BUILD} has no bin/mlir-tblgen -- run script/build_llvm.sh"
setenv LLVM_INSTALL_DIR ${AIEHLC_LLVM_BUILD}
setenv LLVM_DIR ${AIEHLC_LLVM_BUILD}/lib/cmake/llvm
setenv MLIR_DIR ${AIEHLC_LLVM_BUILD}/lib/cmake/mlir
setenv Clang_DIR ${AIEHLC_LLVM_BUILD}/lib/cmake/clang
setenv PATH ${AIEHLC_LLVM_BUILD}/bin:${PATH}
rehash
