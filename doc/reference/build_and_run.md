# Build and run with `script/aiehlc.sh`: platforms and the simulator

(Moved from the root README.md.)

## Examples

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --runtime-source-file example/perf/aieml_perf.cc 
```

Using llvm-aie (experimental):

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --use-llvm-aie --runtime-source-file example/other/multikernel.cc
```

AIE2PS Compilation Support

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --aie-version 5 --runtime-source-file example/perf/aieml_perf.cc
```

PetaLinux Support

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

source script/aiehlc.sh --platform linux --aie-version 2 --runtime-source-file tutorial/example.cpp
```

AIE Simulator Support (no board required)

Select the simulator with `--platform sim`. Key args:

- `--aie-version`: `1` (AIE), `2` (AIE-ML), or `5` (AIE2PS)
- `--sim-tiles "col:row,..."`: tiles to load a stub kernel ELF onto (comma-separated logical-`col`:physical-`row`, e.g. `0:3`). Omit to stub the whole array (default, but seems to crash the simulator often).

```bash
source script/setup.sh
#or
source ./script/setup.sh --bsp-use-git-repo=https://path/to/aie-rt.git

# stub all tiles (seems to crash the simulator often)
source script/aiehlc.sh --platform sim --aie-version 5 --runtime-source-file tutorial/example.cpp
# tutorial/example.cpp programs XAie_TileLoc(4, 4)
source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles "4:4" --runtime-source-file tutorial/example.cpp

# aiehlc only builds the artifacts; launch separately
bash script/runsim.sh
```

`runsim.sh` can be invoked three ways:

```bash
# 1. Default: no args -> reads ./aout/sim_config.sh (same CWD you ran aiehlc.sh from)
bash script/runsim.sh

# 2. Specific aout dir: point it at any aout/ containing a sim_config.sh
bash script/runsim.sh path/to/aout

# TilingLinalg writes its config in the provenance bundle
bash script/runsim.sh aout/worklocal

# 3. Direct (no config file): pass the inputs yourself (not recommended)
bash script/runsim.sh \
    --host-src /path/to/host.cc \
    --kernel-objs "/path/k1.o /path/k2.o" \
    --kernel-names "k1 k2" \
    --aie-gen 5 \
    --stub-all            # or: --stub-tiles "0:3"
```

On success the sim ends with `Sucess: CPU result matches AIE.` / `Sim result: 0`.
