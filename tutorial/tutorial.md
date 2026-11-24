# Tutorial for aiehlc compile

## Clone the Repo and Navigate to the Tutorial Directory

```bash
git clone --recursive https://github.com/Xilinx/aiehlc
cd ./aiehlc/tutorial
```

If you already cloned the repository without submodules then run the following command.

```bash
git submodule update --init --recursive
```

## App Writing

This tutorial walks through the contents of [example.cpp](example.cpp), which demonstrates how to write, load, and execute a simple kernel, including memory management and data movement between host and AIE tiles.

### 1. Write a kernel

A kernel is a function that runs on the AIE tile. In this example, the kernel is defined as loop_kernel. Notice the special \_\_global\_\_ keyword to communicate to the compiler that this is a kernel. Also, the input/output parameters have annotations that provide details about their memory addresses and sizes.

This kernel just writes the values 100, 101, ..., 119 to the output buffer.

```cpp
__global__
void loop_kernel(int *in, int *out) {    
    for (int i = 0; i < 20; ++i)
        *(out++) = 100 + i;
}
```

### 2. Write a Main Function to Initialize AIE

The main function sets up the AIE device and prepares it for kernel execution.

First, the data and instruction caches are disabled.

Then the AIE device instance and configuration are set up.

Lastly, after some specific configuration depending on AIE generation, the host control function is called which runs the kernel and the results are checked.

```cpp
int main(int argc, char* argv[]) {
    // Disable Cache
    Xil_DCacheDisable();
    Xil_ICacheDisable();

    // Device Configuation
    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
                     XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
                     XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
                     XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
                     XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);
    XAie_InstDeclare(DevInst, &ConfigPtr);
    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if(RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

#if AIE_GEN >= 2
    if(DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
    }
    RC = XAie_PartitionInitialize(&DevInst, NULL);
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0); 
#endif

    // Call Host Control Function
    test_kernel(&DevInst);
}
```

### 3. Write a Host Control Function

The host control function manages kernel execution, memory allocation, data movement, and result verification.

#### a. Load Kernel

The core is reset/unreset then the compiled kernel is loaded into the specified tile.

```cpp
XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char *)loop_kernel);
```

#### b. Routing DDR with the AIE Tile

First the routing handler is initialized then the routes between DDR (host) and the AIE tile are configured.

```cpp
XAie_RoutingInstance* routingInstance = XAie_InitRoutingHandler(DevInst);
XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(2,0), /*dest=*/ XAie_TileLoc(4,4));
XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(4,4), /*dest=*/ XAie_TileLoc(35,0));
```

#### c. Move Data

Move input data from DDR to the AIE tile memory.

```cpp
XAie_MoveDataExternal2Aie(routingInstance, /*src=*/ XAie_TileLoc(2,0),
                          in, len*sizeof(u32),
                          CORE_IP_MEM, /*dest=*/ XAie_TileLoc(4,4));
 
```

#### d. Run the AIE Kernel

Start the kernel on the AIE tile.

```cpp
XAie_Run(routingInstance, 1);
```

#### e. Wait for Core Execution to Finish

Wait until the kernel execution is complete.

```cpp
while(XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4,4), 0) != XAIE_OK) {
    // Busy wait
}
```

#### f. Get the Result

First move the output data from the AIE tile memory to DDR. Then for the final step of this example we verify it matches the expected result.

```cpp
// Move Data AIE -> DDR
XAie_MoveDataAie2External(routingInstance, /*src=*/ XAie_TileLoc(4,4),
                            CORE_OP_MEM, len*sizeof(u32),
                            out, /*dest=*/ XAie_TileLoc(35,0));
// Verify Result
for (int i = 0; i < len; i++) {
    if (100 + i != out_ptr[i]) {
        printf("Mismatch at index %d: CPU=%d, AIE=%d\n", i, 100 + i, out_ptr[i]);
    }
}
```

## Deploy

### Set up

```bash
source ../script/setup.sh
```

### Compilation Example

The example.cpp file has both the host code and kernel code in the same file. Notice the special attributes/annotations and the \_\_global\_\_ keyword on lines 19-21.

Using Synopsys tools:

```bash
source aiehlc.sh --runtime-source-file example.cpp
```

Using llvm-aie: (experimental)

```bash
source aiehlc.sh --use-llvm-aie --runtime-source-file example.cpp
```

The resulting code is located in ./aout/

```bash
# To view the host code
vim ./aout/host.cc

# This filename matches the kernel function name
# To view the kernel code
vim ./aout/loop_kernel.cc

# Additionally, the generated linker/config files for each kernel are located in their respective directories
ls ./aout/kernelcfg
ls ./aout/kernelcfg/loop_kernel
```

## Full Example: Compile and Run on vek280 (AIE2)

### Set Up

You can skip this step if you did it previously.

```bash
cd ./tutorial
source ../script/setup.sh
```

### Compile example.cpp and Copy the elf to a Network Folder

```bash
source aiehlc.sh --aie-version 2 --runtime-source-file example.cpp
mkdir -p ~/vek280
cp ./aout/main.elf ~/vek280/
```

### Copy the pdi to a Network Folder

```bash
cp ./hw/vek280.pdi ~/vek280/
```

For the next steps you will need to open two consoles.

### Load the pdi on a Board (console 1)

Replace USER with your username.

```bash
# check for a free board
<systest>/bin/cluster-ping vek280

# assuming vek280-2 is free
<systest>/bin/systest vek280-2
power 0 power 1
xsdb
conn
tar 11
device program /home/USER/vek280/vek280.pdi
```

### Run systest client (console 2)

Open a second console and run the following.

```bash
# The hostname for vek280-2 is galerina13, use the corresponding hostname for your board
ssh galerina13
/opt/systest/common/bin/systest-client
connect com0
```

### Run the ELF (console 1)

Go back to the original console where the pdi was loaded.
Replace USER with your username.

```bash
tar 6
rst -proc
dow -force /home/USER/vek280/main.elf
con
```

Now if you look at the systest client console you should see the output for the application.

## Full Example: Compile and Run on vek385 (AIE2PS)

### Set Up

You can skip this step if you did it previously.

```bash
cd ./tutorial
source ../script/setup.sh
```

### Compile example.cpp and Copy the elf to a Network Folder

```bash
source aiehlc.sh --aie-version 5 --runtime-source-file example.cpp
mkdir -p ~/vek385
cp ./aout/main.elf ~/vek385/
```

### Copy the pdi to a Network Folder

```bash
cp ./hw/vek385.pdi ~/vek385/
```

For the next steps you will need to open two consoles.

### Load the pdi on a Board (console 1)

Replace USER with your username.

```bash
# check for a free board
<systest>/bin/cluster-ping vek385

# assuming vek385-6 is free
<systest>/bin/systest vek385-6
power 0 power 1
xsdb
conn
device program /home/USER/vek385/vek385.pdi
```

### Run telnet (console 2)

Open a second console and run the following.

```bash
# The hostname for vek385-6 is gomphus6, use the corresponding hostname for your board
ssh gomphus6
/opt/systest/common/bin/systest-client
connect com3
```

### Run the ELF (console 1)

Go back to the original console where the pdi was loaded.
Replace USER with your username.

```bash
tar 20
rst -proc
dow -force /home/USER/vek385/main.elf
con
```

Now if you look at the telnet console you should see the output for the application.

## Full Example: Compile and Run on Linux, vek280 (AIE2)

### Set Up

You can skip this step if you did it previously.

```bash
cd ./tutorial
source ../script/setup.sh
```

### Compile example.cpp and Copy the elf to a Network Folder

Make sure to include the `--platform linux` flag.

```bash
source aiehlc.sh --platform linux --aie-version 2 --runtime-source-file example.cpp
mkdir -p ~/vek280
cp ./aout/main.elf ~/vek280/linux.elf
```

### Copy the pdi to a Network Folder

```bash
cp ./hw/vek280.pdi ~/vek280/
```

### Copy libxaiengine to a Network Folder

```bash
cp ../thirdparty/alib/aie-rt/driver/src/libxaiengine.so.3 ~/vek280/
```

### Petalinux Set Up

#### Environment Set Up

```bash
mkdir -p ./bootbin/build
source <petalinux-2025.1>/tool/petalinux-v2025.1-final/settings.sh
```

#### Create/Build Project

This step may take some time.

```bash
petalinux-create -t project -s <petalinux-2025.1>/bsp/release/xilinx-vek280-v2025.1-final.bsp
cd xilinx-vek280-2025.1
export PETALINUX_FOLDER=`pwd`
petalinux-devtool modify linux-xlnx
petalinux-devtool modify ai-engine-driver
petalinux-build
```

### Copy Over Files

```bash
cp ./images/linux/* ../bootbin/build/
cd ../bootbin
source ./build.sh
cp ./BOOT.bin ~/vek280/LINUXBOOT.bin
cp ./build/Image ~/vek280/
cp ./build/rootfs.cpio.gz.u-boot ~/vek280/
cp ./build/system.dtb ~/vek280/
```

For the next steps you will need to open two consoles.

### Connect to Board (console 1)

Replace USER with your username.
This only works on board vek280-7 to vek280-12, so choose a non-reserved board in that range.

```bash
# check for a free board
<systest>/bin/cluster-ping vek280

# assuming vek280-8 is free
<systest>/bin/systest vek280-8
tftpd "/home/USER/vek280"
nfsroot3 "/home/USER/vek280"
xsdb
conn
tar 1
device program /home/USER/vek280/LINUXBOOT.bin
```

### Run systest client (console 2)

Open a second console and run the following.

```bash
# The hostname for vek280-8 is truffle7, use the corresponding hostname for your board
ssh truffle7
/opt/systest/common/bin/systest-client
connect com0
dhcp
tftpb 60000 Image
tftpb 20000000 rootfs.cpio.gz.u-boot
tftpb 36000000 system.dtb
booti 60000 20000000 36000000
# login, username/password petalinux/petalinux
sudo su
mkdir /home/petalinux/nfs
mount -t nfs -o vers=3,nolock 10.10.70.101:/exports/root /home/petalinux/nfs
cd nfs && ls
cp libxaiengine.so.3 /usr/lib/
```

### Run the elf (console 2)

```bash
./linux.elf
```

## Development

### Build Tool (Optional)

If you are editing the aiehlc source then you will need to build it manually. If you are just using aiehlc for compilation then this step is not needed.

```bash
mkdir build
pushd build
cmake ..
make
popd
```

<p align="center">Copyright&copy; 2025 Advanced Micro Devices, Inc</p>