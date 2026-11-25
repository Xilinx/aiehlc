#!/usr/bin/env python3
"""Copyright 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
"""
"""
Palboard ELF Test Script

Prerequisites:
- Follow the README to setup run/debug and SSH public key authentication
- Set environment variables: USERNAME and vek385ip
- Install pexpect: pip install pexpect

This script:
1. Creates two SSH connections to vek385ip
2. First connection: sets up xsdb and programs the device
3. Second connection: connects to VEK385_CONSOLE (com3 by default)
4. First connection: downloads the ELF file
5. Captures and prints console output from second connection
"""

import subprocess
import os
import sys
import time
import threading
import queue
import argparse
import glob
import re
import shlex
import importlib.util

try:
    import pexpect
except ImportError:
    print("Error: pexpect module not found. Install with: pip install pexpect")
    sys.exit(1)

# Read username, IP, and board name from environment variables
username = os.environ.get("USERNAME")
vek385ip = os.environ.get("VEK385IP")
boardname = os.environ.get("BOARDNAME")

if not username or not vek385ip:
    # Try to find and source envlocal.sh
    script_dir = os.path.dirname(os.path.abspath(__file__))
    envlocal_path = os.path.join(script_dir, "envlocal.sh")
    
    if os.path.exists(envlocal_path):
        print(f"Environment variables not set. Found envlocal.sh, sourcing it...")
        try:
            # Source the shell script and capture environment variables
            command = f'source "{envlocal_path}" && env'
            result = subprocess.run(
                ['bash', '-c', command],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Parse environment variables from output
                for line in result.stdout.splitlines():
                    if '=' in line:
                        key, _, value = line.partition('=')
                        os.environ[key] = value
                
                # Recheck the environment variables
                username = os.environ.get("USERNAME")
                vek385ip = os.environ.get("VEK385IP")
            else:
                print(f"Warning: Failed to source envlocal.sh: {result.stderr}")
        except Exception as e:
            print(f"Warning: Error sourcing envlocal.sh: {e}")

if not username or not vek385ip:
    print("Error: Please set USERNAME and VEK385IP environment variables")
    print("Example: export USERNAME=aaaaa && export VEK385IP=boardhostname<crimini2>")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Or create {os.path.join(script_dir, 'envlocal.sh')} with these exports")
    sys.exit(1)

host = f"{username}@{vek385ip}"
console_port = os.environ.get("VEK385_CONSOLE", "com3")

# Configuration
PALBOARD_SCRIPTS_DIR = f"/proj/xsjsswstaff/{username}/palboard_scripts"
# Remote PDI/BOOT image programmed with "device program". Overridable so the
# same runner serves other Versal boards that boot identically (e.g. VEK280,
# which differs only in the image name) -- the debug UI passes this through
# from a debug_ui_config.json entry's hw_env.
VEK385PDI = os.environ.get("VEK385PDI") or f"/home/{username}/aiehlc/vek385.pdi"
# Rev B boards run the segmented-configuration vek385_base platform, whose boot
# image is split in two: the boot PDI (PLM, PS, NoC) must be programmed before
# the PLD PDI (PL + AIE), back to back. Rev A boots from the single VEK385PDI.
VEK385_REV = (os.environ.get("VEK385_REV") or "a").lower()
VEK385_BOOT_PDI = (os.environ.get("VEK385_BOOT_PDI")
                   or f"/home/{username}/aiehlc/vek385revb_boot.pdi")
VEK385_PLD_PDI = (os.environ.get("VEK385_PLD_PDI")
                  or f"/home/{username}/aiehlc/vek385revb_pld.pdi")
ALLGMIOBOOT_PDI_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../thirdparty/allgmioboot",
    "platform/_x/link/vivado/vpl/prj/prj.runs/impl_1"))
#XSDB_ALT_PATH = "/everest/set_vnc_bkup/vnc/t50/es1/tools/Labtools/9999.0/bin/xsdb"
XSDB_ALT_PATH = "/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/Vitis/bin/xsdb"
VITIS_SETTINGS = "/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/Vitis/settings64.sh"

# Seconds to wait for "dow -force" to finish downloading the ELF over JTAG.
# Override with AIEHLC_DOWNLOAD_TIMEOUT for an unusually slow link or large ELF.
DOWNLOAD_TIMEOUT_S = int(os.environ.get("AIEHLC_DOWNLOAD_TIMEOUT", "240"))

# Queue to collect console output from second connection
console_output_queue = queue.Queue()
stop_console_thread = threading.Event()


def find_elf_file(filename=None, auto_yes=False):
    """
    Smart ELF file finder:
    - Full path given: use it directly
    - Relative path given: try current directory first
    - Not found: search for filename in current folder and subfolders, ask which one
    - No argument: show default aout/main.elf and ask for confirmation
    - auto_yes=True: skip all interactive prompts, auto-confirm
    Returns: full path to ELF file or None to exit
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_elf = os.path.normpath(os.path.join(script_dir, "../../", "aout", "main.elf"))
    current_dir = os.getcwd()

    def ask_confirm(file_path, prompt_msg):
        """Helper to ask user confirmation for a file."""
        print(f"\n{prompt_msg}")
        print(f"  {file_path}")
        if auto_yes:
            print("[--yes] Auto-confirmed.")
            return file_path
        response = input("\nDo you want to use this file? [y/n]: ").strip().lower()
        if response in ('y', 'yes'):
            return file_path
        else:
            print("Exiting...")
            return None
    
    if filename is None:
        # No argument provided - show default and ask for confirmation
        if os.path.exists(default_elf):
            return ask_confirm(default_elf, "No ELF file specified. Default ELF file found:")
        else:
            print(f"\nError: Default ELF file not found: {default_elf}")
            print("Please specify an ELF file as argument.")
            return None
    
    # Filename provided - check if it's a full path
    if os.path.isabs(filename):
        # Absolute path provided - use it directly
        if os.path.exists(filename):
            print(f"Using: {filename}")
            return filename
        else:
            print(f"\nError: File not found: {filename}")
            return None
    
    # Relative path provided - try current directory first
    current_path = os.path.join(current_dir, filename)
    if os.path.exists(current_path):
        full_path = os.path.abspath(current_path)
        print(f"Using: {full_path}")
        return full_path
    
    # File not found in current directory - search for the filename
    search_name = os.path.basename(filename)
    print(f"\nFile not found: {filename}")
    print(f"Searching for '{search_name}' in current directory and subdirectories...")
    
    # Use glob to find all matching files recursively
    pattern = os.path.join(current_dir, "**", search_name)
    matches = glob.glob(pattern, recursive=True)
    
    if not matches:
        print(f"\nNo files named '{search_name}' found.")
        # Fall back to default main.elf and ask
        if os.path.exists(default_elf):
            return ask_confirm(default_elf, "Would you like to use the default ELF file instead?")
        else:
            print(f"Error: Default ELF file also not found: {default_elf}")
            return None
    
    # Normalize all paths
    matches = [os.path.normpath(m) for m in matches]
    
    # Single match - show path and ask for confirmation
    if len(matches) == 1:
        return ask_confirm(matches[0], "Found one matching file:")
    
    # Multiple matches - ask user to choose (or auto-pick first in --yes mode)
    print(f"\nFound {len(matches)} matching files:")
    for i, match in enumerate(matches, 1):
        print(f"  [{i}] {match}")

    if auto_yes:
        print(f"[--yes] Auto-selecting first match: {matches[0]}")
        return matches[0]

    print(f"  [0] Exit")

    while True:
        try:
            choice = input("\nSelect file number (or 0 to exit): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                print("Exiting...")
                return None
            elif 1 <= choice_num <= len(matches):
                return matches[choice_num - 1]
            else:
                print(f"Invalid choice. Please enter 0-{len(matches)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def console_reader(child, output_queue, stop_event):
    """Thread function to continuously read console output from Connection 2."""
    buffer = ""
    tail = ""
    while not stop_event.is_set():
        try:
            # Read any available output without blocking for long
            output = child.read_nonblocking(size=4096, timeout=1)
            if output:
                buffer += output
                output_queue.put(output)
                window = tail + output
                tail = window[-16:]
                if "Connection closed by foreign host" in window:
                    output_queue.put("[console_reader] UART session dropped\n")
                    tail = ""
                elif "telnet>" in window:
                    child.sendline("quit")
                    output_queue.put("[console_reader] telnet command mode, quit; UART session dropped\n")
                    tail = ""
        except pexpect.TIMEOUT:
            # No data available, continue polling
            continue
        except pexpect.EOF:
            output_queue.put("[Connection 2 EOF]")
            break
        except Exception as e:
            output_queue.put(f"[Console reader error: {e}]")
            break
    
    # Put any remaining buffer content
    if buffer:
        output_queue.put(f"[Total bytes read: {len(buffer)}]")


def exit_xsdb_and_poweroff(child):
    """Exit xsdb cleanly (stop target first) and power off the board."""
    # Drain stale output
    try:
        child.read_nonblocking(size=65536, timeout=2)
    except pexpect.TIMEOUT:
        pass

    # Stop the running target before exiting — xsdb may refuse to exit
    # while a target is in Running state.
    child.sendline("stop")
    try:
        child.expect(r'xsdb%', timeout=60)
    except pexpect.TIMEOUT:
        print("[Connection 1] Warning: stop command timed out, continuing cleanup...")

    # Drain any extra output (e.g. _exit.c source-not-found messages)
    time.sleep(1)
    try:
        child.read_nonblocking(size=65536, timeout=2)
    except pexpect.TIMEOUT:
        pass

    child.sendline("exit")
    time.sleep(2)
    try:
        child.expect([r'Systest[#>]', r'\$\s*$', r'>\s*$'], timeout=60)
    except pexpect.TIMEOUT:
        print("[Connection 1] Warning: exit xsdb timed out, continuing cleanup...")
    print("[Connection 1] Exited xsdb, powering off...")

    child.sendline("power 0")
    try:
        child.expect(r'Systest[#>]', timeout=60)
    except pexpect.TIMEOUT:
        print("[Connection 1] Warning: power off timed out")
    print("[Connection 1] Power off complete")


def boot_pdis(board_rev):
    """Remote PDIs to `device program`, in order, for this board revision."""
    if board_rev == "b":
        return [VEK385_BOOT_PDI, VEK385_PLD_PDI]
    return [VEK385PDI]


def program_pdi(child, pdi):
    """`device program` one PDI; raise on a PLM stall or an xsdb error."""
    print(f"[Connection 1] Programming PDI: {pdi}")
    # Wait on a result marker, not the next xsdb% prompt: earlier commands
    # (`conn` launching hw_server) leave extra prompts that would match early.
    # `[set r OK]` keeps the echoed command line from matching the marker.
    child.sendline(f'if {{[catch {{device program {pdi}}} e]}} '
                   f'{{puts "PDIRES_[set r FAIL]: $e"}} else {{puts "PDIRES_[set r OK]"}}')
    index = child.expect([r'PDIRES_OK', r'PDIRES_FAIL: ([^\r\n]*)', r'PLM stalled'], timeout=300)
    if index == 2:
        child.expect(r'xsdb%', timeout=60)
        raise RuntimeError(
            f"PLM stalled while programming {pdi}. "
            "The board may need a power cycle. Run 'plm log' for details."
        )
    err = child.match.group(1).strip() if index == 1 else ""
    child.expect(r'xsdb%', timeout=60)
    if index == 1:
        raise RuntimeError(f"device program {pdi} failed: {err}")


def stage_pdis(local_dir, board_rev):
    """SCP rev B's boot/PLD PDIs from local_dir to the remote paths xsdb programs."""
    if board_rev != "b":
        print("Error: --stage-pdi only applies to --board-rev b")
        return False
    pairs = []
    for suffix, remote in (("_boot.pdi", VEK385_BOOT_PDI), ("_pld.pdi", VEK385_PLD_PDI)):
        hits = sorted(glob.glob(os.path.join(local_dir, f"*{suffix}")))
        if len(hits) != 1:
            print(f"Error: expected exactly one *{suffix} in {local_dir}, found {hits}")
            return False
        pairs.append((hits[0], remote))
    try:
        remote_dirs = sorted({os.path.dirname(r) for _, r in pairs})
        subprocess.run(["ssh", host, "mkdir -p " + " ".join(map(shlex.quote, remote_dirs))],
                       check=True, capture_output=True, text=True, timeout=30)
        for local, remote in pairs:
            print(f">>> Staging {local} -> {host}:{remote}")
            subprocess.run(["scp", local, f"{host}:{remote}"],
                           check=True, capture_output=True, text=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error staging PDIs: {e}")
        if getattr(e, "stderr", None):
            print(e.stderr.strip())
        return False
    return True


def setup_first_connection(board_rev="a"):
    """
    First SSH connection: Setup xsdb and program device.
    Returns the pexpect child process for further commands.
    """
    print(f"[Connection 1] Connecting to {host}...")
    
    # Start SSH with X forwarding
    child = pexpect.spawn(f"ssh -X {host}", encoding='utf-8', timeout=60)
    child.logfile_read = sys.stdout
    
    # Wait for shell prompt
    child.expect([r'\$\s*$', r'#\s*$', r'>\s*$'], timeout=60)
    print("[Connection 1] Connected, starting systest...")
    
    # Step 2: Run systest
    child.sendline("/opt/systest/common/bin/systest-client")
    child.expect(r'Systest[#>]', timeout=60)
    print("[Connection 1] In systest, becoming palboard...")
    
    
    # Step 4a: Power off first to ensure clean state
    child.sendline("power 0")
    child.expect(r'Systest[#>]', timeout=60)
    time.sleep(2)  # Wait for power off to complete
    print("[Connection 1] Power off complete, powering on...")
    
    # Step 4b: Power on
    child.sendline("power 1")
    child.expect(r'Systest[#>]', timeout=60)  # Wait for power cycle to complete
    time.sleep(3)  # Extra wait for board to initialize
    print("[Connection 1] Power on complete, sourcing Vitis and starting xsdb...")

    # Step 5b: Source Vitis environment so xsdb is on PATH
    child.sendline(f"source {VITIS_SETTINGS}")
    child.expect(r'Systest[#>]', timeout=30)

    # Step 6: Start xsdb (try default first, then alternative path)
    child.sendline("xsdb")
    index = child.expect([r'xsdb%', r'command not found', r'Unrecognized', pexpect.TIMEOUT], timeout=60)

    if index != 0:
        print("[Connection 1] xsdb not found / timed-out, exiting and trying alternative path...")
        child.sendline("exit")
        try:
            child.expect(r'Systest[#>]', timeout=15)
        except pexpect.TIMEOUT:
            child.sendcontrol('c')
            time.sleep(1)
            child.sendline("exit")
            child.expect(r'Systest[#>]', timeout=15)
        child.sendline(XSDB_ALT_PATH)
        child.expect(r'xsdb%', timeout=60)

    print("[Connection 1] In xsdb, connecting...")

    # Step 7: Connect
    child.sendline("conn")
    child.expect(r'xsdb%', timeout=60)
    print("[Connection 1] Connected, targeting device 1...")

    # Step 8: Target 1
    child.sendline("tar 1")
    child.expect(r'xsdb%', timeout=60)
    # Step 9: Program device (rev B: boot PDI, then PLD PDI)
    for pdi in boot_pdis(board_rev):
        program_pdi(child, pdi)
    print("[Connection 1] PDI programmed, waiting for PS POR release...")
    time.sleep(15)  # allow PLM to fully initialise before continuing
    print("[Connection 1] Targeting APU core (Cortex-A78AE #0.0)...")

    # Step 10: Target 20
    child.sendline("tar 20")
    child.expect(r'xsdb%', timeout=30)
    print("[Connection 1] Resetting processor...")

    # Step 11: Reset processor
    child.sendline("tar 18")
    child.expect(r'xsdb%', timeout=30)
    child.sendline("rst -cores")
    child.expect(r'xsdb%', timeout=60)
    child.sendline("tar 20")
    child.expect(r'xsdb%', timeout=30)

    # Drain any stale xsdb% prompts left in the buffer.
    # xsdb sometimes emits double prompts after rst -proc; if the second
    # one isn't consumed here it will be falsely matched by the next expect.
    time.sleep(0.5)
    try:
        child.read_nonblocking(size=4096, timeout=1)
    except pexpect.TIMEOUT:
        pass

    print("[Connection 1] Setup complete!")
    
    return child


def uart_reconnect_step(conn, chunk, st):
    dropped = "Connection closed by foreign host" in chunk or "UART session dropped" in chunk
    if dropped:
        st["pending"] = True
    if st.get("pending") and re.search(r'Systest[#>]', chunk) and st.get("count", 0) < 4:
        st["pending"] = False
        st["count"] = st.get("count", 0) + 1
        print(f"\n[Connection 2] UART dropped, reconnecting to {console_port} ({st['count']})")
        conn.sendline(f"connect {console_port}")
    return dropped


def setup_second_connection():
    """
    Second SSH connection: Connect to com0 for console output.
    Returns the pexpect child process.
    """
    print(f"[Connection 2] Connecting to {host}...")
    
    # Start SSH with X forwarding
    child = pexpect.spawn(f"ssh -X {host}", encoding='utf-8', timeout=60)
    
    # Wait for shell prompt
    child.expect([r'\$\s*$', r'#\s*$', r'>\s*$'], timeout=60)
    print("[Connection 2] Connected, starting systest...")

    # Step 2: Run systest
    child.sendline("/opt/systest/common/bin/systest-client")
    child.expect(r'Systest[#>]', timeout=60)
    print(f"[Connection 2] In systest, connecting to {console_port}...")

    # Step 3: Connect to the selected console (no output until ELF runs).
    child.sendline(f"connect {console_port}")
    child.expect(rf'Connecting to device {console_port}.*escape', timeout=60)
    print(f"[Connection 2] Connected to {console_port}, listening for output...")

    return child


def download_elf_and_continue(child, elf_path):
    """
    Download the ELF file and continue execution on the first connection.
    Step 12: dow -force <elf_path>
    Step 13: con
    """
    print(f"[Connection 1] Downloading ELF: {elf_path}")

    # Drain stale output from prior commands (e.g. "100%" left over from
    # "device program BOOT.BIN") so they cannot be matched by the expects below.
    try:
        child.read_nonblocking(size=65536, timeout=1)
    except pexpect.TIMEOUT:
        pass

    child.sendline(f"dow -force {elf_path}")

    # Wait for "Successfully downloaded" – this string is emitted only by
    # "dow", never by "device program BOOT.BIN".  Using "100%" was unsafe
    # because device program also emits "100%" which can be buffered and
    # matched prematurely, causing "con" to be sent at 47% download and
    # corrupting the ELF load.
    # Also detect PLM stall — if PLM didn't boot properly, the ELF download
    # aborts and xsdb prints "PLM stalled during programming".  Without this
    # check the script would retry after rst -proc, the download succeeds but
    # UART/PS peripherals are never initialized so we get zero console output
    # and waste the download budget waiting.
    #
    # Timeout sized for a slow JTAG link: observed ~0.1MB/s on a loaded board,
    # where a ~10MB ELF needs ~130s and the old 120s budget expired at ~65%.
    # The expect returns as soon as the match arrives, so a generous ceiling
    # costs nothing on a fast link.
    index = child.expect([r'Successfully downloaded', r'PLM stalled'], timeout=DOWNLOAD_TIMEOUT_S)
    if index == 1:
        # Consume remaining output up to the prompt
        child.expect(r'xsdb%', timeout=60)
        print("\n[ERROR] PLM stalled during ELF download (dow -force).")
        print("BOOT.BIN did not initialize PS peripherals (UART etc.).")
        print("The board needs a full power cycle and re-programming of BOOT.BIN.")
        print("Run 'plm log' in xsdb for details.")
        return False
    child.expect(r'xsdb%', timeout=60)
    print("[Connection 1] ELF download complete!")

    # Step 13: Continue execution
    print("[Connection 1] Continuing execution...")
    child.sendline("con")
    child.expect(r'xsdb%', timeout=60)
    print("[Connection 1] Execution started!")
    return True


# The build produces the same binary under two names -- `build/host` (the linker
# output) and `aout/main.elf` (the copy) -- and the debug UI's ELF search may pick
# either one, so the board only ever received whichever name won. A `dow -force`
# typed against the other name then hit a stale leftover, or nothing at all.
# Whichever of the pair is staged, publish it under both.
ELF_ALIAS_NAMES = ("host", "main.elf")


def elf_alias_names(elf_filename):
    """The other remote names this ELF should also be published under.

    Only the host/main.elf pair, and only when the staged file IS one of them:
    aliasing an unrelated ELF (perf.elf) would silently clobber a main.elf that
    has nothing to do with it.
    """
    if elf_filename not in ELF_ALIAS_NAMES:
        return []
    return [n for n in ELF_ALIAS_NAMES if n != elf_filename]


def publish_elf_aliases(dest_dir, dest_elf):
    """Copy the just-staged ELF to its alias names on the board.

    A remote `cp` rather than a second SCP: the ELF is ~17 MB, and copying from
    the file we just landed is what guarantees the alias holds the bits `dow`
    will load -- re-uploading the local file leaves room for the two to differ.
    Alias failure is a warning, not an error: the primary path is already
    staged and the run can proceed.
    """
    published = []
    for alias in elf_alias_names(os.path.basename(dest_elf)):
        alias_path = os.path.join(dest_dir, alias)
        try:
            subprocess.run(
                ["ssh", host,
                 f"cp -f {shlex.quote(dest_elf)} {shlex.quote(alias_path)}"],
                check=True, capture_output=True, text=True, timeout=60
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Warning: could not publish alias {alias_path}: {e}")
            continue
        print(f"    Also staged as: {host}:{alias_path}")
        published.append(alias_path)
    return published


def copy_elf_to_remote(local_elf):
    """Copy ELF file to remote /home/{username}/aiehlc/ via SCP."""
    dest_dir = f"/home/{username}/aiehlc"
    elf_filename = os.path.basename(local_elf)
    dest_elf = os.path.join(dest_dir, elf_filename)

    print(f">>> Copying ELF file via SCP...")
    print(f"    Source: {local_elf}")
    print(f"    Destination: {host}:{dest_elf}")

    if not os.path.exists(local_elf):
        print(f"Error: Local ELF file not found: {local_elf}")
        return False, None

    try:
        # Ensure remote directory exists
        subprocess.run(
            ["ssh", host, f"mkdir -p {dest_dir}"],
            check=True, capture_output=True, text=True, timeout=30
        )
        # SCP the file
        result = subprocess.run(
            ["scp", local_elf, f"{host}:{dest_elf}"],
            check=True, capture_output=True, text=True, timeout=120
        )
        print(">>> ELF file copied successfully via SCP")
        publish_elf_aliases(dest_dir, dest_elf)
        # The primary path is what gets downloaded; the alias exists so a
        # hand-typed dow against the other name loads the same build.
        return True, dest_elf
    except subprocess.CalledProcessError as e:
        print(f"Error copying ELF file via SCP: {e}")
        if e.stderr:
            print(f"    stderr: {e.stderr}")
        return False, None
    except subprocess.TimeoutExpired:
        print("Error: SCP timed out")
        return False, None


# --netload: run the ELF through the resident netboot supervisor instead of JTAG dow.
# For VEK280 / galerina13, the supervisor is built with --a72 (Cortex-A72 BSP).
# ---------------------------------------------------------------------------
NETBOOT_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "tool", "netboot")
NETBOOT_REMOTE_DIR = f"/home/{username}/aiehlc/netboot"
NETBOOT_READY_TIMEOUT_S = 120
# Multiplexed ssh: one persistent connection reused by all scp/ssh in the netload path.
SSH_MUX = ["-o", "ControlMaster=auto", "-o", "ControlPersist=4h",
           "-o", f"ControlPath=/tmp/appvek385-{username}-%C"]


def ensure_netboot_built():
    """Build netboot_a72.elf when it is missing or older than its sources."""
    proxy = os.path.join(NETBOOT_LOCAL_DIR, "netboot_proxy.py")
    if not os.path.isfile(proxy):
        raise RuntimeError(f"{proxy} missing (it is part of the repo; check the checkout)")
    elf = os.path.join(NETBOOT_LOCAL_DIR, "build", "netboot_a72.elf")
    srcs = [p for pat in ("supervisor/*.c", "supervisor/*.S", "supervisor/*.h", "lwip/*",
                          "bsp_a72/*.h", "netboot_proto.h", "build.sh")
            for p in glob.glob(os.path.join(NETBOOT_LOCAL_DIR, pat))]
    newest = max((os.path.getmtime(p) for p in srcs), default=0)
    if os.path.isfile(elf) and os.path.getmtime(elf) >= newest:
        return
    why = "missing" if not os.path.isfile(elf) else "older than its sources"
    print(f"[netload] netboot_a72.elf {why}: building it (src/tool/netboot/build.sh --a72)...")
    r = subprocess.run(["bash", os.path.join(NETBOOT_LOCAL_DIR, "build.sh"), "--a72"],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.isfile(elf):
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"building netboot_a72.elf failed (exit {r.returncode}):\n{tail}\n"
                           "The A72 cross compiler must be on PATH (source Vitis settings64.sh).")
    print(f"[netload] built {elf}")


def stage_if_changed(local_paths):
    """Copy files to NETBOOT_REMOTE_DIR on the lab host, skipping unchanged ones."""
    import hashlib
    names = [os.path.basename(p) for p in local_paths]
    probe = (f"mkdir -p {NETBOOT_REMOTE_DIR} && cd {NETBOOT_REMOTE_DIR} && "
             f"(sha256sum {' '.join(shlex.quote(n) for n in names)} 2>/dev/null; true)")
    r = subprocess.run(["ssh", *SSH_MUX, host, f"bash -c {shlex.quote(probe)}"],
                       capture_output=True, text=True, timeout=30)
    remote = {ln.split()[1]: ln.split()[0] for ln in r.stdout.splitlines() if len(ln.split()) == 2}
    stale = [p for p, n in zip(local_paths, names)
             if remote.get(n) != hashlib.sha256(open(p, "rb").read()).hexdigest()]
    if stale:
        subprocess.run(["scp", "-q", *SSH_MUX, *stale, f"{host}:{NETBOOT_REMOTE_DIR}/"],
                       check=True, capture_output=True, timeout=120)
    print(f"[netload] staged on {vek385ip}: {', '.join(os.path.basename(p) for p in stale) or 'nothing (up to date)'}")


def netboot_stage_proxy():
    """Stage the relay script that netload.py talks through."""
    stage_if_changed([os.path.join(NETBOOT_LOCAL_DIR, "netboot_proxy.py")])


def netboot_stage_supervisor():
    """Stage netboot_a72.elf for a JTAG bootstrap and return its remote path."""
    local_elf = os.path.join(NETBOOT_LOCAL_DIR, "build", "netboot_a72.elf")
    if not os.path.isfile(local_elf):
        raise RuntimeError(f"{local_elf} missing: run src/tool/netboot/build.sh --a72 first")
    stage_if_changed([local_elf])
    return f"{NETBOOT_REMOTE_DIR}/netboot_a72.elf"


_netload_mod = []


class _Tee:
    """stdout for an in-process netload: shown live AND kept for the caller."""
    def __init__(self, live):
        import io
        self.buf, self.live = io.StringIO(), live

    def write(self, text):
        self.buf.write(text)
        if self.live:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
        return len(text)

    def flush(self):
        pass


def netload_remote(argv, timeout=600, gate=None, live=False):
    """Run netload.py's main() in-process through the multiplexed ssh.

    Returns (exit status, printed text). live=True also prints as they come."""
    import contextlib
    if not _netload_mod:
        spec = importlib.util.spec_from_file_location(
            "netload", os.path.join(NETBOOT_LOCAL_DIR, "netload.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _netload_mod.append(mod)
    out = _Tee(live)
    extra = ["--via", host, "--proxy", f"{NETBOOT_REMOTE_DIR}/netboot_proxy.py"]
    extra += [x for i in range(1, len(SSH_MUX), 2) for x in ("--ssh-opt", SSH_MUX[i])]
    mod = _netload_mod[0]
    mod.send_gate = gate
    try:
        with contextlib.redirect_stdout(out):
            rc = mod.main(argv + extra)
    finally:
        mod.send_gate = None
    return rc, out.buf.getvalue().strip()


def open_console_fast():
    """Open the board console quickly over the multiplexed ssh."""
    child = pexpect.spawn("ssh", [*SSH_MUX, "-tt", host, "/opt/systest/common/bin/systest-client"],
                          encoding='utf-8', codec_errors='replace', timeout=60)
    child.expect(r'Systest[#>]', timeout=60)
    child.sendline(f"connect {console_port}")
    child.expect(rf'Connecting to device {console_port}.*escape', timeout=60)
    return child


def start_console(fast):
    conn2 = open_console_fast() if fast else setup_second_connection()
    stop_console_thread.clear()
    th = threading.Thread(target=console_reader, args=(conn2, console_output_queue, stop_console_thread))
    th.start()
    return conn2, th


def stop_console(conn2, th):
    stop_console_thread.set()
    th.join(timeout=5)
    conn2.close(force=True)


def drain_until(marker, timeout_s, collected):
    """Echo console output until `marker` appears; True if it did."""
    deadline = time.time() + timeout_s
    seen = ""
    while time.time() < deadline:
        while not console_output_queue.empty():
            chunk = console_output_queue.get()
            collected.append(chunk)
            print(chunk, end='', flush=True)
            seen = (seen + chunk)[-4096:]
            if marker in seen:
                return True
        time.sleep(0.2)
    return False


def netboot_park_app_core(park_addr):
    """Free a stuck app core (A72 #0.1) without a full re-bootstrap.

    Uses local JTAG (conn) rather than TCP, because VEK280 is JTAG-attached on
    the lab host (no remote hw_server needed). Returns True when the commands
    went through (the caller checks `status`)."""
    print(f"\n>>> freeing the stuck app core over JTAG (stop, rst -proc, pc=0x{park_addr:x} nb_park, con)...",
          flush=True)
    child = None
    try:
        child = pexpect.spawn("ssh", [*SSH_MUX, "-tt", host, "/opt/systest/common/bin/systest-client"],
                              encoding='utf-8', codec_errors='replace', timeout=60, dimensions=(200, 2000))
        child.expect(r'Systest[#>]', timeout=60)
        child.sendline(f"source {VITIS_SETTINGS}")
        child.expect(r'Systest[#>]', timeout=60)
        child.sendline("xsdb")
        if child.expect([r'xsdb%', r'command not found', pexpect.TIMEOUT], timeout=30) != 0:
            child.sendline(XSDB_ALT_PATH)
            child.expect(r'xsdb%', timeout=60)
        # VEK280: local JTAG, no connect -url needed
        child.sendline("conn")
        child.expect(r'xsdb%', timeout=60)
        child.sendline('targets -set -nocase -filter {name =~ "*A72*#0.1*"}')
        child.expect(r'xsdb%', timeout=30)
        child.sendline("catch {stop}")
        child.expect(r'xsdb%', timeout=30)
        child.sendline("rst -proc")
        child.expect(r'xsdb%', timeout=30)
        child.sendline(f"rwr pc 0x{park_addr:x}")
        child.expect(r'xsdb%', timeout=30)
        child.sendline("con")
        child.expect(r'xsdb%', timeout=30)
        child.sendline("disconnect")
        child.expect(r'xsdb%', timeout=30)
        return True
    except (RuntimeError, pexpect.ExceptionPexpect) as e:
        print(f"[netload] JTAG park of the app core failed: {e}")
        return False
    finally:
        if child is not None:
            child.close(force=True)


def netload_start(args, local_elf, collected):
    """Fast path: console up, then `load` to the resident supervisor.

    Falls back to a JTAG bootstrap when the supervisor is not answering or
    --netload-bootstrap is given. Returns (conn1, conn2, console_thread, ok)."""
    ip = args.netload_ip
    ensure_netboot_built()
    force_now = args.netload_force_now
    load_argv = ["load", local_elf, "--host", ip, "--max-run-ms", str(args.netload_max_run_ms),
                 "--wait", "0" if force_now else str(args.netload_wait)] + \
        ([] if args.netload_no_force and not force_now else ["--force"])
    if not args.netload_bootstrap:
        box = {}
        opener = threading.Thread(target=lambda: box.setdefault("c", start_console(fast=True)))
        opener.start()
        netboot_stage_proxy()

        def console_attached():
            opener.join(timeout=60)
            if "c" not in box:
                raise ConnectionError(f"{console_port} console did not attach")

        t0 = time.time()
        print(f"[netload] asking the supervisor at {ip} to run {os.path.basename(local_elf)} " +
              ("(force now: a running app is taken over)" if force_now
               else f"(waits up to {args.netload_wait:g} s if the board is busy)"), flush=True)
        rc, out = netload_remote(load_argv, gate=console_attached, live=True)
        opener.join(timeout=60)
        if "c" not in box:
            print(f"\n[ERROR] {console_port} console did not attach")
            return None, None, None, False
        conn2, th = box["c"]
        if rc == 0:
            print(f"[netload] streamed {local_elf} through the proxy on {vek385ip} in {time.time() - t0:.2f}s")
            return None, conn2, th, True
        if rc == 3 and not force_now:
            print(f"\n[ERROR] board still busy after --netload-wait {args.netload_wait:g} s with a run "
                  "within its max run time; retry later or raise --netload-wait")
            return None, conn2, th, False
        if rc == 4:
            print("\n[ERROR] the previous app ran past its max run time; drop --netload-no-force to terminate it")
            return None, conn2, th, False
        if rc not in (2, 5) and not (rc == 3 and force_now) and "re-bootstrap" not in out:
            return None, conn2, th, False
        m = re.search(r"park_addr=0x([0-9a-f]+)", out)
        if rc == 3:
            print("[netload] --netload-force-now: taking the app core over", flush=True)
        if rc in (3, 5) and m and netboot_park_app_core(int(m.group(1), 16)):
            time.sleep(0.5)
            st_rc, st = netload_remote(["status", "--host", ip, "--connect-timeout", "3"], timeout=30)
            if st_rc == 0 and "app core parked" in st:
                print("[netload] app core parked by JTAG; sending the ELF again", flush=True)
                rc, out = netload_remote(load_argv, live=True)
                if rc == 0:
                    return None, conn2, th, True
            print(f"[netload] JTAG park did not free the app core; falling back to re-bootstrap", flush=True)
        stop_console(conn2, th)
        why = (f"supervisor not answering at {ip}" if rc == 2 else
               "previous app past its max run time and PDI cannot stop it" if rc == 5
               else "--netload-force-now: running app could not be parked over JTAG" if rc == 3
               else "app core still busy from the previous run")
    else:
        why = "--netload-bootstrap"
    print(f"\n>>> {why}: loading the netboot supervisor over JTAG (one-time)...")
    netboot_stage_proxy()
    # power-cycle + PDI program + open xsdb (same flow as a normal run)
    conn1 = setup_first_connection(board_rev=args.board_rev)
    conn2, th = start_console(fast=False)
    if not download_elf_and_continue(conn1, netboot_stage_supervisor()):
        return conn1, conn2, th, False
    if not drain_until("[netboot] ready", NETBOOT_READY_TIMEOUT_S, collected):
        print(f"\n[ERROR] supervisor did not print '[netboot] ready' within {NETBOOT_READY_TIMEOUT_S}s")
        return conn1, conn2, th, False
    rc, out = netload_remote(load_argv, live=True)
    return conn1, conn2, th, rc == 0


def main():
    """Main test function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Palboard ELF Test Script - Run ELF files on palboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Ask to use default aout/main.elf
  %(prog)s /full/path/test.elf  # Use full path directly
  %(prog)s mykernel.elf         # Try current dir, then search and ask
  %(prog)s ./build/test.elf     # Use relative path from current directory
  %(prog)s -y                   # Use default aout/main.elf without prompting
  %(prog)s -y /path/to/main.elf # Use custom ELF without prompting
  %(prog)s -nonreboot test.elf  # Run ELF but keep board on for manual debug
  %(prog)s --board-rev b        # Rev B: program boot PDI, then PLD PDI
  %(prog)s main.elf --board-rev b --stage-pdi   # Rev B, first SCP freshly built allgmioboot PDIs
        """
    )
    parser.add_argument(
        "elf_file",
        nargs="?",
        default=None,
        help="Path to ELF file (optional, will prompt for default if not specified)"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        default=True,
        help="Non-interactive mode: auto-confirm all prompts"
    )
    parser.add_argument(
        "-nonreboot", "--nonreboot",
        action="store_true",
        default=False,
        help="Keep board powered on after test (no xsdb exit, no power off) for manual debug"
    )
    parser.add_argument(
        "--board-rev",
        choices=("a", "b"),
        default=VEK385_REV,
        help="VEK385 board revision (default: $VEK385_REV or 'a'). "
             "a: program $VEK385PDI. b: program $VEK385_BOOT_PDI then $VEK385_PLD_PDI"
    )
    parser.add_argument(
        "--stage-pdi",
        nargs="?",
        const=ALLGMIOBOOT_PDI_DIR,
        default=None,
        metavar="DIR",
        help="Rev B only: after rebuilding the PDIs, SCP the *_boot.pdi and *_pld.pdi "
             "from DIR over $VEK385_BOOT_PDI / $VEK385_PLD_PDI before programming "
             f"(default DIR: {ALLGMIOBOOT_PDI_DIR})"
    )
    parser.add_argument(
        "--netload",
        action="store_true",
        default=False,
        help="Run via the resident netboot supervisor instead of JTAG dow; bootstraps the "
             "supervisor over JTAG only if it does not answer (src/tool/netboot/README.md)"
    )
    parser.add_argument(
        "--netload-ip",
        default=os.environ.get("VEK280_NETBOOT_IP", ""),
        help="Supervisor IP (set by build.sh NETBOOT_IP; or $VEK280_NETBOOT_IP)"
    )
    parser.add_argument(
        "--netload-bootstrap",
        action="store_true",
        default=False,
        help="Reload the supervisor over JTAG even if one answers (supervisor upgrade)"
    )
    parser.add_argument(
        "--netload-max-run-ms",
        "--netload-timeout-ms",
        dest="netload_max_run_ms",
        type=int,
        default=0,
        help="Max run time of this app on the board (--netload only); 0 = supervisor default (60 s)"
    )
    parser.add_argument(
        "--netload-wait",
        type=float,
        default=120.0,
        help="Seconds to wait for a busy board before giving up (--netload only, default 120)"
    )
    parser.add_argument(
        "--netload-force-now",
        action="store_true",
        default=False,
        help="Take over the app core immediately without waiting (--netload only)"
    )
    parser.add_argument(
        "--netload-no-force",
        action="store_true",
        default=False,
        help="Do not terminate a previous app past its max run time (--netload only)"
    )
    args = parser.parse_args()
    if args.board_rev not in ("a", "b"):
        print(f"Error: VEK385_REV must be 'a' or 'b', got '{args.board_rev}'")
        sys.exit(1)
    if args.netload and not args.netload_ip:
        print("Error: --netload requires --netload-ip <ip> or $VEK280_NETBOOT_IP")
        sys.exit(1)

    # Step 0: Find ELF file using smart selection
    local_elf = find_elf_file(args.elf_file, auto_yes=args.yes)
    if local_elf is None:
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Palboard ELF Test Script")
    print(f"Host: {host}")
    print(f"ELF File: {local_elf}")
    print(f"Board rev: {args.board_rev.upper()}  PDI(s): {' -> '.join(boot_pdis(args.board_rev))}")
    print("=" * 60)

    if args.stage_pdi and not stage_pdis(args.stage_pdi, args.board_rev):
        sys.exit(1)

    # --netload streams the ELF through the proxy; skip the SCP step.
    remote_elf_path = None
    if not args.netload:
        success, remote_elf_path = copy_elf_to_remote(local_elf)
        if not success:
            print("Failed to copy ELF file, exiting...")
            sys.exit(1)

    conn1 = None
    conn2 = None
    console_thread = None
    collected_output = []
    uart_st = {}

    try:
        if args.netload:
            conn1, conn2, console_thread, elf_ok = netload_start(args, local_elf, collected_output)
            if not elf_ok:
                print("\n" + "=" * 60)
                print("Test FAILED (netload)")
                print("=" * 60)
                sys.exit(1)
        else:
            # JTAG path: program PDI, open console, dow ELF, continue.
            print("\n>>> Setting up first connection...")
            conn1 = setup_first_connection(board_rev=args.board_rev)

            print("\n>>> Setting up second connection...")
            conn2 = setup_second_connection()

            console_thread = threading.Thread(
                target=console_reader,
                args=(conn2, console_output_queue, stop_console_thread)
            )
            console_thread.start()
            uart_ready = False
            uart_seen = 0.0
            settle_deadline = time.time() + 45
            while time.time() < settle_deadline:
                while not console_output_queue.empty():
                    chunk = console_output_queue.get()
                    print(chunk, end='', flush=True)
                    if uart_reconnect_step(conn2, chunk, uart_st):
                        uart_seen = 0.0
                    elif not uart_seen and not uart_st.get("pending") and chunk.strip():
                        uart_seen = time.time()
                if uart_seen and time.time() - uart_seen >= 3 and not uart_st.get("pending"):
                    uart_ready = True
                    break
                time.sleep(0.5)
            if not uart_ready:
                print("\n[Connection 2] UART did not stay up; continuing anyway")

            print("\n>>> Downloading ELF file...")
            elf_ok = download_elf_and_continue(conn1, remote_elf_path)

            if not elf_ok:
                print("\n[Skipping console wait due to PLM failure]")
                if not args.nonreboot:
                    print("\n>>> Cleaning up Connection 1...")
                    exit_xsdb_and_poweroff(conn1)
                else:
                    print("\n>>> --nonreboot: board stays powered on for debug.")
                    print(f"    Kill this process when done: Ctrl+C or 'kill {os.getpid()}'")
                    try:
                        while True:
                            time.sleep(60)
                    except KeyboardInterrupt:
                        print("\nReleasing board...")
                print("\n" + "=" * 60)
                print("Test FAILED (PLM stall)")
                print("=" * 60)
                sys.exit(1)

        # ------ console wait (shared by JTAG and netload paths) ------
        MAX_WAIT = 600
        NO_OUTPUT_TIMEOUT = 300
        POLL_INTERVAL = 1

        finished = None
        if args.netload:
            def finished():
                rc, out = netload_remote(["status", "--host", args.netload_ip,
                                         "--connect-timeout", "3"], timeout=30)
                if rc == 0 and ("state=watchdog-timeout" in out or "state=killed" in out):
                    print(f"\n[netload] the supervisor ended the wait: {out.splitlines()[0]}")
                return rc == 0 and any(f"state={st}" in out
                                       for st in ("exited", "watchdog-timeout", "killed"))

        print(f"\n>>> Waiting for console output (max {MAX_WAIT}s, no-output timeout {NO_OUTPUT_TIMEOUT}s)...")

        output_detected = bool(collected_output)
        program_done = False
        start_time = time.time()
        last_output_time = start_time
        last_probe = start_time

        while True:
            elapsed = time.time() - start_time
            since_last_output = time.time() - last_output_time

            got_new = False
            while not console_output_queue.empty():
                chunk = console_output_queue.get()
                collected_output.append(chunk)
                print(chunk, end='', flush=True)
                output_detected = True
                got_new = True
                uart_reconnect_step(conn2, chunk, uart_st)
                if ("device_teardown done" in chunk or "test end" in chunk
                        or "PASS:" in chunk or "FAIL:" in chunk):
                    program_done = True

            if got_new:
                last_output_time = time.time()

            if not program_done and finished and time.time() - last_probe >= 0.5:
                last_probe = time.time()
                if finished():
                    program_done = True

            if program_done:
                time.sleep(2)
                while not console_output_queue.empty():
                    chunk = console_output_queue.get()
                    collected_output.append(chunk)
                    print(chunk, end='', flush=True)
                print(f"\n[Program completed after {elapsed:.0f}s]")
                break

            if elapsed >= MAX_WAIT:
                print(f"\n[Max wait {MAX_WAIT}s reached]")
                break

            if output_detected and since_last_output >= NO_OUTPUT_TIMEOUT:
                print(f"\n[No new output for {NO_OUTPUT_TIMEOUT}s — program may be hung]")
                break

            if not output_detected and since_last_output >= NO_OUTPUT_TIMEOUT:
                print(f"\n[ERROR: No output received for {NO_OUTPUT_TIMEOUT}s]")
                break

            time.sleep(POLL_INTERVAL)

        stop_console_thread.set()
        if console_thread:
            console_thread.join(timeout=5)

        print("\n" + "=" * 60)
        print("CONSOLE OUTPUT SUMMARY:")
        print("=" * 60)
        if not output_detected:
            print(f"[No output received from {console_port} serial console]")

        if args.netload:
            print("\n>>> --netload: board left powered on, netboot supervisor stays resident")
        elif not args.nonreboot:
            print("\n>>> Cleaning up Connection 1...")
            exit_xsdb_and_poweroff(conn1)
        else:
            print("\n>>> --nonreboot: board stays powered on for debug.")
            print("    xsdb and console connections remain active.")
            print(f"    Kill this process when done: Ctrl+C or 'kill {os.getpid()}'")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\nReleasing board...")

        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
        if args.netload and not program_done:
            sys.exit(1)

    except pexpect.TIMEOUT as e:
        print(f"\nError: Command timed out - {e}")
        if conn1 and not args.nonreboot and not args.netload:
            try:
                exit_xsdb_and_poweroff(conn1)
            except Exception:
                pass
        elif conn1:
            print("[--nonreboot/--netload] board left as-is")
        sys.exit(1)
    except pexpect.EOF as e:
        print(f"\nError: Connection closed unexpectedly - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if conn1 and not args.nonreboot and not args.netload:
            try:
                exit_xsdb_and_poweroff(conn1)
            except Exception:
                pass
        elif conn1:
            print("[--nonreboot/--netload] board left as-is")
        sys.exit(1)
    finally:
        stop_console_thread.set()
        if console_thread and console_thread.is_alive():
            console_thread.join(timeout=5)
        if conn1:
            try:
                conn1.close()
            except Exception:
                pass
        if conn2:
            try:
                conn2.close()
            except Exception:
                pass
        print("\nConnections closed.")


if __name__ == "__main__":
    main()