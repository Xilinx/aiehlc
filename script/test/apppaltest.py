#!/usr/bin/env python3
"""Copyright 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
"""
"""
Palboard ELF Test Script

Prerequisites:
- Follow the README to setup run/debug and SSH public key authentication
- Set environment variables: USERNAME and PALIP
- Install pexpect: pip install pexpect

This script:
1. Creates two SSH connections to PALIP
2. First connection: sets up xsdb and programs the device
3. Second connection: connects to com0 for console output
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
import logging
import re
import shlex
import devwatch  # self-watchdog: progress + stall exit (see devwatch.py)
import hws  # hw_server service (hws.py), used through hwspreflight
import hwspreflight  # hws preflight before every run (see hwspreflight.py)

# Configure logging with timestamps so we can see where things get stuck
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("apppaltest")

try:
    import pexpect
except ImportError:
    print("Error: pexpect module not found. Install with: pip install pexpect")
    sys.exit(1)

def board_cli_overrides(argv):
    """-board / -host / -board-rev from the command line, read before the
    environment so `-board vek385 -host gomphus3` needs no BOARDNAME / PALIP.
    main()'s parser declares the same options (for --help and validation)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-board", "--board")
    pre.add_argument("-host", "--host")
    pre.add_argument("-board-rev", "--board-rev")
    return pre.parse_known_args(argv)[0]


_cli = board_cli_overrides(sys.argv[1:])

# Read username, IP, and board name from environment variables; -board / -host win.
username = os.environ.get("USERNAME") or (os.environ.get("USER") if _cli.board and _cli.host else None)
boardname = _cli.board or os.environ.get("BOARDNAME")
palip = _cli.host or (os.environ.get("VEK385IP") if (boardname or "").lower().startswith("vek385") else None) \
    or os.environ.get("PALIP")

if not username or not palip or not boardname:
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
                
                # Recheck the environment variables (the command line still wins)
                username = username or os.environ.get("USERNAME")
                palip = palip or os.environ.get("PALIP")
                boardname = boardname or os.environ.get("BOARDNAME")
            else:
                print(f"Warning: Failed to source envlocal.sh: {result.stderr}")
        except Exception as e:
            print(f"Warning: Error sourcing envlocal.sh: {e}")

if not username or not palip or not boardname:
    print("Error: Please set USERNAME, PALIP, and BOARDNAME environment variables, or pass -board/-host")
    print("Example: export USERNAME=aaaaa && export PALIP=10.23.***.*** && export BOARDNAME=pal***")
    print("Example: apppaltest.py -board vek385 -host gomphus3 -y main.elf")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Or create {os.path.join(script_dir, 'envlocal.sh')} with these exports")
    sys.exit(1)

host = f"{username}@{palip}"

# Configuration
PALBOARD_SCRIPTS_DIR = f"/proj/xsjsswstaff/{username}/palboard_scripts"
# Board profile. BOARDNAME=vek385-* selects the VEK385 farm board: PALIP is its farm
# host (e.g. gomphus3), the boot image is aiebaremetal's vek385_boot.pdi +
# vek385_pld.pdi packed by bootgen (vek385_netboot.BIN), the A78 console is com3
# (as in appvek385.py), and after `device program` the APU cluster is reset with
# `rst -cores` (as appvek385.py does). The netboot supervisor's IP 10.10.70.50 and
# gateway 10.10.70.101 on the farm host's eth1 are shared with palmyra. One
# difference: on the vek385 base platform the GEM (an LPD master, LPD_AXI_NOC0)
# reaches only DDR_CH1/DDR_CH2, not DDR_LOW0, so its supervisor is a separate
# build with the lwIP heap/pools and BD ring at NETBOOT_DMA_BASE (DDR_CH1).
#
# `-board vek385 -host gomphus3` is the same profile from the command line. The
# farm host's systest-client is bound to its one board ("[vek385-3] Systest#"), so
# every VEK385 session uses it, never `become`; `-board vek385` accepts any
# vek385-N there, `-board vek385-3` only that one. The farm host already serves
# hw_server, so xsdb attaches with `connect -url` (HW_SERVER_URL) in both modes.
# -board-rev b (or VEK385_REV=b) programs rev B's boot + PLD PDI pair, as appvek385.py.
IS_VEK385 = boardname.lower().startswith("vek385")
BOARD_REV = (_cli.board_rev or os.environ.get("VEK385_REV") or "a").lower()
if os.environ.get("BOOT_IMAGE"):
    BOOT_IMAGES = [os.environ["BOOT_IMAGE"]]
elif IS_VEK385 and BOARD_REV == "b":
    BOOT_IMAGES = [os.environ.get("VEK385_BOOT_PDI") or f"/home/{username}/aiehlc/vek385revb_boot.pdi",
                   os.environ.get("VEK385_PLD_PDI") or f"/home/{username}/aiehlc/vek385revb_pld.pdi"]
elif IS_VEK385:
    BOOT_IMAGES = [os.environ.get("VEK385PDI") or f"/home/{username}/aiehlc/netboot/vek385_netboot.BIN"]
else:
    BOOT_IMAGES = [f"/home/{username}/palboard/BOOT.BIN"]
HW_SERVER_URL = os.environ.get("HW_SERVER_URL") or f"TCP:{palip}:3121"


def board_matches(found):
    """systest-client's board name ("vek385-3") against -board: the exact name, or
    any board of that family when only the family ("vek385") was given."""
    want, found = boardname.lower(), (found or "").lower()
    return found == want or found.startswith(want + "-")


CONSOLE_PORT = os.environ.get("PAL_CONSOLE") or (os.environ.get("VEK385_CONSOLE", "com3") if IS_VEK385 else "com0")
PLM_SETTLE_S = 15 if IS_VEK385 else 5
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
    log.debug("console_reader thread started")
    buffer = ""
    while not stop_event.is_set():
        try:
            # Read any available output without blocking for long
            output = child.read_nonblocking(size=4096, timeout=0.1)
            if output:
                buffer += output
                output_queue.put(output)
                devwatch.progress("console output")
        except pexpect.TIMEOUT:
            # No data available, continue polling
            continue
        except pexpect.EOF:
            log.debug("console_reader: got EOF")
            output_queue.put("[Connection 2 EOF]")
            break
        except Exception as e:
            log.debug("console_reader: exception: %s", e)
            output_queue.put(f"[Console reader error: {e}]")
            break

    # Put any remaining buffer content
    if buffer:
        output_queue.put(f"[Total bytes read: {len(buffer)}]")
    log.debug("console_reader thread exiting")


def exit_xsdb_and_power_cycle(child):
    """Exit xsdb cleanly (stop target first) and power cycle the board (power 0 then power 1).

    Used in nonreboot mode to reset the board without tearing down the systest session.
    """
    log.debug("exit_xsdb_and_power_cycle: draining stale output...")
    try:
        child.read_nonblocking(size=65536, timeout=2)
    except pexpect.TIMEOUT:
        pass

    log.debug("exit_xsdb_and_power_cycle: sending 'stop'...")
    child.sendline("stop")
    try:
        child.expect(r'xsdb%', timeout=60)
        log.debug("exit_xsdb_and_power_cycle: 'stop' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_power_cycle: 'stop' timed out")
        print("[Connection 1] Warning: stop command timed out, continuing cleanup...")

    time.sleep(1)
    try:
        child.read_nonblocking(size=65536, timeout=2)
    except pexpect.TIMEOUT:
        pass

    log.debug("exit_xsdb_and_power_cycle: sending 'exit'...")
    child.sendline("exit")
    time.sleep(2)
    try:
        child.expect([r'Systest[#>]', r'\$\s*$', r'>\s*$'], timeout=60)
        log.debug("exit_xsdb_and_power_cycle: 'exit' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_power_cycle: 'exit' timed out")
        print("[Connection 1] Warning: exit xsdb timed out, continuing cleanup...")
    print("[Connection 1] Exited xsdb, power cycling board...")

    log.debug("exit_xsdb_and_power_cycle: sending 'power 0'...")
    child.sendline("power 0")
    try:
        child.expect(r'Systest[#>]', timeout=60)
        log.debug("exit_xsdb_and_power_cycle: 'power 0' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_power_cycle: 'power 0' timed out")
        print("[Connection 1] Warning: power 0 timed out")
    print("[Connection 1] Power off complete, powering back on...")

    log.debug("exit_xsdb_and_power_cycle: sending 'power 1'...")
    child.sendline("power 1")
    try:
        child.expect(r'Systest[#>]', timeout=60)
        log.debug("exit_xsdb_and_power_cycle: 'power 1' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_power_cycle: 'power 1' timed out")
        print("[Connection 1] Warning: power 1 timed out")
    print("[Connection 1] Power cycle complete (board cleaned up)")


def exit_xsdb_and_poweroff(child):
    """Exit xsdb cleanly (stop target first) and power off the board."""
    log.debug("exit_xsdb_and_poweroff: draining stale output...")
    # Drain stale output
    try:
        child.read_nonblocking(size=65536, timeout=2)
    except pexpect.TIMEOUT:
        pass

    # Stop the running target before exiting — xsdb may refuse to exit
    # while a target is in Running state.
    log.debug("exit_xsdb_and_poweroff: sending 'stop'...")
    child.sendline("stop")
    try:
        child.expect(r'xsdb%', timeout=60)
        log.debug("exit_xsdb_and_poweroff: 'stop' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_poweroff: 'stop' timed out")
        print("[Connection 1] Warning: stop command timed out, continuing cleanup...")

    # Drain any extra output (e.g. _exit.c source-not-found messages)
    time.sleep(1)
    try:
        child.read_nonblocking(size=65536, timeout=2)
    except pexpect.TIMEOUT:
        pass

    log.debug("exit_xsdb_and_poweroff: sending 'exit'...")
    child.sendline("exit")
    time.sleep(2)
    try:
        child.expect([r'Systest[#>]', r'\$\s*$', r'>\s*$'], timeout=60)
        log.debug("exit_xsdb_and_poweroff: 'exit' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_poweroff: 'exit' timed out")
        print("[Connection 1] Warning: exit xsdb timed out, continuing cleanup...")
    print("[Connection 1] Exited xsdb, powering off...")

    log.debug("exit_xsdb_and_poweroff: sending 'power 0'...")
    child.sendline("power 0")
    try:
        child.expect(r'Systest[#>]', timeout=60)
        log.debug("exit_xsdb_and_poweroff: 'power 0' completed")
    except pexpect.TIMEOUT:
        log.warning("exit_xsdb_and_poweroff: 'power 0' timed out")
        print("[Connection 1] Warning: power off timed out")
    print("[Connection 1] Power off complete")


_xsdb_seq = [0]


def xsdb_run(child, cmd, timeout=60, what=None):
    """Run one xsdb command and wait for ITS completion, not for "a prompt".

    rlwrap echoes the xsdb prompt twice, so a plain expect('xsdb%') matches the
    previous command's prompt: the log showed "Device programmed" printed while
    `device program` was still at 0%, and every later command typed ahead. Here
    the command reports through a unique token. The token is printed via
    [expr], so the echoed command line ("@@OK_[expr {..}]@@") can never match the
    real output ("@@OK_7@@"). Raises RuntimeError on a Tcl error or PLM stall.
    """
    _xsdb_seq[0] += 1
    n = _xsdb_seq[0]
    ok, err = f"@@OK_{n}@@", f"@@ERR_{n}@@"
    child.sendline(f'if {{[catch {{{cmd}}} e]}} {{puts "@@ERR_[expr {{{n - 1}+1}}]@@ $e"}} '
                   f'else {{puts "@@OK_[expr {{{n - 1}+1}}]@@"}}')
    index = child.expect([re.escape(ok), re.escape(err), r"PLM stalled"], timeout=timeout)
    devwatch.progress(f"xsdb `{what or cmd}` returned")
    if index == 0:
        return
    detail = "PLM stalled"
    if index == 1:  # xsdb's message may span lines: take everything up to the prompt
        try:
            child.expect(r"xsdb%", timeout=10)
            detail = " ".join(child.before.split()) or "(no message)"
        except pexpect.TIMEOUT:
            detail = "(no message)"
    raise RuntimeError(f"xsdb `{what or cmd}` failed: {detail}")


# A78 system counter (SCNTRS, the BSP's XIOU_SCNTRS_BASEADDR): CNTCR at +0, base
# frequency at +0x20. After a VEK385 PDI boot both read 0 and stay 0 -- the BSP's
# own enable in XTime_GetTime does not take effect from the A78 -- so CNTPCT is
# frozen and every XTime/usleep wait spins forever (nbspin never ends, an app
# hangs silently). A debugger write starts it (measured: ~100 MHz right after).
SCNTRS_BASE = 0xEC920000
SCNTRS_FREQ_HZ = 99999001  # XPAR_CPU_TIMESTAMP_CLK_FREQ


def vek385_start_system_counter(child):
    xsdb_run(child, 'targets -set -nocase -filter {name =~ "Versal*"}', 60, "select device")
    xsdb_run(child, f"mwr -force 0x{SCNTRS_BASE + 0x20:x} {SCNTRS_FREQ_HZ}", 30, "counter frequency")
    xsdb_run(child, f"mwr -force 0x{SCNTRS_BASE:x} 1", 30, "counter enable")
    print("[Connection 1] A78 system counter enabled (SCNTRS CNTCR=1)")


def vek385_reset_system(child):
    """PMC system reset right before `device program`. A farm board's power cycle
    leaves the previous boot's state behind on VEK385 (the app then exits at once
    with a NUL-only console); `rst -system` from the PMC target clears it, so the
    PDI programs a clean system."""
    # Exactly "PMC": "*PMC*" also hits the ASU target, whose name carries its status
    # ("ASU (Memory write error ... PMC LPD domain isolation)") -> "more than one targets".
    xsdb_run(child, 'targets -set -nocase -filter {name =~ "PMC"}', 60, "select PMC")
    xsdb_run(child, "rst -system", 60, "rst -system (PMC)")
    time.sleep(2)  # the PMC re-runs its ROM; give it a moment before programming
    print("[Connection 1] PMC system reset done (rst -system)")


def setup_first_connection(nonreboot=False):
    """
    First SSH connection: Setup xsdb and program device.
    Returns the pexpect child process for further commands.
    If nonreboot is True, uses systest-client instead of systest.
    """
    log.debug("setup_first_connection: starting (nonreboot=%s)", nonreboot)
    print(f"[Connection 1] Connecting to {host}...")

    # Start SSH with X forwarding
    child = pexpect.spawn(f"ssh -X {host}", encoding='utf-8', codec_errors='replace', timeout=60)
    child.logfile_read = sys.stdout

    # Wait for shell prompt
    log.debug("setup_first_connection: waiting for shell prompt...")
    child.expect([r'\$\s*$', r'#\s*$', r'>\s*$'], timeout=60)
    log.debug("setup_first_connection: shell prompt received")

    # Use systest-client in nonreboot mode so the session doesn't hold the
    # systest server lock, allowing other users/tools to connect.
    if nonreboot or IS_VEK385:  # a VEK385 farm host: systest-client only
        print("[Connection 1] Connected, starting systest-client"
              f"{' (nonreboot mode)' if nonreboot else ''}...")
        log.debug("setup_first_connection: sending systest-client")
        child.sendline("/opt/systest/common/bin/systest-client")
    else:
        print("[Connection 1] Connected, starting systest...")
        log.debug("setup_first_connection: sending systest")
        child.sendline("/bin/systest")
    child.expect(r'Systest[#>]', timeout=60)
    log.debug("setup_first_connection: systest prompt received")
    if IS_VEK385:
        # systest-client lands on the farm host's own board ("[vek385-3] Systest#");
        # refuse to power-cycle anything else (no `become` off palmyra).
        m = re.search(r'\[([^\]\s]+)\]\s*$', child.before or "")
        if not m or not board_matches(m.group(1)):
            child.close(force=True)
            raise RuntimeError(f"systest-client on {palip} is on board {m.group(1) if m else '?'}, "
                               f"not -board/BOARDNAME {boardname}; not power-cycling it")
        print(f"[Connection 1] systest-client is on board {m.group(1)}")

    # Step 3: Become palboard - wait for Systest# prompt after board info.
    # In nonreboot mode the board is already up and owned by an existing
    # session, so skip 'become' to avoid taking over/holding the board.
    if IS_VEK385:
        print("[Connection 1] Farm board: no 'become', powering off first...")
    elif not nonreboot:
        print("[Connection 1] In systest, becoming palboard...")
        child.sendline(f'become "{boardname}"')
        child.expect(r'Systest[#>]', timeout=60)  # Wait for prompt after become completes
        time.sleep(3)  # Extra wait for system controller to stabilize
        print("[Connection 1] Palboard mode, powering off first...")
    else:
        print("[Connection 1] nonreboot mode: skipping 'become' (board already up)")
    
    # Step 4: Power cycle for a clean state. Required before `device program`:
    # programming a live system aborts part-way (measured: ~18%).
    child.sendline("power 0")
    child.expect(r'Systest[#>]', timeout=60)
    time.sleep(2)  # Wait for power off to complete
    print("[Connection 1] Power off complete, powering on...")
    child.sendline("power 1")
    child.expect(r'Systest[#>]', timeout=60)  # Wait for power cycle to complete
    time.sleep(3)  # Extra wait for board to initialize
    if _preflight:  # VEK385: hw_server must survive power 0/1 (hws contract §8); no-op on palmyra
        _preflight.after_power_cycle()
    print("[Connection 1] Power on complete, sourcing Vitis and starting xsdb...")

    # Step 5b: Source Vitis environment so xsdb is on PATH
    child.sendline(f"source {VITIS_SETTINGS}")
    child.expect(r'Systest[#>]', timeout=30)

    # Step 6: Start xsdb (try default first, then alternative path)
    child.sendline("xsdb")
    index = child.expect([r'xsdb%', r'command not found', r'Unrecognized', pexpect.TIMEOUT], timeout=15)

    if index != 0:
        print("[Connection 1] xsdb not found, trying alternative path...")
        child.sendline(XSDB_ALT_PATH)
        child.expect(r'xsdb%', timeout=60)
    
    print("[Connection 1] In xsdb, connecting...")
    
    # Steps 7-11, each waited on by its own completion token (xsdb_run).
    xsdb_run(child, f"connect -url {HW_SERVER_URL}" if nonreboot or IS_VEK385 else "conn", 60, "connect")
    if IS_VEK385:
        vek385_reset_system(child)
    print("[Connection 1] Connected, targeting device 1...")
    xsdb_run(child, "tar 1", 60)
    for image in BOOT_IMAGES:  # rev B VEK385: boot PDI, then PLD PDI, back to back
        print(f"[Connection 1] Programming {image}...")
        try:
            xsdb_run(child, f"device program {image}", 300, "device program")
        except RuntimeError as e:
            raise RuntimeError(f"programming {image} failed ({e}). The board may need a power cycle; "
                               "run 'plm log' for details.") from e
    print("[Connection 1] Device programmed (100%), letting the PLM settle...")
    time.sleep(PLM_SETTLE_S)  # now measured from the END of programming
    if IS_VEK385:
        # Reset A78 #0.0 only, as palmyra does. Not appvek385.py's APU `rst -cores`:
        # that suspends EVERY core of the group, so #0.1 sits in the debugger's Reset
        # Catch and the netboot supervisor's RequestWakeUp of it fails (PM 0x7D2) or
        # the woken core never runs. After `device program` both cores are in Power
        # On Reset and `rst -processor` on #0.0 leaves #0.1 to the PLM (measured).
        vek385_start_system_counter(child)
        xsdb_run(child, 'targets -set -nocase -filter {name =~ "*A78*#0.0*"}', 60, "select A78 #0.0")
        try:
            xsdb_run(child, "rst -processor", 60, "rst -processor (A78 #0.0)")
        except RuntimeError as e:
            print(f"[Connection 1] {e}; falling back to the APU cluster reset (A78 #0.1 stays in Reset Catch)")
            xsdb_run(child, 'targets -set -nocase -filter {name =~ "APU"}', 60, "select APU")
            xsdb_run(child, "rst -cores", 60, "rst -cores")
            xsdb_run(child, 'targets -set -nocase -filter {name =~ "*A78*#0.0*"}', 60, "select A78 #0.0")
    else:
        print("[Connection 1] Targeting device 20...")
        xsdb_run(child, "tar 20", 60)
        print("[Connection 1] Resetting processor...")
        xsdb_run(child, "rst -proc", 60)
    print("[Connection 1] Setup complete!")
    
    return child


def setup_second_connection():
    """
    Second SSH connection: Connect to com0 for console output.
    Returns the pexpect child process.
    """
    log.debug("setup_second_connection: starting")
    print(f"[Connection 2] Connecting to {host}...")

    # No X forwarding: this connection only tails console text (systest-client ->
    # connect com0), so it needs no X GUI. Dropping -X also avoids the tcsh
    # login-script X11 setup (Xinventory/xauth), which could intermittently
    # stall a second concurrent -X session and hang the prompt wait below.
    child = pexpect.spawn(f"ssh {host}", encoding='utf-8', codec_errors='replace', timeout=60)
    # Mirror connection 1 so this connection's raw output is visible; otherwise a
    # prompt-wait hang shows nothing after "waiting for shell prompt".
    child.logfile_read = sys.stdout

    # Wait for shell prompt
    log.debug("setup_second_connection: waiting for shell prompt...")
    try:
        child.expect([r'\$\s*$', r'#\s*$', r'>\s*$'], timeout=60)
    except pexpect.TIMEOUT:
        log.error("setup_second_connection: shell prompt TIMEOUT; buffer so "
                  "far:\n%s", child.before)
        raise
    log.debug("setup_second_connection: shell prompt received")
    print("[Connection 2] Connected, starting systest...")

    # Step 2: Run systest
    log.debug("setup_second_connection: sending systest-client")
    child.sendline("/opt/systest/common/bin/systest-client")
    child.expect(r'Systest[#>]', timeout=60)
    log.debug("setup_second_connection: systest prompt received")
    print(f"[Connection 2] In systest, connecting to {CONSOLE_PORT}...")

    # Step 3: Connect to com0 (no output until ELF runs on first connection)
    log.debug("setup_second_connection: sending 'connect %s'", CONSOLE_PORT)
    child.sendline(f"connect {CONSOLE_PORT}")
    child.expect(rf'Connecting to device {CONSOLE_PORT}.*escape', timeout=60)
    log.debug("setup_second_connection: %s connected", CONSOLE_PORT)
    print(f"[Connection 2] Connected to {CONSOLE_PORT}, listening for output...")

    return child


def download_elf_and_continue(child, elf_path):
    """
    Download the ELF file and continue execution on the first connection.
    Step 12: dow -force <elf_path>
    Step 13: con
    """
    log.debug("download_elf_and_continue: starting with %s", elf_path)
    print(f"[Connection 1] Downloading ELF: {elf_path}")

    # Drain stale output from prior commands (e.g. "100%" left over from
    # "device program BOOT.BIN") so they cannot be matched by the expects below.
    try:
        child.read_nonblocking(size=65536, timeout=1)
    except pexpect.TIMEOUT:
        pass

    log.debug("download_elf_and_continue: sending 'dow -force'...")
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
    # and waste 120 seconds waiting.
    # Timeout sized for a slow JTAG link, not a fast one. Observed ~0.1MB/s on a
    # loaded farm board: a ~10MB ELF then needs ~130s and the old 120s budget
    # expired at ~65% with a pexpect.TIMEOUT, after already having spent two
    # minutes. Downloading is I/O bound and monotonic — a generous ceiling costs
    # nothing on a fast link (the expect returns as soon as the match arrives)
    # and only matters when the alternative is a spurious failure.
    log.debug("download_elf_and_continue: waiting for download result...")
    with devwatch.quiet(DOWNLOAD_TIMEOUT_S + 60, f"dow {os.path.basename(elf_path)}"):
        index = child.expect([r'Successfully downloaded', r'PLM stalled'], timeout=DOWNLOAD_TIMEOUT_S)
    devwatch.progress("dow finished")
    if index == 1:
        # Consume remaining output up to the prompt
        child.expect(r'xsdb%', timeout=60)
        print("\n[ERROR] PLM stalled during ELF download (dow -force).")
        print("BOOT.BIN did not initialize PS peripherals (UART etc.).")
        print("The board needs a full power cycle and re-programming of BOOT.BIN.")
        print("Run 'plm log' in xsdb for details.")
        return False
    log.debug("download_elf_and_continue: download complete")
    print("[Connection 1] ELF download complete!")

    # Step 13: Continue execution
    print("[Connection 1] Continuing execution...")
    log.debug("download_elf_and_continue: sending 'con'...")
    xsdb_run(child, "con", 60)
    log.debug("download_elf_and_continue: 'con' completed, execution started")
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


# --------------------------------------------------------------------------
# --netload: run the ELF through the resident netboot supervisor instead of a
# JTAG `dow` (doc/design/lwip_elf_bootloader.md). The console is still com0,
# read by the same console_reader, so the log is collected exactly as before.
# --------------------------------------------------------------------------
NETBOOT_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "tool", "netboot")
NETBOOT_REMOTE_DIR = f"/home/{username}/aiehlc/netboot"
NETBOOT_READY_TIMEOUT_S = 120
NETBOOT_BOOTSTRAP_TRIES = 2  # full bootstraps per run when the supervisor comes up without a network
# Supervisor variant per board: (build dir, NETBOOT_DMA_BASE, name staged on the lab
# host). /home is shared between the farm hosts, so the variants get distinct names.
NETBOOT_DMA_BASE = os.environ.get("NETBOOT_DMA_BASE", "0xA00000000") if IS_VEK385 else ""
NETBOOT_BUILD_DIR = os.path.join(NETBOOT_LOCAL_DIR, "build", "vek385") if IS_VEK385 \
    else os.path.join(NETBOOT_LOCAL_DIR, "build")
NETBOOT_REMOTE_ELF = "netboot_vek385.elf" if IS_VEK385 else "netboot.elf"
# One persistent ssh connection to the lab host, reused by every netload ssh/scp:
# a fresh connection costs 0.55-0.85 s here, a multiplexed one a few tens of ms.
SSH_MUX = ["-o", "ControlMaster=auto", "-o", "ControlPersist=4h", "-o", f"ControlPath=/tmp/apppal-{username}-%C"]


def ensure_netboot_built():
    """Netload needs two local artifacts: netboot_proxy.py (in the repo) and
    <NETBOOT_BUILD_DIR>/netboot.elf (built by src/tool/netboot/build.sh). Build the ELF when it is
    missing or older than any of its sources, so a fresh checkout or an edited
    supervisor never bootstraps a stale image. The mtime check is free when
    nothing changed."""
    proxy = os.path.join(NETBOOT_LOCAL_DIR, "netboot_proxy.py")
    if not os.path.isfile(proxy):
        raise RuntimeError(f"{proxy} missing (it is part of the repo; check the checkout)")
    elf = os.path.join(NETBOOT_BUILD_DIR, "netboot.elf")
    srcs = [p for pat in ("supervisor/*.c", "supervisor/*.S", "supervisor/*.h", "lwip/*", "netboot_proto.h",
                          "build.sh") for p in glob.glob(os.path.join(NETBOOT_LOCAL_DIR, pat))]
    newest = max((os.path.getmtime(p) for p in srcs), default=0)
    if os.path.isfile(elf) and os.path.getmtime(elf) >= newest:
        return
    why = "missing" if not os.path.isfile(elf) else "older than its sources"
    print(f"[netload] netboot.elf {why}: building it (src/tool/netboot/build.sh)...")
    env = dict(os.environ, NETBOOT_OUT=NETBOOT_BUILD_DIR, NETBOOT_DMA_BASE=NETBOOT_DMA_BASE)
    with devwatch.quiet(1800 + 60, "building netboot.elf"):
        r = subprocess.run(["bash", os.path.join(NETBOOT_LOCAL_DIR, "build.sh")], capture_output=True, text=True,
                           timeout=1800, env=env)
    if r.returncode != 0 or not os.path.isfile(elf):
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"building netboot.elf failed (exit {r.returncode}):\n{tail}\n"
                           "The A78 cross compiler must be on PATH (source Vitis settings64.sh).")
    print(f"[netload] built {elf}")


def stage_if_changed(local_paths, names=None):
    """Copy files to NETBOOT_REMOTE_DIR on the lab host (as `names`, default their
    basenames), skipping any whose remote copy already has the same SHA-256 (one
    ssh round trip to check)."""
    import hashlib
    names = names or [os.path.basename(p) for p in local_paths]
    probe = (f"mkdir -p {NETBOOT_REMOTE_DIR} && cd {NETBOOT_REMOTE_DIR} && "
             f"(sha256sum {' '.join(shlex.quote(n) for n in names)} 2>/dev/null; true)")
    r = subprocess.run(["ssh", *SSH_MUX, host, f"bash -c {shlex.quote(probe)}"], capture_output=True, text=True,
                       timeout=30)
    remote = {ln.split()[1]: ln.split()[0] for ln in r.stdout.splitlines() if len(ln.split()) == 2}
    stale = [(p, n) for p, n in zip(local_paths, names)
             if remote.get(n) != hashlib.sha256(open(p, "rb").read()).hexdigest()]
    for p, n in stale:
        subprocess.run(["scp", "-q", *SSH_MUX, p, f"{host}:{NETBOOT_REMOTE_DIR}/{n}"], check=True,
                       capture_output=True, timeout=120)
    print(f"[netload] staged on {palip}: {', '.join(n for _, n in stale) or 'nothing (up to date)'}")


def netboot_stage_proxy():
    """The relay netload.py talks through; the target ELF itself is never copied."""
    stage_if_changed([os.path.join(NETBOOT_LOCAL_DIR, "netboot_proxy.py")])


def netboot_stage_supervisor():
    """netboot.elf for a JTAG bootstrap (xsdb on the lab host `dow`s it from there)."""
    local_elf = os.path.join(NETBOOT_BUILD_DIR, "netboot.elf")
    if not os.path.isfile(local_elf):
        raise RuntimeError(f"{local_elf} missing: run src/tool/netboot/build.sh first")
    stage_if_changed([local_elf], [NETBOOT_REMOTE_ELF])
    return f"{NETBOOT_REMOTE_DIR}/{NETBOOT_REMOTE_ELF}"


_netload_mod = []


class _Tee:
    """stdout for an in-process netload: shown live AND kept for the caller.
    A busy board makes `load --wait` retry for up to --netload-wait seconds; with
    the output only buffered, those "busy, N s left -- waiting" lines appeared
    after the wait, i.e. apppaltest looked hung."""

    def __init__(self, live, progress=True):
        import io
        self.buf, self.live, self.progress = io.StringIO(), live, progress

    def write(self, text):
        self.buf.write(text)
        if self.progress and text.strip():
            devwatch.progress("netload: " + text.strip()[:80])
        if self.live:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
        return len(text)

    def flush(self):
        pass


def netload_remote(argv, timeout=600, gate=None, live=False, progress=True):
    """Run netload.py's main() in this process and reach the board through
    netboot_proxy.py on the lab host over the multiplexed ssh: the ELF streams
    through the ssh pipe, nothing is scp'd. Returns (exit status, printed text);
    live=True also prints netload's lines as they come (load: busy waits).
    progress=False: a monitoring poll (the 0.5 s `status` probe while waiting for the
    app), which must not feed the self-watchdog -- otherwise a silent hung app never
    trips --stall-secs, because the probe itself always answers."""
    import contextlib
    import importlib.util
    if not _netload_mod:
        spec = importlib.util.spec_from_file_location("netload", os.path.join(NETBOOT_LOCAL_DIR, "netload.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _netload_mod.append(mod)
    out = _Tee(live, progress)
    extra = ["--via", host, "--proxy", f"{NETBOOT_REMOTE_DIR}/netboot_proxy.py"]
    extra += [x for i in range(1, len(SSH_MUX), 2) for x in ("--ssh-opt", SSH_MUX[i])]
    mod = _netload_mod[0]
    mod.send_gate = gate
    step = devwatch.quiet(timeout, "netload " + " ".join(argv[:2])) if progress else contextlib.nullcontext()
    try:
        with step, contextlib.redirect_stdout(out):
            rc = mod.main(argv + extra)
    finally:
        mod.send_gate = None
    return rc, out.buf.getvalue().strip()


def open_console_fast():
    """The console (CONSOLE_PORT) through `ssh -tt host systest-client` on the multiplexed connection:
    no login shell to wait for (that alone was ~0.8 s of setup_second_connection)."""
    # A farm host's long-lived systest session re-opening its com telnets kills a
    # starting systest-client with SIGUSR1 ("User signal 1", EOF); it is transient.
    for attempt in range(3):
        child = pexpect.spawn("ssh", [*SSH_MUX, "-tt", host, "/opt/systest/common/bin/systest-client"],
                              encoding='utf-8', codec_errors='replace', timeout=60)
        try:
            child.expect(r'Systest[#>]', timeout=60)
            child.sendline(f"connect {CONSOLE_PORT}")
            if child.expect([rf'Connecting to device {CONSOLE_PORT}.*escape',
                             rf'Unrecognized device: {CONSOLE_PORT}'], timeout=60) == 1:
                child.close(force=True)  # the farm board is powered off: its com ports are gone
                raise ConnectionError(f"{CONSOLE_PORT}: Unrecognized device (board powered off?)")
            return child
        except pexpect.EOF:
            child.close(force=True)
            if attempt == 2:
                raise
            print(f"[netload] systest-client closed while attaching {CONSOLE_PORT} "
                  f"({' '.join((child.before or '').split())[-60:]}); retrying", flush=True)
            time.sleep(2)


def start_console(fast):
    conn2 = open_console_fast() if fast else setup_second_connection()
    stop_console_thread.clear()
    th = threading.Thread(target=console_reader, args=(conn2, console_output_queue, stop_console_thread))
    th.start()
    return conn2, th


def leave_console(conn2):
    """Leave `connect <com>` the way systest expects (Ctrl-\\, then exit) before the
    ssh closes. Force-closing the ssh alone leaves the farm host's long-lived systest
    holding `telnet <console server> 400N` for that port; those pile up (seen on
    gomphus3 and crimini1), and a com port held that way showed a `telnet>` prompt +
    NULs instead of the board's output on crimini1."""
    try:
        conn2.sendcontrol('\\')
        conn2.expect(r'Systest[#>]', timeout=5)
        conn2.sendline("exit")
        conn2.expect(pexpect.EOF, timeout=5)
    except Exception as e:  # best effort: the ssh is closed next anyway
        log.debug("leave_console: %s", e)
    conn2.close(force=True)


def stop_console(conn2, th):
    stop_console_thread.set()
    th.join(timeout=5)
    leave_console(conn2)


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
    """Free a stuck app core (A78 #0.1) without a re-bootstrap: over JTAG, stop
    it, reset it, point it at the supervisor's nb_park and let it run. It parks
    there like an app that returned from main, and the supervisor reuses it.
    Only core 1 is touched: core 0 keeps the supervisor, which is polled (no
    interrupts), so the GIC state this reset leaves behind does not affect it.
    About 5 s against ~50 s for power cycle + device program + supervisor load.
    Returns True when the commands went through (the caller checks `status`)."""
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
        xsdb_run(child, f"connect -url {HW_SERVER_URL}", 60, "connect")
        xsdb_run(child, 'targets -set -nocase -filter {name =~ "*A78*#0.1*"}', 30, "select app core #0.1")
        xsdb_run(child, "catch {stop}", 30)
        # VEK385: the PLM denies the core power request behind rst -proc; stop + pc +
        # con alone parks the core (jane, 2026-11-05: park_core1.tcl -> exited).
        xsdb_run(child, "catch {rst -proc}" if IS_VEK385 else "rst -proc", 30, "rst -proc (app core)")
        xsdb_run(child, f"rwr pc 0x{park_addr:x}", 30, "pc = nb_park")
        xsdb_run(child, "con", 30, "continue (app core)")
        xsdb_run(child, "disconnect", 30)
        return True
    except (RuntimeError, pexpect.ExceptionPexpect) as e:
        print(f"[netload] JTAG park of the app core failed: {e}")
        return False
    finally:
        if child is not None:
            child.close(force=True)


def netboot_park_symbol():
    """Address of nb_park in the staged supervisor ELF (local build)."""
    elf = os.path.join(NETBOOT_BUILD_DIR, "netboot.elf")
    out = subprocess.run(["nm", elf], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^([0-9a-fA-F]+) T nb_park$", out, re.M)
    if not m:
        raise RuntimeError(f"nb_park not found in {elf}")
    return int(m.group(1), 16)


def vek385_park_app_core(conn1, ip):
    """VEK385 bootstrap: hand A78 #0.1 to the supervisor parked, over the bootstrap's xsdb.

    After `rst -cores` core 1 sits in the debugger's Reset Catch, already powered,
    and the PLM refuses the supervisor's RequestWakeUp (XPM_PM_NO_ACCESS 2002,
    vek385-11 2026-11-05); where rst -cores is refused (vek385-3) the wake works
    but the core still waits in Reset Catch for a `con`. Once core 1 runs (spinning
    in nb_park) the PLM wake of the first load is accepted (vek385-11 run 3), and
    later loads are mailbox releases (no PLM call)."""
    try:
        xsdb_run(conn1, 'targets -set -nocase -filter {name =~ "*A78*#0.1*"}', 30, "select app core #0.1")
        # Only a core the debugger holds needs this. setup_first_connection resets
        # A78 #0.0 alone, leaving #0.1 in Power On Reset for the PLM's wake; it is
        # in Reset Catch only after the `rst -cores` fallback.
        # "[expr 1+1]" keeps xsdb's echo of this line from matching the reply.
        conn1.sendline('puts "@@ST[expr 1+1] [state] @@"')
        conn1.expect(r"@@ST2 ([^@\r\n]*) @@", timeout=30)
        core1 = conn1.match.group(1)
        if "Reset Catch" not in core1:
            print(f"[netload] vek385: A78 #0.1 is '{core1}', not held by the debugger: no park needed", flush=True)
            xsdb_run(conn1, 'targets -set -nocase -filter {name =~ "*A78*#0.0*"}', 30, "select A78 #0.0")
            return True
        park = netboot_park_symbol()
        print(f"[netload] vek385: parking A78 #0.1 at nb_park 0x{park:x} over JTAG", flush=True)
        xsdb_run(conn1, "catch {stop}", 30)
        # The debugger-side reset is what lets the PLM wake succeed afterwards
        # (stop + pc + con alone still got XPM_PM_NO_ACCESS on vek385-11).
        xsdb_run(conn1, "catch {rst -proc}", 30, "rst -proc (app core)")
        xsdb_run(conn1, "catch {stop}", 30)
        xsdb_run(conn1, f"rwr pc 0x{park:x}", 30, "pc = nb_park")
        xsdb_run(conn1, "con", 30, "continue (app core)")
        xsdb_run(conn1, 'targets -set -nocase -filter {name =~ "*A78*#0.0*"}', 30, "select A78 #0.0")
    except (RuntimeError, subprocess.CalledProcessError, pexpect.ExceptionPexpect) as e:
        print(f"\n[ERROR] vek385: JTAG park of A78 #0.1 failed: {e}")
        return False
    time.sleep(0.5)
    st_rc, st = netload_remote(["status", "--host", ip, "--connect-timeout", "3"], timeout=30)
    if st_rc != 0:
        print(f"\n[ERROR] vek385: supervisor stopped answering after the park ({st.strip().splitlines()[-1] if st else 'no answer'})")
        return False
    return True


def netload_start(args, local_elf, collected):
    """Fast path first: console up, then `load` straight to a resident supervisor.

    Only if that is refused -- nothing answers, or core 1 is busy (running/hung;
    this PDI cannot stop it) -- or --netload-bootstrap is given, fall back to a
    JTAG bootstrap of the supervisor. Order on that path matters: `device program`
    resets the board and drops a com0 session opened before it, so the console is
    reopened after programming. Returns (conn1, conn2, console_thread, ok)."""
    ip = args.netload_ip
    ensure_netboot_built()
    # The supervisor acks before the ELF is sent (protocol v2): busy within its
    # max -> wait up to --netload-wait; past its max -> --force terminates it
    # (exit 5 "stuck" when this PDI cannot: the bootstrap below does it).
    # --netload-force-now: no waiting; a run still within its max is taken over below.
    force_now = args.netload_force_now
    load_argv = ["load", local_elf, "--host", ip, "--max-run-ms", str(args.netload_max_run_ms),
                 "--wait", "0" if force_now else str(args.netload_wait)] + \
        ([] if args.netload_no_force and not force_now else ["--force"])
    # A streamed ELF (bf16 weights) is placed as it arrives: the reply can take longer than 60 s.
    reply_timeout = getattr(args, "netload_reply_timeout", 60.0)
    load_argv += ["--reply-timeout", str(reply_timeout)]
    if not args.netload_bootstrap:
        # The console attaches while the ELF streams; netload holds back the last
        # 4 KB until it is attached, so the app cannot print before we listen.
        box = {}
        def open_console():
            try:
                box["c"] = start_console(fast=True)
            except Exception as e:  # reported below; a bootstrap can still bring the board up
                box["err"] = e
        opener = threading.Thread(target=open_console)
        opener.start()
        netboot_stage_proxy()

        def console_attached():
            opener.join(timeout=60)
            if "c" not in box:
                raise ConnectionError(f"{CONSOLE_PORT} console did not attach")

        t0 = time.time()
        print(f"[netload] asking the supervisor at {ip} to run {os.path.basename(local_elf)} " +
              ("(force now: a running app is taken over)" if force_now
               else f"(waits up to {args.netload_wait:g} s if the board is busy)"), flush=True)
        rc, out = netload_remote(load_argv, timeout=max(600, reply_timeout + 120), gate=console_attached, live=True)
        opener.join(timeout=60)
        if "c" not in box and rc != 2:
            print(f"\n[ERROR] {CONSOLE_PORT} console did not attach ({box.get('err', 'timeout')})")
            return None, None, None, False
        if "c" not in box:  # no console AND no supervisor: board off or never booted
            print(f"[netload] {CONSOLE_PORT} not available ({box.get('err', 'timeout')}) and no supervisor "
                  "answering: bootstrapping (power cycle + device program)", flush=True)
        conn2, th = box.get("c", (None, None))
        if rc == 0:
            print(f"[netload] streamed {local_elf} through the proxy on {palip} in {time.time() - t0:.2f}s (no scp)")
            return None, conn2, th, True
        if rc == 3 and not force_now:
            print(f"\n[ERROR] board still busy after --netload-wait {args.netload_wait:g} s with a run that is "
                  f"within its max run time; not power-cycling it. Retry later or raise --netload-wait")
            return None, conn2, th, False
        if rc == 4:
            print("\n[ERROR] the previous app ran past its max run time; drop --netload-no-force to terminate it")
            return None, conn2, th, False
        if rc not in (2, 5) and not (rc == 3 and force_now) and "re-bootstrap" not in out:
            return None, conn2, th, False  # a real refusal (bad ELF etc.), not "no supervisor"
        m = re.search(r"park_addr=0x([0-9a-f]+)", out)
        if rc == 3:
            print("[netload] --netload-force-now: the board is running another app within its max run time; "
                  "taking the app core over", flush=True)
        if rc in (3, 5) and m and hws_need_jtag("app core busy, parking it over JTAG") \
                and netboot_park_app_core(int(m.group(1), 16)):
            time.sleep(0.5)
            st_rc, st = netload_remote(["status", "--host", ip, "--connect-timeout", "3"], timeout=30)
            if st_rc == 0 and "app core parked" in st:
                print("[netload] app core parked by JTAG; sending the ELF again", flush=True)
                rc, out = netload_remote(load_argv, live=True)
                if rc == 0:
                    return None, conn2, th, True
            print(f"[netload] JTAG park did not free the app core ({st.splitlines()[-1] if st else 'no answer'}); "
                  "falling back to a re-bootstrap", flush=True)
        if conn2 is not None:
            stop_console(conn2, th)
        why = (f"supervisor not answering at {ip}" if rc == 2 else
               "the previous app is past its max run time and this PDI cannot stop it (forced run)" if rc == 5
               else "--netload-force-now: the running app could not be parked over JTAG" if rc == 3
               else "app core still busy from the previous run")
    else:
        why = "--netload-bootstrap"
    if not hws_need_jtag(why):
        return None, None, None, False
    return netboot_bootstrap(why, ip, load_argv, collected)


def netboot_bootstrap(why, ip, load_argv, collected):
    """JTAG bootstrap of the supervisor, then the load. Returns (conn1, conn2,
    console_thread, ok).

    A supervisor that came up without a network -- the Ethernet PHY's
    autonegotiation timed out after the power cycle (seen 1 in 3 on 213): no
    `ready` within NETBOOT_READY_TIMEOUT_S, or (a supervisor older than the link
    retry) `ready` and still no answer -- gets another full bootstrap, whose power
    cycle resets the PHY, instead of failing this run and leaving the next run to
    redo everything."""
    print(f"\n>>> {why}: loading the netboot supervisor over JTAG (one-time)...")
    netboot_stage_proxy()
    for attempt in range(1, NETBOOT_BOOTSTRAP_TRIES + 1):
        # `device program` on a live system aborts at ~18%, so the bootstrap power-cycles
        # first, through the non-owning systest-client session. The app finishing never
        # powers anything off, and the fast path never reaches this.
        conn1 = setup_first_connection(nonreboot=True)
        conn2, th = start_console(fast=False)
        if not download_elf_and_continue(conn1, netboot_stage_supervisor()):
            return conn1, conn2, th, False
        if drain_until("[netboot] ready", NETBOOT_READY_TIMEOUT_S, collected):
            if IS_VEK385 and not vek385_park_app_core(conn1, ip):
                return conn1, conn2, th, False
            rc, out = netload_remote(load_argv, live=True)
            if rc != 2:
                return conn1, conn2, th, rc == 0
            why = f"supervisor printed ready but does not answer at {ip}"
        else:
            why = (f"supervisor did not print '[netboot] ready' within {NETBOOT_READY_TIMEOUT_S}s "
                   "(Ethernet link down?)")
        print(f"\n[ERROR] {why}")
        if attempt == NETBOOT_BOOTSTRAP_TRIES:
            return conn1, conn2, th, False
        print(f">>> bootstrapping again, the power cycle resets the PHY "
              f"(attempt {attempt + 1}/{NETBOOT_BOOTSTRAP_TRIES})...", flush=True)
        stop_console(conn2, th)
        conn1.close(force=True)


def build_arg_parser():
    """The command line (main() parses it; -board/-host are also pre-read at import)."""
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
  %(prog)s -board vek385 -host gomphus3 -y main.elf            # VEK385 farm board: full JTAG boot
  %(prog)s -board vek385 -host gomphus3 -y --netload main.elf  # VEK385 through the netboot supervisor
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
        default=False,
        help="Non-interactive mode: auto-confirm all prompts"
    )
    parser.add_argument(
        "-nonreboot", "--nonreboot",
        action="store_true",
        default=False,
        help="Keep board powered on after test (no xsdb exit, no power off) for manual debug"
    )
    parser.add_argument(
        "--netload",
        action="store_true",
        default=False,
        help="Run via the resident netboot supervisor instead of JTAG dow; bootstraps the "
             "supervisor over JTAG only if it does not answer. The board is left powered on "
             "with the supervisor resident (src/tool/netboot/README.md)."
    )
    parser.add_argument(
        "--netload-ip",
        default="10.10.70.50",
        help="Supervisor IP (netboot build.sh NETBOOT_IP)"
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
        help="Max run time of this app on the board (--netload only); 0 = the supervisor default (60 s). "
             "Past it the supervisor marks the run timed out, and a later forced run may take the core over"
    )
    parser.add_argument(
        "--netload-reply-timeout",
        type=float,
        default=60.0,
        help="Seconds netload waits for the supervisor's reply after the ELF is sent (--netload only, default 60); "
             "raise it for a large streamed ELF (bf16 weights) whose placement outlasts the default"
    )
    parser.add_argument(
        "--netload-wait",
        type=float,
        default=120.0,
        help="If the board is busy with a run still within its max run time, wait up to this many seconds "
             "for it (--netload only, default 120); the supervisor says how long the run has left"
    )
    parser.add_argument(
        "--netload-force-now",
        action="store_true",
        default=False,
        help="Run this ELF immediately (--netload only): do not wait for a run still within its max run time -- "
             "free the app core over JTAG (stop, rst -proc, park; ~5-17 s) and send this ELF, falling back to a "
             "supervisor re-bootstrap if that fails. Kills whatever is running, including another user's run"
    )
    parser.add_argument(
        "--netload-no-force",
        action="store_true",
        default=False,
        help="Do not terminate a previous app that ran past its max run time (default: force it off and run "
             "this ELF; on a PDI that cannot stop the app core that means a re-bootstrap)"
    )
    parser.add_argument(
        "-board", "--board",
        default=None,
        help="Board: palmyra (from BOARDNAME) or vek385 / vek385-N (a VEK385 farm board: systest-client "
             "on -host, its running hw_server, PDI boot, console com3, VEK385 netboot supervisor). "
             "Overrides BOARDNAME"
    )
    parser.add_argument(
        "-host", "--host",
        default=None,
        help="Lab host the board hangs off (palmyra's PALIP, or the VEK385 farm host, e.g. gomphus3). "
             "Overrides PALIP / VEK385IP"
    )
    parser.add_argument(
        "-board-rev", "--board-rev",
        choices=("a", "b"),
        default=None,
        help="VEK385 revision (default a, or VEK385_REV): a programs VEK385PDI (default "
             "~/aiehlc/netboot/vek385_netboot.BIN), b programs VEK385_BOOT_PDI then VEK385_PLD_PDI"
    )
    parser.add_argument(
        "--lock-wait", type=int, default=int(os.environ.get("APPPAL_LOCK_WAIT", "1800")),
        help="Seconds to wait for the per-device lock before giving up (default 1800). "
             "Only one apppaltest/board.py run per device at a time; a crashed holder "
             "releases it automatically (kernel flock)."
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="Skip the per-device lock (only for read-only debugging; can corrupt another run)"
    )
    parser.add_argument(
        "--stall-secs", type=int, default=int(os.environ.get("APPPAL_STALL_SECS", "600")),
        help="Self-watchdog: exit with STALLED (code 124, lock released) after this many seconds "
             "with no progress (console output, an xsdb command returning, a netload line). "
             "Known-slow steps (netboot build, dow) extend it. 0 = off. Default 600"
    )
    parser.add_argument(
        "--max-run", type=int, default=int(os.environ.get("APPPAL_MAX_RUN", "3600")),
        help="Self-watchdog hard cap on the whole run in seconds (default 3600, 0 = off)"
    )
    parser.add_argument(
        "--lock-break", action="store_true",
        help="Kill a HUNG holder of this device's lock (must run on the holder's host) and exit"
    )
    parser.add_argument(
        "--hws", choices=("status", "up", "down"), default=None,
        help="Only run this hw_server-service (hws.py) command on the board and exit with its code "
             "(0 ok, 1 down, 3 degraded, 4 lock busy, 5 foreign, 6 unreachable). up/down take the board lock"
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Manual debugging: skip the hws preflight (status/up/restart, netload smoke test, palmyra "
             "full-mode hws down/up)"
    )
    parser.add_argument(
        "--hws-eager", action="store_true",
        help="--netload: check hw_server and run the smoke test before the run. Default: only when the "
             "supervisor cannot take the ELF and JTAG is needed (park a stuck app core, bootstrap)"
    )
    parser.add_argument(
        "--hws-restarts", type=int, default=hwspreflight.DEFAULT_MAX_RESTARTS,
        help="Preflight retry limit: clean + restart attempts when hws is DEGRADED (default 1, the hws contract)"
    )
    parser.add_argument(
        "--smoke-elf", default=os.environ.get("APPPAL_SMOKE_ELF"),
        help="--hws-eager netload smoke-test ELF (default <netboot build>/nbspin5.elf; skipped with a note "
             "when that default is missing)"
    )
    return parser


def main():
    """Parse args, take the per-device lock, then run."""
    import devlock
    args = build_arg_parser().parse_args()
    device = palip
    if args.lock_break:
        print(f"[devlock] {devlock.break_lock(device)}")
        return
    if args.hws:
        sys.exit(hws_passthrough(args))
    if args.no_lock:
        print("[devlock] WARNING: --no-lock, not serialising with other runs on this device")
        with _watchdog(args, None):
            return _run_with_preflight(args, None)
    with devlock.DeviceLock(device, wait_secs=args.lock_wait) as lk:
        with _watchdog(args, lk.heartbeat):
            return _run_with_preflight(args, lk)


# --------------------------------------------------------------------------
# hws preflight (plan-1109-b1597a s5): hw_server healthy before every run, under
# this run's board lock. Logic and tests: hwspreflight.py / test_hwspreflight.py.
# --------------------------------------------------------------------------
HWS_TYPE = "vek385" if IS_VEK385 else "palmyra"
SMOKE_MAX_RUN_MS = 30000
SMOKE_DONE_S = 60
_preflight = None  # the running Preflight; setup_first_connection calls after_power_cycle()


def hws_passthrough(args):
    """--hws status|up|down: one hws command on this board; returns its exit code."""
    try:
        if args.hws == "status":
            h = hws.status(palip, HWS_TYPE)
            rc = hws.RC_DOWN if h.state == hws.DOWN else h.rc
        else:
            h = getattr(hws, args.hws)(palip, HWS_TYPE, wait_lock=args.lock_wait)
            rc = h.rc
    except hws.HwsError as e:
        print(f"HWS ERROR {palip} rc={e.rc} {e.msg}")
        return e.rc
    print(f"HWS {h.state} {h.board} rc={rc} {h.msg} checks={h.checks}")
    return rc


def preflight_mode(args):
    if args.netload:
        return hwspreflight.NETLOAD
    if not IS_VEK385 and args.nonreboot:  # systest-client + connect -url to the resident hw_server
        return hwspreflight.ATTACH
    return hwspreflight.FULL


def hws_smoke(args):
    """Netload smoke test: run a short ELF (nbspin5) through the supervisor and wait
    until it has exited. Returns (True|False|None, msg); None = skipped."""
    elf = args.smoke_elf or os.path.join(NETBOOT_BUILD_DIR, "nbspin5.elf")
    name = os.path.basename(elf)
    if not os.path.isfile(elf):
        if args.smoke_elf:
            return False, f"--smoke-elf {elf} not found"
        return None, f"no {elf} (build it with src/tool/netboot/test/build.sh 5, or pass --smoke-elf)"
    ip = args.netload_ip
    t0 = time.time()
    rc, out = netload_remote(["load", elf, "--host", ip, "--max-run-ms", str(SMOKE_MAX_RUN_MS),
                              "--wait", "30", "--force"], timeout=120)
    last = out.splitlines()[-1] if out else "no output"
    if rc == 2:
        return None, (f"netboot supervisor not answering at {ip} ({last}); "
                      "the netload run bootstraps it over the healthy hw_server")
    if rc != 0:
        return False, f"load {name} rc={rc}: {last}"
    m = re.search(r"runs=(\d+)", out)
    while time.time() - t0 < SMOKE_DONE_S:
        st_rc, st = netload_remote(["status", "--host", ip, "--connect-timeout", "3"], timeout=30, progress=False)
        if st_rc == 0 and (not m or f"runs={m.group(1)}" in st):
            if "state=exited" in st:
                return True, f"{name} ran and exited in {time.time() - t0:.1f}s"
            if "state=watchdog-timeout" in st or "state=killed" in st:
                return False, f"{name} did not exit by itself: {st.splitlines()[0]}"
        time.sleep(0.5)
    return False, f"{name} still running after {SMOKE_DONE_S}s"


def hws_need_jtag(why):
    """Deferred netload preflight: hw_server is checked only before the first JTAG
    use (the fast path never touches it). False, with the reason printed, if it
    cannot be made healthy."""
    if _preflight is None:  # --skip-preflight
        return True
    try:
        _preflight.need_jtag(why)
        return True
    except hwspreflight.PreflightError as e:
        print(f"\n{e.msg}")
        return False


def _run_with_preflight(args, lk):
    """hws preflight (inside the lock), the run, then palmyra full mode's hws up."""
    global _preflight
    if args.skip_preflight:
        print("[hws] --skip-preflight: not checking hw_server")
        return _main(args)
    pf = hwspreflight.Preflight(palip, HWS_TYPE, lock=lk, smoke=lambda: hws_smoke(args),
                                max_restarts=args.hws_restarts, quiet=devwatch.quiet)
    mode = preflight_mode(args)
    print(f"[hws] preflight ({HWS_TYPE}, {mode}) on {palip}")
    try:
        pf.before_run(mode, defer_netload=not args.hws_eager)
    except hwspreflight.PreflightError as e:
        print(f"\n{e.msg}")
        print("=" * 60 + "\nTest FAILED (hws preflight)\n" + "=" * 60)
        sys.exit(e.rc)
    _preflight = pf
    try:
        return _main(args)
    finally:
        _preflight = None
        pf.after_run()


def _watchdog(args, heartbeat):
    """Start the self-watchdog only after the lock is held (waiting is not a stall)."""
    wd = devwatch.Watchdog(stall_secs=args.stall_secs, max_run=args.max_run, heartbeat=heartbeat)
    devwatch.current = wd
    return wd


def _main(args):
    """Main test function."""

    # Step 0: Find ELF file using smart selection
    local_elf = find_elf_file(args.elf_file, auto_yes=args.yes)
    if local_elf is None:
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Palboard ELF Test Script")
    print(f"Board: {boardname}" + (f" (rev {BOARD_REV.upper()})" if IS_VEK385 else "") + f", console {CONSOLE_PORT}")
    print(f"Host: {host}")
    print(f"ELF File: {local_elf}")
    print("=" * 60)
    
    # Step 1: Copy ELF file to remote server
    # --netload streams the ELF through the proxy: no copy of the target ELF.
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
    try:
        if args.netload:
            conn1, conn2, console_thread, elf_ok = netload_start(args, local_elf, collected_output)
            if not elf_ok:
                print("\n" + "=" * 60)
                print("Test FAILED (netload)")
                print("=" * 60)
                sys.exit(1)
        else:
            conn1, conn2, elf_ok = jtag_start(args, remote_elf_path)
            if elf_ok is None:
                sys.exit(1)

        if console_thread is None:
            log.debug("main: starting console_reader thread...")
            console_thread = threading.Thread(
                target=console_reader,
                args=(conn2, console_output_queue, stop_console_thread)
            )
            console_thread.start()
            log.debug("main: console_reader thread started")

        finished = None
        if args.netload:
            def finished():  # the app returned through its patched _exit, or ran past its max run time
                rc, out = netload_remote(["status", "--host", args.netload_ip, "--connect-timeout", "3"], timeout=30,
                                         progress=False)
                if rc == 0 and ("state=watchdog-timeout" in out or "state=killed" in out):
                    print(f"\n[netload] the supervisor ended the wait: {out.splitlines()[0]}")
                return rc == 0 and any(f"state={st}" in out for st in ("exited", "watchdog-timeout", "killed"))
        max_wait = None
        if args.netload:
            max_wait = args.netload_max_run_ms / 1000.0 + 120  # watchdog + margin for the status probe
        if max_wait:
            # the supervisor watchdog bounds a silent run; don't let --stall-secs cut it first
            with devwatch.quiet(max_wait, "netload app run (supervisor watchdog bounds it)"):
                program_done, output_detected = wait_console(collected_output, finished, max_wait)
        else:
            program_done, output_detected = wait_console(collected_output, finished, max_wait)

        # Stop console reader
        log.debug("\nmain: signaling console_reader thread to stop...")
        stop_console_thread.set()
        if console_thread:
            log.debug("main: joining console_reader thread (timeout=5)...")
            console_thread.join(timeout=5)
            if console_thread.is_alive():
                log.warning("main: console_reader thread did not stop within 5s")
            else:
                log.debug("main: console_reader thread joined")

        # Print summary
        print("\n" + "=" * 60)
        print("CONSOLE OUTPUT SUMMARY:")
        print("=" * 60)

        if not output_detected:
            print("[No output received from com0 serial console]")

        # Step 6: Go back to conn1, exit xsdb and power off
        if args.netload:
            print("\n>>> --netload: board left powered on, netboot supervisor stays resident")
        elif not args.nonreboot:
            log.debug("main: cleaning up Connection 1 (exit_xsdb_and_poweroff)...")
            print("\n>>> Cleaning up Connection 1...")
            exit_xsdb_and_poweroff(conn1)
            log.debug("main: exit_xsdb_and_poweroff done")
        else:
            log.debug("main: nonreboot mode, skipping cleanup entirely")
            print("\n>>> --nonreboot: skipping cleanup, board left powered on for manual debug")

        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
        if args.netload and not program_done:
            sys.exit(1)

    except pexpect.TIMEOUT as e:
        log.error("main: pexpect.TIMEOUT: %s", e)
        print(f"\nError: Command timed out - {e}")
        if conn1 and not args.nonreboot and not args.netload:
            try:
                exit_xsdb_and_poweroff(conn1)
            except Exception:
                pass
        elif conn1:
            print("[--nonreboot/--netload] Skipping cleanup, board left as-is")
        sys.exit(1)
    except pexpect.EOF as e:
        log.error("main: pexpect.EOF: %s", e)
        print(f"\nError: Connection closed unexpectedly - {e}")
        sys.exit(1)
    except Exception as e:
        log.error("main: exception: %s", e, exc_info=True)
        print(f"\nError: {e}")
        if conn1 and not args.nonreboot and not args.netload:
            try:
                exit_xsdb_and_poweroff(conn1)
            except Exception:
                pass
        elif conn1:
            print("[--nonreboot/--netload] Skipping cleanup, board left as-is")
        sys.exit(1)
    finally:
        _close_all(console_thread, conn1, conn2)


def jtag_start(args, remote_elf_path):
    """The JTAG flow: program BOOT.BIN, open com0, dow -force, con.

    Returns (conn1, conn2, elf_ok); elf_ok None means the run already failed."""
    log.debug("main: setting up first connection...")
    print("\n>>> Setting up first connection...")
    conn1 = setup_first_connection(nonreboot=args.nonreboot)
    log.debug("main: first connection ready")

    log.debug("main: setting up second connection...")
    print("\n>>> Setting up second connection...")
    conn2 = setup_second_connection()
    log.debug("main: second connection ready")

    log.debug("main: downloading ELF file...")
    print("\n>>> Downloading ELF file...")
    elf_ok = download_elf_and_continue(conn1, remote_elf_path)
    log.debug("main: download_elf_and_continue returned %s", elf_ok)
    if elf_ok:
        return conn1, conn2, True

    # PLM stalled -- skip console wait, go straight to cleanup
    print("\n[Skipping console wait due to PLM failure]")
    if not args.nonreboot:
        print("\n>>> Cleaning up Connection 1...")
        exit_xsdb_and_poweroff(conn1)
    else:
        print("\n>>> --nonreboot: skipping cleanup, board left as-is for manual debug")
    print("\n" + "=" * 60)
    print("Test FAILED (PLM stall)")
    print("=" * 60)
    return conn1, conn2, None


def wait_console(collected_output, finished=None, max_wait=None):
    """Echo com0 output until the program completes, goes quiet, or MAX_WAIT.

    Returns (program_done, output_detected).
      - Finish early when "device_teardown done" is seen (program completed), or
        when `finished()` (polled every 0.5 s; --netload passes "supervisor reports
        the app exited") says so -- apps that print neither marker then end on
        their real exit instead of the 300 s no-output timeout
      - Abort early if no output arrives within NO_OUTPUT_TIMEOUT seconds
      - Hard cap at MAX_WAIT seconds as safety bound
    """
    MAX_WAIT = 600        # absolute maximum wait (seconds)
    NO_OUTPUT_TIMEOUT = 300  # abort if no output for this long (debug snapshot reads many regs)
    if max_wait is not None and max_wait > MAX_WAIT:
        # --netload: the supervisor watchdog (--netload-max-run-ms) bounds the run and
        # `finished()` reports its end, so a long silent graph_run must not be cut at
        # 600 s / 300 s quiet.
        MAX_WAIT = int(max_wait)
        NO_OUTPUT_TIMEOUT = int(max_wait)
    POLL_INTERVAL = 0.05  # check queue every N seconds

    print(f"\n>>> Waiting for console output (max {MAX_WAIT}s, no-output timeout {NO_OUTPUT_TIMEOUT}s)...")
    output_detected = bool(collected_output)
    program_done = False
    tail = ""  # markers can straddle two console chunks
    last_probe = time.time()
    start_time = time.time()
    last_output_time = start_time

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
            tail = (tail + chunk)[-512:]
            if "device_teardown done" in tail or "Not tearing down partition" in tail:
                program_done = True
        if got_new:
            last_output_time = time.time()
        # 0.5 s: one STATUS over the multiplexed ssh costs ~0.1 s, and for apps that
        # print no teardown line this probe IS the completion signal.
        if finished and not program_done and time.time() - last_probe >= 0.5:
            last_probe = time.time()
            program_done = finished()

        if program_done:
            log.debug("main: program_done detected, draining trailing output...")
            time.sleep(0.05)  # "device_teardown done" is the app's last line
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
            print(f"\n[ERROR: No output received for {NO_OUTPUT_TIMEOUT}s — "
                  "UART/PS may not be initialized (PLM issue or board problem)]")
            break
        time.sleep(POLL_INTERVAL)
    return program_done, output_detected


def _close_all(console_thread, conn1, conn2):
    """Stop the console reader and close both ssh sessions."""
    log.debug("main finally: stopping console thread...")
    stop_console_thread.set()
    if console_thread and console_thread.is_alive():
        console_thread.join(timeout=5)
        if console_thread.is_alive():
            log.warning("main finally: console_reader thread still alive after join!")
    for name, conn in (("conn1", conn1), ("conn2", conn2)):
        if conn:
            try:
                leave_console(conn) if conn is conn2 else conn.close(force=True)
            except Exception as e:
                log.debug("main finally: %s.close() error: %s", name, e)
    log.debug("main finally: all cleanup done")
    print("\nConnections closed.")


if __name__ == "__main__":
    main()