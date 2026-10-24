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

# Read username, IP, and board name from environment variables
username = os.environ.get("USERNAME")
palip = os.environ.get("PALIP")
boardname = os.environ.get("BOARDNAME")

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
                
                # Recheck the environment variables
                username = os.environ.get("USERNAME")
                palip = os.environ.get("PALIP")
                boardname = os.environ.get("BOARDNAME")
            else:
                print(f"Warning: Failed to source envlocal.sh: {result.stderr}")
        except Exception as e:
            print(f"Warning: Error sourcing envlocal.sh: {e}")

if not username or not palip or not boardname:
    print("Error: Please set USERNAME, PALIP, and BOARDNAME environment variables")
    print("Example: export USERNAME=aaaaa && export PALIP=10.23.***.*** && export BOARDNAME=pal***")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Or create {os.path.join(script_dir, 'envlocal.sh')} with these exports")
    sys.exit(1)

host = f"{username}@{palip}"

# Configuration
PALBOARD_SCRIPTS_DIR = f"/proj/xsjsswstaff/{username}/palboard_scripts"
PALBOARD_BIN = f"/home/{username}/palboard/BOOT.BIN"
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
    if nonreboot:
        print("[Connection 1] Connected, starting systest-client (nonreboot mode)...")
        log.debug("setup_first_connection: sending systest-client")
        child.sendline("/opt/systest/common/bin/systest-client")
    else:
        print("[Connection 1] Connected, starting systest...")
        log.debug("setup_first_connection: sending systest")
        child.sendline("/bin/systest")
    child.expect(r'Systest[#>]', timeout=60)
    log.debug("setup_first_connection: systest prompt received")

    # Step 3: Become palboard - wait for Systest# prompt after board info.
    # In nonreboot mode the board is already up and owned by an existing
    # session, so skip 'become' to avoid taking over/holding the board.
    if not nonreboot:
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
    xsdb_run(child, "connect -url TCP:10.23.224.213:3121" if nonreboot else "conn", 60, "connect")
    print("[Connection 1] Connected, targeting device 1...")
    xsdb_run(child, "tar 1", 60)
    print("[Connection 1] Programming Palboard.BIN...")
    try:
        xsdb_run(child, f"device program {PALBOARD_BIN}", 120, "device program")
    except RuntimeError as e:
        raise RuntimeError(f"BOOT.BIN programming failed ({e}). The board may need a power cycle; "
                           "run 'plm log' for details.") from e
    print("[Connection 1] Device programmed (100%), letting the PLM settle...")
    time.sleep(5)  # now measured from the END of programming
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
    print("[Connection 2] In systest, connecting to com0...")

    # Step 3: Connect to com0 (no output until ELF runs on first connection)
    log.debug("setup_second_connection: sending 'connect com0'")
    child.sendline("connect com0")
    child.expect(r'Connecting to device com0.*escape', timeout=60)
    log.debug("setup_second_connection: com0 connected")
    print("[Connection 2] Connected to com0, listening for output...")

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
    index = child.expect([r'Successfully downloaded', r'PLM stalled'], timeout=DOWNLOAD_TIMEOUT_S)
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
# One persistent ssh connection to the lab host, reused by every netload ssh/scp:
# a fresh connection costs 0.55-0.85 s here, a multiplexed one a few tens of ms.
SSH_MUX = ["-o", "ControlMaster=auto", "-o", "ControlPersist=4h", "-o", f"ControlPath=/tmp/apppal-{username}-%C"]


def ensure_netboot_built():
    """Netload needs two local artifacts: netboot_proxy.py (in the repo) and
    build/netboot.elf (built by src/tool/netboot/build.sh). Build the ELF when it is
    missing or older than any of its sources, so a fresh checkout or an edited
    supervisor never bootstraps a stale image. The mtime check is free when
    nothing changed."""
    proxy = os.path.join(NETBOOT_LOCAL_DIR, "netboot_proxy.py")
    if not os.path.isfile(proxy):
        raise RuntimeError(f"{proxy} missing (it is part of the repo; check the checkout)")
    elf = os.path.join(NETBOOT_LOCAL_DIR, "build", "netboot.elf")
    srcs = [p for pat in ("supervisor/*.c", "supervisor/*.S", "supervisor/*.h", "lwip/*", "netboot_proto.h",
                          "build.sh") for p in glob.glob(os.path.join(NETBOOT_LOCAL_DIR, pat))]
    newest = max((os.path.getmtime(p) for p in srcs), default=0)
    if os.path.isfile(elf) and os.path.getmtime(elf) >= newest:
        return
    why = "missing" if not os.path.isfile(elf) else "older than its sources"
    print(f"[netload] netboot.elf {why}: building it (src/tool/netboot/build.sh)...")
    r = subprocess.run(["bash", os.path.join(NETBOOT_LOCAL_DIR, "build.sh")], capture_output=True, text=True,
                       timeout=1800)
    if r.returncode != 0 or not os.path.isfile(elf):
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"building netboot.elf failed (exit {r.returncode}):\n{tail}\n"
                           "The A78 cross compiler must be on PATH (source Vitis settings64.sh).")
    print(f"[netload] built {elf}")


def stage_if_changed(local_paths):
    """Copy files to NETBOOT_REMOTE_DIR on the lab host, skipping any whose remote
    copy already has the same SHA-256 (one ssh round trip to check)."""
    import hashlib
    names = [os.path.basename(p) for p in local_paths]
    probe = (f"mkdir -p {NETBOOT_REMOTE_DIR} && cd {NETBOOT_REMOTE_DIR} && "
             f"(sha256sum {' '.join(shlex.quote(n) for n in names)} 2>/dev/null; true)")
    r = subprocess.run(["ssh", *SSH_MUX, host, f"bash -c {shlex.quote(probe)}"], capture_output=True, text=True,
                       timeout=30)
    remote = {ln.split()[1]: ln.split()[0] for ln in r.stdout.splitlines() if len(ln.split()) == 2}
    stale = [p for p, n in zip(local_paths, names)
             if remote.get(n) != hashlib.sha256(open(p, "rb").read()).hexdigest()]
    if stale:
        subprocess.run(["scp", "-q", *SSH_MUX, *stale, f"{host}:{NETBOOT_REMOTE_DIR}/"], check=True,
                       capture_output=True, timeout=120)
    print(f"[netload] staged on {palip}: {', '.join(os.path.basename(p) for p in stale) or 'nothing (up to date)'}")


def netboot_stage_proxy():
    """The relay netload.py talks through; the target ELF itself is never copied."""
    stage_if_changed([os.path.join(NETBOOT_LOCAL_DIR, "netboot_proxy.py")])


def netboot_stage_supervisor():
    """netboot.elf for a JTAG bootstrap (xsdb on the lab host `dow`s it from there)."""
    local_elf = os.path.join(NETBOOT_LOCAL_DIR, "build", "netboot.elf")
    if not os.path.isfile(local_elf):
        raise RuntimeError(f"{local_elf} missing: run src/tool/netboot/build.sh first")
    stage_if_changed([local_elf])
    return f"{NETBOOT_REMOTE_DIR}/netboot.elf"


_netload_mod = []


class _Tee:
    """stdout for an in-process netload: shown live AND kept for the caller.
    A busy board makes `load --wait` retry for up to --netload-wait seconds; with
    the output only buffered, those "busy, N s left -- waiting" lines appeared
    after the wait, i.e. apppaltest looked hung."""

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
    """Run netload.py's main() in this process and reach the board through
    netboot_proxy.py on the lab host over the multiplexed ssh: the ELF streams
    through the ssh pipe, nothing is scp'd. Returns (exit status, printed text);
    live=True also prints netload's lines as they come (load: busy waits)."""
    import contextlib
    import importlib.util
    if not _netload_mod:
        spec = importlib.util.spec_from_file_location("netload", os.path.join(NETBOOT_LOCAL_DIR, "netload.py"))
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
    """com0 through `ssh -tt host systest-client` on the multiplexed connection:
    no login shell to wait for (that alone was ~0.8 s of setup_second_connection)."""
    child = pexpect.spawn("ssh", [*SSH_MUX, "-tt", host, "/opt/systest/common/bin/systest-client"],
                          encoding='utf-8', codec_errors='replace', timeout=60)
    child.expect(r'Systest[#>]', timeout=60)
    child.sendline("connect com0")
    child.expect(r'Connecting to device com0.*escape', timeout=60)
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
        xsdb_run(child, f"connect -url TCP:{palip}:3121", 60, "connect")
        xsdb_run(child, 'targets -set -nocase -filter {name =~ "*A78*#0.1*"}', 30, "select app core #0.1")
        xsdb_run(child, "catch {stop}", 30)
        xsdb_run(child, "rst -proc", 30, "rst -proc (app core)")
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
    load_argv = ["load", local_elf, "--host", ip, "--max-run-ms", str(args.netload_max_run_ms),
                 "--wait", str(args.netload_wait)] + ([] if args.netload_no_force else ["--force"])
    if not args.netload_bootstrap:
        # The console attaches while the ELF streams; netload holds back the last
        # 4 KB until it is attached, so the app cannot print before we listen.
        box = {}
        opener = threading.Thread(target=lambda: box.setdefault("c", start_console(fast=True)))
        opener.start()
        netboot_stage_proxy()

        def console_attached():
            opener.join(timeout=60)
            if "c" not in box:
                raise ConnectionError("com0 console did not attach")

        t0 = time.time()
        print(f"[netload] asking the supervisor at {ip} to run {os.path.basename(local_elf)} "
              f"(waits up to {args.netload_wait:g} s if the board is busy)", flush=True)
        rc, out = netload_remote(load_argv, gate=console_attached, live=True)
        opener.join(timeout=60)
        if "c" not in box:
            print("\n[ERROR] com0 console did not attach")
            return None, None, None, False
        conn2, th = box["c"]
        if rc == 0:
            print(f"[netload] streamed {local_elf} through the proxy on {palip} in {time.time() - t0:.2f}s (no scp)")
            return None, conn2, th, True
        if rc == 3:
            print(f"\n[ERROR] board still busy after --netload-wait {args.netload_wait:g} s with a run that is "
                  f"within its max run time; not power-cycling it. Retry later or raise --netload-wait")
            return None, conn2, th, False
        if rc == 4:
            print("\n[ERROR] the previous app ran past its max run time; drop --netload-no-force to terminate it")
            return None, conn2, th, False
        if rc not in (2, 5) and "re-bootstrap" not in out:
            return None, conn2, th, False  # a real refusal (bad ELF etc.), not "no supervisor"
        m = re.search(r"park_addr=0x([0-9a-f]+)", out)
        if rc == 5 and m and netboot_park_app_core(int(m.group(1), 16)):
            time.sleep(0.5)
            st_rc, st = netload_remote(["status", "--host", ip, "--connect-timeout", "3"], timeout=30)
            if st_rc == 0 and "app core parked" in st:
                print("[netload] app core parked by JTAG; sending the ELF again", flush=True)
                rc, out = netload_remote(load_argv, live=True)
                if rc == 0:
                    return None, conn2, th, True
            print(f"[netload] JTAG park did not free the app core ({st.splitlines()[-1] if st else 'no answer'}); "
                  "falling back to a re-bootstrap", flush=True)
        stop_console(conn2, th)
        why = (f"supervisor not answering at {ip}" if rc == 2 else
               "the previous app is past its max run time and this PDI cannot stop it (forced run)" if rc == 5
               else "app core still busy from the previous run")
    else:
        why = "--netload-bootstrap"
    print(f"\n>>> {why}: loading the netboot supervisor over JTAG (one-time)...")
    netboot_stage_proxy()
    # `device program` on a live system aborts at ~18%, so the bootstrap power-cycles
    # first, through the non-owning systest-client session. The app finishing never
    # powers anything off, and the fast path above never reaches this.
    conn1 = setup_first_connection(nonreboot=True)
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
        "--netload-wait",
        type=float,
        default=120.0,
        help="If the board is busy with a run still within its max run time, wait up to this many seconds "
             "for it (--netload only, default 120); the supervisor says how long the run has left"
    )
    parser.add_argument(
        "--netload-no-force",
        action="store_true",
        default=False,
        help="Do not terminate a previous app that ran past its max run time (default: force it off and run "
             "this ELF; on a PDI that cannot stop the app core that means a re-bootstrap)"
    )
    args = parser.parse_args()

    # Step 0: Find ELF file using smart selection
    local_elf = find_elf_file(args.elf_file, auto_yes=args.yes)
    if local_elf is None:
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Palboard ELF Test Script")
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
                rc, out = netload_remote(["status", "--host", args.netload_ip, "--connect-timeout", "3"], timeout=30)
                if rc == 0 and ("state=watchdog-timeout" in out or "state=killed" in out):
                    print(f"\n[netload] the supervisor ended the wait: {out.splitlines()[0]}")
                return rc == 0 and any(f"state={st}" in out for st in ("exited", "watchdog-timeout", "killed"))
        program_done, output_detected = wait_console(collected_output, finished)

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


def wait_console(collected_output, finished=None):
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
                conn.close(force=True)
            except Exception as e:
                log.debug("main finally: %s.close() error: %s", name, e)
    log.debug("main finally: all cleanup done")
    print("\nConnections closed.")


if __name__ == "__main__":
    main()