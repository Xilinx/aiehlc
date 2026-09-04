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
            output = child.read_nonblocking(size=4096, timeout=1)
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
    index = child.expect([r'xsdb%', r'command not found', r'Unrecognized', pexpect.TIMEOUT], timeout=15)

    if index != 0:
        print("[Connection 1] xsdb not found, trying alternative path...")
        child.sendline(XSDB_ALT_PATH)
        child.expect(r'xsdb%', timeout=60)
    
    print("[Connection 1] In xsdb, connecting...")
    
    # Step 7: Connect
    if nonreboot:
        #child.sendline("hw_server -stcp:0.0.0.0:3121 &")
        #child.expect(r'xsdb%', timeout=60)
        child.sendline("connect -url TCP:10.23.224.213:3121")
    else:
        child.sendline("conn")
    child.expect(r'xsdb%', timeout=60)
    print("[Connection 1] Connected, targeting device 1...")
    
    # Step 8: Target 1
    child.sendline("tar 1")
    child.expect(r'xsdb%', timeout=60)
    print("[Connection 1] Programming Palboard.BIN...")
    
    # Step 9: Program device – detect PLM stall and abort early
    child.sendline(f"device program {PALBOARD_BIN}")
    index = child.expect([r'xsdb%', r'PLM stalled'], timeout=120)
    if index == 1:
        # Consume the rest of the error output up to the prompt
        child.expect(r'xsdb%', timeout=60)
        raise RuntimeError(
            "PLM stalled during BOOT.BIN programming. "
            "The board may need a power cycle. Run 'plm log' for details."
        )
    time.sleep(5)  # allow PLM to fully initialise before continuing
    print("[Connection 1] Device programmed, targeting device 20...")
    
    # Step 10: Target 20
    child.sendline("tar 20")
    child.expect(r'xsdb%', timeout=60)
    print("[Connection 1] Resetting processor...")
    
    # Step 11: Reset processor
    child.sendline("rst -proc")
    child.expect(r'xsdb%', timeout=60)

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
    log.debug("download_elf_and_continue: waiting for download result...")
    index = child.expect([r'Successfully downloaded', r'PLM stalled'], timeout=120)
    if index == 1:
        # Consume remaining output up to the prompt
        child.expect(r'xsdb%', timeout=60)
        print("\n[ERROR] PLM stalled during ELF download (dow -force).")
        print("BOOT.BIN did not initialize PS peripherals (UART etc.).")
        print("The board needs a full power cycle and re-programming of BOOT.BIN.")
        print("Run 'plm log' in xsdb for details.")
        return False
    child.expect(r'xsdb%', timeout=60)
    log.debug("download_elf_and_continue: download complete")
    print("[Connection 1] ELF download complete!")

    # Step 13: Continue execution
    print("[Connection 1] Continuing execution...")
    log.debug("download_elf_and_continue: sending 'con'...")
    child.sendline("con")
    child.expect(r'xsdb%', timeout=60)
    log.debug("download_elf_and_continue: 'con' completed, execution started")
    print("[Connection 1] Execution started!")
    return True


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
        return True, dest_elf
    except subprocess.CalledProcessError as e:
        print(f"Error copying ELF file via SCP: {e}")
        if e.stderr:
            print(f"    stderr: {e.stderr}")
        return False, None
    except subprocess.TimeoutExpired:
        print("Error: SCP timed out")
        return False, None


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
    success, remote_elf_path = copy_elf_to_remote(local_elf)
    if not success:
        print("Failed to copy ELF file, exiting...")
        sys.exit(1)
    
    conn1 = None
    conn2 = None
    console_thread = None

    try:
        # Step 2-3: Setup first connection and program device
        log.debug("main: setting up first connection...")
        print("\n>>> Setting up first connection...")
        conn1 = setup_first_connection(nonreboot=args.nonreboot)
        log.debug("main: first connection ready")

        # Step 4: Setup second connection for console output
        log.debug("main: setting up second connection...")
        print("\n>>> Setting up second connection...")
        conn2 = setup_second_connection()
        log.debug("main: second connection ready")

        # Step 5: Download ELF file
        log.debug("main: downloading ELF file...")
        print("\n>>> Downloading ELF file...")
        elf_ok = download_elf_and_continue(conn1, remote_elf_path)
        log.debug("main: download_elf_and_continue returned %s", elf_ok)

        if not elf_ok:
            # PLM stalled — skip console wait, go straight to cleanup
            print("\n[Skipping console wait due to PLM failure]")
            if not args.nonreboot:
                print("\n>>> Cleaning up Connection 1...")
                exit_xsdb_and_poweroff(conn1)
            else:
                print("\n>>> --nonreboot: skipping cleanup, board left as-is for manual debug")
            print("\n" + "=" * 60)
            print("Test FAILED (PLM stall)")
            print("=" * 60)
            sys.exit(1)

        # Start console reader thread after ELF is running
        log.debug("main: starting console_reader thread...")
        console_thread = threading.Thread(
            target=console_reader,
            args=(conn2, console_output_queue, stop_console_thread)
        )
        console_thread.start()
        log.debug("main: console_reader thread started")

        # Poll console output instead of fixed sleep:
        #   - Finish early when "device_teardown done" is seen (program completed)
        #   - Abort early if no output arrives within NO_OUTPUT_TIMEOUT seconds
        #   - Hard cap at MAX_WAIT seconds as safety bound
        MAX_WAIT = 600        # absolute maximum wait (seconds)
        NO_OUTPUT_TIMEOUT = 300  # abort if no output for this long (debug snapshot reads many regs)
        POLL_INTERVAL = 1     # check queue every N seconds

        print(f"\n>>> Waiting for console output (max {MAX_WAIT}s, no-output timeout {NO_OUTPUT_TIMEOUT}s)...")

        collected_output = []
        output_detected = False
        program_done = False
        start_time = time.time()
        last_output_time = start_time

        while True:
            #print("\ndebug: polling console output queue...\n")
            elapsed = time.time() - start_time
            since_last_output = time.time() - last_output_time

            # Check for new output in the queue
            got_new = False
            while not console_output_queue.empty():
                #print("\ndebug: new console output detected\n")
                chunk = console_output_queue.get()
                collected_output.append(chunk)
                print(chunk, end='', flush=True)
                output_detected = True
                got_new = True
                # Check for completion marker
                if "device_teardown done" in chunk or "Not tearing down partition" in chunk:
                    program_done = True

            #print(f"\ndebug: elapsed={elapsed:.1f}s, since_last_output={since_last_output:.1f}s, "
            #      f"output_detected={output_detected}, program_done={program_done}\n")
            if got_new:
                last_output_time = time.time()

            # Exit conditions (in priority order)
            if program_done:
                # Give a short grace period for any trailing output
                log.debug("main: program_done detected, draining trailing output...")
                time.sleep(2)
                while not console_output_queue.empty():
                    chunk = console_output_queue.get()
                    collected_output.append(chunk)
                    print(chunk, end='', flush=True)
                print(f"\n[Program completed after {elapsed:.0f}s]")
                log.debug("main: exiting poll loop (program_done)")
                break

            if elapsed >= MAX_WAIT:
                log.debug("main: exiting poll loop (MAX_WAIT reached)")
                print(f"\n[Max wait {MAX_WAIT}s reached]")
                break

            if output_detected and since_last_output >= NO_OUTPUT_TIMEOUT:
                log.debug("main: exiting poll loop (no new output for %ds)", NO_OUTPUT_TIMEOUT)
                print(f"\n[No new output for {NO_OUTPUT_TIMEOUT}s — program may be hung]")
                break

            if not output_detected and since_last_output >= NO_OUTPUT_TIMEOUT:
                log.debug("main: exiting poll loop (no output ever received, %ds)", NO_OUTPUT_TIMEOUT)
                print(f"\n[ERROR: No output received for {NO_OUTPUT_TIMEOUT}s — "
                      "UART/PS may not be initialized (PLM issue or board problem)]")
                break

            time.sleep(POLL_INTERVAL)

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
        if not args.nonreboot:
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

    except pexpect.TIMEOUT as e:
        log.error("main: pexpect.TIMEOUT: %s", e)
        print(f"\nError: Command timed out - {e}")
        if conn1 and not args.nonreboot:
            try:
                exit_xsdb_and_poweroff(conn1)
            except Exception:
                pass
        elif conn1 and args.nonreboot:
            print("[--nonreboot] Skipping cleanup, board left as-is")
        sys.exit(1)
    except pexpect.EOF as e:
        log.error("main: pexpect.EOF: %s", e)
        print(f"\nError: Connection closed unexpectedly - {e}")
        sys.exit(1)
    except Exception as e:
        log.error("main: exception: %s", e, exc_info=True)
        print(f"\nError: {e}")
        if conn1 and not args.nonreboot:
            try:
                exit_xsdb_and_poweroff(conn1)
            except Exception:
                pass
        elif conn1 and args.nonreboot:
            print("[--nonreboot] Skipping cleanup, board left as-is")
        sys.exit(1)
    finally:
        # Cleanup - ensure thread is stopped and connections closed
        log.debug("main finally: stopping console thread...")
        stop_console_thread.set()

        if console_thread and console_thread.is_alive():
            log.debug("main finally: joining console_reader thread (timeout=5)...")
            console_thread.join(timeout=5)
            if console_thread.is_alive():
                log.warning("main finally: console_reader thread still alive after join!")
            else:
                log.debug("main finally: console_reader thread joined")

        if conn1:
            log.debug("main finally: closing conn1...")
            try:
                conn1.close()
                log.debug("main finally: conn1 closed")
            except Exception as e:
                log.debug("main finally: conn1.close() error: %s", e)

        if conn2:
            log.debug("main finally: closing conn2...")
            try:
                conn2.close()
                log.debug("main finally: conn2 closed")
            except Exception as e:
                log.debug("main finally: conn2.close() error: %s", e)

        log.debug("main finally: all cleanup done")
        print("\nConnections closed.")


if __name__ == "__main__":
    main()