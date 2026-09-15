"""Shared pexpect helpers for driving an interactive ssh -> systest session.

Extracted so both apppaltest.py and schedule_debug_server.py build the ssh
command and wait for the post-login shell prompt in exactly the same way.
"""


# Match a bare interactive shell prompt at end of buffer: $, # or > possibly
# followed by trailing whitespace. Kept in sync with setup_first_connection in
# apppaltest.py.
_SHELL_PROMPT_PATTERNS = [r'\$\s*$', r'#\s*$', r'>\s*$']


def ssh_command(ssh_target):
    """Return the ssh command string used to open the interactive session.

    X forwarding (-X) is required because the downstream xsdb/hw_server flow
    expects a forwarded display, matching the manual apppaltest.py path.
    """
    return f"ssh -X {ssh_target}"


def expect_shell_after_ssh(child, timeout=60):
    """Wait for the remote shell prompt after ssh login.

    Returns the pexpect match index for the prompt pattern that matched.
    """
    return child.expect(_SHELL_PROMPT_PATTERNS, timeout=timeout)
