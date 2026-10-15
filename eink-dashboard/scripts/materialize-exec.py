#!/usr/bin/env python3
"""Run a program with permission to download Dropbox online-only files (DECISIONS #72).

Dropbox keeps files "online-only" as APFS dataless files. A program started from Terminal
downloads them on first read, but a launchd job is not allowed to: its reads fail with
EDEADLK ("Unknown system error -11" in Node). This sets the process's I/O policy to
materialize dataless files, then replaces itself with the real program, whose children
(the import tool, sips, claude) inherit the policy.

    /usr/bin/python3 materialize-exec.py /opt/homebrew/bin/node series-mailbox.mjs run
"""
import ctypes
import os
import sys

IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES = 3
IOPOL_SCOPE_PROCESS = 0
IOPOL_MATERIALIZE_DATALESS_FILES_ON = 2

libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
if libc.setiopolicy_np(IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS,
                       IOPOL_MATERIALIZE_DATALESS_FILES_ON) != 0:
    sys.exit("setiopolicy_np failed: errno %d" % ctypes.get_errno())
os.execv(sys.argv[1], sys.argv[1:])
