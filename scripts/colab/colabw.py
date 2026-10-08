"""Runs the google-colab-cli on Windows: console.py imports the POSIX-only
termios/tty at import time (only the interactive console uses them)."""
import sys
import types

for name in ('termios', 'tty'):
    sys.modules.setdefault(name, types.ModuleType(name))

from colab_cli.cli import main  # noqa: E402

sys.exit(main())
