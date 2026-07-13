"""
path_utils.py

Turn a path typed (or dragged, or pasted) on the command line into something the
filesystem will actually resolve -- with particular care for Windows, where the
shell and drag-and-drop routinely decorate paths in ways that make an otherwise
valid file look "not found".
"""

import os


def clean_path_arg(raw: str) -> str:
    """Normalize a filesystem path received as a CLI argument.

    Windows users frequently hit "file not found" on a path that is plainly
    correct, because of how the path reached the program:

      * Dragging a file onto a terminal, or copying its path from Explorer,
        wraps it in double quotes. Depending on the shell (notably PowerShell,
        or a quoted argument inside a ``.bat``/``.cmd`` wrapper) those quotes can
        survive into ``sys.argv``. ``os.path.abspath('"C:\\clips\\a.mp4"')`` then
        treats the leading quote as a *relative* path and prepends the current
        directory, so the file is never found.
      * Copied commands can carry trailing whitespace or a stray newline.
      * Paths may start with ``~`` for the user's home directory.

    Stripping surrounding quotes and whitespace, then expanding ``~``, fixes all
    of these and is a no-op on paths that were already clean.
    """
    if not raw:
        return raw

    s = raw.strip()
    # Peel off one or more layers of surrounding matching quotes. Handles the
    # single pair a shell leaves behind as well as the rare double-wrapped path.
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    return os.path.expanduser(s)
