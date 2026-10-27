"""Keeping a downloaded model from becoming a problem.

Two halves, deliberately separate:

* :mod:`localcode.security.sandbox` — *containment*: the process that reads a
  model has no internet and no access to your files beyond the model itself,
  so a bug in a parser cannot become a bug in your home directory.
* :mod:`localcode.security.scan` — *inspection*: what a model's files are,
  before anything opens them, with a real check of the one format that executes
  code when read (Python pickles).

Neither is antivirus in the marketing sense and the app never calls it that:
there are no signatures of known malware here. What there is: a format policy,
a pickle disassembler, and a sandbox that makes the whole question much less
interesting.
"""

from localcode.security.sandbox import (
    SandboxPlan,
    SandboxUnavailable,
    available,
    build_profile,
    describe,
    wrap,
)
from localcode.security.scan import (
    Finding,
    ScanReport,
    scan_path,
    scan_pickle_bytes,
    scan_pickle_stream,
)

__all__ = [
    "SandboxPlan",
    "SandboxUnavailable",
    "available",
    "build_profile",
    "describe",
    "wrap",
    "Finding",
    "ScanReport",
    "scan_path",
    "scan_pickle_bytes",
    "scan_pickle_stream",
]
