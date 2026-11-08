"""Make the interactive MuJoCo viewer (--viewer) work from anaconda's Python on Linux.

Anaconda bundles an old libstdc++. Recent Mesa drivers (e.g. radeonsi built against LLVM 20)
need a newer one, so inside conda they fail to load and GLX reports "No GLXFBConfigs
returned" / mujoco "gladLoadGL error". Preloading the system libstdc++ fixes it, but
LD_PRELOAD only takes effect at process start, so we restart the script once with it set.

Call relaunch_for_viewer() at the very top of a script, before importing mujoco.
It does nothing unless: Linux, --viewer was passed, Python is a conda Python, the system
library exists, and LD_PRELOAD doesn't already contain libstdc++ (which also prevents loops).
"""

from __future__ import annotations

import os
import sys

SYSTEM_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


def _is_conda_python() -> bool:
    # sys.base_prefix: a venv made from conda's Python (e.g. the repo's .venv) still uses conda's
    # old libstdc++, but its sys.prefix is the venv
    return any(os.path.isdir(os.path.join(p, "conda-meta")) for p in {sys.prefix, sys.base_prefix})


def relaunch_for_viewer() -> None:
    if sys.platform != "linux" or "--viewer" not in sys.argv:
        return
    if not _is_conda_python() or not os.path.exists(SYSTEM_LIBSTDCXX):
        return
    preload = os.environ.get("LD_PRELOAD", "")
    if "libstdc++" in preload:
        return
    os.environ["LD_PRELOAD"] = SYSTEM_LIBSTDCXX + (":" + preload if preload else "")
    print(f"[viewer_env] conda Python: restarting with LD_PRELOAD={SYSTEM_LIBSTDCXX} for GLX", flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)
