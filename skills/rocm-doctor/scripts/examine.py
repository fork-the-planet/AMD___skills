#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Read-only system examination for the `rocm-doctor` skill.

This is the first script the skill runs once it has decided the user's
framework actually touches the system ROCm install (so: PyTorch, llama.cpp,
and anything built against `/opt/rocm`, but NOT Lemonade / LM Studio /
Ollama, which ship their own runtime).

The script collects the minimum set of facts needed to disambiguate the
twelve known misconfigurations in `reference.md`. It never installs or
removes packages, never changes group membership, and never edits files.

Exit codes:
  0 = examination ran; results emitted. The agent should pass the JSON to
      `diagnose.py` next.
  2 = wrong platform (not Linux, no AMD GPU, NVIDIA-only, etc.). The
      agent should stop and route the user instead of running diagnose.
  3 = examination ran but something prevented a key probe from completing
      and the agent should warn the user before continuing.

Usage:
    python scripts/examine.py
    python scripts/examine.py --json
    python scripts/examine.py --framework pytorch
    python scripts/examine.py --framework llama-cpp --json

The optional `--framework` flag scopes the framework-specific probes
(e.g. running PyTorch's `torch.version.hip`). When omitted the script
probes everything it can detect without launching a Python interpreter
for a framework that may not be installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Environment variables that silently change ROCm/HIP behaviour. We record
# every one of these, even when empty, so `diagnose.py` can see both the
# value and the fact that it is unset (which is itself a signal for some
# misconfigurations -- e.g. ROCM_PATH being unset is fine, but the user
# having set HSA_OVERRIDE_GFX_VERSION on a supported GPU is suspicious).
TRACKED_ENV_VARS = (
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",          # PyTorch HIP also honours this name.
    "GPU_DEVICE_ORDINAL",
    "ROCM_PATH",
    "ROCM_HOME",
    "PYTORCH_ROCM_ARCH",
    "HCC_AMDGPU_TARGET",
    "AMDGPU_TARGETS",
    "LD_LIBRARY_PATH",
    "PATH",
)

# Files the amdgpu-install pipeline drops on APT-based systems. Presence
# of these tells us "installed via amdgpu-install", absence + apt-installed
# ROCm packages tells us "installed via plain apt", and absence of both
# with a populated /opt/rocm typically means "tarball or pip wheel".
AMDGPU_INSTALL_MARKERS = (
    "/etc/apt/sources.list.d/amdgpu.list",
    "/etc/apt/sources.list.d/rocm.list",
    "/etc/apt/sources.list.d/radeon.list",
    "/etc/yum.repos.d/amdgpu.repo",
    "/etc/yum.repos.d/rocm.repo",
)

# Containers we can detect cheaply from /proc/1/cgroup or marker files.
CONTAINER_MARKERS = {
    "/.dockerenv": "docker",
    "/run/.containerenv": "podman",
}


@dataclass
class GPU:
    name: str = ""
    gfx_target: str = ""        # e.g. gfx1151
    pci_id: str = ""
    is_apu: bool | None = None
    is_amd: bool = False


@dataclass
class Device:
    path: str
    exists: bool
    mode: str = ""              # e.g. "crw-rw----"
    owner_user: str = ""
    owner_group: str = ""
    user_can_read: bool | None = None
    user_can_write: bool | None = None


@dataclass
class Examination:
    # --- platform ---
    os_family: str = "unknown"          # linux | windows | other
    os_version: str = ""
    distro_id: str = ""                 # ubuntu, debian, rhel, fedora, ...
    distro_version: str = ""
    kernel_release: str = ""
    kernel_cmdline: str = ""

    # --- hardware ---
    cpu_vendor: str = "unknown"
    cpu_model: str = ""
    gpus: list[GPU] = field(default_factory=list)
    has_amd_gpu: bool = False
    has_nvidia_gpu: bool = False
    has_apu: bool = False
    has_discrete_amd: bool = False

    # --- driver / runtime ---
    amdgpu_loaded: bool | None = None
    amdgpu_blacklisted_in: list[str] = field(default_factory=list)
    amdkfd_loaded: bool | None = None
    secure_boot: str = "unknown"        # enabled | disabled | unknown
    iommu_kernel_param: str = ""        # value of iommu=, empty if unset
    kfd: Device | None = None
    render_devices: list[Device] = field(default_factory=list)

    # --- user / groups ---
    user_name: str = ""
    user_groups: list[str] = field(default_factory=list)
    in_render_group: bool | None = None
    in_video_group: bool | None = None

    # --- ROCm install ---
    rocm_version: str = ""              # e.g. 6.4.1
    rocm_install_method: str = ""       # amdgpu-install | apt | dnf | pip-only | unknown | none
    rocm_path: str = ""                 # /opt/rocm typically
    rocminfo_present: bool = False
    rocminfo_status: str = ""           # ok | not-loaded | permission-denied | missing
    hip_libs_on_ld_path: bool | None = None
    rocm_repos_seen: list[str] = field(default_factory=list)

    # --- framework ---
    framework: str = "unknown"          # pytorch | llama-cpp | unknown | skipped
    framework_version: str = ""
    framework_rocm_version: str = ""    # e.g. PyTorch's torch.version.hip
    framework_arch_list: list[str] = field(default_factory=list)
    framework_notes: list[str] = field(default_factory=list)

    # --- environment ---
    env: dict[str, str] = field(default_factory=dict)

    # --- container ---
    in_container: bool = False
    container_kind: str = ""

    # --- evidence captured for diagnose.py ---
    dmesg_amdgpu_tail: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    probe_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shell helpers (never raise)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    """Run `cmd`; return (rc, stdout, stderr). Never raises."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return 127, "", ""


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# Platform probes
# ---------------------------------------------------------------------------

def _probe_os(e: Examination) -> None:
    sysname = platform.system().lower()
    e.os_version = platform.platform()
    if sysname == "linux":
        e.os_family = "linux"
        e.kernel_release = platform.release()
        e.kernel_cmdline = _read_text("/proc/cmdline").strip()
        # /etc/os-release is the standard for distro identity since 2012.
        osr = _read_text("/etc/os-release")
        for line in osr.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"')
            if k == "ID":
                e.distro_id = v
            elif k == "VERSION_ID":
                e.distro_version = v
        m = re.search(r"\biommu=(\w+)", e.kernel_cmdline)
        if m:
            e.iommu_kernel_param = m.group(1)
    elif sysname == "windows":
        e.os_family = "windows"
    else:
        e.os_family = "other"


def _probe_cpu(e: Examination) -> None:
    if e.os_family != "linux":
        return
    txt = _read_text("/proc/cpuinfo")
    for line in txt.splitlines():
        if line.startswith("vendor_id") and not e.cpu_vendor or e.cpu_vendor == "unknown":
            val = line.split(":", 1)[1].strip()
            e.cpu_vendor = "amd" if "AMD" in val else ("intel" if "Intel" in val else val.lower())
        if line.startswith("model name") and not e.cpu_model:
            e.cpu_model = line.split(":", 1)[1].strip()
        if e.cpu_vendor != "unknown" and e.cpu_model:
            break


# ---------------------------------------------------------------------------
# GPU probes
# ---------------------------------------------------------------------------

# Strix Halo / Phoenix / Hawk Point / Strix Point marketing names commonly
# seen in `lspci`. Used to flag the GPU as an APU when rocminfo isn't
# available.
_APU_KEYWORDS = (
    "strix halo", "ryzen ai max", "phoenix", "hawk point", "strix point",
    "krackan", "rembrandt", "raphael", "barcelo", "lucienne", "renoir",
    "cezanne",
)


def _classify_amd_marketing_name(name: str) -> tuple[str, bool]:
    """Return (best-effort gfx_target, is_apu) for an AMD GPU marketing name.

    Falls back to ("", False) when we can't tell, in which case `rocminfo`
    output (when available) is the source of truth for the gfx target.
    """
    n = name.lower()
    if "ryzen ai max" in n or "strix halo" in n:
        return "gfx1151", True
    if "radeon 880m" in n or "radeon 890m" in n or "strix point" in n or "krackan" in n:
        return "gfx1150", True
    if "radeon 780m" in n or "radeon 760m" in n or "phoenix" in n or "hawk point" in n:
        return "gfx1103", True
    return "", any(kw in n for kw in _APU_KEYWORDS)


def _probe_gpus_lspci(e: Examination) -> None:
    """Enumerate AMD/NVIDIA display+3D controllers via lspci."""
    if not _have("lspci"):
        e.probe_failures.append("lspci not found; cannot enumerate PCI GPUs")
        return
    rc, out, _ = _run(["lspci", "-nn", "-D"], timeout=8)
    if rc != 0:
        e.probe_failures.append("lspci returned non-zero; PCI enumeration incomplete")
        return
    for line in out.splitlines():
        # Match VGA, 3D, and Display controllers.
        if not re.search(r"(VGA compatible controller|3D controller|Display controller)", line):
            continue
        pci_id = line.split()[0] if line.split() else ""
        is_amd = "[1002" in line or "Advanced Micro Devices" in line or "AMD" in line
        is_nvidia = "[10de" in line or "NVIDIA" in line
        # Marketing name lives between the controller-kind colon and the
        # `[vendor:device]` tail.
        m = re.match(
            r"\S+\s+(?:VGA compatible controller|3D controller|Display controller)"
            r"\s*\[\w+\]:\s*(.+?)\s*\[[\da-f]{4}:[\da-f]{4}\]",
            line,
            re.IGNORECASE,
        )
        name = m.group(1).strip() if m else line
        if is_nvidia:
            e.has_nvidia_gpu = True
            e.gpus.append(GPU(name=name, pci_id=pci_id, is_amd=False, is_apu=False))
            continue
        if not is_amd:
            continue
        gfx_guess, is_apu_guess = _classify_amd_marketing_name(name)
        e.gpus.append(GPU(
            name=name, gfx_target=gfx_guess, pci_id=pci_id,
            is_apu=is_apu_guess, is_amd=True,
        ))


def _probe_gpus_rocminfo(e: Examination) -> None:
    """Refine the AMD GPU list with rocminfo's authoritative gfx targets.

    rocminfo's output is the ground truth for the LLVM gfx target the
    runtime will load kernels for. When the binary is present but exits
    non-zero we capture the failure (it's a signal in its own right --
    e.g. "ROCk module is NOT loaded" means amdkfd isn't loaded).
    """
    if not _have("rocminfo"):
        e.rocminfo_present = False
        e.rocminfo_status = "missing"
        return
    e.rocminfo_present = True
    rc, out, err = _run(["rocminfo"], timeout=15)
    if rc != 0:
        merged = (out + "\n" + err).lower()
        if "rock module is not loaded" in merged:
            e.rocminfo_status = "not-loaded"
        elif "permission denied" in merged or "operation not permitted" in merged:
            e.rocminfo_status = "permission-denied"
        else:
            e.rocminfo_status = f"error rc={rc}"
        return
    e.rocminfo_status = "ok"

    # Parse GPU agents. rocminfo blocks look like:
    #   Agent 2
    #     Name:            gfx1151
    #     Marketing Name:  AMD Radeon Graphics
    #     Device Type:     GPU
    gfx_targets: list[tuple[str, str]] = []
    cur_name = ""
    cur_marketing = ""
    cur_is_gpu = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Agent "):
            if cur_is_gpu and cur_name.startswith("gfx"):
                gfx_targets.append((cur_name, cur_marketing))
            cur_name = ""
            cur_marketing = ""
            cur_is_gpu = False
            continue
        if s.startswith("Name:"):
            cur_name = s.split(":", 1)[1].strip()
        elif s.startswith("Marketing Name:"):
            cur_marketing = s.split(":", 1)[1].strip()
        elif s.startswith("Device Type:"):
            cur_is_gpu = "GPU" in s
    if cur_is_gpu and cur_name.startswith("gfx"):
        gfx_targets.append((cur_name, cur_marketing))

    if not gfx_targets:
        return

    # Reconcile with the lspci-derived list: prefer rocminfo's gfx target
    # for any AMD entry that didn't already have one.
    amd_entries = [g for g in e.gpus if g.is_amd]
    for idx, (gfx, marketing) in enumerate(gfx_targets):
        if idx < len(amd_entries):
            amd_entries[idx].gfx_target = gfx
            if marketing and not amd_entries[idx].name:
                amd_entries[idx].name = marketing
            # APU classification: gfx115x/gfx110x/gfx103x are APU families
            # the doctor cares about. The rest are discrete.
            amd_entries[idx].is_apu = bool(re.match(r"gfx11[05]\d", gfx))
        else:
            e.gpus.append(GPU(
                name=marketing or "AMD GPU", gfx_target=gfx,
                is_amd=True, is_apu=bool(re.match(r"gfx11[05]\d", gfx)),
            ))


def _summarise_gpu_categories(e: Examination) -> None:
    e.has_amd_gpu = any(g.is_amd for g in e.gpus)
    e.has_apu = any(g.is_amd and g.is_apu for g in e.gpus)
    e.has_discrete_amd = any(g.is_amd and g.is_apu is False for g in e.gpus)


# ---------------------------------------------------------------------------
# Kernel module / device probes
# ---------------------------------------------------------------------------

def _probe_modules(e: Examination) -> None:
    if e.os_family != "linux":
        return
    rc, out, _ = _run(["lsmod"], timeout=5)
    if rc == 0:
        modules = {line.split()[0] for line in out.splitlines()[1:] if line.split()}
        e.amdgpu_loaded = "amdgpu" in modules
        e.amdkfd_loaded = "amdkfd" in modules
    else:
        # /proc/modules is always readable and is the source of truth for lsmod.
        txt = _read_text("/proc/modules")
        if txt:
            modules = {line.split()[0] for line in txt.splitlines() if line.split()}
            e.amdgpu_loaded = "amdgpu" in modules
            e.amdkfd_loaded = "amdkfd" in modules

    # Blacklist files. We don't try to parse every modprobe.d directive
    # perfectly; we just flag any file that contains a literal "blacklist
    # amdgpu" line so the agent can ask the user to inspect it.
    for d in ("/etc/modprobe.d", "/usr/lib/modprobe.d", "/run/modprobe.d"):
        try:
            for f in Path(d).glob("*.conf"):
                body = _read_text(str(f))
                if re.search(r"^\s*blacklist\s+amdgpu\b", body, re.MULTILINE):
                    e.amdgpu_blacklisted_in.append(str(f))
        except OSError:
            continue


def _probe_devices(e: Examination) -> None:
    if e.os_family != "linux":
        return
    e.kfd = _stat_device("/dev/kfd")
    try:
        for path in sorted(Path("/dev/dri").glob("renderD*")):
            e.render_devices.append(_stat_device(str(path)))
    except OSError:
        pass


def _stat_device(path: str) -> Device:
    d = Device(path=path, exists=os.path.exists(path))
    if not d.exists:
        return d
    try:
        st = os.stat(path)
    except OSError as exc:
        d.mode = f"stat failed: {exc}"
        return d
    d.mode = stat.filemode(st.st_mode)
    # Resolve uid/gid to names via /etc/passwd & /etc/group; we do this by
    # hand because the pwd / grp modules are unavailable inside `uv run`
    # sandboxes on some systems.
    d.owner_user = _uid_to_name(st.st_uid)
    d.owner_group = _gid_to_name(st.st_gid)
    d.user_can_read = os.access(path, os.R_OK)
    d.user_can_write = os.access(path, os.W_OK)
    return d


def _uid_to_name(uid: int) -> str:
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except (KeyError, ImportError, OSError):
        return str(uid)


def _gid_to_name(gid: int) -> str:
    try:
        import grp
        return grp.getgrgid(gid).gr_name
    except (KeyError, ImportError, OSError):
        return str(gid)


def _probe_user(e: Examination) -> None:
    if e.os_family != "linux":
        return
    e.user_name = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    rc, out, _ = _run(["id", "-Gn"], timeout=3)
    if rc == 0:
        e.user_groups = out.strip().split()
    else:
        # Fallback: scan /etc/group for our uid.
        try:
            import grp
            uid = os.getuid()
            import pwd
            primary_gid = pwd.getpwuid(uid).pw_gid
            e.user_groups = [
                g.gr_name for g in grp.getgrall()
                if e.user_name in g.gr_mem or g.gr_gid == primary_gid
            ]
        except (ImportError, KeyError, OSError):
            pass
    e.in_render_group = "render" in e.user_groups
    e.in_video_group = "video" in e.user_groups


def _probe_secure_boot(e: Examination) -> None:
    if e.os_family != "linux":
        return
    if _have("mokutil"):
        rc, out, _ = _run(["mokutil", "--sb-state"], timeout=3)
        if rc == 0:
            o = out.lower()
            if "enabled" in o:
                e.secure_boot = "enabled"
            elif "disabled" in o:
                e.secure_boot = "disabled"


# ---------------------------------------------------------------------------
# ROCm install probes
# ---------------------------------------------------------------------------

def _probe_rocm_install(e: Examination) -> None:
    if e.os_family != "linux":
        return
    # 1) Canonical install path. `/opt/rocm` is a symlink to /opt/rocm-X.Y.Z
    # on every supported install pattern, including pip wheels that ship a
    # bundled runtime (the wheel sets ROCM_PATH instead).
    rocm_dir = ""
    for candidate in ("/opt/rocm", os.environ.get("ROCM_PATH", "")):
        if candidate and os.path.isdir(candidate):
            rocm_dir = candidate
            break
    e.rocm_path = rocm_dir

    # 2) Version. Modern installs put a .info/version-* file; older ones
    # only have it inside /opt/rocm-X.Y.Z. Walk both.
    if rocm_dir:
        for fname in ("version", "version-utils", "version-libs"):
            f = Path(rocm_dir) / ".info" / fname
            if f.exists():
                e.rocm_version = f.read_text(encoding="utf-8", errors="replace").strip()
                break
        if not e.rocm_version:
            # /opt/rocm-X.Y.Z symlink target.
            try:
                real = os.path.realpath(rocm_dir)
                m = re.search(r"rocm-(\d+(?:\.\d+)+)", real)
                if m:
                    e.rocm_version = m.group(1)
            except OSError:
                pass

    # 3) Install method. We check in priority order: amdgpu-install repo
    # files, packaged ROCm on the distro repos, and finally "looks like a
    # pip wheel" if /opt/rocm doesn't exist but a torch wheel bundles HIP.
    for marker in AMDGPU_INSTALL_MARKERS:
        if os.path.exists(marker):
            e.rocm_install_method = "amdgpu-install"
            e.rocm_repos_seen.append(marker)

    if not e.rocm_install_method:
        if _have("dpkg"):
            rc, out, _ = _run(["dpkg", "-l", "rocm-hip-runtime"], timeout=8)
            if rc == 0 and "rocm-hip-runtime" in out:
                e.rocm_install_method = "apt"
        if not e.rocm_install_method and _have("rpm"):
            rc, out, _ = _run(["rpm", "-q", "rocm-hip-runtime"], timeout=8)
            if rc == 0 and "rocm-hip-runtime" in out:
                e.rocm_install_method = "dnf"
    if not e.rocm_install_method:
        if rocm_dir:
            e.rocm_install_method = "tarball-or-other"
        else:
            e.rocm_install_method = "none"

    # 4) Stale repo detection: more than one ROCm repo file at the same
    # time. Common after `amdgpu-install` reruns with different `--rocmrelease`.
    extra = []
    try:
        for d in ("/etc/apt/sources.list.d", "/etc/yum.repos.d"):
            for f in Path(d).glob("*"):
                if re.search(r"(rocm|amdgpu|radeon)", f.name, re.IGNORECASE):
                    extra.append(str(f))
    except OSError:
        pass
    # Deduplicate while preserving order.
    for x in extra:
        if x not in e.rocm_repos_seen:
            e.rocm_repos_seen.append(x)


# ---------------------------------------------------------------------------
# Framework probes
# ---------------------------------------------------------------------------

# Inline Python the PyTorch probe pipes into the user's interpreter. Kept
# tiny so it works even on Python interpreters with broken site-packages.
_PYTORCH_PROBE = (
    "import json,sys\n"
    "out={'ok':False}\n"
    "try:\n"
    "  import torch\n"
    "  out['ok']=True\n"
    "  out['version']=torch.__version__\n"
    "  out['hip']=getattr(torch.version,'hip',None)\n"
    "  out['cuda']=getattr(torch.version,'cuda',None)\n"
    "  out['is_available']=bool(torch.cuda.is_available())\n"
    "  try: out['device_count']=int(torch.cuda.device_count())\n"
    "  except Exception: out['device_count']=0\n"
    "  try: out['arch_list']=list(torch.cuda.get_arch_list())\n"
    "  except Exception: out['arch_list']=[]\n"
    "except Exception as ex:\n"
    "  out['error']=type(ex).__name__+': '+str(ex)\n"
    "sys.stdout.write(json.dumps(out))\n"
)


def _probe_pytorch(e: Examination) -> None:
    """Try to introspect PyTorch in the user's default python."""
    py = sys.executable or shutil.which("python") or shutil.which("python3")
    if not py:
        e.framework_notes.append("No python interpreter found to probe torch.")
        return
    rc, out, err = _run([py, "-c", _PYTORCH_PROBE], timeout=20)
    if rc != 0 or not out.strip():
        # Try `python3` explicitly in case `sys.executable` is uv's own env
        # and the user's torch lives elsewhere.
        py2 = shutil.which("python3")
        if py2 and py2 != py:
            rc, out, err = _run([py2, "-c", _PYTORCH_PROBE], timeout=20)
    if not out.strip():
        e.framework_notes.append(
            "Could not import torch; if PyTorch is in a venv, activate it "
            "and re-run examine.py inside that venv."
        )
        if err:
            e.framework_notes.append(f"python stderr: {err.strip().splitlines()[-1][:200]}")
        return
    try:
        data = json.loads(out.strip())
    except json.JSONDecodeError:
        e.framework_notes.append(f"torch probe returned non-JSON: {out[:200]}")
        return
    if not data.get("ok"):
        e.framework_notes.append(f"torch import failed: {data.get('error', 'unknown')}")
        return
    e.framework = "pytorch"
    e.framework_version = data.get("version", "")
    hip = data.get("hip")
    cuda = data.get("cuda")
    if hip:
        e.framework_rocm_version = f"hip={hip}"
    elif cuda:
        e.framework_rocm_version = f"cuda={cuda}"
        e.framework_notes.append(
            "This torch wheel is a CUDA build, not a ROCm build. Reinstall "
            "from the ROCm wheel index."
        )
    arch = data.get("arch_list") or []
    e.framework_arch_list = [a for a in arch if isinstance(a, str)]
    if data.get("is_available") is False:
        e.framework_notes.append(
            "torch.cuda.is_available() returned False -- runtime can't see a GPU."
        )


def _probe_llama_cpp(e: Examination) -> None:
    """Best-effort probe of a llama.cpp build on PATH."""
    binary = None
    for name in ("llama-cli", "llama-server", "main"):
        p = shutil.which(name)
        if p:
            binary = p
            break
    if not binary:
        e.framework_notes.append("No llama.cpp binary (llama-cli/llama-server/main) on PATH.")
        return
    rc, out, err = _run([binary, "--version"], timeout=10)
    body = out + err
    if rc != 0 and not body:
        e.framework_notes.append(f"{binary} --version exited rc={rc}")
        return
    e.framework = "llama-cpp"
    e.framework_version = body.strip().splitlines()[0][:200] if body.strip() else "unknown"
    # Newer builds print "ROCm" or "HIP" in --version when GGML_HIP=ON.
    if "HIP" in body or "ROCm" in body or "hipBLAS" in body:
        e.framework_rocm_version = "GGML_HIP=ON"
    else:
        e.framework_notes.append(
            "llama.cpp binary doesn't advertise HIP/ROCm support; was it built "
            "with `cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx>`?"
        )


def _probe_framework(e: Examination, requested: str | None) -> None:
    if requested == "skip":
        e.framework = "skipped"
        return
    if requested == "pytorch":
        _probe_pytorch(e)
        return
    if requested == "llama-cpp":
        _probe_llama_cpp(e)
        return
    # Auto-detect: prefer PyTorch (the common case for the doctor), then
    # llama.cpp. We don't probe both to keep the script fast and to avoid
    # spawning a python interpreter when the user clearly meant llama.cpp.
    py = sys.executable or shutil.which("python") or shutil.which("python3")
    if py:
        _probe_pytorch(e)
        if e.framework == "pytorch":
            return
    _probe_llama_cpp(e)


# ---------------------------------------------------------------------------
# Misc probes
# ---------------------------------------------------------------------------

def _probe_env(e: Examination) -> None:
    for k in TRACKED_ENV_VARS:
        v = os.environ.get(k)
        if v is None:
            continue
        # Truncate enormous PATHs so JSON output stays human-scale.
        if k in ("PATH", "LD_LIBRARY_PATH") and len(v) > 4000:
            v = v[:4000] + "...[truncated]"
        e.env[k] = v
    # Quick check: does any path in LD_LIBRARY_PATH carry a libamdhip64?
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    hits = []
    for d in ld.split(os.pathsep):
        if not d:
            continue
        try:
            for hit in Path(d).glob("libamdhip64*"):
                hits.append(str(hit))
                break
        except OSError:
            continue
    if hits:
        e.hip_libs_on_ld_path = True
        e.notes.append(f"libamdhip64 visible via LD_LIBRARY_PATH: {hits[0]}")
    else:
        e.hip_libs_on_ld_path = False if ld else None


def _probe_container(e: Examination) -> None:
    for marker, kind in CONTAINER_MARKERS.items():
        if os.path.exists(marker):
            e.in_container = True
            e.container_kind = kind
            return
    cg = _read_text("/proc/1/cgroup")
    if cg and any(x in cg for x in ("docker", "containerd", "lxc", "kubepods", "podman")):
        e.in_container = True
        e.container_kind = e.container_kind or "container"


def _probe_dmesg_amdgpu(e: Examination) -> None:
    if e.os_family != "linux":
        return
    # We try journalctl first because it works for unprivileged users when
    # the systemd journal is world-readable; dmesg is usually root-only on
    # modern kernels (`kernel.dmesg_restrict=1`).
    text = ""
    rc, out, _ = _run(["journalctl", "-k", "--no-pager", "-n", "400"], timeout=8)
    if rc == 0 and out:
        text = out
    else:
        rc, out, _ = _run(["dmesg"], timeout=5)
        if rc == 0:
            text = out
    if not text:
        return
    # Keep at most ~15 amdgpu/amdkfd lines as evidence so the JSON stays
    # small. We prioritise lines containing well-known failure substrings.
    interesting = (
        "page fault", "RAS Controller", "vm_fault", "amdgpu_device_init",
        "OUT_OF_REGISTERS", "ring", "GPU reset", "PSP", "HW_FAULT",
    )
    hits: list[str] = []
    for line in text.splitlines():
        if "amdgpu" not in line and "amdkfd" not in line:
            continue
        if any(s.lower() in line.lower() for s in interesting):
            hits.append(line.strip()[:300])
    e.dmesg_amdgpu_tail = hits[-15:]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def examine(requested_framework: str | None) -> Examination:
    e = Examination()
    _probe_os(e)
    if e.os_family != "linux":
        # Phase 0 only supports Linux for diagnostics. Windows is intentionally
        # out of scope (the WSL/HIP SDK path is its own can of worms).
        e.notes.append(
            f"rocm-doctor only supports Linux (got {e.os_family}). "
            "On Windows, use the HIP SDK installer + Adrenalin path; "
            "this skill cannot help here."
        )
        return e
    _probe_cpu(e)
    _probe_gpus_lspci(e)
    _probe_gpus_rocminfo(e)
    _summarise_gpu_categories(e)
    _probe_modules(e)
    _probe_devices(e)
    _probe_user(e)
    _probe_secure_boot(e)
    _probe_rocm_install(e)
    _probe_env(e)
    _probe_container(e)
    _probe_dmesg_amdgpu(e)
    _probe_framework(e, requested_framework)
    return e


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------

def _fmt_yesno(v: bool | None) -> str:
    return "unknown" if v is None else ("yes" if v else "no")


def _print_human(e: Examination) -> None:
    print("rocm-doctor -- system examination (read-only)")
    print("=" * 60)
    print(f"OS:               {e.os_family} {e.distro_id} {e.distro_version}".strip())
    if e.os_family != "linux":
        for n in e.notes:
            print(f"  note: {n}")
        return
    print(f"Kernel:           {e.kernel_release}")
    if e.iommu_kernel_param:
        print(f"  iommu=          {e.iommu_kernel_param}")
    print(f"CPU:              {e.cpu_model} (vendor: {e.cpu_vendor})")
    if e.secure_boot != "unknown":
        print(f"Secure Boot:      {e.secure_boot}")
    if e.in_container:
        print(f"Container:        yes ({e.container_kind})")

    print("\nGPUs:")
    if not e.gpus:
        print("  (none detected; lspci returned no VGA/3D/Display controllers)")
    for g in e.gpus:
        flag = ""
        if g.is_amd and g.is_apu:
            flag = " [AMD APU]"
        elif g.is_amd:
            flag = " [AMD dGPU]"
        elif "NVIDIA" in g.name.upper():
            flag = " [NVIDIA]"
        print(f"  - {g.pci_id}  {g.name or 'unknown'}  gfx={g.gfx_target or '?'}{flag}")

    print("\nDriver & devices:")
    print(f"  amdgpu loaded:   {_fmt_yesno(e.amdgpu_loaded)}")
    if e.amdgpu_blacklisted_in:
        print(f"  amdgpu blacklisted in: {', '.join(e.amdgpu_blacklisted_in)}")
    print(f"  amdkfd loaded:   {_fmt_yesno(e.amdkfd_loaded)}")
    print(f"  rocminfo:        {e.rocminfo_status}")
    if e.kfd:
        print(f"  /dev/kfd:        exists={e.kfd.exists} mode={e.kfd.mode} "
              f"owner={e.kfd.owner_user}:{e.kfd.owner_group} "
              f"r={_fmt_yesno(e.kfd.user_can_read)} w={_fmt_yesno(e.kfd.user_can_write)}")
    for d in e.render_devices:
        print(f"  {d.path}:  mode={d.mode} owner={d.owner_user}:{d.owner_group} "
              f"r={_fmt_yesno(d.user_can_read)} w={_fmt_yesno(d.user_can_write)}")

    print("\nUser:")
    print(f"  name:            {e.user_name or 'unknown'}")
    print(f"  in render group: {_fmt_yesno(e.in_render_group)}")
    print(f"  in video group:  {_fmt_yesno(e.in_video_group)}")
    if e.user_groups:
        print(f"  all groups:      {' '.join(e.user_groups)}")

    print("\nROCm install:")
    print(f"  path:            {e.rocm_path or 'not found'}")
    print(f"  version:         {e.rocm_version or 'unknown'}")
    print(f"  install method:  {e.rocm_install_method or 'unknown'}")
    if e.rocm_repos_seen:
        print(f"  repos seen:      {len(e.rocm_repos_seen)} file(s)")
        for r in e.rocm_repos_seen:
            print(f"    - {r}")

    print("\nFramework:")
    print(f"  detected:        {e.framework}")
    if e.framework_version:
        print(f"  version:         {e.framework_version}")
    if e.framework_rocm_version:
        print(f"  rocm/hip:        {e.framework_rocm_version}")
    if e.framework_arch_list:
        print(f"  arch list:       {' '.join(e.framework_arch_list)}")
    for n in e.framework_notes:
        print(f"  note: {n}")

    if e.env:
        print("\nRelevant environment variables (set in current shell):")
        for k, v in e.env.items():
            display = v if len(v) <= 200 else (v[:200] + "...")
            print(f"  {k}={display}")

    if e.dmesg_amdgpu_tail:
        print("\nRecent amdgpu/amdkfd kernel messages (last few interesting lines):")
        for line in e.dmesg_amdgpu_tail:
            print(f"  | {line}")

    if e.probe_failures:
        print("\nProbes that did not complete:")
        for p in e.probe_failures:
            print(f"  - {p}")

    if e.notes:
        print("\nNotes:")
        for n in e.notes:
            print(f"  - {n}")

    print("\nNext step: feed this examination into diagnose.py:")
    print("  python scripts/examine.py --json > exam.json")
    print("  python scripts/diagnose.py --exam exam.json --symptom \"<paste user's error>\"")


def _to_jsonable(e: Examination) -> dict:
    """asdict() handles nested dataclasses; we just rename Optional[Device]."""
    d = asdict(e)
    return d


def _exit_code(e: Examination) -> int:
    if e.os_family != "linux":
        return 2
    if not e.has_amd_gpu:
        # NVIDIA-only or no GPU at all -- this skill can't help.
        return 2
    # Probes that didn't complete are a soft warning, not a hard fail.
    if e.probe_failures and not e.rocminfo_present and not e.gpus:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON for diagnose.py.")
    parser.add_argument(
        "--framework",
        choices=["pytorch", "llama-cpp", "skip", "auto"],
        default="auto",
        help="Which framework probe to run (default: auto-detect).",
    )
    args = parser.parse_args(argv)

    requested = None if args.framework == "auto" else args.framework
    e = examine(requested)
    if args.json:
        print(json.dumps(_to_jsonable(e), indent=2))
    else:
        _print_human(e)
    return _exit_code(e)


if __name__ == "__main__":
    raise SystemExit(main())
