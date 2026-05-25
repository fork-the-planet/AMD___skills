---
name: rocm-doctor
description: >-
  Diagnoses why ROCm, PyTorch, or llama.cpp isn't working on an AMD GPU
  by matching the symptom against twelve known misconfigurations and
  either applying a low-risk fix with consent or handing back the exact
  next step. Use when the user says "ROCm/HIP isn't working",
  "torch.cuda.is_available() is False on Radeon/Ryzen AI",
  "rocminfo can't find my GPU", "hipErrorNoBinaryForGpu",
  "HSA_STATUS_ERROR_INVALID_ISA", "invalid device function",
  "Unable to open /dev/kfd", "ROCk module is NOT loaded",
  "libamdhip64.so cannot open shared object file", "amdgpu-install broke
  apt", "ROCm wheel doesn't see my gfx1151/gfx1150/gfx1103 (Strix Halo,
  Phoenix)", "iGPU/dGPU collision", "multi-GPU hang"; or mentions
  HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES, PYTORCH_ROCM_ARCH,
  render group / /dev/kfd permissions, amdgpu blacklist, or Secure Boot
  blocking amdgpu. Do NOT use for non-AMD GPUs, fresh ROCm installs,
  performance tuning, or Lemonade/LM Studio/Ollama -- those ship their
  own ROCm; route upstream.
---

# ROCm Doctor

Given a "ROCm/PyTorch/llama.cpp isn't working on my AMD GPU" complaint,
identify which of a fixed list of **twelve known misconfigurations** is
the cause and either fix it or hand back the exact next step.

This is a diagnose-and-fix skill, not a setup or tuning skill. The closed
list is deliberate: if the user's symptom doesn't match one of the twelve,
the skill explicitly routes upstream rather than guessing.

## When to use this skill

Use it when **all** of the following are true:

- The user has an **AMD** GPU (APU or discrete). NVIDIA / Intel / Apple
  Silicon are out of scope; exit cleanly and route the user.
- The user's framework is **PyTorch**, **llama.cpp**, or anything else
  built directly against the system ROCm (`/opt/rocm` or a pip wheel that
  bundles HIP). Lemonade, LM Studio, and Ollama ship their own runtimes
  and bypass the system install entirely; skip examination and route
  upstream (see [Framework routing](#framework-routing)).
- There is a **functional** error (import fails, `torch.cuda.is_available()`
  is `False`, `rocminfo` errors, a kernel can't launch). Pure performance
  complaints belong in `mi-tuner` / `omniperf-tune` / `apu-memory-tuner`.

Do not use it for fresh installs on a clean machine. That is a setup task;
point the user at `amdgpu-install` from the [AMD ROCm install
guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-overview.html).

## Prerequisites

- **OS:** Linux. Phase 0 is Linux-only; on Windows, the HIP SDK / Adrenalin
  path is its own ecosystem and this skill cannot help.
- **Tools the agent will invoke as part of examination** (best-effort; the
  script degrades when one is missing):
  - `lspci` (always present on desktop distros)
  - `rocminfo` (when ROCm is installed)
  - `journalctl` or `dmesg` (for amdgpu kernel-ring evidence)
  - `python` / `python3` to introspect PyTorch
  - `llama-cli` / `llama-server` / `main` to introspect llama.cpp
- **Permissions:** examination is fully read-only and works as a regular
  user. Some fixes (`fix-4-render-group`, `fix-5-amdgpu-load`,
  `fix-7-stale-repos`, `fix-11-iommu`, `fix-12-installer`) need `sudo`;
  the script always prints the command before asking for consent.

Silent footguns to surface explicitly when relevant:

- `HSA_OVERRIDE_GFX_VERSION` -- forcing an unsupported gfx target works
  for `rocminfo` but causes page faults at runtime. Diagnosis
  `fix-2-unset-override` is the response when this is set on a GPU that
  already has a native wheel.
- `HIP_VISIBLE_DEVICES` -- on dual-GPU systems (APU + dGPU) the iGPU is
  often index 0 and destabilises HIP unless explicitly hidden.
- `PYTORCH_ROCM_ARCH` -- only honored during a *build* of PyTorch. Setting
  it at runtime does nothing for a prebuilt wheel.
- `LD_LIBRARY_PATH` -- a wheel-bundled `libamdhip64.so` shadowed by a
  system one (or vice versa) gives confusing `cannot open shared object
  file` errors that look like fix-8 but are really a load-order bug.

## The three-step flow

Run these in order. The first two are read-only. The third asks before
changing anything.

```
[ ] 1. Identify the framework, then examine (read-only).
[ ] 2. Diagnose: match examination + symptom against the twelve known cases.
[ ] 3. Propose the fix; only apply with explicit consent; re-verify.
```

### Step 1: identify the framework and examine

If the user hasn't said, ask which framework they are running. Use the
`AskQuestion` tool with PyTorch / llama.cpp / Lemonade / LM Studio /
Ollama / other as the options. The routing in [Framework routing](#framework-routing)
keys off the answer.

If the framework is in the "skip examination" bucket, jump straight to
the upstream link and exit. Otherwise run:

```bash
python scripts/examine.py --framework pytorch --json > exam.json
```

Replace `pytorch` with `llama-cpp`, or pass `--framework auto` to let the
script pick. Exit codes:

| Exit | Meaning | Next action |
|---|---|---|
| 0 | Examined; AMD GPU present | Continue to Step 2. |
| 2 | Not Linux / no AMD GPU | Stop. Route the user. |
| 3 | Probes partially failed | Continue but warn the user. |

For a quick read-only summary without piping JSON, drop `--json`:

```bash
python scripts/examine.py --framework pytorch
```

`examine.py` collects exactly the facts the twelve-case decision tree
needs: OS / kernel, AMD GPUs and gfx targets, `amdgpu` / `amdkfd`
status, `/dev/kfd` ownership and group, user's group membership, system
ROCm version and install method, framework version and arch list, the
silent-footgun env vars, container/IOMMU state, and recent `amdgpu`
kernel log lines. It deliberately does NOT spawn heavy probes (no kernel
launches, no model downloads).

### Step 2: diagnose

Hand the JSON snapshot plus the user's error text to `diagnose.py`:

```bash
python scripts/diagnose.py --exam exam.json \
  --symptom "HIP error: invalid device function on gfx1151"
```

The script runs the twelve checkers, scores each from 0..100, and prints
a ranked list. Each match has a stable `fix-N-...` id used by
`apply_fix.py`.

Score tiers:

- `>= 75` (`HIGH`) -- propose the fix and (if auto-applicable) ask for
  consent to apply it.
- `>= 50` (`LIKELY`) -- describe the match and ask the user to confirm one
  more piece of evidence before applying.
- Below `50` -- print but do **not** act. If nothing scores `>= 50`, the
  script exits 1 with a single-line route to the right upstream tracker.
  Do not speculate.

JSON output (`--json`) is the same data the agent should use programmatically:

```bash
python scripts/diagnose.py --exam exam.json --symptom "..." --json
```

### Step 3: apply the fix (with consent)

Show the user the proposed fix (it's already printed by `diagnose.py`).
If they consent, run:

```bash
python scripts/apply_fix.py --fix-id fix-4-render-group --dry-run
python scripts/apply_fix.py --fix-id fix-4-render-group --yes
```

`--dry-run` prints the exact commands without executing. `--yes` skips
the interactive `[y/N]` prompt (only pass this after the user has agreed
in chat).

Five of the twelve fixes are auto-applicable; the rest are deliberately
print-only because the risk of a half-applied state is too high for an
agent to take:

| Fix-id | Auto? | Why |
|---|---|---|
| `fix-1-arch` | Print-only | Reinstalls a framework; user must approve and pick the wheel index. |
| `fix-2-unset-override` | Auto | Just unsets an env var + flags persistent rc lines. |
| `fix-3-rocm-kernel` | Print-only | Upgrading kernels needs the user. |
| `fix-4-render-group` | Auto | `usermod -a -G render,video $USER` is well-bounded. |
| `fix-5-amdgpu-load` | Print-only | Editing modprobe.d + initramfs regen needs the user. |
| `fix-6-path` | Auto | Appends one line to `~/.bashrc` / `~/.zshrc`. |
| `fix-7-stale-repos` | Print-only | Moving repo files is destructive enough to require the user. |
| `fix-8-wheel-rocm` | Print-only | Reinstalls a framework. |
| `fix-9-igpu-dgpu` | Auto | Adds `export HIP_VISIBLE_DEVICES=N` (user supplies N via `--device-index`). |
| `fix-10-container` | Print-only | Re-launches a container. |
| `fix-11-iommu` | Print-only | Edits GRUB and reboots. |
| `fix-12-installer` | Print-only | Reinstalls system packages. |

After every fix, re-run the `verify` command the recipe printed. Only
declare success when the user's *original* failing command now succeeds
(e.g. `torch.cuda.is_available()` returns `True`, `rocminfo` lists the
GPU, the llama.cpp build runs).

## Framework routing

The skill's first decision is which framework the user runs. Some
frameworks ship their own ROCm and bypass the system install -- examining
the host is the wrong question for them.

| Framework | Examine the system? | Where to send the user |
|---|---|---|
| PyTorch | Yes | `python scripts/examine.py --framework pytorch` |
| llama.cpp (built against system ROCm) | Yes | `python scripts/examine.py --framework llama-cpp` |
| Lemonade | No -- ships its own ROCm | <https://github.com/lemonade-sdk/lemonade> + [Discord](https://discord.gg/5xXzkMu8Zk) |
| LM Studio | No -- ships its own runtime | <https://lmstudio.ai/docs/app> + Discord |
| Ollama | No -- ships its own runtime | <https://github.com/ollama/ollama> + Discord |
| vLLM / SGLang | Out of scope until phase 1+ | Route to the project's own issue tracker. |

If a Lemonade / LM Studio / Ollama user really does have a system ROCm
problem (rare), it shows up when their app fails AND a standalone
`rocminfo` also fails. Only then run the full examination.

## Safety rules

- Read-only by default. Examination and diagnosis never change state.
- Always print before applying. `apply_fix.py` shows every command before
  asking for consent, even with `--yes`.
- Never reboot, never touch BIOS, never flash firmware.
- Never reinstall system packages without an interactive prompt or `--yes`.
- Never set `HSA_OVERRIDE_GFX_VERSION` as the *first* fix when a native
  wheel exists. That is `fix-2-unset-override`'s entire reason for being.
- Never silently fall back to a different fix when the requested one
  isn't applicable. Exit 3 and tell the user why.
- When nothing matches the twelve known cases, **do not speculate**. Hand
  the user the upstream tracker URL from `diagnose.py --json`.

## Verification checklist

Mark this skill complete only when **all** are true:

- [ ] `python scripts/examine.py` exits 0 (or 3 with the user's explicit
      go-ahead to continue despite a partial probe).
- [ ] `python scripts/diagnose.py --exam exam.json --symptom "..."` exits 0
      and surfaced exactly one HIGH-confidence diagnosis, OR it exited 1
      and the user has been routed to the right upstream tracker.
- [ ] If a fix was applied: the recipe's `verify` command exits cleanly.
- [ ] The user's *original* failing command now succeeds end-to-end (run
      it again in their original shell).
- [ ] If any fix needed a re-login or reboot, the user has actually done
      it before declaring success.

If any box is unchecked, the failure isn't resolved -- say so out loud
rather than declaring victory.

## Reference

For the full table of twelve known misconfigurations, every fix-id and
its verify command, the silent-footgun env-var reference, and the
upstream-routing table in machine-readable form, see
[reference.md](reference.md).
