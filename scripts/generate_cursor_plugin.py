#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate Cursor plugin manifest from existing repo metadata.

Outputs:
- .cursor-plugin/plugin.json

Design goals:
- Keep Claude + Cursor metadata in sync.
- Reuse `.claude-plugin/plugin.json` as the primary metadata source.
- Discover skills from `skills/*/SKILL.md` so the manifest tracks the
  catalog automatically.

Usage:
    uv run scripts/generate_cursor_plugin.py            # write
    uv run scripts/generate_cursor_plugin.py --check    # validate only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_PLUGIN_MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
CURSOR_PLUGIN_DIR = ROOT / ".cursor-plugin"
CURSOR_PLUGIN_MANIFEST = CURSOR_PLUGIN_DIR / "plugin.json"

# Fields copied verbatim from the Claude plugin manifest into the Cursor
# manifest so the two stay in lock-step.
COPIED_FIELDS = (
    "description",
    "version",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "logo",
)

PLUGIN_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_frontmatter(text: str) -> dict[str, str]:
    """Return the YAML frontmatter at the top of `text` as a flat mapping.

    Only top-level scalar keys are extracted. That is sufficient for `name`,
    which is all this script needs.
    """
    match = re.search(r"^---\s*\n(.*?)\n---\s*", text, re.DOTALL)
    if not match:
        return {}
    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        # Skip continuation lines so multi-line `description: >-` values
        # don't get parsed as keys.
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def collect_skills() -> list[str]:
    skills: list[str] = []
    for skill_md in sorted(ROOT.glob("skills/*/SKILL.md")):
        meta = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = meta.get("name", "").strip()
        if not name:
            continue
        skills.append(name)
    return skills


def validate_plugin_name(name: str) -> None:
    if not PLUGIN_NAME_RE.match(name):
        raise ValueError(
            "Invalid plugin name in .claude-plugin/plugin.json: "
            f"'{name}'. Must be lowercase and match {PLUGIN_NAME_RE.pattern}"
        )


def build_cursor_plugin_manifest() -> dict:
    src = load_json(CLAUDE_PLUGIN_MANIFEST)

    name = src.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(".claude-plugin/plugin.json must define a non-empty 'name'")
    validate_plugin_name(name)

    skills = collect_skills()
    if not skills:
        raise ValueError("No skills discovered under skills/*/SKILL.md")

    manifest: dict = {"name": name, "skills": "skills"}
    for key in COPIED_FIELDS:
        if key in src:
            manifest[key] = src[key]

    return manifest


def render_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    """Return True when the file is already up-to-date.

    In write mode (check=False) the file is written first, so the return
    value is always True in that branch.
    """
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return True

    if check:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Cursor plugin manifest from .claude-plugin/plugin.json"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated manifest is up to date without writing changes.",
    )
    args = parser.parse_args()

    plugin_manifest = render_json(build_cursor_plugin_manifest())
    ok_plugin = write_or_check(CURSOR_PLUGIN_MANIFEST, plugin_manifest, check=args.check)

    if args.check:
        if not ok_plugin:
            print("Generated Cursor manifest is out of date:", file=sys.stderr)
            print(f"  - {CURSOR_PLUGIN_MANIFEST.relative_to(ROOT)}", file=sys.stderr)
            print("Run: uv run scripts/generate_cursor_plugin.py", file=sys.stderr)
            sys.exit(1)

        print("Cursor plugin manifest is up to date.")
        return

    print(f"Wrote {CURSOR_PLUGIN_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
