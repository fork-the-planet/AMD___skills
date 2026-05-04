#!/usr/bin/env bash
# Regenerate every committed artifact derived from skills/ and the
# Claude plugin manifest.
#
# Usage:
#   ./scripts/publish.sh            Regenerate all derived artifacts.
#   ./scripts/publish.sh --check    Verify derived artifacts are up to date.
#   ./scripts/publish.sh -h|--help  Print this help.
#
# Currently regenerates:
#   - .cursor-plugin/plugin.json    (from .claude-plugin/plugin.json + skills/)
#
# `.claude-plugin/marketplace.json` and `.cursor-plugin/marketplace.json`
# are hand-maintained because their human-facing descriptions intentionally
# differ from the SKILL.md routing descriptions; ./scripts/check.sh enforces
# that the marketplace listing matches skills/ on disk.
#
# Requires `uv` (https://github.com/astral-sh/uv).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed -n '/^Usage:/,/^Requires/p'
}

case "${1:-}" in
  "")
    uv run scripts/generate_cursor_plugin.py
    echo "Publish artifacts generated successfully."
    ;;
  --check)
    uv run scripts/generate_cursor_plugin.py --check
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac
