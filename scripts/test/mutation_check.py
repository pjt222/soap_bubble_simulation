#!/usr/bin/env python3
"""Check that a test command catches a deliberate bug (a mutation).

Replaces OLD with NEW in FILE (OLD must occur exactly once), runs COMMAND, then restores
FILE byte-for-byte, also on Ctrl-C or error, and verifies the restore. A mutation counts
as caught when COMMAND exits non-zero; the failing tests are read from cargo's
`failures:` blocks in the output, so `--no-fail-fast` lists the tests of every binary.

    scripts/test/mutation_check.py --name drop-scale \\
        --file src/render/shaders/caustics_compute.wgsl \\
        --old 'thickness_field[idx] * params.film_thickness_scale' \\
        --new 'thickness_field[idx]' \\
        -- scripts/test-local.sh --no-fail-fast --lib -- --ignored --test-threads=1 caustic

The full output goes to target/mutation-check/<name>.log. Exit status: 0 if caught,
1 if not caught (COMMAND passed with the mutation in place), 2 on a usage or restore
error. Run from the project root. Standard library only.
"""

import argparse
import pathlib
import re
import subprocess
import sys

LOG_DIR = pathlib.Path("target/mutation-check")


def failing_tests(output: str) -> list[str]:
    """Test names listed in cargo's `failures:` summary blocks."""
    names: list[str] = []
    in_block = False
    for line in output.splitlines():
        if line == "failures:":
            in_block = True
            continue
        if line.startswith("test result:"):
            in_block = False
            continue
        match = re.fullmatch(r" {4}(\S+)", line)
        if in_block and match:
            names.append(match.group(1).rsplit("::", 1)[-1])
    return sorted(set(names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="label for the report and log file")
    parser.add_argument("--file", required=True, type=pathlib.Path, help="file to mutate")
    parser.add_argument("--old", required=True, help="exact text to replace (once)")
    parser.add_argument("--new", required=True, help="replacement text")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- test command")
    args = parser.parse_args()

    if not (pathlib.Path("Cargo.toml").is_file() and pathlib.Path("src/render").is_dir()):
        print("error: run from the project root", file=sys.stderr)
        return 2
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("error: no test command after --", file=sys.stderr)
        return 2

    original = args.file.read_bytes()
    text = original.decode("utf-8")
    occurrences = text.count(args.old)
    if occurrences != 1:
        print(f"error: --old occurs {occurrences} times in {args.file}", file=sys.stderr)
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.name)}.log"
    try:
        args.file.write_bytes(text.replace(args.old, args.new).encode("utf-8"))
        result = subprocess.run(command, capture_output=True, text=True)
    finally:
        args.file.write_bytes(original)
    if args.file.read_bytes() != original:
        print(f"error: {args.file} was NOT restored; check `git diff`", file=sys.stderr)
        return 2

    output = result.stdout + result.stderr
    log_path.write_text(output)
    failures = failing_tests(output)
    if result.returncode == 0:
        print(f"{args.name}: NOT CAUGHT (command passed; log {log_path})")
        return 1
    caught_by = ", ".join(failures) if failures else f"exit {result.returncode}, no test names"
    print(f"{args.name}: caught by {caught_by} (log {log_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
