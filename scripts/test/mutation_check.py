#!/usr/bin/env python3
"""Check that a test command catches a deliberate bug (a mutation).

First runs COMMAND unmutated and requires it to pass (skip with --skip-baseline only if
you just ran the same command and it passed). Then replaces OLD with NEW in FILE (OLD must
occur exactly once), runs COMMAND again, and restores FILE byte-for-byte, verifying the
restore. A mutation counts as caught only when the mutated run exits non-zero AND cargo
names at least one failing test in its `failures:` summary; a non-zero exit without test
names (a build error, a crashed test binary, a missing driver) is inconclusive. Pass
`--no-fail-fast` to cargo so every test binary runs.

    scripts/test/mutation_check.py --name drop-scale \\
        --file src/render/shaders/caustics_compute.wgsl \\
        --old 'thickness_field[idx] * params.film_thickness_scale' \\
        --new 'thickness_field[idx]' \\
        -- scripts/test-local.sh --no-fail-fast --lib -- --ignored --test-threads=1 caustic

Restoring: FILE is restored in a `finally` block, which also runs on Ctrl-C, SIGTERM and
SIGHUP. Before mutating, the original is copied byte-for-byte to
target/mutation-check/<name>.orig, with FILE's path in <name>.orig.path; both are deleted
after a verified restore. If a run is killed with SIGKILL they remain, and the next run
refuses to start until you restore FILE (`cp <name>.orig "$(cat <name>.orig.path)"`) and
delete both.

Outputs go to target/mutation-check/<name>.baseline.log and <name>.log. Exit status:
0 caught, 1 not caught (the mutated run passed), 2 anything else (usage error, baseline
failed, inconclusive result, restore failure). Run from the project root. Standard
library only.
"""

import argparse
import pathlib
import re
import signal
import subprocess
import sys

LOG_DIR = pathlib.Path("target/mutation-check")


class Interrupted(Exception):
    """Raised by the SIGTERM / SIGHUP handler so the `finally` restore runs."""


def raise_interrupted(signal_number, _frame):
    raise Interrupted(f"signal {signal_number}")


def failing_tests(output: str) -> list[str]:
    """Full test paths listed in cargo's `failures:` summary blocks.

    cargo prints two blocks headed `failures:`. The first holds each failed test's
    captured output under `---- path stdout ----` lines; the second lists the paths, one
    per line, indented by four spaces. Only the second kind is read.
    """
    names: list[str] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line != "failures:":
            continue
        following = lines[index + 1 :]
        first_content = next((entry for entry in following if entry.strip()), "")
        if first_content.startswith("---- "):
            continue  # the block with captured output, not the summary
        for entry in following:
            if not entry.strip():
                if names:
                    break
                continue
            match = re.fullmatch(r" {4}(\S+)", entry)
            if not match:
                break
            names.append(match.group(1))
    return sorted(set(names))


def run(command: list[str], log_path: pathlib.Path) -> tuple[int, str]:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = result.stdout + result.stderr
    log_path.write_text(output, encoding="utf-8")
    return result.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="label for the report and log files")
    parser.add_argument("--file", required=True, type=pathlib.Path, help="file to mutate")
    parser.add_argument("--old", required=True, help="exact text to replace (once)")
    parser.add_argument("--new", required=True, help="replacement text")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="do not run COMMAND unmutated first (only if it just passed)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- test command")
    args = parser.parse_args()

    if not (pathlib.Path("Cargo.toml").is_file() and pathlib.Path("src/render").is_dir()):
        print("error: run from the project root", file=sys.stderr)
        return 2
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("error: no test command after --", file=sys.stderr)
        return 2
    if args.old == args.new:
        print("error: --old and --new are identical", file=sys.stderr)
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", args.name)
    leftover_backups = sorted(LOG_DIR.glob("*.orig"))
    if leftover_backups:
        leftover = leftover_backups[0]
        print(
            f"error: {leftover} exists: an earlier run did not finish. Restore with "
            f'cp {leftover} "$(cat {leftover}.path)", then delete {leftover} and '
            f"{leftover}.path.",
            file=sys.stderr,
        )
        return 2

    try:
        original = args.file.read_bytes()
        text = original.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"error: cannot read {args.file} as UTF-8 text: {error}", file=sys.stderr)
        return 2
    occurrences = text.count(args.old)
    if occurrences != 1:
        print(f"error: --old occurs {occurrences} times in {args.file}", file=sys.stderr)
        return 2

    signal.signal(signal.SIGTERM, raise_interrupted)
    signal.signal(signal.SIGHUP, raise_interrupted)
    try:
        if not args.skip_baseline:
            baseline_status, _ = run(command, LOG_DIR / f"{stem}.baseline.log")
            if baseline_status != 0:
                print(
                    f"{args.name}: baseline FAILED (exit {baseline_status}) before mutating; "
                    f"log {LOG_DIR / f'{stem}.baseline.log'}",
                    file=sys.stderr,
                )
                return 2
        backup = LOG_DIR / f"{stem}.orig"
        backup_path_note = LOG_DIR / f"{stem}.orig.path"
        backup.write_bytes(original)
        backup_path_note.write_text(f"{args.file}\n", encoding="utf-8")
        try:
            args.file.write_bytes(text.replace(args.old, args.new).encode("utf-8"))
            status, output = run(command, LOG_DIR / f"{stem}.log")
        finally:
            args.file.write_bytes(original)
            if args.file.read_bytes() == original:
                backup.unlink()
                backup_path_note.unlink()
        if backup.exists():
            print(f"error: {args.file} was NOT restored; see {backup}", file=sys.stderr)
            return 2
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (OSError, Interrupted) as error:
        print(f"error: stopped ({error}); check `git diff` on {args.file}", file=sys.stderr)
        return 2

    log_path = LOG_DIR / f"{stem}.log"
    if status == 0:
        print(f"{args.name}: NOT CAUGHT (command passed; log {log_path})")
        return 1
    failures = failing_tests(output)
    if not failures:
        print(
            f"{args.name}: INCONCLUSIVE (exit {status}, no failing test named; "
            f"log {log_path})",
            file=sys.stderr,
        )
        return 2
    print(f"{args.name}: caught by {', '.join(failures)} (log {log_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
