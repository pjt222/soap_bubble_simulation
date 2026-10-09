#!/usr/bin/env python3
"""Audit findings toolbox: tables, issue-grouping checks, issue drafts, and filing.

Works on the findings JSON produced by an audit workflow (see
docs/audit-2026-10-09-findings.json) plus a hand-written grouping spec that
batches findings into GitHub issues or comments on existing issues (see
docs/audit-2026-10-09-issues.json). Standard library only.

Run from the project root:

    python3 scripts/audit/findings.py table  FINDINGS
    python3 scripts/audit/findings.py check  FINDINGS GROUPS
    python3 scripts/audit/findings.py draft  FINDINGS GROUPS --out DIR
    python3 scripts/audit/findings.py file   FINDINGS GROUPS --out DIR [--yes]

`file` without --yes is a dry run. With --yes it creates issues / posts
comments through the `gh` CLI, in spec order, and records each result in
DIR/filed.json so that a re-run skips what was already filed.

Grouping spec format (JSON):

    {
      "context": "footer text appended to every issue body",
      "groups": [
        {"key": "airy-phase", "kind": "issue",
         "title": "[CRITICAL] Optics: ...", "labels": ["bug", "physics"],
         "summary": "...", "acceptance": ["...", "..."],
         "findings": ["optics-01", "geomfoam-03"],
         "lead_findings": ["extra finding text not in the JSON"],
         "related": "Related: {#other-key}, #38"},
        {"key": "status-35", "kind": "comment", "target": 35,
         "summary": "...", "findings": ["gpuperf-11"]}
      ]
    }

`{#key}` placeholders in summary/acceptance/related resolve to the issue
number filed for that key. They may only point at groups earlier in the
list, which `check` enforces, so a single in-order pass can resolve them.
Every finding id must appear in exactly one group.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
GITHUB_BODY_LIMIT = 65536
PLACEHOLDER = re.compile(r"\{#([a-z0-9-]+)\}")


def require_project_root():
    if not (Path("Cargo.toml").is_file() and Path("src/render").is_dir()):
        sys.exit("error: run from the soap_bubble_simulation project root")


def load_findings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {finding["id"]: finding for finding in data["findings"]}


def load_groups(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effective_severity(finding):
    verification = finding.get("verification", {})
    return verification.get("adjusted_severity") or finding["severity"]


def verdict_label(finding):
    verdict = finding.get("verification", {}).get("verdict", "")
    if verdict in ("", "not-selected-for-verification", "no-verdict"):
        return "unverified"
    return verdict


def escape_cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------- table


def command_table(arguments):
    findings = load_findings(arguments.findings)
    ordered = sorted(
        findings.values(),
        key=lambda f: (SEVERITY_ORDER[effective_severity(f)], f["dimension"], f["id"]),
    )
    print("| id | sev | verdict | evidence | location | title | issue |")
    print("|---|---|---|---|---|---|---|")
    for finding in ordered:
        location = f"`{finding['file']}:{finding['line']}`"
        print(
            f"| {finding['id']} | {effective_severity(finding)} | {verdict_label(finding)} "
            f"| {finding['evidence_kind']} | {location} | {escape_cell(finding['title'])} "
            f"| {finding['related_issue']} |"
        )
    return 0


# ---------------------------------------------------------------- check


def validate(findings, spec):
    problems = []
    seen_keys = []
    assignment = {}
    for group in spec["groups"]:
        key = group["key"]
        if key in seen_keys:
            problems.append(f"duplicate group key: {key}")
        kind = group.get("kind")
        if kind not in ("issue", "comment"):
            problems.append(f"{key}: kind must be 'issue' or 'comment'")
        if kind == "issue" and not group.get("title"):
            problems.append(f"{key}: issue group needs a title")
        if kind == "issue" and not group.get("acceptance"):
            problems.append(f"{key}: issue group needs acceptance criteria")
        if kind == "comment" and not isinstance(group.get("target"), int):
            problems.append(f"{key}: comment group needs an integer target issue")
        for finding_id in group.get("findings", []):
            if finding_id not in findings:
                problems.append(f"{key}: unknown finding id {finding_id}")
            elif finding_id in assignment:
                problems.append(f"{finding_id}: in both {assignment[finding_id]} and {key}")
            else:
                assignment[finding_id] = key
        text = " ".join(
            [group.get("summary", ""), group.get("related", "")] + group.get("acceptance", [])
        )
        for referenced in PLACEHOLDER.findall(text):
            if referenced not in seen_keys:
                problems.append(f"{key}: placeholder {{#{referenced}}} must reference an earlier group")
        seen_keys.append(key)
    unassigned = sorted(set(findings) - set(assignment))
    for finding_id in unassigned:
        problems.append(f"unassigned finding: {finding_id}")
    return problems


def command_check(arguments):
    findings = load_findings(arguments.findings)
    spec = load_groups(arguments.groups)
    problems = validate(findings, spec)
    issue_count = sum(1 for g in spec["groups"] if g["kind"] == "issue")
    comment_count = sum(1 for g in spec["groups"] if g["kind"] == "comment")
    print(f"{len(findings)} findings, {issue_count} issues, {comment_count} comments")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


# ---------------------------------------------------------------- draft


def resolve_placeholders(text, filed_numbers):
    def replace(match):
        key = match.group(1)
        if key in filed_numbers:
            return f"#{filed_numbers[key]}"
        return f"(audit issue '{key}', not yet filed)"

    return PLACEHOLDER.sub(replace, text)


def render_finding(index, finding, include_derivation):
    verification = finding.get("verification", {})
    lines = [
        f"### {index}. [{effective_severity(finding).upper()}] {finding['title']}",
        "",
        f"`{finding['file']}:{finding['line']}` · verdict: {verdict_label(finding)} · "
        f"evidence: {finding['evidence_kind']} · audit id: `{finding['id']}`",
        "",
        f"**Claim.** {finding['claim']}",
    ]
    if verification.get("corrected_claim"):
        lines += ["", f"**Verifier correction.** {verification['corrected_claim']}"]
    lines += ["", f"**Recommendation.** {finding['recommendation']}"]
    if include_derivation:
        lines += [
            "",
            "<details><summary>Derivation / evidence</summary>",
            "",
            finding["derivation"],
            "",
            "</details>",
        ]
    return "\n".join(lines)


def render_group(group, findings, context, filed_numbers):
    member_findings = sorted(
        (findings[finding_id] for finding_id in group.get("findings", [])),
        key=lambda f: (SEVERITY_ORDER[effective_severity(f)], f["id"]),
    )
    summary = resolve_placeholders(group["summary"], filed_numbers)
    for include_derivation in (True, False):
        parts = ([] if group["kind"] == "comment" else ["## Summary", ""]) + [summary, ""]
        if member_findings or group.get("lead_findings"):
            parts += ["## Findings", ""]
            index = 0
            for lead_text in group.get("lead_findings", []):
                index += 1
                parts += [f"### {index}. {lead_text}", ""]
            for finding in member_findings:
                index += 1
                parts += [render_finding(index, finding, include_derivation), ""]
        if group.get("acceptance"):
            parts += ["## Acceptance Criteria", ""]
            parts += [f"- [ ] {resolve_placeholders(item, filed_numbers)}" for item in group["acceptance"]]
            parts += [""]
        parts += ["## Context", "", context]
        if group.get("related"):
            parts += ["", resolve_placeholders(group["related"], filed_numbers)]
        body = "\n".join(parts) + "\n"
        if len(body) <= GITHUB_BODY_LIMIT:
            return body
    sys.exit(f"error: {group['key']} body exceeds {GITHUB_BODY_LIMIT} chars even without derivations")


def draft_all(findings, spec, out_dir, filed_numbers):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for position, group in enumerate(spec["groups"], start=1):
        body = render_group(group, findings, spec.get("context", ""), filed_numbers)
        path = out_dir / f"{position:02d}-{group['key']}.md"
        path.write_text(body, encoding="utf-8")
        paths.append((group, path, len(body)))
    return paths


def command_draft(arguments):
    findings = load_findings(arguments.findings)
    spec = load_groups(arguments.groups)
    problems = validate(findings, spec)
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        return 1
    out_dir = Path(arguments.out)
    filed_numbers = read_filed(out_dir)
    for group, path, size in draft_all(findings, spec, out_dir, filed_numbers):
        target = group.get("title") or f"comment on #{group['target']}"
        print(f"{path}  ({size} chars, {len(group.get('findings', []))} findings)  {target}")
    return 0


# ---------------------------------------------------------------- file


def read_filed(out_dir):
    filed_path = out_dir / "filed.json"
    if not filed_path.is_file():
        return {}
    filed = json.loads(filed_path.read_text(encoding="utf-8"))
    return {key: record["number"] for key, record in filed.items() if record.get("number")}


def run_gh(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"gh exited {completed.returncode}")
    return completed.stdout.strip()


def command_file(arguments):
    findings = load_findings(arguments.findings)
    spec = load_groups(arguments.groups)
    problems = validate(findings, spec)
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        return 1
    out_dir = Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    filed_path = out_dir / "filed.json"
    filed = json.loads(filed_path.read_text(encoding="utf-8")) if filed_path.is_file() else {}
    failures = 0
    for position, group in enumerate(spec["groups"], start=1):
        key = group["key"]
        if key in filed:
            print(f"skip {key}: already filed -> {filed[key]['url']}")
            continue
        filed_numbers = {k: r["number"] for k, r in filed.items() if r.get("number")}
        body = render_group(group, findings, spec.get("context", ""), filed_numbers)
        body_path = out_dir / f"{position:02d}-{key}.md"
        body_path.write_text(body, encoding="utf-8")
        if group["kind"] == "issue":
            command = ["gh", "issue", "create", "--title", group["title"], "--body-file", str(body_path)]
            for label in group.get("labels", []):
                command += ["--label", label]
        else:
            command = ["gh", "issue", "comment", str(group["target"]), "--body-file", str(body_path)]
        if not arguments.yes:
            print("DRY RUN:", " ".join(command))
            continue
        try:
            url = run_gh(command)
        except RuntimeError as error:
            failures += 1
            print(f"FAILED {key}: {error}")
            continue
        number = int(url.rstrip("/").split("/")[-1]) if group["kind"] == "issue" else None
        filed[key] = {"url": url, "number": number, "kind": group["kind"]}
        filed_path.write_text(json.dumps(filed, indent=1) + "\n", encoding="utf-8")
        print(f"filed {key}: {url}")
    print_summary(spec, filed)
    return 1 if failures else 0


def print_summary(spec, filed):
    print()
    print("| # | key | title / target | findings |")
    print("|---|---|---|---|")
    for group in spec["groups"]:
        record = filed.get(group["key"], {})
        reference = record.get("url", "not filed")
        label = group.get("title") or f"comment on #{group['target']}"
        print(f"| {reference} | {group['key']} | {label} | {len(group.get('findings', []))} |")


# ---------------------------------------------------------------- main


def main():
    require_project_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = parser.add_subparsers(dest="command", required=True)

    table_parser = subcommands.add_parser("table", help="print a markdown table of all findings")
    table_parser.add_argument("findings")
    table_parser.set_defaults(handler=command_table)

    check_parser = subcommands.add_parser("check", help="validate a grouping spec against the findings")
    check_parser.add_argument("findings")
    check_parser.add_argument("groups")
    check_parser.set_defaults(handler=command_check)

    draft_parser = subcommands.add_parser("draft", help="render issue/comment bodies to a directory")
    draft_parser.add_argument("findings")
    draft_parser.add_argument("groups")
    draft_parser.add_argument("--out", required=True)
    draft_parser.set_defaults(handler=command_draft)

    file_parser = subcommands.add_parser("file", help="create issues / post comments via gh (dry run unless --yes)")
    file_parser.add_argument("findings")
    file_parser.add_argument("groups")
    file_parser.add_argument("--out", required=True)
    file_parser.add_argument("--yes", action="store_true", help="actually call gh")
    file_parser.set_defaults(handler=command_file)

    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
