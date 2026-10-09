import json, sys, pathlib
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    data = json.loads(path.read_text())
    crate = data["crate"]
    versions = [v for v in data["versions"] if not v["yanked"]]
    stable = [v for v in versions if "-" not in v["num"]]
    newest = versions[0] if versions else None
    newest_stable = max(stable, key=lambda v: v["created_at"]) if stable else None
    print(f'{crate["name"]:14s} max_stable={crate.get("max_stable_version")} newest={newest["num"] if newest else None}@{newest["created_at"][:10] if newest else ""} newest_stable_by_date={newest_stable["num"] if newest_stable else None}@{newest_stable["created_at"][:10] if newest_stable else ""} updated={crate["updated_at"][:10]} repo={crate.get("repository")}')
