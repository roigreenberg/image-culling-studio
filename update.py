"""CLI helper to check for and install ImageCulling Studio updates from GitHub Releases."""

from __future__ import annotations

import argparse
import json
import sys

from src.updater import check_for_updates, download_and_apply_update, get_version_info


def main() -> int:
    parser = argparse.ArgumentParser(description="ImageCulling Studio Updater")
    parser.add_argument("--check", action="store_true", help="Check GitHub Releases for a newer version")
    parser.add_argument("--apply", action="store_true", help="Download and install the latest release")
    parser.add_argument("--repo", type=str, default=None, help="Override GitHub owner/repo")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    if args.apply:
        res = download_and_apply_update(repo=args.repo, progress_cb=lambda m, p: print(f"[{p:3.0f}%] {m}"))
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(res.get("message") or res.get("error"))
        return 0 if res.get("ok") else 1

    # Default: check for updates
    res = check_for_updates(repo=args.repo)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Current version : v{res['current_version']}")
        print(f"Latest version  : v{res['latest_version']}")
        print(f"GitHub repo     : {res['github_repo']}")
        if res.get("error"):
            print(f"Status          : {res['error']}")
        elif res["update_available"]:
            print(f"Update available! Run `python update.py --apply` to update to v{res['latest_version']}.")
            if res.get("changelog"):
                print("\n--- Release Notes ---")
                print(res["changelog"])
        else:
            print("Status          : You are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
