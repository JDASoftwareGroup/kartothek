#!/usr/bin/env python3
# Get pyarrow nightly build from GitHub API.
#
# The issue is that https://issues.apache.org/jira/browse/ARROW-1581 is not solved yet and the crossbow build system
# creates releases for different kind of artifacts. So we need to find the correct release and the linked wheel.
#
# IMPORTANT: Only use python builtin libs here since no venv is bootstrapped at this point!
import json
import os
import re
import sys
import urllib.request

GITHUB_REPO = "ursa-labs/crossbow"
VERSION_TAG = "{}{}".format(sys.version_info.major, sys.version_info.minor)
ASSET_PATTERN = re.compile(
    r"^pyarrow-[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+-cp"
    + VERSION_TAG
    + r"-cp"
    + VERSION_TAG
    + r"m-manylinux1_x86_64\.whl$"
)


def main():
    nightly_builds = {}

    # we need to authenticate this request because the travis IP range might easily hit the rate limits
    client_id = os.environ["GITHUB_CLIENT_ID"]
    client_secret = os.environ["GITHUB_CLIENT_SECRET"]
    request = urllib.request.Request(
        url="https://api.github.com/repos/{}/releases?client_id={}&client_secret={}".format(
            GITHUB_REPO, client_id, client_secret
        ),
        headers={"Accept": "application/vnd.github.v3+json"},
    )

    with urllib.request.urlopen(request) as url:
        data = json.loads(url.read().decode())
        for release in data:
            for asset in release["assets"]:
                if ASSET_PATTERN.match(asset["name"]):
                    nightly_builds[asset["updated_at"]] = asset["browser_download_url"]

    if not nightly_builds:
        raise RuntimeError("No nightly build found!")

    _release_date, release_file = sorted(nightly_builds.items(), reverse=True)[0]
    print(release_file)


if __name__ == "__main__":
    main()
