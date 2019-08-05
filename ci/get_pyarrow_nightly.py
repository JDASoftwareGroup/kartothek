import json
import urllib.request

dict_nightly_builds = {}

with urllib.request.urlopen(
    "https://api.github.com/repos/ursa-labs/crossbow/releases"
) as url:
    data = json.loads(url.read().decode())
    for eachBlock in data:
        for eachAsset in eachBlock["assets"]:
            if str(eachAsset["name"]).startswith("pyarrow-") & str(
                eachAsset["name"]
            ).endswith(".whl"):
                dict_nightly_builds[eachAsset["updated_at"]] = eachAsset[
                    "browser_download_url"
                ]

if len(dict_nightly_builds) == 0:
    print("NULL")
else:
    sorted(dict_nightly_builds.items(), reverse=True)
    for key, value in dict_nightly_builds.items():
        if str(value).__contains__("manylinux") & str(value).__contains__("cp37"):
            print(value)
            break
