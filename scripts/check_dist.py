"""Check built Python distributions: license files present, metadata as intended.

    python scripts/check_dist.py DIST_FILE...      # wheels (.whl) and/or sdists (.tar.gz)
"""

import sys
import tarfile
import zipfile

LICENSES = ["LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY_NOTICES.md", "libjpeg-turbo-LICENSE.md",
            "libjpeg-turbo-README.ijg", "rust-dependencies.txt"]


def check(path):
    if path.endswith(".whl"):
        names = zipfile.ZipFile(path).namelist()
        meta = zipfile.ZipFile(path).read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
        want = [f".dist-info/licenses/licenses/{f}" for f in LICENSES]
    else:
        tf = tarfile.open(path)
        names = tf.getnames()
        meta = tf.extractfile(next(n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO"))).read().decode()
        want = [f"/licenses/{f}" for f in LICENSES]
    missing = [w for w in want if not any(n.endswith(w) for n in names)]
    for line in ("License-Expression: MIT OR Apache-2.0", "Classifier: Development Status :: 3 - Alpha"):
        if line not in meta:
            missing.append(f"METADATA line {line!r}")
    for f in LICENSES:
        if f"License-File: licenses/{f}" not in meta:
            missing.append(f"METADATA License-File {f}")
    status = "ok" if not missing else "MISSING " + ", ".join(missing)
    print(f"{path}: {len(names)} files, {status}")
    return not missing


if __name__ == "__main__":
    sys.exit(0 if all([check(p) for p in sys.argv[1:]]) else 1)
