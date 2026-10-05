"""Download a sample of real photos for the end-to-end check (see the README, "Testing").

    python scripts/real/fetch_photos.py OUT_DIR [N=200]

Source: Wikimedia Commons "Quality images" (a curated category of real photographs), JPEG
files whose license is CC0 or public domain. Files up to 6 MB / 24 MP are fetched as
uploaded (camera JPEGs, EXIF and all); larger ones as Wikimedia's 1280-px JPEG thumbnails.
OUT_DIR/manifest.json records the title, license, author, URL and SHA-256 of every file.
The photos are not committed to the repository.
"""

import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "hf-processors-rs-eval/0.1 (https://github.com/Dat-cool-repo/hf-processors-rs)"
OK_LICENSES = ("CC0", "Public domain", "PD")


def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - network hiccups: back off and retry
            if i == retries - 1:
                raise
            time.sleep(2 + 3 * i)


def search(offset):
    q = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "gsroffset": offset,
        "gsrsearch": "haswbstatement:P275=Q6938433 filemime:image/jpeg incategory:Quality_images",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|Artist",
        "iiurlwidth": 1280,
    }
    return json.loads(get(API + "?" + urllib.parse.urlencode(q)))


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out.mkdir(parents=True, exist_ok=True)
    manifest, offset = [], 0
    while len(manifest) < n:
        res = search(offset)
        pages = sorted(res.get("query", {}).get("pages", {}).values(), key=lambda p: p["index"])
        for p in pages:
            if len(manifest) >= n:
                break
            ii = p["imageinfo"][0]
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if ii.get("mime") != "image/jpeg" or not lic.startswith(OK_LICENSES):
                continue
            original = ii["size"] <= 6_000_000 and ii["width"] * ii["height"] <= 24_000_000
            url = (ii["url"] if original else ii["thumburl"]).split("?")[0]
            name = f"{len(manifest):03d}_{p['pageid']}.jpg"
            data = get(url)
            (out / name).write_bytes(data)
            artist = re.sub(r"<[^>]+>", "", html.unescape(meta.get("Artist", {}).get("value", ""))).strip()
            manifest.append({
                "file": name,
                "title": p["title"],
                "license": lic,
                "artist": artist,
                "source": ii["descriptionurl"],
                "url": url,
                "variant": "original" if original else "thumbnail-1280",
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            print(f"{name} {len(data) // 1024} KB {lic} {p['title'][:70]}", flush=True)
            time.sleep(0.2)
        if "continue" not in res:
            break
        offset = res["continue"]["gsroffset"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} photos in {out}")


if __name__ == "__main__":
    main()
