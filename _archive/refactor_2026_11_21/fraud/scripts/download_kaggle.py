#!/usr/bin/env python3
"""Kaggle dataset'i indir.

`kaggle` CLI'ina veya ek Python paketine guvenmez; yalnizca standart
kutuphane (urllib + hashlib) ile Kaggle REST API'na HMAC auth ile baglanir.
Kimlik `~/.kaggle/kaggle.json` (username + key) icinden okunur.

Kullanim:
    python3 fraud/scripts/download_kaggle.py <owner/slug> <cikis_dizini>
    python3 fraud/scripts/download_kaggle.py ealtman2019/credit-card-transactions fraud/data
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile


def load_creds():
    path = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(path) as f:
        d = json.load(f)
    if "username" not in d or "key" not in d:
        raise SystemExit("kaggle.json 'username' ve 'key' alanlari gerektirir")
    return d["username"], d["key"]


def auth_headers(username, key):
    token = hashlib.sha256(f"{username}:{key}".encode()).hexdigest()
    ts = str(int(time.time()))
    sig = hashlib.sha256(f"{ts}:{username}:{key}".encode()).hexdigest()
    return {
        "X-Kaggle-Api-V1": "1",
        "X-Kaggle-Api-Token": token,
        "X-Kaggle-Api-Hash": sig,
        "X-Kaggle-Api-Timestamp": ts,
        "User-Agent": "kaggle-api/1 (custom-stdlib-downloader)",
        "Accept": "*/*",
    }


def download(url, out_path, username, key, chunk=1 << 20):
    req = urllib.request.Request(url, headers=auth_headers(username, key))
    with urllib.request.urlopen(req, timeout=180) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(out_path, "wb") as f:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if total:
                    print(f"\r{done / 1e6:8.1f} / {total / 1e6:.1f} MB "
                          f"({100 * done / total:.0f}%)", end="", flush=True)
    print()
    return done


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    slug = sys.argv[1].strip("/")
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    username, key = load_creds()
    url = f"https://www.kaggle.com/api/v1/datasets/download?dataset={slug}"
    out_zip = os.path.join(out_dir, "dataset.zip")

    print(f"Indiriliyor: {url}\n  -> {out_zip}")
    download(url, out_zip, username, key)

    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
        print("Ariv icindeki dosyalar:", names)
        z.extractall(out_dir)
    print("Tamam. Zip korunuyor:", out_zip)


if __name__ == "__main__":
    main()
