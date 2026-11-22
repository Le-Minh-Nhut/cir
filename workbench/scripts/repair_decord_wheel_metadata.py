#!/usr/bin/env python3
"""Repair the mislabeled CPython ABI tag in the official Decord 0.6.0 wheel.

The package's Python binding uses ctypes.CDLL, while official Linux wheel embeds a
C API shared library. This changes only wheel compatibility metadata, never payload.
"""
import argparse
import base64
import csv
import hashlib
import io
import sys
import zipfile
import urllib.request
from pathlib import Path

OFFICIAL_SHA256 = "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"
REPAIRED_SHA256 = "07dd2c3491a8ee92786fe41a71798b2057e22a47c8a66774b9aea72ac7215d1a"
OLD_TAG = b"Tag: cp36-cp36m-manylinux2010_x86_64"
NEW_TAG = b"Tag: py3-none-manylinux2010_x86_64"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def repair(source: Path, destination: Path) -> None:
    if not source.is_file() or sha256(source.read_bytes()) != OFFICIAL_SHA256:
        raise ValueError(f"unexpected official wheel SHA256: {source}")
    with zipfile.ZipFile(source) as wheel:
        infos = [info for info in wheel.infolist() if not info.is_dir()]
        files = {info.filename: wheel.read(info) for info in infos}

    wheel_name = next(n for n in files if n.endswith(".dist-info/WHEEL"))
    if files[wheel_name].count(OLD_TAG) != 1:
        raise ValueError("official wheel WHEEL metadata did not contain expected cp36 tag")
    files[wheel_name] = files[wheel_name].replace(OLD_TAG, NEW_TAG, 1)
    record = wheel_name.rsplit("/", 1)[0] + "/RECORD"

    rows = []
    for name, payload in sorted(files.items()):
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        rows.append((name, "sha256=" + digest, str(len(payload))))
    rows.append((record, "", ""))
    text = io.StringIO(newline="")
    csv.writer(text, lineterminator="\n").writerows(rows)
    files[record] = text.getvalue().encode("utf-8")

    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w") as output:
        info_by_name = {info.filename: info for info in original.infolist() if not info.is_dir()}
        for name, payload in sorted(files.items()):
            info = info_by_name[name]
            output.writestr(info, payload, compress_type=info.compress_type, compresslevel=9)
    with zipfile.ZipFile(destination) as repaired:
        record_rows = list(csv.reader(io.StringIO(repaired.read(record).decode("utf-8"))))
        if [row for row in record_rows if row[0] == record] != [[record, "", ""]]:
            raise ValueError("RECORD must contain one blank-digest self entry")
        if len({row[0] for row in record_rows}) != len(record_rows):
            raise ValueError("RECORD contains duplicate paths")
        for name, digest, size in record_rows:
            if name == record:
                continue
            content = repaired.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            if digest != "sha256=" + expected or size != str(len(content)):
                raise ValueError(f"RECORD entry mismatch: {name}")

    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination) as repaired:
        for info in original.infolist():
            if info.is_dir() or info.filename.endswith((".dist-info/WHEEL", ".dist-info/RECORD")):
                continue
            if repaired.read(info.filename) != original.read(info.filename):
                raise ValueError(f"wheel payload changed: {info.filename}")
        if repaired.read(wheel_name).count(NEW_TAG) != 1:
            raise ValueError("repaired wheel tag validation failed")

    payload = destination.read_bytes()
    repaired_sha256 = sha256(payload)
    if REPAIRED_SHA256 and repaired_sha256 != REPAIRED_SHA256:
        raise ValueError(f"unexpected repaired wheel SHA256: {repaired_sha256}")
    print(f"official_sha256={OFFICIAL_SHA256}")
    print(f"repaired_path={destination}")
    print(f"repaired_bytes={len(payload)}")
    print(f"repaired_sha256={repaired_sha256}")
    print(f"tag={NEW_TAG.decode().removeprefix('Tag: ')}")
    print("payload_comparison=all entries except WHEEL and regenerated RECORD byte-identical")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="official wheel download/cache path")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if not args.source.exists():
            args.source.parent.mkdir(parents=True, exist_ok=True)
            url = "https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41b6a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if sha256(payload) != OFFICIAL_SHA256:
                raise ValueError("downloaded official wheel SHA256 mismatch")
            args.source.write_bytes(payload)
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        repair(args.source, args.destination)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"repair failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
