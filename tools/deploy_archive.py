#!/usr/bin/env python3
"""Сборка архива, деплой на 118, распаковка и проверка актуальности."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
HOST = os.environ.get("SFU_REMOTE_HOST", "192.168.52.118")
USER = os.environ.get("SFU_REMOTE_USER", "dmd")
PASSWORD = os.environ.get("SFU_REMOTE_PASSWORD", "")
REMOTE_HOME = "/home/dmd"
REMOTE = f"{REMOTE_HOME}/sfu-kaggle"

MARKERS = (
    "docker/compose.yaml",
    "notebooks/1_chto-vidit-mashina.ipynb",
    "stand/run.py",
    "tools/build_notebook.py",
    "requirements.txt",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_remote(client: paramiko.SSHClient, cmd: str, timeout: int = 7200) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def latest_archive() -> Path:
    pattern = str(PARENT / "sfu-kaggle-*.zip")
    files = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"архив не найден: {pattern}")
    return Path(files[0])


def pack() -> Path:
    ps1 = ROOT / "tools" / "pack_transfer.ps1"
    print("=== упаковка ===")
    r = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
        cwd=ROOT,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)
    return latest_archive()


def upload(client: paramiko.SSHClient, local_zip: Path) -> None:
    sha_local = Path(f"{local_zip}.sha256")
    if not sha_local.exists():
        raise FileNotFoundError(f"нет контрольной суммы: {sha_local}")

    remote_zip = f"{REMOTE_HOME}/{local_zip.name}"
    remote_sha = f"{remote_zip}.sha256"

    print(f"=== загрузка {local_zip.name} ({local_zip.stat().st_size / 1e9:.2f} ГБ) ===")
    sftp = client.open_sftp()
    t0 = time.time()
    sftp.put(str(local_zip), remote_zip)
    sftp.put(str(sha_local), remote_sha)
    sftp.close()
    print(f"загружено за {time.time() - t0:.0f} с")


def deploy_on_remote(client: paramiko.SSHClient, zip_name: str) -> None:
    remote_zip = f"{REMOTE_HOME}/{zip_name}"
    remote_sha = f"{remote_zip}.sha256"

    print("=== sha256sum ===")
    code, out, err = run_remote(client, f"cd {REMOTE_HOME} && sha256sum -c {Path(remote_sha).name}")
    print(out.strip())
    if code != 0:
        print(err, file=sys.stderr)
        sys.exit(code)

    print("=== распаковка ===")
    script = f"""
set -e
cd {REMOTE_HOME}
if [ -d {REMOTE} ]; then
  (cd {REMOTE} && (docker compose -f docker/compose.yaml down 2>/dev/null || docker-compose -f docker/compose.yaml down 2>/dev/null) || true)
  docker run --rm -v {REMOTE}:/w alpine sh -c 'rm -rf /w/* /w/.[!.]*' 2>/dev/null || rm -rf {REMOTE}/* {REMOTE}/.[!.]* 2>/dev/null || true
  rmdir {REMOTE} 2>/dev/null || docker run --rm -v {REMOTE_HOME}:/h alpine sh -c 'rm -rf /h/sfu-kaggle'
fi
if command -v unzip >/dev/null 2>&1; then
  unzip -q {remote_zip}
else
  python3 -m zipfile -e {remote_zip} {REMOTE_HOME}
fi
sed -i 's/\\r$//' {REMOTE}/tools/start_stand.sh
cd {REMOTE} && (docker compose -f docker/compose.yaml up -d || docker-compose -f docker/compose.yaml up -d)
"""
    code, out, err = run_remote(client, script, timeout=600)
    print(out)
    if err.strip():
        print(err, file=sys.stderr)
    if code != 0:
        sys.exit(code)


def local_markers() -> dict[str, str]:
    return {m: sha256_file(ROOT / m) for m in MARKERS}


def remote_markers(client: paramiko.SSHClient) -> dict[str, str]:
    result = {}
    for m in MARKERS:
        cmd = f"sha256sum {REMOTE}/{m} 2>/dev/null | cut -d' ' -f1"
        _, out, _ = run_remote(client, cmd)
        result[m] = out.strip()
    return result


def verify(client: paramiko.SSHClient) -> None:
    time.sleep(8)
    print("\n=== контрольные суммы ключевых файлов ===")
    loc = local_markers()
    rem = remote_markers(client)
    ok = True
    for m in MARKERS:
        match = loc[m] == rem[m]
        mark = "OK" if match else "РАСХОЖДЕНИЕ"
        if not match:
            ok = False
        print(f"  {mark}  {m}")
        if not match:
            print(f"         local:  {loc[m][:16]}...")
            print(f"         remote: {rem[m][:16]}...")

    checks = [
        ("контейнер", "docker ps --filter name=sfu-cv-stand --format '{{.Status}}'"),
        ("cuda", '''docker exec sfu-cv-stand python -c "import torch; print(torch.cuda.is_available())"'''),
        ("gpu", "docker exec sfu-cv-stand nvidia-smi -L 2>&1 | head -1"),
        ("jupyter", "curl -s -o /dev/null -w '%{http_code}' http://localhost:8888/api"),
        ("data_mb", f"du -sm {REMOTE}/data {REMOTE}/models | awk '{{print $1}}' | paste -sd/ -"),
    ]
    print("\n=== проверки стенда ===")
    for label, cmd in checks:
        _, out, _ = run_remote(client, cmd)
        print(f"  {label}: {out.strip()}")

    if not ok:
        sys.exit(2)
    print("\nЛокаль и 118 совпадают по ключевым файлам.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-only", action="store_true")
    p.add_argument("--deploy-only", metavar="ZIP")
    p.add_argument("--skip-pack", action="store_true")
    p.add_argument("--verify-only", action="store_true", help="только сверить локаль и 118")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verify_only:
        if not PASSWORD:
            print("Задайте SFU_REMOTE_PASSWORD", file=sys.stderr)
            sys.exit(1)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False)
        try:
            verify(client)
        finally:
            client.close()
        return

    if args.pack_only:
        pack()
        return

    if args.deploy_only:
        archive = Path(args.deploy_only)
    elif args.skip_pack:
        archive = latest_archive()
        print(f"архив: {archive}")
    else:
        archive = pack()

    if not PASSWORD:
        print("Задайте SFU_REMOTE_PASSWORD", file=sys.stderr)
        sys.exit(1)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False)
    try:
        upload(client, archive)
        deploy_on_remote(client, archive.name)
        verify(client)
    finally:
        client.close()

    print(f"\nJupyter: http://{HOST}:8888/lab")
    print(f"Ноутбук: notebooks/1_chto-vidit-mashina.ipynb")


if __name__ == "__main__":
    main()
