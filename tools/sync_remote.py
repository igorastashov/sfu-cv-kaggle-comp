#!/usr/bin/env python3
"""Синхронизация репозитория на удалённую машину и управление стендом.

Примеры:
  # Первый раз: всё (код + data + models, ~4 ГБ)
  python tools/sync_remote.py --full --start

  # Быстрое обновление кода и ноутбука (секунды)
  python tools/sync_remote.py

  # Обновить код и перезапустить контейнер
  python tools/sync_remote.py --restart

  # Только статус
  python tools/sync_remote.py --status
"""
from __future__ import annotations

import argparse
import getpass
import os
import stat
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST = "192.168.52.118"
DEFAULT_USER = "dmd"
DEFAULT_REMOTE = "~/sfu-kaggle"

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
    ".pytest_cache", ".cache", ".idea", ".vscode",
}
SKIP_FILES = {".pyc", ".csv"}
CODE_PATHS = ("stand", "notebooks", "tools", "docker", "docs", "requirements.txt")
FULL_EXTRA = ("data", "models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("SFU_REMOTE_HOST", DEFAULT_HOST))
    p.add_argument("--user", default=os.environ.get("SFU_REMOTE_USER", DEFAULT_USER))
    p.add_argument("--password", default=os.environ.get("SFU_REMOTE_PASSWORD"))
    p.add_argument("--remote", default=os.environ.get("SFU_REMOTE_DIR", DEFAULT_REMOTE))
    p.add_argument("--full", action="store_true", help="также data/ и models/")
    p.add_argument("--start", action="store_true", help="поднять стенд после синхронизации")
    p.add_argument("--restart", action="store_true", help="docker compose restart после синхронизации")
    p.add_argument("--status", action="store_true", help="только проверить удалённый стенд")
    p.add_argument("--quiet", action="store_true", help="не печатать ссылку и токен")
    return p.parse_args()


def resolve_remote_dir(client: paramiko.SSHClient, remote: str) -> str:
    remote = remote.strip()
    if remote.startswith("~/"):
        _, out, _ = run(client, "printf '%s' \"$HOME\"")
        home = out.strip()
        if not home:
            raise RuntimeError("не удалось определить HOME на удалённой машине")
        return home + remote[1:]
    if remote == "~":
        _, out, _ = run(client, "printf '%s' \"$HOME\"")
        return out.strip()
    return remote


def connect(host: str, user: str, password: str) -> paramiko.SSHClient:
    if not password:
        password = getpass.getpass(f"Пароль {user}@{host}: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=30,
                   allow_agent=False, look_for_keys=False)
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 3600) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return code, out, err


def remote_path(remote: str, *parts: str) -> str:
    base = remote.rstrip("/")
    if parts:
        return base + "/" + "/".join(parts)
    return base


def collect_files(paths: tuple[str, ...]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for rel in paths:
        src = ROOT / rel
        if src.is_file():
            files.append((src, rel.replace("\\", "/")))
            continue
        if not src.is_dir():
            print(f"пропуск (нет): {rel}", file=sys.stderr)
            continue
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            parts = set(path.relative_to(src).parts)
            if parts & SKIP_DIRS:
                continue
            if path.suffix in SKIP_FILES:
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
                continue
            rel_path = path.relative_to(ROOT).as_posix()
            files.append((path, rel_path))
    return sorted(files, key=lambda x: x[1])


def ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}" if cur else f"/{part}" if path.startswith("/") else part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_files(client: paramiko.SSHClient, remote: str, files: list[tuple[Path, str]]) -> None:
    run(client, f"mkdir -p {remote}")
    sftp = client.open_sftp()
    total = len(files)
    total_bytes = sum(p.stat().st_size for p, _ in files)
    done_bytes = 0
    t0 = time.time()

    for i, (local, rel) in enumerate(files, 1):
        rpath = remote_path(remote, rel)
        rdir = str(Path(rpath).parent).replace("\\", "/")
        ensure_remote_dir(sftp, rdir)
        size = local.stat().st_size
        sftp.put(str(local), rpath)
        done_bytes += size
        elapsed = max(time.time() - t0, 0.1)
        speed = done_bytes / elapsed / 1024 / 1024
        pct = 100 * done_bytes / max(total_bytes, 1)
        print(f"\r[{i}/{total}] {pct:5.1f}%  {speed:5.1f} МБ/с  {rel[:60]:<60}", end="", flush=True)

    sftp.close()
    print(f"\nЗагружено: {total} файлов, {done_bytes / 1024 / 1024 / 1024:.2f} ГБ за {time.time() - t0:.0f} с")


def fix_scripts(client: paramiko.SSHClient, remote: str) -> None:
    run(client, f"sed -i 's/\\r$//' {remote}/tools/start_stand.sh")


def start_stand(client: paramiko.SSHClient, remote: str) -> None:
    fix_scripts(client, remote)
    print("\nСборка и запуск стенда (первая сборка может занять 20–40 мин)...")
    code, out, err = run(client, f"cd {remote} && bash tools/start_stand.sh", timeout=7200)
    print(out, end="")
    if err:
        print(err, file=sys.stderr)
    if code != 0:
        sys.exit(code)


def restart_stand(client: paramiko.SSHClient, remote: str) -> None:
    code, out, err = run(client,
        f"cd {remote} && docker compose -f docker/compose.yaml restart stand")
    print(out, end="")
    if err:
        print(err, file=sys.stderr)
    if code != 0:
        sys.exit(code)
    _, out, _ = run(client, f"grep '^JUPYTER_TOKEN=' {remote}/docker/.env | cut -d= -f2-")
    token = out.strip()
    print(f"\nСтенд перезапущен. Jupyter: http://{client.get_transport().getpeername()[0]}:8888/lab?token={token}")


def show_status(client: paramiko.SSHClient, remote: str, host: str) -> None:
    cmds = [
        f"test -d {remote} && du -sm {remote}/data {remote}/models 2>/dev/null || echo 'репозиторий не найден'",
        f"docker ps --filter name=sfu-cv-stand --format '{{{{.Status}}}}\\t{{{{.Ports}}}}'",
        f"test -f {remote}/docker/.env && grep JUPYTER_TOKEN {remote}/docker/.env || true",
        "curl -s -o /dev/null -w '%{http_code}' http://localhost:8888/api || echo fail",
    ]
    labels = ["данные/модели", "контейнер", "токен", "jupyter api"]
    for label, cmd in zip(labels, cmds):
        _, out, _ = run(client, cmd)
        print(f"{label}: {out.strip()}")
    _, out, _ = run(client, f"grep '^JUPYTER_TOKEN=' {remote}/docker/.env 2>/dev/null | cut -d= -f2-")
    token = out.strip()
    if token:
        print(f"\nСсылка: http://{host}:8888/lab?token={token}")
        print(f"Ноутбук: notebooks/1_chto-vidit-mashina.ipynb")


def main() -> None:
    args = parse_args()
    client = connect(args.host, args.user, args.password or "")

    try:
        remote = resolve_remote_dir(client, args.remote)
        if args.status:
            show_status(client, remote, args.host)
            return

        paths = list(CODE_PATHS)
        if args.full:
            paths.extend(FULL_EXTRA)

        files = collect_files(tuple(paths))
        print(f"Синхронизация -> {args.user}@{args.host}:{remote}")
        print(f"Файлов: {len(files)}, режим: {'полный' if args.full else 'код'}")
        upload_files(client, remote, files)

        if args.start:
            start_stand(client, remote)
        elif args.restart:
            restart_stand(client, remote)

        if not args.quiet:
            show_status(client, remote, args.host)
    finally:
        client.close()


if __name__ == "__main__":
    main()
