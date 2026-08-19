"""Синхронизация, пересоздание контейнера и проверка GPU/Jupyter на 118."""
import os
import sys

import paramiko

HOST = os.environ.get("SFU_REMOTE_HOST", "192.168.52.118")
USER = os.environ.get("SFU_REMOTE_USER", "dmd")
PASSWORD = os.environ.get("SFU_REMOTE_PASSWORD", "")
REMOTE = "/home/dmd/sfu-kaggle"


def run(client, cmd, timeout=600):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    code = o.channel.recv_exit_status()
    return code, o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")


def main():
    if not PASSWORD:
        print("Задайте SFU_REMOTE_PASSWORD", file=sys.stderr)
        sys.exit(1)

    # sync via subprocess
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sync = os.path.join(root, "tools", "sync_remote.py")
    r = subprocess.run([sys.executable, sync, "--quiet"], env=os.environ, cwd=root)
    if r.returncode != 0:
        sys.exit(r.returncode)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False)

    print("\n=== compose.yaml (gpu) ===")
    _, out, _ = run(c, f"grep -E 'privileged|runtime|nvidia' {REMOTE}/docker/compose.yaml")
    print(out)

    print("=== docker compose down / up ===")
    cmd = (
        f"cd {REMOTE} && "
        "(docker compose -f docker/compose.yaml down || docker-compose -f docker/compose.yaml down) && "
        "(docker compose -f docker/compose.yaml up -d || docker-compose -f docker/compose.yaml up -d)"
    )
    code, out, err = run(c, cmd, timeout=300)
    print(out)
    if err:
        print(err, file=sys.stderr)
    if code != 0:
        sys.exit(code)

    import time
    time.sleep(8)

    checks = [
        ("docker ps", "docker ps --filter name=sfu-cv-stand --format '{{.Status}}\\t{{.Ports}}'"),
        ("runtime", "docker inspect sfu-cv-stand --format '{{.HostConfig.Runtime}} privileged={{.HostConfig.Privileged}}'"),
        ("nvidia-smi", "docker exec sfu-cv-stand nvidia-smi -L 2>&1"),
        ("torch cuda", '''docker exec sfu-cv-stand python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"'''),
        ("jupyter", "curl -s -o /dev/null -w '%{http_code}' http://localhost:8888/api"),
        ("data", '''docker exec sfu-cv-stand python -c "from stand import data; data.segment_summary()"'''),
        ("token_set", f"test -s {REMOTE}/docker/.env && echo yes || echo no"),
    ]
    for label, cmd in checks:
        print(f"\n=== {label} ===")
        _, out, err = run(c, cmd, timeout=120)
        print(out.rstrip() or "(empty)")
        if err.strip():
            print("ERR:", err.rstrip())

    c.close()
    print(f"\nJupyter: http://{HOST}:8888/lab  (токен: grep JUPYTER_TOKEN {REMOTE}/docker/.env на 118)")
    print("Готово.")


if __name__ == "__main__":
    main()
