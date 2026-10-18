#!/usr/bin/env python3
"""Pause only this campaign's active transfer children if Intern becomes too full."""

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/m2v_intern/zhenglin/omni_art_ablation_20261016")
MIN_FREE_BYTES = int(os.environ.get("OMNI_MIN_FREE_GIB", "250")) * 1024**3
ROOT_PROCESSES = {
    184791: "hf download diffusers/LTX-2.3-Diffusers",
    297694: "download_reward_assets.sh",
    316564: "install_runtime.sh",
}


def parse_proc_stat(raw_stat):
    # The parenthesized comm field can contain spaces, so split only after
    # its final ')'. Fields 3, 4 and 22 then have tail indexes 0, 1, 19.
    stat = raw_stat[raw_stat.rfind(")") + 2 :].split()
    return int(stat[1]), int(stat[19])


def process(pid):
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text()
        ppid, start_ticks = parse_proc_stat(raw_stat)
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return {"pid": pid, "ppid": ppid, "start_ticks": start_ticks, "cmd": cmd.strip()}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def utc():
    return datetime.now(timezone.utc).isoformat()


def free_bytes():
    stat = os.statvfs(ROOT)
    return stat.f_bavail * stat.f_frsize


def main():
    roots = {}
    for pid, fragment in ROOT_PROCESSES.items():
        record = process(pid)
        if record is None or fragment not in record["cmd"]:
            raise RuntimeError(f"transfer root pid={pid} identity mismatch")
        roots[pid] = record["start_ticks"]
    print(f"{utc()} guard armed; threshold={MIN_FREE_BYTES // 1024**3} GiB", flush=True)
    while True:
        available = free_bytes()
        if available >= MIN_FREE_BYTES:
            print(f"{utc()} Intern free={available // 1024**3} GiB", flush=True)
            time.sleep(30)
            continue
        for pid, ticks in roots.items():
            current = process(pid)
            if current is not None and current["start_ticks"] != ticks:
                raise RuntimeError(f"transfer root pid={pid} was reused")
        all_processes = {}
        for name in os.listdir("/proc"):
            if name.isdecimal() and (record := process(int(name))) is not None:
                all_processes[record["pid"]] = record

        def belongs_to_task(pid):
            seen = set()
            while pid in all_processes and pid not in seen:
                seen.add(pid)
                if pid in roots:
                    return all_processes[pid]["start_ticks"] == roots[pid]
                pid = all_processes[pid]["ppid"]
            return False

        paused = []
        for pid, record in all_processes.items():
            if not belongs_to_task(pid):
                continue
            if "hf download " not in record["cmd"] and "uv pip install " not in record["cmd"]:
                continue
            current = process(pid)
            if current is None or current["start_ticks"] != record["start_ticks"]:
                continue
            os.kill(pid, signal.SIGSTOP)
            paused.append({"pid": pid, "start_ticks": record["start_ticks"], "cmd": record["cmd"]})
        state = {"utc": utc(), "free_bytes": available, "paused": paused}
        path = ROOT / "logs/storage_guard_pause.json"
        temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(path)
        print(f"{utc()} low space: free={available // 1024**3} GiB; paused {len(paused)} task transfer processes", flush=True)
        return


if __name__ == "__main__":
    main()
