# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stream one owned child's combined console into strictly bounded log files."""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import time

MEDIA_INTERVAL_SECONDS = 120
SIGNAL_MARKERS = {signal.SIGTERM: b"T", signal.SIGINT: b"I", signal.SIGHUP: b"H"}


class RotatingConsole:
    def __init__(self, path, max_bytes, backup_count):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for index in range(backup_count + 1):
            candidate = self.path if index == 0 else Path(f"{self.path}.{index}")
            if candidate.exists() and candidate.stat().st_size > max_bytes:
                raise ValueError("existing console file exceeds configured byte bound")
        self.stream = self.path.open("ab", buffering=0)
        self.size = self.stream.tell()

    def rotate(self):
        self.stream.close()
        if self.backup_count:
            Path(f"{self.path}.{self.backup_count}").unlink(missing_ok=True)
            for index in range(self.backup_count - 1, 0, -1):
                source = Path(f"{self.path}.{index}")
                if source.exists():
                    source.replace(Path(f"{self.path}.{index + 1}"))
            self.path.replace(Path(f"{self.path}.1"))
        self.stream = self.path.open("wb", buffering=0)
        self.size = 0

    def write(self, content):
        remaining = memoryview(content)
        while remaining:
            if self.size == self.max_bytes:
                self.rotate()
            written = self.stream.write(remaining[:self.max_bytes - self.size])
            if not written:
                raise OSError("console write made no progress")
            self.size += written
            remaining = remaining[written:]

    def close(self):
        self.stream.close()


def receive_exact(channel, count):
    chunks = []
    while count:
        chunk = channel.recv(count)
        if not chunk:
            if chunks:
                raise EOFError("incomplete console frame")
            return None
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def sink(descriptor, path, max_bytes, backup_count, output_root):
    channel = socket.socket(fileno=descriptor)

    def group_signal(signum, frame):
        try:
            channel.send(SIGNAL_MARKERS[signum], socket.MSG_DONTWAIT)
        except (BlockingIOError, BrokenPipeError):
            pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, group_signal)
    writer = RotatingConsole(path, max_bytes, backup_count)
    retention = Path(__file__).with_name("media_retention.py")
    prune = None
    import_error = None
    try:
        if retention.is_file():
            spec = importlib.util.spec_from_file_location("omni_media_retention", retention)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            prune = module.prune_completed_media
    except Exception as error:
        import_error = type(error).__name__

    def prune_media():
        result = {"status": "skipped", "reason": "helper_absent"}
        try:
            if import_error is not None:
                result = {"status": "error", "error_type": import_error}
            elif prune is not None:
                result = prune(Path(output_root))
        except Exception as error:
            result = {"status": "error", "error_type": type(error).__name__}
        temporary = None
        try:
            state = json.dumps({"observed_unix": time.time(), "result": result}).encode()
            if len(state) > 65536:
                state = b'{"result":{"status":"error","error_type":"RetentionResultTooLarge"}}'
            directory = Path(output_root) / "provenance"
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / ".media_retention_state.tmp"
            with temporary.open("wb") as stream:
                stream.write(state + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(directory / "media_retention_state.json")
        except (OSError, TypeError, ValueError) as error:
            writer.write(f"Media retention state failed: {type(error).__name__}\n".encode())
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    writer.write(f"Media retention temporary cleanup failed: {type(error).__name__}\n".encode())

    try:
        prune_media()
        channel.sendall(b"R")
        while True:
            header = receive_exact(channel, 4)
            if header is None:
                return
            count = struct.unpack("!I", header)[0]
            if not 0 <= count <= 65536:
                raise ValueError("invalid console frame size")
            if count:
                content = receive_exact(channel, count)
                if content is None:
                    raise EOFError("missing console frame")
                writer.write(content)
            else:
                prune_media()
            channel.sendall(b"A")
    finally:
        prune_media()
        writer.close()
        channel.close()


class BoundedSink:
    def __init__(self, path, max_bytes, backup_count, timeout, output_root):
        self.timeout = timeout
        self.group_signals = Counter()
        self.channel, peer = socket.socketpair()
        self.channel.setblocking(False)
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).absolute()), "--sink", str(peer.fileno()),
                 str(path), str(max_bytes), str(backup_count), str(output_root)],
                pass_fds=(peer.fileno(),), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except BaseException:
            self.channel.close()
            raise
        finally:
            peer.close()
        try:
            self.acknowledge(b"R", time.monotonic() + self.timeout)
        except BaseException:
            self.close(check=False)
            raise

    def ready(self, deadline, writing=False):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("console filesystem I/O deadline exceeded")
            readable, writable, _ = select.select(
                [] if writing else [self.channel], [self.channel] if writing else [], [], min(remaining, 0.1)
            )
            if readable or writable:
                return
            if self.process.poll() is not None:
                raise OSError("console sink exited before acknowledging I/O")

    def acknowledge(self, expected, deadline):
        while True:
            self.ready(deadline)
            message = self.channel.recv(1)
            if message in SIGNAL_MARKERS.values():
                self.group_signals.update(signum for signum, marker in SIGNAL_MARKERS.items() if marker == message)
            elif message == expected:
                return
            else:
                raise OSError("console sink did not acknowledge I/O")

    def collect_group_signals(self):
        while select.select([self.channel], [], [], 0)[0]:
            message = self.channel.recv(1)
            if message not in SIGNAL_MARKERS.values():
                raise OSError("unexpected console sink control message")
            self.group_signals.update(signum for signum, marker in SIGNAL_MARKERS.items() if marker == message)

    def write(self, content):
        deadline = time.monotonic() + self.timeout
        frame = memoryview(struct.pack("!I", len(content)) + content)
        while frame:
            self.ready(deadline, writing=True)
            written = self.channel.send(frame)
            if not written:
                raise OSError("console sink connection closed")
            frame = frame[written:]
        self.acknowledge(b"A", deadline)

    def close(self, check=True):
        try:
            try:
                self.channel.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            deadline = time.monotonic() + self.timeout
            while self.process.poll() is None and time.monotonic() < deadline:
                if select.select([self.channel], [], [], 0.05)[0]:
                    if not self.channel.recv(65536):
                        break
            code = self.process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.kill()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            if check:
                raise TimeoutError("console sink finalization exceeded I/O deadline")
            return
        finally:
            self.channel.close()
        if check and code:
            raise OSError("console sink failed during finalization")


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass


def run(command, path, max_bytes, backup_count, io_timeout, drain_timeout, output_root=None):
    writer = None
    child = None
    previous_handlers = {}
    pending_signals = []

    def forward(signum, frame):
        pending_signals.append((signum, time.monotonic()))

    try:
        writer = BoundedSink(path, max_bytes, backup_count, io_timeout,
                             Path(path).parent.parent if output_root is None else output_root)
        print(f"Training console: {path} (max_bytes={max_bytes}, backups={backup_count})", flush=True)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, forward)
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        os.set_blocking(child.stdout.fileno(), False)
        exited_at = None
        pipe_open = True
        next_retention = time.monotonic() + MEDIA_INTERVAL_SECONDS
        while pipe_open or child.poll() is None:
            writer.collect_group_signals()
            for signum, received_at in pending_signals[:]:
                if writer.group_signals[signum] > 0:
                    writer.group_signals[signum] -= 1
                    if writer.group_signals[signum] == 0:
                        del writer.group_signals[signum]
                    pending_signals.remove((signum, received_at))
                elif time.monotonic() - received_at >= 0.1:
                    if child.poll() is None:
                        try:
                            child.send_signal(signum)
                        except ProcessLookupError:
                            pass
                    pending_signals.remove((signum, received_at))
            if time.monotonic() >= next_retention:
                writer.write(b"")
                next_retention = time.monotonic() + MEDIA_INTERVAL_SECONDS
            if child.poll() is not None:
                if exited_at is None:
                    exited_at = time.monotonic()
                if time.monotonic() - exited_at >= drain_timeout:
                    break
            if not pipe_open:
                time.sleep(0.05)
                continue
            readable, _, _ = select.select([child.stdout], [], [], 0.05)
            if readable:
                content = os.read(child.stdout.fileno(), 65536)
                if content:
                    writer.write(content)
                else:
                    pipe_open = False
        return child.wait(timeout=1)
    finally:
        if child is not None:
            stop_child(child)
            child.stdout.close()
        if writer is not None:
            writer.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    if sys.argv[1:2] == ["--sink"]:
        sink(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--backup-count", type=int, default=3)
    parser.add_argument("--io-timeout", type=float, default=10.0)
    parser.add_argument("--drain-timeout", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_bytes <= 0 or not 0 <= args.backup_count <= 16:
        parser.error("command, positive max-bytes and backup-count in [0,16] required")
    if not 0 < args.io_timeout <= 60 or not 0 < args.drain_timeout <= 60:
        parser.error("I/O and drain deadlines must be in (0,60] seconds")
    try:
        code = run(command, args.log, args.max_bytes, args.backup_count, args.io_timeout,
                   args.drain_timeout, args.output_root)
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as error:
        print(f"Bounded console failed: {type(error).__name__}", file=sys.stderr, flush=True)
        return 74
    if code < 0:
        signum = -code
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
