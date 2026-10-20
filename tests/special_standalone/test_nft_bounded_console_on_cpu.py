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

"""Bounded console subprocess, retention and owned process-group contracts."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts/omni_art_joint32/bounded_console.py"
spec = importlib.util.spec_from_file_location("nft_console", HELPER)
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)


class BoundedConsoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "logs/train.log"

    def command(self, code, max_bytes=257, backups=2, helper=HELPER, log=None):
        return [sys.executable, str(helper), "--log", str(log or self.log),
                "--max-bytes", str(max_bytes), "--backup-count", str(backups),
                "--io-timeout", "3", "--drain-timeout", "0.15", "--output-root", str(self.root),
                "--", sys.executable, "-c", code]

    def execute(self, code, **kwargs):
        return subprocess.run(self.command(code, **kwargs), capture_output=True, timeout=10)

    def retained(self, backups=2):
        paths = [Path(f"{self.log}.{index}") for index in range(backups, 0, -1)] + [self.log]
        return b"".join(path.read_bytes() for path in paths if path.exists())

    def test_success_stderr_order_and_no_duplicate_owner_stream(self):
        result = self.execute("import os; os.write(1,b'out\\n'); os.write(2,b'err\\n')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.retained(), b"out\nerr\n")
        self.assertEqual(result.stdout.count(b"Training console:"), 1)
        self.assertNotIn(b"out\n", result.stdout)
        self.assertNotIn(b"err\n", result.stderr)

    def test_failure_preserves_exit_17_exactly(self):
        result = self.execute("import sys; print('last error'); sys.exit(17)")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertIn(b"last error", self.retained())

    def test_large_binary_output_and_oversized_line_stay_strictly_bounded(self):
        for payload in (bytes(range(256)) * 1024 + b"END", b"x" * 100003 + b"TAIL\n"):
            with self.subTest(length=len(payload)):
                payload_path = self.root / "payload.bin"
                payload_path.write_bytes(payload)
                result = self.execute(f"import sys; sys.stdout.buffer.write(open({str(payload_path)!r},'rb').read())",
                                      max_bytes=4096)
                self.assertEqual(result.returncode, 0, result.stderr)
                files = list(self.log.parent.glob("train.log*"))
                self.assertLessEqual(len(files), 3)
                self.assertTrue(all(path.stat().st_size <= 4096 for path in files))
                self.assertLessEqual(sum(path.stat().st_size for path in files), 4096 * 3)
                self.assertTrue(payload.endswith(self.retained()))
                self.assertTrue(self.retained().endswith(payload[-4:]))
                self.assertLess(len(result.stdout) + len(result.stderr), 1024)

    def test_zero_backup_count_is_also_bounded(self):
        result = self.execute("print('x'*1000)", max_bytes=257, backups=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLessEqual(self.log.stat().st_size, 257)
        self.assertEqual(list(self.log.parent.glob("train.log.*")), [])

    def test_child_signal_status_is_preserved(self):
        for signum in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=signum):
                result = self.execute(f"import os; os.kill(os.getpid(),{int(signum)})")
                self.assertEqual(result.returncode, -signum, result.stderr)

    def signal_fixture(self, signum, group=False):
        ready = self.root / "ready.json"
        ready.unlink(missing_ok=True)
        code = (
            "import json,os,signal,sys,time\n"
            f"signal.signal({int(signum)},lambda *args:sys.exit(23))\n"
            f"with open({str(ready) + '.tmp'!r},'w') as stream: stream.write(json.dumps([os.getpid(),os.getpgrp()]))\n"
            f"os.replace({str(ready) + '.tmp'!r},{str(ready)!r})\n"
            "time.sleep(10)\n"
        )
        process = subprocess.Popen(self.command(code), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 4
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            child_pid, child_group = json.loads(ready.read_text())
            self.assertEqual(child_group, os.getpgid(process.pid))
            self.assertEqual(os.getpgid(child_pid), child_group)
            if group:
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 23, stderr)
            self.assertLess(len(stdout) + len(stderr), 1024)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=2)

    def test_direct_signals_forward_with_original_process_group(self):
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=signum):
                self.signal_fixture(signum)

    def test_owner_group_term_preserves_child_handler_status(self):
        self.signal_fixture(signal.SIGTERM, group=True)

    def test_group_terms_do_not_suppress_later_direct_term(self):
        for group_count in (1, 2):
            with self.subTest(group_count=group_count):
                state = self.root / "signal_deliveries.json"
                state.unlink(missing_ok=True)
                code = (
                    "import json,os,signal,sys,time\n"
                    "deliveries=0\n"
                    "def record():\n"
                    f" with open({str(state) + '.tmp'!r},'w') as stream:\n"
                    "  json.dump({'pid':os.getpid(),'deliveries':deliveries},stream)\n"
                    f" os.replace({str(state) + '.tmp'!r},{str(state)!r})\n"
                    "def handle(*args):\n"
                    " global deliveries\n"
                    " deliveries+=1\n"
                    " record()\n"
                    f" if deliveries=={group_count + 1}:sys.exit(23)\n"
                    "signal.signal(signal.SIGTERM,handle)\n"
                    "record()\n"
                    "time.sleep(15)\n"
                )
                process = subprocess.Popen(self.command(code), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True)

                def wait_for_delivery(expected):
                    deadline = time.monotonic() + 4
                    while time.monotonic() < deadline:
                        if state.exists():
                            observed = json.loads(state.read_text())
                            if observed["deliveries"] >= expected:
                                self.assertEqual(observed["deliveries"], expected)
                                return
                        if process.poll() is not None:
                            break
                        time.sleep(0.01)
                    self.fail(f"child did not record signal delivery {expected}")

                try:
                    wait_for_delivery(0)
                    for delivery in range(1, group_count + 1):
                        os.killpg(process.pid, signal.SIGTERM)
                        wait_for_delivery(delivery)
                        time.sleep(0.2)
                        self.assertIsNone(process.poll())
                        self.assertEqual(json.loads(state.read_text())["deliveries"], delivery)
                    process.send_signal(signal.SIGTERM)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 23, stderr)
                    self.assertEqual(json.loads(state.read_text())["deliveries"], group_count + 1)
                    self.assertLess(len(stdout) + len(stderr), 1024)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate(timeout=2)

    def test_sink_term_marker_is_drained_before_bounded_eof_shutdown(self):
        writer = console.BoundedSink(self.log, 257, 2, 1, self.root)
        try:
            os.kill(writer.process.pid, signal.SIGTERM)
            self.assertTrue(select.select([writer.channel], [], [], 1)[0])
            self.assertIsNone(writer.process.poll())
            writer.close()
            self.assertEqual(writer.process.returncode, 0)
        finally:
            if writer.process.poll() is None:
                writer.process.kill()
                writer.process.wait(timeout=2)

    def test_exited_child_does_not_wait_for_descendant_pipe_eof(self):
        code = (
            "import subprocess,sys\n"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.5)'])\n"
            "print('direct child done',flush=True)\n"
            "sys.exit(17)\n"
        )
        started = time.monotonic()
        result = self.execute(code)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertLess(time.monotonic() - started, 1.4)
        self.assertIn(b"direct child done", self.retained())

    def test_log_probe_failure_prevents_child_launch(self):
        blocked = self.root / "blocked"
        blocked.write_text("not a directory")
        marker = self.root / "must_not_exist"
        result = self.execute(f"open({str(marker)!r},'w').write('started')", log=blocked / "console.log")
        self.assertEqual(result.returncode, 74)
        self.assertFalse(marker.exists())
        self.assertIn(b"Bounded console failed", result.stderr)

    def test_stalled_sink_startup_has_bounded_deadline(self):
        stalled = self.root / "stall.py"
        stalled.write_text(
            "import importlib.util,sys,time\n"
            f"spec=importlib.util.spec_from_file_location('actual_console',{str(HELPER)!r})\n"
            "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
            "class Stalled:\n"
            " def __init__(self,*args): time.sleep(30)\n"
            "module.RotatingConsole=Stalled\n"
            "sys.exit(module.main())\n"
        )
        started = time.monotonic()
        with patch.object(console, "__file__", str(stalled)):
            with self.assertRaises(TimeoutError):
                console.BoundedSink(self.log, 257, 2, 0.1, self.root)
        self.assertLess(time.monotonic() - started, 2)

    def test_log_write_failure_stops_only_direct_test_child(self):
        ready = self.root / "child.pid"
        code = (f"import os,time;open({str(ready)!r},'w').write(str(os.getpid()));"
                "print('output',flush=True);time.sleep(10)")
        with patch.object(console.BoundedSink, "write", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                console.run([sys.executable, "-c", code], self.log, 257, 2, 1, 0.15, self.root)
        self.assertTrue(ready.exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(int(ready.read_text()), 0)

    def test_optional_retention_start_periodic_exit_and_atomic_error_state(self):
        isolated = self.root / "isolated"
        isolated.mkdir()
        helper = isolated / "bounded_console.py"
        shutil.copyfile(HELPER, helper)
        retention = isolated / "media_retention.py"
        events = self.root / "retention_events"
        retention.write_text(
            "def prune_completed_media(root):\n"
            f" with open({str(events)!r},'a') as stream: stream.write('called\\n')\n"
            " return {'status':'skipped','reason':'cpu_fixture'}\n"
        )
        with patch.object(console, "__file__", str(helper)), patch.object(console, "MEDIA_INTERVAL_SECONDS", 0.05):
            self.assertEqual(console.run([sys.executable, "-c", "import time;time.sleep(.18)"],
                                         self.log, 257, 2, 1, 0.15, self.root), 0)
        self.assertGreaterEqual(len(events.read_text().splitlines()), 3)
        state = self.root / "provenance/media_retention_state.json"
        self.assertEqual(json.loads(state.read_text())["result"]["reason"], "cpu_fixture")
        self.assertEqual(list(state.parent.glob(".media_retention*")), [])
        retention.write_text("def prune_completed_media(root):\n raise ValueError('guard rejected')\n")
        result = self.execute("import sys;sys.exit(17)", helper=helper)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(json.loads(state.read_text())["result"], {"status": "error", "error_type": "ValueError"})


if __name__ == "__main__":
    unittest.main()
