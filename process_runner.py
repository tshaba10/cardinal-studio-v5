"""Run trusted local workers with cancellation, not an elapsed-time limit."""

import json
import subprocess
import sys
from queue import Queue, Empty
from threading import Thread


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_worker(script, payload, control, env):
    """Exchange JSON with a trusted local script; poll cancellation, not a deadline.

    Reader/writer threads prevent a full pipe from blocking the UI's Stop path.
    Only a final result event is returned; progress events update telemetry.
    The worker is always reaped, including on cancellation or malformed output.
    """
    control.check()
    process = subprocess.Popen(
        [sys.executable, "-I", "-u", "-B", str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=script.parent,
    )
    events = Queue()

    def read():
        try:
            for line in process.stdout:
                events.put(json.loads(line))
        except (ValueError, OSError):
            events.put({"error": "The local worker returned an unreadable response."})
        finally:
            events.put(None)

    def write():
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=True))
            process.stdin.close()
        except (OSError, ValueError):
            pass

    reader = Thread(target=read, daemon=True)
    writer = Thread(target=write, daemon=True)
    reader.start()
    writer.start()
    try:
        while True:
            control.check()
            try:
                event = events.get(timeout=0.2)
            except Empty:
                continue
            control.check()
            if event is None:
                raise ValueError(
                    "The local worker ended before returning a complete answer. No partial result was applied."
                )
            if "error" in event:
                raise ValueError(event["error"])
            if "result" in event:
                return event["result"]
            if "progress" in event:
                control.activity(event["progress"])
    finally:
        stop_process(process)
        reader.join(timeout=2)
        writer.join(timeout=2)
        for pipe in (process.stdout, process.stdin):
            if pipe and not pipe.closed:
                pipe.close()
