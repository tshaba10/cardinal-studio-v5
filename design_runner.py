"""Background graph runs with real stage telemetry, no overall deadline, and Stop."""

from copy import deepcopy
from threading import Event, Lock, Thread
import time


class RunStopped(ValueError):
    pass


class Control:
    def __init__(self):
        self.started = time.monotonic()
        self.cancelled = Event()
        self.lock = Lock()
        self.label = "Starting"
        self.node = "starting"
        self.detail = "Waiting for the worker to start"
        self.last_activity = self.started
        self.history = []

    def check(self):
        if self.cancelled.is_set():
            raise RunStopped(
                "Stopped by you. No partial answer was applied. An already-sent AI request may still be billed."
            )

    def stage(self, label, node=None):
        self.check()
        with self.lock:
            self.label, self.node = label, node or label
            self.detail = (
                "Working locally" if self.node == "validator" else "Waiting for Claude to respond"
            )
            self.last_activity = time.monotonic()
            self.history.append(
                dict(
                    node=self.node, label=label, elapsed=round(self.last_activity - self.started, 1)
                )
            )

    def activity(self, detail):
        with self.lock:
            self.detail = detail
            self.last_activity = time.monotonic()

    def snapshot(self):
        with self.lock:
            return dict(
                stage=self.label,
                node=self.node,
                detail=self.detail,
                elapsed=round(time.monotonic() - self.started),
                quiet_seconds=round(time.monotonic() - self.last_activity),
                history=deepcopy(self.history),
            )


class GraphRun:
    def __init__(self, initial, graph_factory):
        self.control = Control()
        self.lock = Lock()
        self.state = deepcopy(initial)
        self.status, self.error = "running", None
        self.thread = Thread(target=self._run, args=(graph_factory,), daemon=True)
        self.thread.start()

    def _run(self, graph_factory):
        try:
            initial = dict(self.state, control=self.control)
            for update in graph_factory().stream(initial, stream_mode="updates"):
                self.control.check()
                with self.lock:
                    for values in update.values():
                        self.state.update(
                            {k: deepcopy(v) for k, v in values.items() if k != "control"}
                        )
            self.control.check()
            with self.lock:
                self.status = "complete"
        except RunStopped as exc:
            with self.lock:
                self.status, self.error = "stopped", str(exc)
        except ValueError as exc:
            with self.lock:
                self.status, self.error = "failed", str(exc)
        except Exception:
            with self.lock:
                self.status, self.error = (
                    "failed",
                    "Unexpected worker failure. Your previously saved requirements are unchanged; no partial response was applied.",
                )

    def snapshot(self):
        with self.lock:
            stopped = self.control.cancelled.is_set()
            return dict(
                status="stopped" if stopped else self.status,
                error="Stopped by you. An already-sent AI request may still be billed."
                if stopped
                else self.error,
                state=deepcopy(self.state),
                **self.control.snapshot(),
            )

    def stop(self):
        self.control.cancelled.set()

    def alive(self):
        return self.thread.is_alive()


class DesignRun(GraphRun):
    def __init__(self, brief, graph_factory=None, feedback=None):
        from workflow import design_graph

        super().__init__(
            dict(brief=brief, attempt=0, usage=[], trace=[], attempts=[], feedback=feedback or {}),
            graph_factory or design_graph,
        )


class InterviewRun(GraphRun):
    def __init__(self, initial):
        from workflow import interview_graph

        super().__init__(initial, interview_graph)
