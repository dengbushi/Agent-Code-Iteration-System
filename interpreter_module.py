import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import humanize
import multiprocessing as mp
from dataclasses_json import DataClassJsonMixin


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """Result of executing a code snippet in the interpreter."""

    term_out: list[str]
    exec_time: float
    exc_type: Optional[str]
    exc_info: Optional[dict] = None
    exc_stack: Optional[list[tuple]] = None


def exception_summary(e, exec_file_name):
    """Generates a string that summarizes an exception and its stack trace"""
    tb_lines = traceback.format_exception(e)
    tb_str = "".join(tb_lines)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, out_queue, timeout=5):
        self.queue = out_queue
        self.timeout = timeout

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            logging.warning("Queue write timed out")

    def flush(self):
        pass


class Interpreter:
    def __init__(self, timeout: int = 3600, agent_file_name: str = "runfile.py"):
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.process: Optional[mp.Process] = None

    def child_proc_setup(self, result_outq: mp.Queue) -> None:
        # Redirect both stdout and stderr to the provided result queue.
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(self, code_inq: mp.Queue, result_outq: mp.Queue, event_outq: mp.Queue) -> None:
        self.child_proc_setup(result_outq)
        global_scope: dict = {}
        while True:
            code = code_inq.get()
            with open(self.agent_file_name, "w", encoding="utf-8") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(e, self.agent_file_name)
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            try:
                os.remove(self.agent_file_name)
            except Exception:
                pass

            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        ctx = mp.get_context("spawn")
        self.code_inq, self.result_outq, self.event_outq = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.process = ctx.Process(target=self._run_session, args=(self.code_inq, self.result_outq, self.event_outq))
        self.process.daemon = True
        self.process.start()

    def cleanup_session(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.join(timeout=0.5)
            if self.process.exitcode is None:
                self.process.kill()
                self.process.join(timeout=0.5)
                if self.process.exitcode is None:
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception as e:
            print(f"Error during process cleanup: {e}")
        finally:
            if self.process is not None:
                self.process.close()
                self.process = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        if reset_session:
            if self.process is not None:
                self.cleanup_session()
            self.create_process()
        else:
            assert self.process is not None

        assert self.process.is_alive()
        self.code_inq.put(code)

        try:
            state = self.event_outq.get(timeout=30)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            while not self.result_outq.empty():
                break
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        child_in_overtime = False
        while True:
            try:
                state = self.event_outq.get(timeout=1)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                if not child_in_overtime and not self.process.is_alive():
                    raise RuntimeError("REPL child process died unexpectedly") from None
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    try:
                        self.process.terminate()
                    except Exception:
                        pass
                    child_in_overtime = True
                    if running_time > self.timeout + 5:
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                if time.time() - start_collect > 5:
                    break
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                continue
        if output and output[-1] == "<|EOF|>":
            output.pop()

        e_cls_name, exc_info, exc_stack = state[1:]
        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )

        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)


