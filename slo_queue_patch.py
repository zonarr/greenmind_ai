"""GreenMind: SLO-first waiting queue for vLLM 0.32.0.

What it changes
  vLLM normally serves waiting requests oldest first (FCFS). When the Final's
  peak overloads the GPU, the oldest waiting requests have already missed their
  TTFT deadline, so FCFS spends the whole peak on requests that can no longer
  count, and every new request waits behind them and is late too.

  This queue picks the next request to start in this order:
    1. requests that already started (preempted or resumed), oldest first
    2. new requests that can still meet their TTFT deadline, newest first
    3. new requests already past their deadline, oldest first
  Every request is still answered in full (accuracy is unchanged); the late
  ones are simply finished when the load drops instead of during the peak.

  It is only active when the server starts with GREENMIND_SLO_QUEUE=1, so the
  patched vLLM behaves exactly like stock vLLM otherwise.

Usage (inside the venv):
  python slo_queue_patch.py          # install (keeps a .orig backup)
  python slo_queue_patch.py --undo   # restore the original file
"""

import importlib.util
import pathlib
import shutil
import sys

MARKER = "# --- GreenMind SLO-first queue ---"

ANCHOR = (
    "def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:\n"
    '    """Create request queue based on scheduling policy."""\n'
)

HOOK = (
    "    if (\n"
    "        policy == SchedulingPolicy.FCFS\n"
    '        and _gm_os.environ.get("GREENMIND_SLO_QUEUE") == "1"\n'
    "    ):\n"
    "        return SLOFirstRequestQueue()\n"
)

QUEUE_CLASS = MARKER + '''
import bisect as _gm_bisect
import os as _gm_os
import time as _gm_time

# TTFT SLO from the briefing: 0.5 s + 0.25 s per 1,000 prompt tokens.
_GM_TTFT_BASE_S = 0.5
_GM_TTFT_PER_TOKEN_S = 0.25 / 1000
# Rough time to prefill a prompt once started, so we do not pick a request
# that will miss anyway. Override with GREENMIND_PREFILL_TOK_S.
_GM_PREFILL_TOK_S = float(_gm_os.environ.get("GREENMIND_PREFILL_TOK_S", "8000"))
_GM_START_OVERHEAD_S = 0.05


def _gm_started(request: Request) -> bool:
    return (
        request.num_computed_tokens > 0
        or getattr(request, "num_preemptions", 0) > 0
        or getattr(request, "num_output_tokens", 0) > 0
    )


def _gm_can_still_make_ttft(request: Request, now: float) -> bool:
    n = request.num_prompt_tokens
    deadline = request.arrival_time + _GM_TTFT_BASE_S + _GM_TTFT_PER_TOKEN_S * n
    return now + _GM_START_OVERHEAD_S + n / _GM_PREFILL_TOK_S < deadline


class SLOFirstRequestQueue(RequestQueue):
    """Start the requests that can still be on time first (see module notes)."""

    def __init__(self) -> None:
        self._started: deque[Request] = deque()
        # New requests, kept sorted by arrival time (oldest at index 0).
        self._new: list[Request] = []
        # peek_request() and the following pop_request() must agree even if a
        # deadline passes in between, so pop returns what peek chose.
        self._peeked: Request | None = None

    def _insert_new(self, request: Request) -> None:
        _gm_bisect.insort(self._new, request, key=lambda r: r.arrival_time)

    def _select(self) -> Request:
        if self._started:
            return self._started[0]
        if not self._new:
            raise IndexError("peek from an empty queue")
        now = _gm_time.time()
        for request in reversed(self._new):
            if _gm_can_still_make_ttft(request, now):
                return request
        return self._new[0]

    def _contains(self, request: Request) -> bool:
        return any(r is request for r in self._started) or any(
            r is request for r in self._new
        )

    def _discard(self, request: Request) -> None:
        for i, r in enumerate(self._started):
            if r is request:
                del self._started[i]
                return
        for i, r in enumerate(self._new):
            if r is request:
                del self._new[i]
                return
        raise ValueError("request not in queue")

    def add_request(self, request: Request) -> None:
        if _gm_started(request):
            self._started.append(request)
        else:
            self._insert_new(request)

    def pop_request(self) -> Request:
        request = self._peeked
        self._peeked = None
        if request is None or not self._contains(request):
            request = self._select()
        self._discard(request)
        return request

    def peek_request(self) -> Request:
        request = self._select()
        self._peeked = request
        return request

    def prepend_request(self, request: Request) -> None:
        if _gm_started(request):
            self._started.appendleft(request)
        else:
            self._insert_new(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        # Same order as FCFSRequestQueue.prepend_requests (deque.extendleft).
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        self._discard(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = {id(r) for r in requests}
        self._started = deque(r for r in self._started if id(r) not in to_remove)
        self._new = [r for r in self._new if id(r) not in to_remove]

    def __bool__(self) -> bool:
        return bool(self._started) or bool(self._new)

    def __len__(self) -> int:
        return len(self._started) + len(self._new)

    def __iter__(self) -> Iterator[Request]:
        yield from list(self._started)
        yield from list(self._new)


if _gm_os.environ.get("GREENMIND_SLO_QUEUE") == "1":
    print("GreenMind SLO-first waiting queue: ENABLED", flush=True)
'''


def target_file() -> pathlib.Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("vllm not found: run `source ~/venv/bin/activate` first")
    root = pathlib.Path(list(spec.submodule_search_locations)[0])
    return root / "v1" / "core" / "sched" / "request_queue.py"


def main() -> None:
    path = target_file()
    backup = path.with_name(path.name + ".orig")

    if "--undo" in sys.argv:
        if not backup.exists():
            sys.exit(f"no backup at {backup}, nothing to undo")
        shutil.copy2(backup, path)
        print(f"restored original {path}")
        return

    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return
    if ANCHOR not in text:
        sys.exit(
            "this vLLM's request_queue.py is not the 0.32.0 one I expected; "
            "not patching"
        )

    if not backup.exists():
        shutil.copy2(path, backup)
    patched = text.replace(ANCHOR, ANCHOR + HOOK) + "\n\n" + QUEUE_CLASS
    compile(patched, str(path), "exec")  # refuse to write a broken file
    path.write_text(patched)
    print(f"patched {path}\nbackup  {backup}")


if __name__ == "__main__":
    main()
