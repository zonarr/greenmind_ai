"""GreenMind: SLO-first waiting queue for vLLM 0.32.0.

What it changes
  vLLM normally serves waiting requests oldest first (FCFS). When the Final's
  peak overloads the GPU, the oldest waiting requests have already missed their
  TTFT deadline, so FCFS spends the whole peak on requests that can no longer
  count, and every new request waits behind them and is late too.

  This queue picks the next request to start in this order:
    1. requests that already started (preempted or resumed), oldest first
    2. new requests that can still meet their TTFT deadline, shortest prompt
       first (ties: newest first). Under overload one 20K-token prompt costs
       as much GPU time as ~15 chat requests, so short ones go first and more
       requests make their deadline. Set GREENMIND_SLO_ORDER=newest for v1's
       plain newest-first order.
    3. new requests already past their deadline, oldest first
  Every request is still answered in full (accuracy is unchanged); the late
  ones are simply finished when the load drops instead of during the peak.

  It is only active when the server starts with GREENMIND_SLO_QUEUE=1, so the
  patched vLLM behaves exactly like stock vLLM otherwise.

  v2 also estimates the start time from how vLLM really prefills: a prompt is
  cut into chunks of at most GREENMIND_CHUNK tokens (match
  --long-prefill-token-threshold, default 768), one chunk per scheduler step
  of about GREENMIND_STEP_S seconds (default 0.16, from Final 12's log).

Usage (inside the venv):
  python slo_queue_patch.py          # install or upgrade (keeps a .orig backup)
  python slo_queue_patch.py --undo   # restore the original file
"""

import importlib.util
import pathlib
import shutil
import sys

MARKER_PREFIX = "# --- GreenMind SLO-first queue"
MARKER = MARKER_PREFIX + " v2 ---"

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
# Time to prefill a prompt once started: one chunk of at most _GM_CHUNK tokens
# per scheduler step (--long-prefill-token-threshold), each step ~_GM_STEP_S.
_GM_CHUNK = max(1, int(_gm_os.environ.get("GREENMIND_CHUNK", "768")))
_GM_STEP_S = float(_gm_os.environ.get("GREENMIND_STEP_S", "0.16"))
_GM_START_OVERHEAD_S = 0.05
_GM_ORDER = _gm_os.environ.get("GREENMIND_SLO_ORDER", "short")


def _gm_started(request: Request) -> bool:
    return (
        request.num_computed_tokens > 0
        or getattr(request, "num_preemptions", 0) > 0
        or getattr(request, "num_output_tokens", 0) > 0
    )


def _gm_can_still_make_ttft(request: Request, now: float) -> bool:
    n = request.num_prompt_tokens
    deadline = request.arrival_time + _GM_TTFT_BASE_S + _GM_TTFT_PER_TOKEN_S * n
    steps = -(-n // _GM_CHUNK)  # ceil
    return now + _GM_START_OVERHEAD_S + steps * _GM_STEP_S < deadline


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
        best = None
        for request in reversed(self._new):  # newest first
            if not _gm_can_still_make_ttft(request, now):
                continue
            if _GM_ORDER == "newest":
                return request
            if best is None or request.num_prompt_tokens < best.num_prompt_tokens:
                best = request
        return best if best is not None else self._new[0]

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
    print(
        f"GreenMind SLO-first waiting queue v2: ENABLED (order={_GM_ORDER}, "
        f"chunk={_GM_CHUNK}, step={_GM_STEP_S}s)",
        flush=True,
    )
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
        print(f"already patched (v2): {path}")
        return
    if MARKER_PREFIX in text:  # older version: start again from the original
        if not backup.exists():
            sys.exit(f"old patch found but no backup at {backup}; reinstall vllm")
        text = backup.read_text()
        print("upgrading the old patch to v2")
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
