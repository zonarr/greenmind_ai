"""GreenMind: local stand-in for the Final test.

Sends 300 synthetic requests to your vLLM server in the same SHAPE as the
Final (from the briefing): open-loop ramp 0 -> 4 req/s, a 4 req/s peak, then a
1 req/s quiet phase; the same class mix; thinking on, max_tokens 8192,
temperature 0, streaming; multi-turn tool-calling episodes. It then applies the
same SLO rules and scoring formula and prints an estimated score.

The prompts are made up here (no hidden test items). Treat the estimated score
as a way to COMPARE configs on your own VM, not as a prediction of the real
Final. Never run it while a graded submission is running.

Usage (in the venv, server already running):
  python final_sim.py                       # full run, ~4 min
  python final_sim.py --scale 0.3           # quick smoke test (90 requests)
  python final_sim.py --label slo64 --out ~/greenmind/sim
"""

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time

import httpx

# ---------------------------------------------------------------- constants

CLASS_MIX = {  # Final track: 300 requests by class (briefing slide 4)
    "chat": 90,
    "rag": 60,
    "agentic": 60,
    "reasoning": 45,
    "thai": 30,
    "long": 15,
}
MAX_TOKENS = 8192
TIMEOUT_S = 900
TPOT_SLO_S = 0.050
# Baseline efficiency, back-solved from your own Finals 4, 5 and 11
# (on-time requests / (J per request x 300) / H all give ~1.63e-3 per joule).
ETA_BASELINE = 1.63e-3

WORDS = (
    "river market council engine harvest station village report museum bridge "
    "factory teacher farmer festival railway library budget harbour province "
    "temple garden survey network climate highway hospital orchard archive "
    "district treaty canal mountain island delegation reservoir monastery"
).split()
VERBS = (
    "reported built expanded closed restored funded studied measured opened "
    "renamed visited recorded approved surveyed moved replaced"
).split()
NAMES = (
    "Arin Bodin Chanok Dara Ekk Fahsai Krit Lalita Mali Nawin Ploy Rattana "
    "Somchai Tida Udom Wichai"
).split()
CITIES = (
    "Bangkok ChiangMai Phuket KhonKaen HatYai Udon Korat Pattaya Hanoi "
    "Singapore Tokyo Seoul Taipei Manila Jakarta Sydney"
).split()

THAI_PROMPTS = [
    "ช่วยอธิบายว่าทำไมท้องฟ้าถึงเป็นสีฟ้า ให้เด็กประถมเข้าใจ",
    "สรุปข้อดีและข้อเสียของการทำงานจากบ้าน เป็นข้อๆ",
    "แต่งกลอนสั้นๆ สี่บรรทัดเกี่ยวกับฤดูฝนในกรุงเทพ",
    "ให้คำแนะนำการออมเงินสำหรับนักศึกษามหาวิทยาลัย 5 ข้อ",
    "อธิบายความแตกต่างระหว่างพลังงานแสงอาทิตย์กับพลังงานลมอย่างง่าย",
    "เขียนอีเมลขอลาป่วยถึงหัวหน้าอย่างสุภาพ",
    "ช่วยวางแผนเที่ยวเชียงใหม่ 3 วัน 2 คืน งบประมาณไม่เกิน 5,000 บาท",
    "อธิบายหลักการทำงานของปัญญาประดิษฐ์แบบภาษาใหญ่ให้คนทั่วไปเข้าใจ",
    "ถ้ามีเงิน 1,200 บาท ซื้อของ 3 ชิ้นราคา 250, 375 และ 410 บาท จะเหลือเงินเท่าไร",
    "แปลประโยคนี้เป็นภาษาอังกฤษ: การประหยัดพลังงานเริ่มต้นที่ตัวเรา",
]

CHAT_TASKS = [
    "Rewrite this note as a short, polite email to a customer.",
    "Summarize the note in three bullet points.",
    "List the action items in this note and who owns each one.",
    "Translate the main point of this note into simple English for a child.",
    "Suggest a catchy title for this note and explain your choice briefly.",
    "Point out anything unclear in this note and suggest fixes.",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Find the cheapest flight between two cities on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]

# ---------------------------------------------------------------- prompts


def filler(rng: random.Random, approx_tokens: int) -> str:
    """Varied made-up prose, roughly approx_tokens long (~4 chars/token)."""
    out, n_chars = [], 0
    while n_chars < approx_tokens * 4:
        s = (
            f"In {rng.randint(1850, 2025)}, the {rng.choice(WORDS)} of "
            f"{rng.choice(CITIES)} {rng.choice(VERBS)} a {rng.choice(WORDS)} "
            f"near the old {rng.choice(WORDS)}, according to {rng.choice(NAMES)}, "
            f"who counted {rng.randint(2, 9999)} {rng.choice(WORDS)}s."
        )
        out.append(s)
        n_chars += len(s) + 1
    return " ".join(out)


def make_messages(cls: str, rng: random.Random) -> tuple[list, int]:
    """Return (messages, number of turns) for one request."""
    if cls == "chat":
        note = filler(rng, rng.randint(80, 350))
        return [{"role": "user", "content": f"Note:\n{note}\n\n{rng.choice(CHAT_TASKS)}"}], 1
    if cls == "thai":
        return [{"role": "user", "content": rng.choice(THAI_PROMPTS)}], 1
    if cls == "reasoning":
        a, b, c = rng.randint(12, 99), rng.randint(3, 40), rng.randint(2, 9)
        task = rng.choice(
            [
                f"A tank holds {a * 10} litres. It drains {b} litres per hour and is "
                f"refilled by {c * 3} litres every {c} hours. After how many hours is it "
                f"empty? Show your reasoning.",
                f"Find all integers x with 1 <= x <= {a * 7} such that x^2 leaves "
                f"remainder {c} when divided by {b + 11}.",
                f"Write a Python function that returns the {a}th prime number and "
                f"explain its time complexity.",
                f"Prove that the sum of the first {a} odd numbers is a perfect square, "
                f"then give the value.",
            ]
        )
        return [{"role": "user", "content": task}], 1
    if cls in ("rag", "long"):
        tokens = rng.randint(1500, 6000) if cls == "rag" else rng.randint(8000, 28000)
        doc = filler(rng, tokens)
        name = rng.choice(NAMES)
        fact = f"{name} recorded exactly {rng.randint(100, 999)} visitors at the {rng.choice(WORDS)}."
        cut = rng.randint(0, len(doc))
        doc = doc[:cut] + " " + fact + " " + doc[cut:]
        q = f"According to the document, how many visitors did {name} record? Answer briefly."
        return [{"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {q}"}], 1
    if cls == "agentic":
        turns = rng.randint(2, 12)
        cities = rng.sample(CITIES, k=min(turns - 1, len(CITIES)))
        task = (
            "I am planning a trip. Use the tools, one call per step. First check the "
            f"weather in each of these cities, one at a time: {', '.join(cities)}. "
            "Then recommend the best city and estimate a flight price from Bangkok "
            "in Thai baht."
        )
        system = (
            "You are a travel assistant. Always use the tools for facts; never "
            "guess weather or prices."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ], turns
    raise ValueError(cls)


def tool_result(rng: random.Random, name: str, args: str) -> str:
    try:
        a = json.loads(args or "{}")
    except json.JSONDecodeError:
        a = {}
    if name == "get_weather":
        return json.dumps({"city": a.get("city", "?"), "temp_c": rng.randint(18, 36),
                           "condition": rng.choice(["sunny", "rain", "cloudy", "storm"])})
    if name == "search_flights":
        return json.dumps({"price_usd": rng.randint(60, 900), "airline": rng.choice(NAMES) + " Air"})
    if name == "convert_currency":
        amt = float(a.get("amount", 100) or 100)
        return json.dumps({"result": round(amt * 35.7, 2)})
    return json.dumps({"error": "unknown tool"})


# ---------------------------------------------------------------- one call


async def one_call(client, url, model, messages, use_tools, budget):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if budget > 0:
        body["thinking_token_budget"] = budget
    if use_tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"

    rec = {"t_send": time.time(), "ttft": None, "error": None, "finish": None,
           "prompt_tokens": 0, "completion_tokens": 0}
    content, tool_calls = [], {}
    t_first = t_last = None
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                rec["error"] = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return rec, content, tool_calls
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    rec["prompt_tokens"] = chunk["usage"].get("prompt_tokens", 0)
                    rec["completion_tokens"] = chunk["usage"].get("completion_tokens", 0)
                for ch in chunk.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    got = False
                    for key in ("content", "reasoning_content", "reasoning"):
                        if d.get(key):
                            got = True
                            if key == "content":
                                content.append(d[key])
                    for tc in d.get("tool_calls") or []:
                        got = True
                        slot = tool_calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                    if got:
                        now = time.time()
                        t_first = t_first or now
                        t_last = now
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
    except Exception as e:  # timeouts, disconnects
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    if t_first:
        rec["ttft"] = t_first - rec["t_send"]
        n = rec["completion_tokens"]
        rec["tpot"] = (t_last - t_first) / (n - 1) if n > 1 else None
    return rec, content, tool_calls


def judge(rec) -> str | None:
    """Return why a call missed the SLO, or None if on time."""
    if rec["error"]:
        return "error"
    if rec["ttft"] is None:
        return "no stream"
    if rec["finish"] == "length" or rec["completion_tokens"] >= MAX_TOKENS:
        return "cut off"
    allow = 0.5 + 0.25 * rec["prompt_tokens"] / 1000
    rec["ttft_ratio"] = rec["ttft"] / allow
    if rec["ttft"] > allow:
        return "ttft"
    if rec.get("tpot") is not None and rec["tpot"] > TPOT_SLO_S:
        return "tpot"
    return None


# ---------------------------------------------------------------- one request


async def one_request(i, cls, t_arrive, phase, args, client, model, t0, results):
    rng = random.Random(args.seed * 100003 + i)
    messages, turns = make_messages(cls, rng)
    await asyncio.sleep(max(0.0, t0 + t_arrive - time.time()))
    calls = []
    for turn in range(turns):
        rec, content, tool_calls = await one_call(
            client, args.url + "/chat/completions", model, messages,
            use_tools=(cls == "agentic"), budget=args.thinking_budget,
        )
        rec["miss"] = judge(rec)
        calls.append(rec)
        if cls != "agentic" or not tool_calls or turn == turns - 1 or rec["error"]:
            break
        msg = {"role": "assistant", "content": "".join(content) or None, "tool_calls": []}
        for k in sorted(tool_calls):
            tc = tool_calls[k]
            tc_id = tc["id"] or f"call_{i}_{turn}_{k}"
            msg["tool_calls"].append({"id": tc_id, "type": "function",
                                      "function": {"name": tc["name"], "arguments": tc["args"] or "{}"}})
        messages.append(msg)
        for tc in msg["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": tool_result(rng, tc["function"]["name"], tc["function"]["arguments"])})
    misses = [c["miss"] for c in calls if c["miss"]]
    results.append({"i": i, "cls": cls, "phase": phase, "arrive": t_arrive,
                    "on_time": not misses, "miss": misses[0] if misses else None,
                    "calls": calls})
    done = len(results)
    if done % 25 == 0:
        print(f"  {done} requests finished ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- schedule


def schedule(args):
    """Open-loop arrival times: ramp 0->peak rate, peak, quiet."""
    times = []
    n_ramp = round(args.ramp * args.peak_rate / 2)
    for k in range(1, n_ramp + 1):  # N(t) = rate*t^2/(2*ramp)
        times.append((math.sqrt(2 * args.ramp * k / args.peak_rate), "ramp"))
    n_peak = round(args.peak * args.peak_rate)
    for k in range(n_peak):
        times.append((args.ramp + k / args.peak_rate, "peak"))
    n_quiet = round(args.quiet * args.quiet_rate)
    for k in range(n_quiet):
        times.append((args.ramp + args.peak + k / args.quiet_rate, "quiet"))
    return times


# ---------------------------------------------------------------- energy


class Energy:
    def __init__(self):
        self.h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"(no GPU energy reading: {e})")

    def joules(self):
        if self.h is None:
            return None
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0


# ---------------------------------------------------------------- main


def p95(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="default: first model the server lists")
    ap.add_argument("--ramp", type=float, default=50, help="ramp seconds (0 -> peak rate)")
    ap.add_argument("--peak", type=float, default=35, help="peak seconds")
    ap.add_argument("--quiet", type=float, default=60, help="quiet seconds")
    ap.add_argument("--peak-rate", type=float, default=4.0)
    ap.add_argument("--quiet-rate", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink all phases, e.g. 0.3 for a smoke test")
    ap.add_argument("--thinking-budget", type=int, default=512,
                    help="thinking_token_budget sent with each call (0 = don't send)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=os.path.expanduser("~/greenmind/sim"))
    args = ap.parse_args()
    args.ramp *= args.scale
    args.peak *= args.scale
    args.quiet *= args.scale

    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    async with httpx.AsyncClient(timeout=TIMEOUT_S, limits=limits) as client:
        model = args.model or (await client.get(args.url + "/models")).json()["data"][0]["id"]
        slots = schedule(args)
        rng = random.Random(args.seed)
        n = len(slots)
        mix = []
        for cls, k in CLASS_MIX.items():
            mix += [cls] * max(1, round(k * n / 300))
        rng.shuffle(mix)
        mix = (mix * 2)[:n]
        print(f"{n} requests to {model}: ramp {args.ramp:.0f}s, peak {args.peak:.0f}s, "
              f"quiet {args.quiet:.0f}s, thinking budget {args.thinking_budget}")

        energy = Energy()
        e0, t0 = energy.joules(), time.time() + 1.0
        results = []
        await asyncio.gather(*[
            one_request(i, mix[i], t, phase, args, client, model, t0, results)
            for i, (t, phase) in enumerate(slots)
        ])
        t_end, e1 = time.time(), energy.joules()

    # ---------------------------------------------------------- report
    calls = [c for r in results for c in r["calls"]]
    on_time = sum(r["on_time"] for r in results)
    peak = [r for r in results if r["phase"] == "peak"]
    s_t = sum(r["on_time"] for r in peak) / len(peak) if peak else 0.0
    p95_ttft_ratio = p95([c.get("ttft_ratio") for c in calls])
    p95_tpot = p95([c.get("tpot") for c in calls])
    sl_ttft = min(1.0, 1 / p95_ttft_ratio) if p95_ttft_ratio else 0.0
    sl_tpot = min(1.0, TPOT_SLO_S / p95_tpot) if p95_tpot else 0.0
    s_l = math.sqrt(sl_ttft * sl_tpot)

    print(f"\n=== {args.label}: {on_time}/{n} on time, {len(calls)} calls, "
          f"{t_end - t0:.0f} s wall ===")
    for phase in ("ramp", "peak", "quiet"):
        rs = [r for r in results if r["phase"] == phase]
        if rs:
            print(f"  {phase:5s}: {sum(r['on_time'] for r in rs):3d}/{len(rs)} on time")
    print("  by class: " + ", ".join(
        f"{c} {sum(r['on_time'] for r in results if r['cls'] == c)}/"
        f"{sum(1 for r in results if r['cls'] == c)}" for c in CLASS_MIX))
    reasons = {}
    for r in results:
        if r["miss"]:
            reasons[r["miss"]] = reasons.get(r["miss"], 0) + 1
    print(f"  missed because: {reasons or 'nothing'}")
    print(f"  p95 TTFT {p95([c['ttft'] for c in calls]) or 0:.2f} s "
          f"({p95_ttft_ratio or 0:.1f}x its SLO), p95 TPOT {(p95_tpot or 0) * 1000:.0f} ms")
    print(f"  avg prompt {statistics.mean(c['prompt_tokens'] for c in calls):.0f} tokens, "
          f"avg output {statistics.mean(c['completion_tokens'] for c in calls):.0f} tokens per call "
          f"(real Final: ~1130 / ~375)")
    errors = [c["error"] for c in calls if c["error"]]
    if errors:
        print(f"  {len(errors)} errors, first: {errors[0]}")

    est = None
    if e0 is not None and e1 is not None:
        joules = e1 - e0
        eta = on_time / joules
        h = eta / ETA_BASELINE
        s_e = min(1.0, max(0.0, math.log(h) / math.log(3))) if h > 0 else 0.0
        est = 100 * (s_e ** 0.6) * (s_t ** 0.2) * (s_l ** 0.2)
        print(f"  energy {joules / 1000:.1f} kJ = {joules / n:.0f} J/request, H {h:.2f}x")
        print(f"  sE {s_e:.2f}  sT {s_t:.2f}  sL {s_l:.2f}  ->  estimated S {est:.1f} (assumes A = 1)")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{time.strftime('%Y%m%d-%H%M')}-{args.label}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "estimated_S": est, "on_time": on_time,
                   "results": results}, f, indent=1, ensure_ascii=False)
    print(f"  saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
