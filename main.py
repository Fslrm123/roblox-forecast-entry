#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun program/v1).

Reads /input/history.ndjson (hourly-ish snapshots of the chart, oldest first)
and /input/task.json, writes /output/forecast.json: for each of the top 10
places, a probability for each game named being at that place, plus the
forecaster's top 10 in order.

Method: an empirical transition kernel, estimated from the history itself.
For every pair of snapshots, games are pooled by (time gap, rank band) and
their later rank recorded. A game now at rank r is forecast by the pooled
distribution of where games at a similar rank were after a similar time gap,
smoothed with a small Gaussian kernel over ranks, averaged over twelve
horizons spread uniformly across the scored window, then blended with a
persistence spike on its current rank (the current leader is stickier than
the average game at its rank). Each place keeps a share unnamed for games
the history cannot name (new entrants, deep climbers). Games that will age
out of the 28-day chart window (from the games/v1 `created` date, when the
history carries details) have their weight scaled by the fraction of the
window they can still be on the chart.

No network, no model calls, no randomness: two runs on the same history
write the same forecast byte for byte. Standard library only.
"""

import json
import math
import os
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (chosen on the practice history + rolling backtests) ---
U_BASE = 0.15          # share of each place left unnamed
SIGMA_SMOOTH = 1.5     # Gaussian kernel width (ranks) over pooled future ranks
BLEND_PERSIST = 0.4    # weight of the persistence spike on the current rank
CAND_MAX_RANK = 50     # candidates: last chart's ranks 1..50 + recent top 10s
MIN_POOL = 40          # minimum pooled transitions before widening the kernel
DT_BUCKET_HOURS = 6    # time-gap bucket width for the transition pools
N_HORIZONS = 12        # horizons sampled uniformly across the window
LEFT_RANK = 110.0      # expected-rank placeholder for a game off the chart


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rank_band(rank):
    if rank <= 3:
        return 0
    if rank <= 6:
        return 1
    if rank <= 10:
        return 2
    if rank <= 15:
        return 3
    if rank <= 20:
        return 4
    if rank <= 30:
        return 5
    if rank <= 50:
        return 6
    return 7


def build_pools(snaps):
    """pools[(dt_bucket, rank_band)] = list of later ranks (None = left chart)."""
    times = [parse_time(s["at"]) for s in snaps]
    maps = [{r["universe_id"]: r["rank"] for r in s["up_and_coming"]} for s in snaps]
    pools = {}
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt > 84:
                continue
            db = min(int(dt // DT_BUCKET_HOURS), 13)
            later = maps[j]
            for uid, r0 in maps[i].items():
                pools.setdefault((db, rank_band(r0)), []).append(later.get(uid))
    return pools


def pool_distribution(pools, dt_bucket, band, places):
    """Smoothed distribution over places 1..places for one (gap, band) pool.

    Widen the gap/band window until at least MIN_POOL transitions are pooled,
    then average a Gaussian kernel over the pooled later ranks. Games that
    left the chart contribute nothing (their mass leaves the named rows).
    """
    vals = []
    for dspan in (0, 1, 2, 3, 5):
        for bspan in (0, 1, 2):
            vals = []
            for d2 in range(max(0, dt_bucket - dspan), min(14, dt_bucket + dspan + 1)):
                for b2 in range(max(0, band - bspan), min(8, band + bspan + 1)):
                    vals.extend(pools.get((d2, b2), []))
            if len(vals) >= MIN_POOL:
                break
        if len(vals) >= MIN_POOL:
            break
    dist = [0.0] * (places + 1)
    kept = 0
    # Group pooled ranks first: the kernel is evaluated once per distinct
    # rank instead of once per pooled transition (same result, far less work
    # on the final history, where the hourly recorder makes pools large).
    counts = {}
    for future_rank in vals:
        if future_rank is None:
            continue
        kept += 1
        counts[future_rank] = counts.get(future_rank, 0) + 1
    for future_rank, n in counts.items():
        for k in range(1, places + 1):
            dk = (k - future_rank) / SIGMA_SMOOTH
            dist[k] += n * math.exp(-0.5 * dk * dk)
    if kept == 0:
        return dist
    return [d / kept for d in dist]


def age_factor(created, uid, t_end, window_hours):
    """Fraction of the window a game can still be on the 28-day chart."""
    born = created.get(uid)
    if born is None:
        return 1.0
    remain = (born + timedelta(days=28) - t_end).total_seconds() / 3600.0
    if remain <= 0:
        return 0.0
    if remain < window_hours:
        return max(0.05, remain / window_hours)
    return 1.0


def persistence_forecast(last_ids, places):
    """Fallback: the house baseline's shape (chart will not move)."""
    probs = []
    for place in range(1, places + 1):
        probs.append({"universe_id": last_ids[place - 1], "place": place, "p": 0.5})
        if place == 1:
            if len(last_ids) > 1:
                probs.append({"universe_id": last_ids[1], "place": place, "p": 0.3})
        else:
            probs.append({"universe_id": last_ids[place - 2], "place": place, "p": 0.15})
            if place < len(last_ids):
                probs.append({"universe_id": last_ids[place], "place": place, "p": 0.15})
    return {"probabilities": probs, "top10": last_ids[:places]}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    places = task.get("places", 10)
    window = task.get("window", {})
    window_hours = window.get("slots", 72) * window.get("slot_minutes", 60) / 60.0
    output_name = task.get("output", "forecast.json")

    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    if not snaps:
        raise SystemExit("empty history")

    last_rows = sorted(snaps[-1]["up_and_coming"], key=lambda r: r["rank"])
    last_ids = [r["universe_id"] for r in last_rows]
    last_map = {r["universe_id"]: r["rank"] for r in last_rows}

    if len(snaps) < 2 or not last_ids:
        fc = persistence_forecast(last_ids, places)
    else:
        pools = build_pools(snaps)
        # Precompute one smoothed distribution per (gap bucket, rank band).
        dists = {}
        for db in range(14):
            for band in range(8):
                dists[(db, band)] = pool_distribution(pools, db, band, places)

        candidates = sorted(
            {uid for uid, r in last_map.items() if r <= CAND_MAX_RANK}
            | {r["universe_id"] for s in snaps[-3:] for r in s["up_and_coming"][:10]
               if r["universe_id"] in last_map}
        )
        created = {}
        for s in snaps:
            for uid, det in (s.get("details") or {}).items():
                if det and det.get("created"):
                    created[int(uid)] = parse_time(det["created"])
        t_end = parse_time(snaps[-1]["at"])

        step = window_hours / N_HORIZONS
        horizons = [step * (i + 0.5) for i in range(N_HORIZONS)]
        weights = {u: [0.0] * (places + 1) for u in candidates}
        for h in horizons:
            db = min(int(h // DT_BUCKET_HOURS), 13)
            for u in candidates:
                af = age_factor(created, u, t_end, window_hours)
                if af == 0.0:
                    continue
                dist = dists[(db, rank_band(last_map[u]))]
                for k in range(1, places + 1):
                    weights[u][k] += af * dist[k] / N_HORIZONS

        if BLEND_PERSIST > 0:
            for u in candidates:
                r0 = last_map[u]
                if 1 <= r0 <= places:
                    weights[u][r0] = (1 - BLEND_PERSIST) * weights[u][r0] + BLEND_PERSIST

        probs = []
        for k in range(1, places + 1):
            total = sum(weights[u][k] for u in candidates)
            if total <= 0:
                continue
            for u in candidates:
                if weights[u][k] > 0:
                    probs.append({
                        "universe_id": u,
                        "place": k,
                        "p": round((1 - U_BASE) * weights[u][k] / total, 6),
                    })

        def expected_rank(u):
            inside = sum(weights[u][k] for k in range(1, places + 1))
            return (sum(k * weights[u][k] for k in range(1, places + 1))
                    + (1 - inside) * LEFT_RANK)

        top10 = sorted(candidates, key=lambda u: (expected_rank(u), last_map[u]))[:places]
        fc = {"probabilities": probs, "top10": top10}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(fc) + "\n")


if __name__ == "__main__":
    main()
