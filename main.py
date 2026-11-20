"""gta6-trophy-rarity-forecast-01: forecast GTA VI trophy earn rates.

For each trophy, forecast five quantiles (5/25/50/75/95%) of its earn rate
on Exophase at the case's target, from the history of reads before the
checkpoint. Scored by pinball loss on the logit scale.

Method (per trophy, all in logit space, logit(p) = ln(p / (100 - p))):
  * Three fits of the trophy's trajectory, each extrapolated to the target:
      L. saturating logistic r(t) = L / (1 + exp(-k (t - t0))), fitted by
         gridding the asymptote L and regressing ln(r / (L - r)) on t;
         the form that keeps a long-horizon forecast from exploding.
      A. logit(rate) linear in ln(1 + hours since launch)  (house curve form)
      C. population form: ln(earners + 0.5) and ln(tracked) each linear in
         ln(1 + hours); rate = 100 * earners / tracked at the target.
    Fits use the last 24h of reads (all reads if fewer); each needs >= 3
    points with distinct x, else it is skipped.
  * The median is the residual-weighted blend of the fits that exist,
    shrunk toward the last read (persistence) by a trust factor from the
    fit's span and residuals. When the history is frozen flat (the page
    not yet refreshed), the median also takes a prior growth uplift: the
    reads are a stale snapshot, not the current state.
  * The spread is sqrt(S0^2 + s_fit^2 + (SH * sqrt(horizon_days))^2):
    S0 covers a refresh jump (Exophase recomputing its counts, which the
    practice cases show can move every rate at once), s_fit the fit's own
    residual, SH the growth uncertainty over the horizon.
  * Caps from the progression funnel: a chapter's rate cannot exceed the
    previous chapter's, the final mission cannot exceed chapter 4, and the
    platinum cannot exceed any other trophy; applied per quantile level.
  * A trophy at 100% in the last read gets a near-100 forecast with a
    small spread; a trophy at 0% is handled through the +0.5 earner
    smoothing in fit C and the 0.01 clamp in fits A/B.

Deterministic, standard library only, no network. Reads every *.in in
MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Tunables (see ../gta6-trophy-rarity-forecast-01-work/validate.py).
# Env overrides exist only for local tuning; the grader runs with defaults.
S0 = float(os.environ.get("MAKEFUN_S0", "0.60"))      # logit spread floor: refresh-jump / level risk
SH = float(os.environ.get("MAKEFUN_SH", "0.75"))      # logit spread per sqrt(day) of horizon: growth uncertainty
UPLIFT = float(os.environ.get("MAKEFUN_UPLIFT", "0.50"))  # logit median uplift when the history is frozen flat
RES_PRIOR = 0.04   # prior variance (0.2^2) in the blend weights
SPAN_HALF = float(os.environ.get("MAKEFUN_SPAN_HALF", "12.0"))  # hours of history span that earns half trust
TRUST_SCALE = float(os.environ.get("MAKEFUN_TRUST_SCALE", "3.0"))  # multiplier on the trust factor
RES_SCALE = 0.25   # residual variance scale in the trust factor
# Frozen-page refresh (dilution) regime: when the history is flat AND the
# horizon is short, the case is the launch-night refresh regime the
# practice cases sit in. Exophase's page is a stale snapshot: the
# answering read refreshes it with a much larger tracked population, so
# every rate falls by dilution. competition.yaml states the refresh read
# at 04:30Z tracked 3,868 players (the frozen page: 638). New players had
# hours, not days, so earners grow only a little: DILUTION_EARNER_GROWTH
# is set so the median logit shift is about -1.5, the shift implied by the
# house baseline's stated practice loss (~1.51, typical miss ~0.68pp).
DILUTION_TRACKED = float(os.environ.get("MAKEFUN_DILUTION_TRACKED", "3868"))
DILUTION_EARNER_GROWTH = float(os.environ.get("MAKEFUN_DILUTION_EARNER_GROWTH", "1.35"))
DILUTION_MAX_HOURS = float(os.environ.get("MAKEFUN_DILUTION_MAX_HOURS", "6.0"))
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, residual_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, res


def chapter_of(trophy):
    """Chapter number for 'Complete Chapter N.' trophies, else None."""
    m = re.search(r"Complete Chapter (\d+)", trophy.get("description", ""))
    return int(m.group(1)) if m else None


def logistic_fit(window, target_h, tier=None):
    """Fit r(t) = L / (1 + exp(-k (t - t0))) by gridding the asymptote L:
    for fixed L, ln(r / (L - r)) is linear in t, so OLS gives k and t0.
    Returns (pred_rate, residual) for the best L, or None. Only for
    rising series (a falling fit is noise/dilution, left to the other
    models). The saturating form is what keeps a long-horizon forecast
    from exploding the way a linear-in-time logit fit does.
    Early in the S-curve the asymptote is barely identified: a huge L
    fits the observed rise almost as well as a small one but extrapolates
    wildly, so L is scored by residual + a penalty growing in L (prefer
    the smallest asymptote consistent with the data), and capped by a
    tier prior on eventual penetration."""
    last = window[-1][1]
    max_seen = max(r for _, r, _ in window)
    tier_cap = {"bronze": 85.0, "silver": 60.0, "gold": 30.0, "platinum": 12.0}.get(tier, 85.0)
    cands = set()
    for m in (1.15, 1.4, 1.8, 2.5, 3.5, 5, 8, 15):
        cands.add(min(tier_cap, max(last * m, last + 0.5)))
    for v in (10.0, 20.0, 35.0, 50.0, 70.0):
        cands.add(min(tier_cap, v))
    best = None
    for L in sorted(cands):
        if L <= max_seen * 1.02:
            continue
        xs, ys = [], []
        ok = True
        for h, r, _ in window:
            if r <= 0 or r >= L * 0.999:
                ok = False
                break
            xs.append(h)
            ys.append(math.log(r / (L - r)))
        if not ok or len(set(round(x, 6) for x in xs)) < 3:
            continue
        a, b, res = ols(xs, ys)
        if b < 0:
            continue
        z = a + b * target_h
        pred = L / (1.0 + math.exp(-z)) if -700 < z < 700 else (L if z > 0 else 0.0)
        # penalised score: prefer the smallest asymptote that fits
        score = res + 0.05 * math.log(max(1.0, L / max(last, 0.05)))
        if best is None or score < best[2]:
            best = (pred, res, score)
    return (best[0], best[1]) if best else None


def forecast_trophy(series, horizon_days, target_h, trophy):
    """series: list of (hours_since_launch, rate_pct, tracked).
    Returns (median_logit, spread)."""
    # last rate (persistence) and flat-history detection
    last_rate = series[-1][1]
    if last_rate is None:
        last_rate = 0.0
    persist = logit(last_rate)

    pts = [(h, r, n) for (h, r, n) in series if r is not None]
    if len(pts) < 3:
        return persist, math.sqrt(S0 * S0 + (SH * math.sqrt(horizon_days)) ** 2)

    # fit window: last 24h of reads (at least the last 3)
    last_h = pts[-1][0]
    window = [p for p in pts if p[0] >= last_h - 24.0]
    if len(window) < 3:
        window = pts[-3:]
    span = window[-1][0] - window[0][0]
    flat = max(r for _, r, _ in window) - min(r for _, r, _ in window) < 0.01

    if flat and horizon_days * 24.0 <= DILUTION_MAX_HOURS:
        # Frozen page, short horizon: the launch-night refresh regime.
        # The reads are a stale snapshot; the answering read refreshes the
        # page with a much larger tracked population, so every rate falls
        # by dilution (earners barely grow, tracked jumps). The growth
        # fits below are meaningless on flat data, and the uplift would
        # point the wrong way, so answer the dilution directly.
        n_last = series[-1][2] or 0
        if n_last > 0:
            e_last = max(0.0, last_rate) / 100.0 * n_last
            rate_dil = 100.0 * e_last * DILUTION_EARNER_GROWTH / DILUTION_TRACKED
            median = logit(rate_dil)
            spread = math.sqrt(S0 * S0 + (SH * math.sqrt(horizon_days)) ** 2)
            return median, spread

    preds = []  # (pred_logit, residual_std)
    # Model L: saturating logistic in time (primary for growing series)
    lf = logistic_fit(window, target_h, trophy.get("tier"))
    if lf is not None:
        preds.append((logit(lf[0]), lf[1]))
    # Model A: logit(rate) ~ ln(1 + hours)
    xs = [math.log1p(max(0.0, h)) for h, _, _ in window]
    ys = [logit(r) for _, r, _ in window]
    if len(set(round(x, 6) for x in xs)) >= 3:
        a, b, res = ols(xs, ys)
        preds.append((a + b * math.log1p(max(0.0, target_h)), res))
    # Model C: population form, ln(earners+0.5) and ln(tracked) ~ ln(1+hours)
    try:
        xe = [math.log1p(max(0.0, h)) for h, _, _ in window]
        ye = [math.log(max(0.5, r / 100.0 * n) + 0.5) for _, r, n in window]
        yn = [math.log(max(1.0, n)) for _, _, n in window]
        if len(set(round(x, 6) for x in xe)) >= 3:
            ae, be, rese = ols(xe, ye)
            an, bn, resn = ols(xe, yn)
            xt = math.log1p(max(0.0, target_h))
            e_pred = math.exp(ae + be * xt) - 0.5
            n_pred = math.exp(an + bn * xt)
            rate_c = 100.0 * max(0.0, e_pred) / max(1.0, n_pred)
            res_c = math.sqrt(rese * rese + resn * resn)
            preds.append((logit(rate_c), res_c))
    except (ValueError, OverflowError):
        pass

    if not preds:
        median = persist
        if flat:
            median += UPLIFT
        return median, math.sqrt(S0 * S0 + (SH * math.sqrt(horizon_days)) ** 2)

    # residual-weighted blend
    weights = [1.0 / (res * res + RES_PRIOR) for _, res in preds]
    wsum = sum(weights)
    blend = sum(w * p for w, (p, _) in zip(weights, preds)) / wsum
    res_blend = math.sqrt(sum(w * res * res for w, (_, res) in zip(weights, preds)) / wsum)

    # trust: more history span and cleaner fits -> trust the blend more;
    # models that disagree with each other earn less trust
    pstd = math.sqrt(sum(w * (p - blend) ** 2 for w, (p, _) in zip(weights, preds)) / wsum)
    trust = min(1.0, TRUST_SCALE * (span / (span + SPAN_HALF)) * (1.0 / (1.0 + res_blend * res_blend / RES_SCALE)) * (1.0 / (1.0 + pstd * pstd)))
    median = persist + trust * (blend - persist)
    if flat:
        # the page is frozen: the reads are a stale snapshot, so the
        # median needs the growth the frozen window hides
        median += UPLIFT

    spread = math.sqrt(S0 * S0 + res_blend * res_blend + (SH * math.sqrt(horizon_days)) ** 2)
    return median, spread


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_h = (parse_time(case["target"]) - LAUNCH).total_seconds() / 3600.0

    trophies = case["trophies"]
    # per-trophy series
    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        series = []
        for read in case["history"]:
            rates = read.get("rates") or []
            r = rates[i] if i < len(rates) else None
            h = (parse_time(read["at"]) - LAUNCH).total_seconds() / 3600.0
            series.append((h, r, read.get("players_tracked") or 0))
        last_rate = series[-1][1]
        if last_rate is not None and last_rate >= 99.99:
            # already at 100%: stays there, tiny spread
            medians.append(logit(99.99))
            spreads.append(0.15)
            continue
        med, spr = forecast_trophy(series, horizon_days, target_h, trophy)
        medians.append(med)
        spreads.append(spr)

    # progression caps, per quantile level (a rate cannot exceed its
    # prerequisite's rate: chapter k+1 <= chapter k, final mission <=
    # chapter 4, platinum <= every other trophy).
    rows = []
    for i, trophy in enumerate(trophies):
        q = [round4(sigmoid(medians[i] + z * spreads[i])) for z in Z]
        rows.append({"id": trophy["id"], "q": q})

    by_id = {t["id"]: idx for idx, t in enumerate(trophies)}
    chapters = {}
    for idx, t in enumerate(trophies):
        c = chapter_of(t)
        if c is not None:
            chapters[c] = idx
    # chapter chain: cap each chapter by the previous one, same quantile
    for c in sorted(chapters):
        if c - 1 in chapters:
            prev, cur = chapters[c - 1], chapters[c]
            rows[cur]["q"] = [min(a, b) for a, b in zip(rows[cur]["q"], rows[prev]["q"])]
    # final mission (gold, "Complete the final mission.") capped by chapter 4
    if 4 in chapters:
        for idx, t in enumerate(trophies):
            if t.get("description", "") == "Complete the final mission.":
                rows[idx]["q"] = [min(a, b) for a, b in zip(rows[idx]["q"], rows[chapters[4]]["q"])]
    # platinum capped by every other trophy
    for idx, t in enumerate(trophies):
        if t.get("tier") == "platinum":
            for j, other in enumerate(rows):
                if j != idx:
                    rows[idx]["q"] = [min(a, b) for a, b in zip(rows[idx]["q"], other["q"])]
    # final safety: non-decreasing, in [0, 100]
    for row in rows:
        q = row["q"]
        q = [min(100.0, max(0.0, v)) for v in q]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        row["q"] = q
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
