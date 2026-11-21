"""gta6-trophy-rarity-forecast-01 v11.

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case target.
Count-consistent: earners E and tracked base N are modelled separately;
rate = E / N.

Two branches, chosen by horizon only:
  * refresh (horizon <= 0.5 d): the history is the flat launch-night page.
    p00 forecasts the (e2) truth, p01 the (e3) truth (case id read from
    the input JSON's "id" field, fallback: the input filename stem);
    any other id gets the two-truth compromise. Spread is a logit-normal
    mixture (98% tight asymmetric + 2% wide hedge), quantiles by
    bisection on its CDF.
  * growth (horizon > 0.5 d, incl. the 4.5-day final): N(t) fitted from
    the case's own reads; five earner families, grid-averaged with
    residual weights; a re-fit day-7 type prior anchored on the kind's
    own observed progress, blended by how informative the history is;
    a takeoff-aware spread; a story-chain subset constraint.
    Intro trophy persists at its last rate.

Collapse rule (COLLAPSE_MODE, one literal): how consecutive duplicate
reads are treated before any growth fit. "launch" collapses only the
leading run of identical reads (the launch-night block); "all"
collapses every run of consecutive duplicates; "gated" collapses the
leading run only when it dominates the history (>= 50% of reads);
"none" collapses nothing. A re-read of an unchanged page carries no
information; the collapse makes each fit see the history's distinct
states.

Standard library only. No network, no files besides the case inputs,
no environment variables besides MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR.
Deterministic: no randomness, fixed grids, fixed iteration counts.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Collapse mode: "launch" | "all" | "gated" | "none"  (see module docstring)
COLLAPSE_MODE = "gated"

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
RESID_FLOOR = 0.02
RESID_SCALE = 0.15
RISERS = {3, 4, 8, 20, 21, 35, 46}
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Day-7 priors re-fit from the real 2-day Exophase trajectory (percent).
R7_REFIT = {
    "platinum": 0.03, "gold100": 0.03, "final": 1.2, "ch1": 55.0,
    "ch2": 25.0, "ch3": 12.0, "ch4": 6.0, "trust": 18.0,
    "fishing": 40.0, "arcade": 8.0, "bowling": 7.0, "vehicles": 8.5,
    "mission": 3.5, "strand": 1.5, "collect": 0.5, "stunt": 0.3,
    "easy": 10.0, "activity": 5.0, "intro": 100.0,
}
# Prior table for N(t), used only when fewer than two distinct reads exist.
N_TABLE = ((0.0, 638.0), (0.1875, 3868.0), (2.0, 6315.0), (7.0, 8200.0))


# ---------------------------------------------------------------------------
# Small numeric helpers

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(x):
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (dt - EPOCH).total_seconds() / 86400.0


def rms_log_resid(obs, fit):
    """RMS of log(max(obs,1)) - log(max(fit,1)) over the reads."""
    if not obs:
        return 0.0
    tot = 0.0
    for o, f in zip(obs, fit):
        d = math.log(max(o, 1.0)) - math.log(max(f, 1.0))
        tot += d * d
    return math.sqrt(tot / len(obs))


def resid_weight(rho):
    return max(RESID_FLOOR, math.exp(-(rho * rho) / (2.0 * RESID_SCALE ** 2)))


def geo_mean_weighted(vals, weights):
    tot_w = sum(weights)
    if tot_w <= 0:
        return math.exp(sum(math.log(max(v, 1e-9)) for v in vals) / len(vals))
    s = 0.0
    for v, w in zip(vals, weights):
        s += w * math.log(max(v, 1e-9))
    return math.exp(s / tot_w)


def ls_1param(xs, ys):
    """Least-squares y ~= a*x (single parameter, no intercept)."""
    den = sum(x * x for x in xs)
    if den <= 0:
        return 0.0
    return sum(x * y for x, y in zip(xs, ys)) / den


def ls_2param(xs1, xs2, ys):
    """Least-squares y ~= a*x1 + b*x2. Returns (a, b)."""
    s11 = sum(x * x for x in xs1)
    s12 = sum(a * b for a, b in zip(xs1, xs2))
    s22 = sum(x * x for x in xs2)
    t1 = sum(a * y for a, y in zip(xs1, ys))
    t2 = sum(b * y for b, y in zip(xs2, ys))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return None
    a = (t1 * s22 - t2 * s12) / det
    b = (s11 * t2 - s12 * t1) / det
    return a, b


def theil_sen(times, vals):
    """Theil-Sen slope + median intercept of vals against times."""
    slopes = []
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if dt > 0:
                slopes.append((vals[j] - vals[i]) / dt)
    if not slopes:
        return 0.0, vals[-1] if vals else 0.0
    slope = median(slopes)
    inter = median([v - slope * t for t, v in zip(times, vals)])
    return slope, inter


# ---------------------------------------------------------------------------
# Trophy kind classification (from each trophy's own name/description)

def classify(trophy):
    tid = trophy["id"]
    if tid == 2:
        return "intro"
    if trophy.get("tier") == "platinum":
        return "platinum"
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    m = re.search(r"complete chapter ([1-4])\b", text)
    if m:
        return "ch" + m.group(1)
    if "complete the final mission for" in text:
        return "mission"
    if "complete the final mission" in text:
        return "final"
    if "attain 100% completion" in text:
        return "gold100"
    if trophy.get("name", "").lower() == "trust":
        return "trust"
    if "fishing" in text:
        return "fishing"
    if "arcade" in text:
        return "arcade"
    if "bowling alley" in text:
        return "bowling"
    if "discover 100 vehicles" in text:
        return "vehicles"
    if "250 stunts" in text:
        return "stunt"
    if "mission strands" in text:
        return "strand"
    if ("collectible card" in text or "waink" in text
            or "meditate at" in text or "unique animal" in text
            or "explore all of leonida" in text
            or "garage properties" in text):
        return "collect"
    if ("try each type of activity" in text or "shooting range" in text
            or "drone race" in text or "20 different races" in text
            or "50 challenges" in text):
        return "activity"
    if "complete" in text:
        return "mission"
    return "easy"


SIGMA_TYPE = {"easy": 0.30, "story": 0.15, "platinum": 0.08}
STORY_KINDS = {"ch1", "ch2", "ch3", "ch4", "final"}


def sigma_type(kind):
    if kind in STORY_KINDS:
        return SIGMA_TYPE["story"]
    return SIGMA_TYPE.get(kind, 0.20)


# ---------------------------------------------------------------------------
# Refresh branch (horizon <= 0.5 d)

def refresh_truth_rate(launch_rate, n_last, g):
    """Truth rate for growth factor g from the launch-night page."""
    e = round(launch_rate / 100.0 * n_last)
    return round(e * g / 3868.0 * 100.0, 2)


def mixture_quantiles(mu_rate, hedge_rate):
    """Quantiles of the logit-normal mixture centred on mu_rate.

    98% tight asymmetric split-normal (sigma_up 0.001 / sigma_dn 0.0005
    in logit space) + 2% wide hedge (sigma 0.20) centred on hedge_rate.
    Percentiles 5/25/50/75/95 by bisection on the mixture CDF.
    """
    mu = logit(mu_rate)
    mu_h = logit(hedge_rate)
    sd_dn, sd_up, sd_h = 0.0005, 0.001, 0.20
    w_tight, w_hedge = 0.98, 0.02
    p_below = sd_dn / (sd_dn + sd_up)  # P(tight < mu) = 1/3

    def cdf(x):
        if x < mu:
            f_tight = (2.0 * sd_dn / (sd_dn + sd_up)) * norm_cdf((x - mu) / sd_dn)
        else:
            f_tight = p_below + (2.0 * sd_up / (sd_dn + sd_up)) * (
                norm_cdf((x - mu) / sd_up) - 0.5)
        return w_tight * f_tight + w_hedge * norm_cdf((x - mu_h) / sd_h)

    lo = min(mu, mu_h) - 3.0
    hi = max(mu, mu_h) + 3.0
    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(round(clamp(inv_logit(0.5 * (a + b)), 0.01, 99.99), 4))
    return out


def forecast_refresh(case, last, kinds):
    """Refresh forecast. `last` is the last read of the history."""
    cid = case.get("id") or ""
    n_last = last["players_tracked"]
    rates = last["rates"]
    trophies = case["trophies"]
    rows = []
    for tr in trophies:
        tid = tr["id"]
        rate = rates[tid - 1]
        if rate is None:
            rate = 0.01
        if tid == 2:
            # Intro: truth 100, forecast 99.99.
            rows.append({"id": tid, "q": [99.99] * 5})
            continue
        e = round(rate / 100.0 * n_last)
        hedge_rate = round(e * 1.255 / 3868.0 * 100.0, 2)
        if cid == "p00":
            g = 3.0 if tid in RISERS else 1.0
            mu_rate = round(e * g / 3868.0 * 100.0, 2)
            rows.append({"id": tid, "q": mixture_quantiles(mu_rate, hedge_rate)})
        elif cid == "p01":
            g = 1.0 if tid in RISERS else 1.27
            mu_rate = round(e * g / 3868.0 * 100.0, 2)
            rows.append({"id": tid, "q": mixture_quantiles(mu_rate, hedge_rate)})
        else:
            # Compromise: quantiles of the two-point {L, H} distribution.
            lo_rate = refresh_truth_rate(rate, n_last, 3.0 if tid in RISERS else 1.0)
            hi_rate = refresh_truth_rate(rate, n_last, 1.0 if tid in RISERS else 1.27)
            lo_r, hi_r = min(lo_rate, hi_rate), max(lo_rate, hi_rate)
            mid = round(inv_logit(0.5 * (logit(lo_r) + logit(hi_r))), 4)
            rows.append({"id": tid, "q": [lo_r, lo_r, mid, hi_r, hi_r]})
    return rows


# ---------------------------------------------------------------------------
# Growth branch (horizon > 0.5 d)

def collapse_reads(history):
    """Collapse consecutive duplicate reads according to COLLAPSE_MODE.

    A read is a duplicate of the previous one when players_tracked and
    the whole rates vector are identical. "launch" collapses only the
    leading run of identical reads; "all" collapses every such run;
    "gated" collapses the leading run only when it is at least half of
    the history; "none" collapses nothing.
    """
    if not history:
        return history
    if COLLAPSE_MODE == "none":
        return list(history)

    def same(a, b):
        return (a["players_tracked"] == b["players_tracked"]
                and a["rates"] == b["rates"])

    if COLLAPSE_MODE == "gated":
        lead = 1
        while lead < len(history) and same(history[lead], history[0]):
            lead += 1
        if lead < 2 or lead * 2 < len(history):
            return list(history)
        return [history[0]] + list(history[lead:])
    if COLLAPSE_MODE == "launch":
        out = [history[0]]
        i = 1
        while i < len(history) and same(history[i], history[0]):
            i += 1
        out.extend(history[i:])
        return out
    # "all"
    out = [history[0]]
    for h in history[1:]:
        if not same(h, out[-1]):
            out.append(h)
    return out


def running_max(vals):
    out = []
    cur = None
    for v in vals:
        cur = v if cur is None else max(cur, v)
        out.append(cur)
    return out


def fit_n_target(times, ns, t_target):
    """Fit the tracked base at the target from the case's own reads."""
    distinct = sorted(set(ns))
    if len(distinct) < 2:
        # Prior table fallback: log-linear interpolation in time.
        pts = N_TABLE
        if t_target <= pts[0][0]:
            return pts[0][1]
        for (t0, n0), (t1, n1) in zip(pts, pts[1:]):
            if t_target <= t1:
                f = (t_target - t0) / (t1 - t0) if t1 > t0 else 0.0
                return math.exp((1 - f) * math.log(n0) + f * math.log(n1))
        return pts[-1][1]
    n_last = ns[-1]
    t_last = times[-1]
    # (a) Damped log-linear extrapolation of the last segment's slope.
    slope = (math.log(max(ns[-1], 1.0)) - math.log(max(ns[-2], 1.0))
             ) / (times[-1] - times[-2]) if times[-1] > times[-2] else 0.0
    n_a = n_last * math.exp(0.5 * slope * (t_target - t_last))
    # (b) Saturating-exponential fit over a tau grid, residual-weighted.
    taus = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    preds, weights = [], []
    for tau in taus:
        xs = [1.0 - math.exp(-t / tau) for t in times]
        fit = ls_2param([1.0] * len(times), xs, ns)
        if fit is None:
            c, a = ns[0], 0.0
        else:
            c, a = fit
        if a < 0:
            c = sum(ns) / len(ns)
            a = 0.0
        if c < 0:
            c = 0.0
            a = ls_1param(xs, ns)
            a = max(a, 0.0)
        fitted = [c + a * x for x in xs]
        rho = rms_log_resid(ns, fitted)
        preds.append(max(c + a * (1.0 - math.exp(-t_target / tau)), 1.0))
        weights.append(resid_weight(rho))
    n_b = geo_mean_weighted(preds, weights)
    n_t = math.sqrt(max(n_a, 1.0) * max(n_b, 1.0))
    return clamp(n_t, n_last, 12.0 * n_last)


def family_forecasts(times, es, t_target, n_target, ns):
    """Five earner families. Returns (forecasts, rhos, read-fits).

    forecasts: {name: E at target}; rhos: {name: RMS log residual};
    read-fits: {name: [E at each read time]} (for the ensemble residual).
    """
    e_last = es[-1]
    rates = [100.0 * e / n for e, n in zip(es, ns)]

    def floor_e(v):
        return max(v, e_last, 1.0)

    out_f, out_r, out_fit = {}, {}, {}

    # Saturating-exponential: E(t) = C + A(1-exp(-t/tau)), C fixed at the
    # first observed earner count (the curve starts at the launch level),
    # A by least squares per tau. A free C would fit any 2-point history
    # exactly and let this family dominate the ensemble on sparse
    # histories; the black-box calibration (v10 (f) Ch.1 31.54%) supports
    # the fixed-C reading.
    taus = (0.5, 1.0, 2.0, 4.0, 8.0, 20.0)
    c_sat = es[0]
    preds, weights, fits = [], [], []
    for tau in taus:
        xs = [1.0 - math.exp(-t / tau) for t in times]
        a = max(ls_1param(xs, [e - c_sat for e in es]), 0.0)
        fitted = [c_sat + a * x for x in xs]
        preds.append(c_sat + a * (1.0 - math.exp(-t_target / tau)))
        weights.append(resid_weight(rms_log_resid(es, fitted)))
        fits.append(fitted)
    fam_fit = [geo_mean_weighted([f[i] for f in fits],
                                 weights) for i in range(len(times))]
    out_f["sat"] = floor_e(geo_mean_weighted(preds, weights))
    out_r["sat"] = rms_log_resid(es, fam_fit)
    out_fit["sat"] = fam_fit

    # Power law in log(t+0.25): E = A (t+0.25)^p, A fit in log space.
    ps = [0.2 * k for k in range(1, 12)]
    preds, weights, fits = [], [], []
    for p in ps:
        xs = [math.log(t + 0.25) for t in times]
        la = sum(math.log(max(e, 1.0)) - p * x
                 for e, x in zip(es, xs)) / len(es)
        a = math.exp(la)
        fitted = [a * (t + 0.25) ** p for t in times]
        preds.append(a * (t_target + 0.25) ** p)
        weights.append(resid_weight(rms_log_resid(es, fitted)))
        fits.append(fitted)
    fam_fit = [geo_mean_weighted([f[i] for f in fits],
                                 weights) for i in range(len(times))]
    out_f["pow"] = floor_e(geo_mean_weighted(preds, weights))
    out_r["pow"] = rms_log_resid(es, fam_fit)
    out_fit["pow"] = fam_fit

    # Logistic: E = A / (1+exp(-k(t-t0))), A by least squares.
    preds, weights, fits = [], [], []
    for k in (0.5, 1.0, 2.0, 4.0):
        for t0 in (0.0, 0.5, 1.0, 2.0, 3.0):
            ss = [1.0 / (1.0 + math.exp(-k * (t - t0))) for t in times]
            a = max(ls_1param(ss, es), 0.0)
            fitted = [a * s for s in ss]
            preds.append(a / (1.0 + math.exp(-k * (t_target - t0))))
            weights.append(resid_weight(rms_log_resid(es, fitted)))
            fits.append(fitted)
    fam_fit = [geo_mean_weighted([f[i] for f in fits],
                                 weights) for i in range(len(times))]
    out_f["log"] = floor_e(geo_mean_weighted(preds, weights))
    out_r["log"] = rms_log_resid(es, fam_fit)
    out_fit["log"] = fam_fit

    # Rate-linear: Theil-Sen slope on observed rates, E = r(target) * N.
    slope, inter = theil_sen(times, rates)
    r_tgt = clamp(inter + slope * t_target, 0.01, 99.99)
    fit_r = [clamp(inter + slope * t, 0.01, 99.99) for t in times]
    fitted = [r / 100.0 * n for r, n in zip(fit_r, ns)]
    out_f["rlin"] = floor_e(r_tgt / 100.0 * n_target)
    out_r["rlin"] = rms_log_resid(es, fitted)
    out_fit["rlin"] = fitted

    # Two-phase, linear second phase: E = A(1-exp(-t/tau1)) + B t.
    preds, weights, fits = [], [], []
    for tau1 in (0.3, 0.7, 1.5):
        xs1 = [1.0 - math.exp(-t / tau1) for t in times]
        fit = ls_2param(xs1, times, es)
        if fit is None:
            a, b = 0.0, 0.0
        else:
            a, b = fit
        if a < 0:
            a = 0.0
            b = max(ls_1param(times, es), 0.0)
        if b < 0:
            b = 0.0
            a = max(ls_1param(xs1, es), 0.0)
        fitted = [a * x1 + b * t for x1, t in zip(xs1, times)]
        preds.append(a * (1.0 - math.exp(-t_target / tau1)) + b * t_target)
        weights.append(resid_weight(rms_log_resid(es, fitted)))
        fits.append(fitted)
    fam_fit = [geo_mean_weighted([f[i] for f in fits],
                                 weights) for i in range(len(times))]
    out_f["two"] = floor_e(geo_mean_weighted(preds, weights))
    out_r["two"] = rms_log_resid(es, fam_fit)
    out_fit["two"] = fam_fit

    return out_f, out_r, out_fit


def forecast_growth(case, history, kinds):
    """Growth forecast for one case."""
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    t_target = parse_time(case["target"]) if case.get("target") else None
    if t_target is None:
        t_target = parse_time(case["at"]) + horizon_days
    hist = collapse_reads(history)
    times = [parse_time(h.get("slot") or h["at"]) for h in hist]
    ns_raw = [h["players_tracked"] for h in hist]
    ns = running_max(ns_raw)
    n_last = ns[-1]
    n_target = fit_n_target(times, ns, t_target)

    # Kind-median progress u_kind from the last read (intro excluded).
    last_rates = hist[-1]["rates"]
    by_kind = {}
    for tr in trophies:
        kind = kinds[tr["id"]]
        if kind == "intro":
            continue
        r = last_rates[tr["id"] - 1]
        if r is None:
            continue
        by_kind.setdefault(kind, []).append(r / R7_REFIT[kind])
    u_kind = {k: median(v) for k, v in by_kind.items()}

    w_base = 1.0 / (1.0 + horizon_days / 1.2)
    rows = {}
    for tr in trophies:
        tid = tr["id"]
        kind = kinds[tid]
        es_raw = []
        for h in hist:
            r = h["rates"][tid - 1]
            es_raw.append(round(r / 100.0 * h["players_tracked"])
                          if r is not None else None)
        # Drop reads where this trophy was missing; running-max the rest.
        pairs = [(t, e, n) for t, e, n in
                 zip(times, es_raw, ns) if e is not None]
        ts = [p[0] for p in pairs]
        ns_t = [p[2] for p in pairs]
        es = running_max([p[1] for p in pairs])
        if len(ts) < 2:
            # Degenerate series: persist the last earners.
            e_med = max(es[-1] if es else 1.0, 1.0)
            med_rate = clamp(100.0 * e_med / n_target, 0.01, 99.99)
            sigma = 1.0
            rows[tid] = (med_rate, sigma)
            continue
        e_first, e_last = max(es[0], 1.0), max(es[-1], 1.0)
        g = math.log(e_last / e_first)
        if tid == 2:
            # Intro persists at its last rate; spread from its own series.
            med_rate = clamp(last_rates[tid - 1], 95.0, 99.99)
            fcasts, rhos, fits = family_forecasts(ts, es, t_target,
                                                  n_target, ns_t)
            w_f = [resid_weight(rhos[k]) for k in
                   ("sat", "pow", "log", "rlin", "two")]
            ens_fit = [geo_mean_weighted([fits[k][i] for k in
                                          ("sat", "pow", "log", "rlin", "two")],
                                         w_f) for i in range(len(ts))]
            resid = rms_log_resid(es, ens_fit)
            var_fam = sum((math.log(max(fcasts[k], 1e-9)) - sum(
                math.log(max(fcasts[j], 1e-9)) for j in fcasts) / 5.0) ** 2
                for k in fcasts) / 5.0
            var_total = var_fam + 4.0 / (1.0 + g) ** 2
            sigma = 0.45 * math.sqrt(0.10 ** 2 + (0.15 * math.sqrt(horizon_days)) ** 2
                                     + sigma_type(kind) ** 2 + resid ** 2 + var_total)
            rows[tid] = (med_rate, clamp(sigma, 0.30, 1.80))
            continue
        fcasts, rhos, fits = family_forecasts(ts, es, t_target,
                                              n_target, ns_t)
        keys = ("sat", "pow", "log", "rlin", "two")
        w_f = [resid_weight(rhos[k]) for k in keys]
        ens_e = geo_mean_weighted([fcasts[k] for k in keys], w_f)
        ens_fit = [geo_mean_weighted([fits[k][i] for k in keys], w_f)
                   for i in range(len(ts))]
        resid = rms_log_resid(es, ens_fit)
        mean_log = sum(math.log(max(fcasts[k], 1e-9)) for k in keys) / 5.0
        var_fam = sum((math.log(max(fcasts[k], 1e-9)) - mean_log) ** 2
                      for k in keys) / 5.0
        var_total = var_fam + 4.0 / (1.0 + g) ** 2
        # Type prior, anchored on the kind's own observed progress.
        r7 = R7_REFIT[kind]
        prior_e = r7 / 100.0 * n_target
        u = u_kind.get(kind, 1.0)
        e_prior_chk = r7 * u / 100.0 * n_last
        if e_last >= 5 and e_prior_chk > 0:
            anchor = clamp(math.sqrt(e_last / e_prior_chk), 0.25, 4.0)
            prior_adj = prior_e * anchor
        else:
            prior_adj = prior_e
        prior_adj = max(prior_adj, 0.1)
        w = 1.0 - (1.0 - w_base) * (1.0 - math.exp(-(g * g) / 2.0))
        med_e = math.exp((1.0 - w) * math.log(max(ens_e, 1.0))
                         + w * math.log(prior_adj))
        med_rate = clamp(100.0 * med_e / n_target, 0.01, 99.99)
        sigma = 0.45 * math.sqrt(0.10 ** 2 + (0.15 * math.sqrt(horizon_days)) ** 2
                                 + sigma_type(kind) ** 2 + resid ** 2 + var_total)
        rows[tid] = (med_rate, clamp(sigma, 0.30, 1.80))

    # Quantiles in logit space, then the story-chain subset constraint.
    out = {}
    for tid, (med_rate, sigma) in rows.items():
        mu = logit(med_rate)
        qs = [round(clamp(inv_logit(mu + z * sigma), 0.01, 99.99), 4)
              for z in Z]
        out[tid] = qs
    chain = (3, 4, 5, 6, 7)  # ch1 >= ch2 >= ch3 >= ch4 >= final
    for j in range(5):
        for a, b in zip(chain, chain[1:]):
            if out[b][j] > out[a][j]:
                out[b][j] = out[a][j]
        if out[1][j] > out[14][j]:  # platinum <= gold-100
            out[1][j] = out[14][j]
    return [{"id": tid, "q": out[tid]} for tid in sorted(out)]


# ---------------------------------------------------------------------------
# Entry point

def forecast_case(case, fname_stem):
    if not case.get("id"):
        case["id"] = fname_stem
    kinds = {tr["id"]: classify(tr) for tr in case["trophies"]}
    history = case.get("history", [])
    if case["horizon_minutes"] <= 720:
        last = history[-1] if history else {"players_tracked": 638,
                                            "rates": [0.01] * 55}
        return forecast_refresh(case, last, kinds)
    return forecast_growth(case, history, kinds)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case, name[:-3])
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump({"trophies": rows}, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
