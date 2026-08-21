"""
bin_optimizer.py
================
Optimal histogram binning for EMCCD photon-counting flux retrieval.

Question answered
-----------------
Given a detector (bias, EM gain, read-out noise, saturation level) and a flux
range of interest (here mu < 8 e-/read), what is the *smallest* number of
histogram bins K -- ideally a power of two -- whose maximum-likelihood flux
estimate retains >95 % of the accuracy of the full, un-degraded 1-ADU
histogram?  And where should those K bin edges sit?

Short answer: K = 8, and the cuts go
    one (or a few) cuts just above the read-noise peak, at bias + c*sigma,
    then a ladder of rungs uniform in sqrt(u), u = (x-bias)/G, from u ~ 1 to
    u_top = min( (sat-bias)/G , mu_max + 3*sqrt(mu_max) ),
    then one open bin above u_top.
See ``design_bins()`` for the general-purpose entry point, and the
accompanying document for the derivation.

Method
------
1.  The pixel-value distribution is discrete: the detector digitises to 1 ADU.
    The "full histogram" is therefore the set of 1-ADU cells

        cell i = [i, i+1)   for i = 0 ... sat-1
        cell sat = [sat, inf)          (the saturation catch-all)

    Any K-bin histogram is a *merge* of contiguous runs of these cells, so by
    the data-processing inequality its Fisher information can never exceed
    that of the full histogram.  Efficiency is bounded by 1 by construction.

2.  Fisher information for the mean flux mu, per pixel per read:

        I(mu) = sum_bins  (dp_i/dmu)^2 / p_i

    The Cramer-Rao bound gives sigma_mu = 1 / sqrt(N * I), so the accuracy
    ratio between a K-bin histogram and the full one is

        sigma_full / sigma_K = sqrt( I_K(mu) / I_full(mu) ) = sqrt(eta(mu))

    with eta the *information efficiency*.

3.  Because I is a plain sum over bins, a weighted objective

        V = sum_mu  w_mu * I_K(mu) / I_full(mu)

    is additive over bins, so the best K-bin partition (over a dense
    candidate-edge set) is found exactly by dynamic programming.  A
    multiplicative-weights outer loop drives w_mu towards the max-min
    solution, i.e. the design that maximises the *worst-case* efficiency over
    the flux range.

4.  That numerical optimum is then matched by a simple, interpretable
    closed-form family (``physical_edges``), which recovers ~99.9 % of it.

Physics model
-------------
    n ~ Poisson(mu + cic)                       photo-electrons + CIC
    g ~ Gamma(n, G)   (g = 0 if n = 0)          EM register
    x = bias + g + N(0, ron)                    read-out
    x >= sat  ->  saturated, indistinguishable

References: Basden et al. 2003; Harpsoe et al. 2012; Daigle et al. 2009.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erf, gammaln

# ---------------------------------------------------------------------------
# Timestamped, colour-coded logging (house convention)
# ---------------------------------------------------------------------------
_C = {'info': '\033[92m', 'value': '\033[94m', 'skip': '\033[93m',
      'stop': '\033[91m', 'end': '\033[0m'}


def log(message, level='info'):
    """YYMMDD HH:MM:SS.SS | message, colour-coded by semantic role."""
    stamp = time.strftime('%y%m%d %H:%M:%S') + f'.{int((time.time() % 1) * 100):02d}'
    sys.stdout.write(f"{_C.get(level, '')}{stamp} | {message}{_C['end']}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Detector description
# ---------------------------------------------------------------------------
@dataclass
class Detector:
    """EMCCD signal-chain parameters, all in raw ADU unless noted."""
    bias: float = 1000.0        # bias level [ADU]
    gain: float = 5000.0        # mean EM gain [ADU/e-]
    ron: float = 30.0           # read-out noise, 1 sigma [ADU]
    sat_adu: float = 60000.0    # saturation threshold [ADU]
    cic: float = 0.001          # clock-induced charge [e-/read]
    nmax: int = 60              # electrons carried in the Poisson sum

    # --- the two dimensionless numbers that actually set the design ---
    @property
    def r(self):
        """Read noise in units of the EM gain, sigma / G."""
        return self.ron / self.gain

    @property
    def S(self):
        """Saturation head-room in units of the EM gain, (sat - bias) / G."""
        return (self.sat_adu - self.bias) / self.gain

    @classmethod
    def from_config(cls, config):
        """Build from the repository's emccd_config.yaml dictionary."""
        det = config['detector']
        hist = config.get('histogram', {})
        return cls(bias=float(det['bias']), gain=float(det['gain']),
                   ron=float(det['ron']),
                   sat_adu=float(hist.get('saturation_adu', det['full_well'])),
                   cic=float(det.get('cic', 0.0)),
                   nmax=max(60, int(det.get('nmax', 60))))


# ---------------------------------------------------------------------------
# Exact cell-level model  (1 ADU resolution = the "full histogram")
# ---------------------------------------------------------------------------
def _norm_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def per_electron_cell_probs(det: Detector):
    """
    P(pixel lands in 1-ADU cell i | exactly n electrons), for n = 0 ... nmax.

    Cells are  [i, i+1)  for i < sat  and  [sat, inf)  for i = sat, so the
    returned matrix rows sum to exactly 1 and the last column holds the
    saturation probability.

    Returns
    -------
    pn : (nmax+1, sat+1) float array
    """
    sat = int(round(det.sat_adu))
    ncell = sat + 1
    pn = np.zeros((det.nmax + 1, ncell), dtype=np.float64)

    # --- n = 0 : pure Gaussian around the bias, analytic ------------------
    edges = np.arange(0, sat + 1, dtype=np.float64)          # 0 ... sat
    cdf0 = _norm_cdf((edges - det.bias) / det.ron)
    pn[0, :sat] = np.diff(cdf0)
    pn[0, 0] += cdf0[0]                                      # everything below 0
    pn[0, sat] = 1.0 - cdf0[-1]

    # --- n >= 1 : Gamma(n, G) shifted by bias, smeared by the RON --------
    # Discrete RON kernel: exact per-ADU probabilities of the Gaussian,
    # including any fractional part of the bias.
    b0 = int(np.floor(det.bias))
    frac = det.bias - b0
    khalf = int(np.ceil(6.0 * det.ron)) + 1
    kx = np.arange(-khalf, khalf + 1, dtype=np.float64)
    kern = np.diff(_norm_cdf((np.append(kx, kx[-1] + 1.0) - 0.5 - frac) / det.ron))
    kern /= kern.sum()

    # Gamma support: everything up to saturation plus a read-noise margin.
    dmax = max(16, sat - b0 + khalf + 2)
    dmid = np.arange(dmax, dtype=np.float64) + 0.5           # cell mid-points
    log_dmid = np.log(dmid)

    for n in range(1, det.nmax + 1):
        # Gamma(n, G) density at the cell centres, in log space for stability
        logpdf = ((n - 1) * log_dmid - dmid / det.gain
                  - n * np.log(det.gain) - gammaln(n))
        gpdf = np.exp(logpdf)                                # per-ADU probability
        smeared = fftconvolve(gpdf, kern, mode='full')
        x0 = b0 - khalf                     # ADU of element 0 of `smeared`
        lo = max(0, -x0)
        hi = min(len(smeared), sat - x0)
        if hi > lo:
            pn[n, x0 + lo:x0 + hi] = np.clip(smeared[lo:hi], 0.0, None)
        # anything below ADU 0 folds into cell 0 (the ADC clips there);
        # anything at or above `sat` is the saturation cell
        if -x0 > 0:
            pn[n, 0] += float(np.clip(smeared[:min(-x0, len(smeared))],
                                      0.0, None).sum())
        pn[n, sat] = max(0.0, 1.0 - pn[n, :sat].sum())

    # renormalise against accumulated discretisation error (~1e-12)
    pn /= pn.sum(axis=1, keepdims=True)
    return pn


def _poisson_pmf(lam, nmax):
    k = np.arange(nmax + 1, dtype=np.float64)
    return np.exp(-lam + k * np.log(lam) - gammaln(k + 1.0))


def cell_model(det: Detector, mu_grid, pn=None):
    """
    Full-resolution cell probabilities and their flux derivative.

    Returns
    -------
    p     : (n_mu, ncell)  P(cell | mu)
    dp    : (n_mu, ncell)  dP(cell | mu) / dmu
    i_ref : (n_mu,)        Fisher information of the full 1-ADU histogram
    """
    if pn is None:
        pn = per_electron_cell_probs(det)
    mu_grid = np.atleast_1d(np.asarray(mu_grid, dtype=np.float64))
    ncell = pn.shape[1]
    p = np.zeros((mu_grid.size, ncell))
    dp = np.zeros((mu_grid.size, ncell))

    for j, mu in enumerate(mu_grid):
        lam = mu + det.cic
        w = _poisson_pmf(lam, det.nmax)
        # d/dmu Poisson(n | lam) = P(n-1) - P(n)
        dw = np.concatenate([[0.0], w[:-1]]) - w
        p[j] = w @ pn
        dp[j] = dw @ pn
        p[j, -1] += max(0.0, 1.0 - w.sum())   # Poisson tail: always saturates
        p[j] /= p[j].sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        contrib = np.where(p > 0.0, dp ** 2 / np.where(p > 0.0, p, 1.0), 0.0)
    return p, dp, contrib.sum(axis=1)


# ---------------------------------------------------------------------------
# Efficiency of an arbitrary set of bin edges
# ---------------------------------------------------------------------------
def _prefix(p, dp):
    """Cumulative sums with a leading zero, for O(1) bin queries."""
    P = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
    D = np.concatenate([np.zeros((dp.shape[0], 1)), np.cumsum(dp, axis=1)], axis=1)
    return P, D


def fisher_of_edges(edge_idx, P, D):
    """Fisher information per flux point for a partition given as cell indices."""
    e = np.asarray(edge_idx, dtype=np.int64)
    dP = P[:, e[1:]] - P[:, e[:-1]]
    dD = D[:, e[1:]] - D[:, e[:-1]]
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(dP > 0.0, dD ** 2 / np.where(dP > 0.0, dP, 1.0), 0.0)
    return g.sum(axis=1)


def efficiency(edge_idx, P, D, i_ref):
    """Information efficiency eta(mu) = I_bins(mu) / I_full(mu)."""
    return fisher_of_edges(edge_idx, P, D) / i_ref


# ---------------------------------------------------------------------------
# Near-global optimum by dynamic programming
# ---------------------------------------------------------------------------
def candidate_edges(det: Detector, n_fine=140, n_log=420):
    """
    Candidate cell indices at which a bin edge may be placed.

    Dense (~0.1 sigma) through the read-noise peak, where the interesting
    structure is, and log-spaced above it where the Gamma tail is smooth.
    """
    sat = int(round(det.sat_adu))
    lo = max(0.0, det.bias - 7.0 * det.ron)
    hi = det.bias + 7.0 * det.ron
    fine = np.linspace(lo, hi, n_fine)
    top = np.geomspace(max(hi + 1.0, det.bias + 1.0) - det.bias,
                       sat - det.bias, n_log) + det.bias
    cand = np.unique(np.round(np.concatenate([[0.0], fine, top,
                                              [sat, sat + 1.0]])).astype(np.int64))
    return cand[(cand >= 0) & (cand <= sat + 1)]


def bin_value_tensor(cand, P, D):
    """
    g[j, a, b] = Fisher information contributed at flux j by the single bin
    spanning cells [cand[a], cand[b]).  Computed once and reused by the DP.
    """
    Pc, Dc = P[:, cand], D[:, cand]
    dP = Pc[:, None, :] - Pc[:, :, None]
    dD = Dc[:, None, :] - Dc[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(dP > 0.0, dD ** 2 / np.where(dP > 0.0, dP, 1.0), 0.0)


def dp_best_partition(K, cand, g, weights):
    """
    Best K-bin partition for the additive objective

        V = sum_mu weights[mu] * I_bins(mu)

    restricted to edges drawn from `cand`.  Returns (edge_idx, V).

    Additivity of the Fisher information over bins is what makes an exact
    dynamic program possible: the value of a partition is the sum of the
    values of its bins, so the usual "best prefix ending at edge a" recursion
    applies and the optimum is global within the candidate-edge set.
    """
    C = len(cand)
    value = np.tensordot(weights, g, axes=(0, 0))            # (C, C)
    value[np.tril_indices(C)] = -np.inf                      # b must exceed a

    best = np.full((K + 1, C), -np.inf)
    back = np.zeros((K + 1, C), dtype=np.int64)
    best[0, 0] = 0.0
    for k in range(1, K + 1):
        with np.errstate(invalid='ignore'):
            m = best[k - 1][:, None] + value
        back[k] = np.argmax(m, axis=0)
        best[k] = m[back[k], np.arange(C)]

    end = C - 1                                              # must cover everything
    if not np.isfinite(best[K, end]):
        raise RuntimeError(f'no feasible {K}-bin partition on this candidate set')
    nodes, node = [end], end
    for k in range(K, 0, -1):
        node = int(back[k, node])
        nodes.append(node)
    return np.array(sorted(cand[np.array(nodes[::-1])])), float(best[K, end])


def maximin_partition(K, det, mu_grid, P, D, i_ref, cand=None,
                      n_iter=40, step=8.0, verbose=True):
    """
    Max-min-efficiency K-bin partition via multiplicative weights.

    Each round solves the exactly-optimal weighted problem by DP, then pushes
    weight towards the flux values that are currently worst served.  The best
    worst-case design seen over all rounds is returned.
    """
    if cand is None:
        cand = candidate_edges(det)
    g = bin_value_tensor(cand, P, D)
    w = np.ones(len(mu_grid)) / len(mu_grid)
    best_edges, best_worst = None, -np.inf
    for it in range(n_iter):
        edges, _ = dp_best_partition(K, cand, g, w / i_ref)
        eta = efficiency(edges, P, D, i_ref)
        worst = float(eta.min())
        if worst > best_worst:
            best_worst, best_edges = worst, edges
        w = w * np.exp(step * (eta.mean() - eta))
        w /= w.sum()
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            log(f'  K={K:2d} iter {it:3d}: worst-case eta = {worst:.4f} '
                f'(best so far {best_worst:.4f})', 'value')
    return best_edges, best_worst


# ---------------------------------------------------------------------------
# The interpretable closed-form bin families
# ---------------------------------------------------------------------------
def ladder_edges(u_start, u_stop, n, shape='sqrt'):
    """
    n+1 rungs from u_start to u_stop in gain units u = (x-bias)/G.

    shape='sqrt'   : uniform in sqrt(u) -- matches the local width of the Gamma
                     distribution, which grows like sqrt(u).  This is the
                     variance-stabilising variable of the EM register, and it
                     is what the numerical optimum reproduces.
    shape='log'    : uniform in log u   (constant fractional width)
    shape='linear' : uniform in u
    """
    if shape == 'log':
        return np.geomspace(u_start, u_stop, n + 1)
    q = {'sqrt': 0.5, 'linear': 1.0}.get(shape)
    if q is None:
        raise ValueError(f'unknown ladder shape {shape!r}')
    return np.linspace(u_start ** q, u_stop ** q, n + 1) ** (1.0 / q)


def physical_edges(det: Detector, K, c_hi=4.0, c_lo=1.0, m=1, u1=1.0,
                   u_top=None, shape='sqrt'):
    """
    The recommended closed-form bin design.  Three zones, in this order:

      zone A -- the read-noise peak.  ``m`` cuts placed UNIFORMLY IN UNITS OF
                THE READ NOISE, from ``bias + c_lo*sigma`` to
                ``bias + c_hi*sigma``.  Bin 0, everything below the first cut,
                holds the zero-electron peak plus any underflow.  When
                sigma/G is very small the peak carries no flux-dependent
                internal structure and a single cut (m = 1, placed at c_hi)
                is optimal.

      zone B -- the EM tail.  ``n = K - m - 1`` rungs placed uniformly in
                sqrt(u), u = (x - bias)/G, from ``u1`` (about one electron) up
                to ``u_top``.  The wide bin between the last peak cut and
                ``u1`` is the single-electron shoulder and is deliberately
                left unsplit.

      zone C -- the open bin ``[bias + G*u_top, inf)``, which absorbs both real
                saturation and the (empty) region above the flux range.

    ``u_top`` defaults to the saturation head-room S = (sat-bias)/G, but if the
    flux range stops well below saturation it should be lowered to
    ``mu_max + 3*sqrt(mu_max)``: an edge placed where no counts ever land is an
    edge wasted.

    Returns integer cell indices of length K+1, or None if the parameters
    cannot yield K distinct edges.
    """
    sat = int(round(det.sat_adu))
    if u_top is None:
        u_top = det.S
    u_top = min(float(u_top), det.S)
    n = K - m - 1
    if n < 2 or m < 1:
        return None

    core = (det.bias + det.ron * (np.array([c_hi], dtype=np.float64) if m == 1
                                  else np.linspace(c_lo, c_hi, m)))
    rungs = det.bias + det.gain * ladder_edges(u1, u_top, n - 1, shape=shape)

    # Edges land on whole ADU because the detector itself is digitised.  The
    # core cuts are rounded UP, never down: rounding a cut down towards the
    # bias lets the Gaussian tail of the zero-electron peak leak across it,
    # and that leak is what destroys the low-flux information.  Sub-ADU read
    # noise makes this rounding direction decisive.
    edges = np.concatenate([[0], np.ceil(core).astype(np.int64),
                            np.round(rungs).astype(np.int64), [sat + 1]])
    edges = np.unique(np.clip(edges, 0, sat + 1))
    return edges if len(edges) == K + 1 else None


def fit_physical_family(det, K, P, D, i_ref, mu_max=None, verbose=True,
                        shapes=('sqrt', 'log', 'linear')):
    """
    Search the four free numbers of ``physical_edges`` (m, c_lo, c_hi, u1) plus
    the ladder shape, maximising the worst-case efficiency.

    Coarse pass over the whole space, then a local refinement around the
    winner; the objective is smooth in c_lo/c_hi/u1, so this matches a brute
    force to better than 0.1 % at a fraction of the cost.
    """
    u_tops = [det.S]
    if mu_max is not None:
        u_tops += [min(det.S, mu_max + c * np.sqrt(mu_max))
                   for c in (2.0, 3.0, 4.0)]
    u_tops = sorted(set(round(float(u), 4) for u in u_tops))

    def scan(m_vals, c_lo_vals, c_hi_vals, u1_vals):
        best = (None, -np.inf, None)
        for shape in shapes:
            for m in m_vals:
                if m < 1 or m > K - 3:
                    continue
                los = [0.0] if m == 1 else c_lo_vals
                for c_hi in c_hi_vals:
                    for c_lo in los:
                        if m > 1 and c_lo >= c_hi:
                            continue
                        for u1 in u1_vals:
                            for u_top in u_tops:
                                e = physical_edges(det, K, c_hi=c_hi, c_lo=c_lo,
                                                   m=m, u1=u1, u_top=u_top,
                                                   shape=shape)
                                if e is None:
                                    continue
                                w = float(efficiency(e, P, D, i_ref).min())
                                if w > best[1]:
                                    best = (e, w, dict(
                                        m=int(m), c_lo=float(c_lo),
                                        c_hi=float(c_hi), u1=float(u1),
                                        u_top=float(u_top), shape=shape))
        return best

    best = scan(range(1, min(6, K - 2)), np.arange(0.5, 3.01, 0.5),
                np.arange(1.0, 6.01, 0.5), np.arange(0.4, 2.41, 0.4))
    if best[0] is None:
        return best
    q = best[2]
    refined = scan([q['m']],
                   np.arange(max(0.0, q['c_lo'] - 0.5), q['c_lo'] + 0.51, 0.125),
                   np.arange(max(0.25, q['c_hi'] - 0.5), q['c_hi'] + 0.51, 0.125),
                   np.arange(max(0.1, q['u1'] - 0.4), q['u1'] + 0.41, 0.1))
    best = max(best, refined, key=lambda t: t[1])
    if verbose and best[0] is not None:
        log(f'  K={K:2d} closed form: worst-case eta = {best[1]:.4f} '
            f'at {best[2]}', 'value')
    return best


def analytic_edges(det: Detector, K, a=3.0, b=3.0, m=2, x_lo_frac=None,
                   shape='log'):
    """
    Earlier, simpler family, kept for backward compatibility with
    ``binning_code/embin_config.yaml``:

        bin 0            : [0, bias - a*sigma)              low tail
        bins 1 .. m      : uniform, bias - a*sigma -> bias + b*sigma
        bins m+1 .. K-2  : ladder in u, from b*sigma/G to S
        bin K-1          : [sat, inf)

    ``physical_edges`` supersedes it: it drops the useless low-side extent,
    puts the ladder in sqrt(u), and lets the top bin start below saturation.
    """
    sat = int(round(det.sat_adu))
    n_log = K - m - 2
    if n_log < 1:
        raise ValueError('not enough bins left for the upper ladder')

    hi = det.bias + b * det.ron
    if m == 0:
        core = np.array([hi])
    else:
        core = np.linspace(max(0.0, det.bias - a * det.ron), hi, m + 1)

    start = hi - det.bias if x_lo_frac is None else x_lo_frac * det.gain
    start = max(start, 0.25 * det.ron)
    ladder = ladder_edges(start / det.gain, (sat - det.bias) / det.gain,
                          n_log, shape=shape)[1:] * det.gain + det.bias

    edges = np.concatenate([[0.0], core, ladder, [sat + 1.0]])
    edges = np.unique(np.clip(np.round(edges).astype(np.int64), 0, sat + 1))
    return edges if len(edges) == K + 1 else None


# ---------------------------------------------------------------------------
# Maximum-likelihood machinery, for the Monte-Carlo demonstration
# ---------------------------------------------------------------------------
def binned_pn(pn, edge_idx):
    """Collapse the per-electron 1-ADU table onto K bins: (nmax+1, K)."""
    e = np.asarray(edge_idx, dtype=np.int64)
    cum = np.concatenate([np.zeros((pn.shape[0], 1)), np.cumsum(pn, axis=1)],
                         axis=1)
    return cum[:, e[1:]] - cum[:, e[:-1]]


def bin_probability_table(det, edge_idx, mu_grid, pn=None):
    """
    log P(bin | mu) on a flux grid, ready for a matmul-based MLE.

    Built from the per-electron table collapsed onto the K bins, so the cost
    is O(n_mu * nmax * K) rather than O(n_mu * nmax * n_ADU): the flux grid can
    be as dense as the fit needs without ever materialising a
    (n_mu x 60000) array.
    """
    if pn is None:
        pn = per_electron_cell_probs(det)
    bpn = binned_pn(pn, edge_idx)
    mu_grid = np.atleast_1d(np.asarray(mu_grid, dtype=np.float64))

    # Poisson weights for every flux at once: (n_mu, nmax+1) @ (nmax+1, K)
    lam = (mu_grid + det.cic)[:, None]
    k = np.arange(det.nmax + 1, dtype=np.float64)[None, :]
    W = np.exp(-lam + k * np.log(lam) - gammaln(k + 1.0))
    bp = W @ bpn
    # Poisson mass beyond nmax: those reads always saturate
    bp[:, -1] += np.clip(1.0 - W.sum(axis=1), 0.0, None)
    bp = np.clip(bp, 1e-300, None)
    bp /= bp.sum(axis=1, keepdims=True)
    return np.log(bp)


def mle_from_counts(counts, mu_grid, log_bp):
    """Grid maximum-likelihood flux for one or many histograms."""
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    return mu_grid[np.argmin(-(log_bp @ counts.T), axis=0)]


def simulate_adu(det, mu, n, rng):
    """Draw n raw ADU pixel values at mean flux mu (exact signal-chain model)."""
    ns = rng.poisson(mu + det.cic, size=n)
    x = np.zeros(n, dtype=np.float64)
    nz = ns > 0
    if nz.any():
        x[nz] = rng.gamma(ns[nz], det.gain)
    x += det.bias + rng.normal(0.0, det.ron, size=n)
    np.clip(x, 0.0, None, out=x)
    np.minimum(x, det.sat_adu, out=x)        # saturated pixels pile up at sat
    return x


def histogram_from_adu(x, edge_idx):
    """Bin raw ADU into the K-bin histogram defined by integer cell edges."""
    e = np.asarray(edge_idx, dtype=np.float64).copy()
    e[0], e[-1] = -np.inf, np.inf
    return np.histogram(x, bins=e)[0]


# ---------------------------------------------------------------------------
# Top-level API:  detector properties in  ->  bin edges out
# ---------------------------------------------------------------------------
@dataclass
class BinDesign:
    """The answer: where to cut, and how good that is."""
    edges: np.ndarray            # K+1 bin edges [ADU]; edges[-1] is +inf
    cell_edges: np.ndarray       # the same edges as integer cell indices
    detector: Detector
    nbins: int
    mu_grid: np.ndarray          # flux values the design was optimised over
    eta: np.ndarray              # information efficiency at each mu
    method: str
    params: dict | None = None   # closed-form parameters, when applicable

    @property
    def worst_eta(self):
        """Worst-case Fisher-information efficiency over the flux range."""
        return float(self.eta.min())

    @property
    def worst_accuracy(self):
        """Worst-case accuracy ratio sigma_full / sigma_K = sqrt(eta)."""
        return float(np.sqrt(self.eta.min()))

    @property
    def u_edges(self):
        """Bin edges in gain units u = (x - bias)/G."""
        return (self.edges - self.detector.bias) / self.detector.gain

    def edge_list(self, sentinel=None):
        """
        Edges as a plain list of ints, with the open top bin written as a
        sentinel one ADU above the last finite edge -- the convention used by
        ``binning_code/embin.py`` and by ``emccd_histo.get_bins``.
        """
        e = self.edges.copy()
        e[-1] = (e[-2] + 1.0) if sentinel is None else sentinel
        return [int(round(v)) for v in e]

    def __repr__(self):
        return (f'<BinDesign K={self.nbins} method={self.method} '
                f'worst eta={self.worst_eta:.4f} '
                f'accuracy={100 * self.worst_accuracy:.1f}% of full>')

    def summary(self):
        """Human-readable report of where every cut sits."""
        det = self.detector
        out = [f'{self.nbins}-bin design   (method: {self.method})',
               f'  detector : bias={det.bias:g}  G={det.gain:g} ADU/e-  '
               f'RON={det.ron:g} ADU  sat={det.sat_adu:g} ADU',
               f'  reduced  : r = RON/G = {det.r:.4f}    '
               f'S = (sat-bias)/G = {det.S:.2f} e-',
               f'  flux range : {self.mu_grid[0]:.4g} to {self.mu_grid[-1]:.4g} e-/read',
               f'  worst-case eta = {self.worst_eta:.4f}  ->  '
               f'sigma_full/sigma_K = {self.worst_accuracy:.4f}  '
               f'({100 * self.worst_accuracy:.1f} % of the full histogram)']
        if self.params:
            out.append(f'  closed-form parameters : {self.params}')
        out += ['',
                '   bin    lower [ADU]    upper [ADU]        u_lo       u_hi'
                '     width',
                '  ' + '-' * 68]
        for i in range(self.nbins):
            lo, hi = self.edges[i], self.edges[i + 1]
            ulo = (lo - det.bias) / det.gain
            uhi = (hi - det.bias) / det.gain
            his = '        inf' if not np.isfinite(hi) else f'{hi:11.0f}'
            uhs = '       inf' if not np.isfinite(uhi) else f'{uhi:10.3f}'
            wid = '      inf' if not np.isfinite(hi) else f'{hi - lo:9.0f}'
            out.append(f'  {i:4d}    {lo:11.0f}    {his}   {ulo:10.3f} {uhs} {wid}')
        ns = [(e - det.bias) / det.ron
              for e in self.edges[1:min(5, self.nbins)]]
        out += ['', '  first cuts, in RON sigmas above bias: '
                + ', '.join(f'{v:+.2f}' for v in ns)]
        return '\n'.join(out)


def design_bins(bias, gain, ron, saturation, nbins=8, mu_min=1e-3, mu_max=8.0,
                cic=0.0, method='both', n_mu=21, nmax=60, verbose=False):
    """
    General-purpose entry point: give it the detector, get the bin edges.

    Parameters
    ----------
    bias       : float  bias / pedestal level [ADU]
    gain       : float  mean EM gain [ADU/e-]
    ron        : float  read-out noise, 1 sigma [ADU]
    saturation : float  ADU at and above which the pixel is saturated
    nbins      : int    number of bins K.  8 is enough to retain >95 % of the
                        full-histogram accuracy in the usual regime.
    mu_min,
    mu_max     : float  flux range to optimise for [e-/read].  The design
                        maximises the WORST-CASE efficiency over this range, so
                        do not make it wider than you need.
    cic        : float  clock-induced charge [e-/read], added to the flux
    method     : 'dp'     max-min dynamic programming (near-global optimum)
                 'closed' the interpretable closed-form family
                 'both'   compute both, return whichever wins (default)

    Returns
    -------
    BinDesign  -- ``.edges`` are the K+1 edges in ADU (last one +inf),
                  ``.edge_list()`` the integer list with a sentinel,
                  ``.summary()`` a printable report of every cut,
                  ``.worst_accuracy`` the fraction of the full-histogram
                  accuracy that these K bins retain.
    """
    det = Detector(bias=float(bias), gain=float(gain), ron=float(ron),
                   sat_adu=float(saturation), cic=float(cic), nmax=int(nmax))
    if nbins < 4:
        raise ValueError('nbins must be at least 4')
    if not 0 < mu_min < mu_max:
        raise ValueError('need 0 < mu_min < mu_max')

    mu = np.geomspace(mu_min, mu_max, n_mu)
    if verbose:
        log(f'detector r={det.r:.4f} S={det.S:.2f}; optimising over '
            f'mu = {mu_min:g} ... {mu_max:g} e-/read', 'info')
    pn = per_electron_cell_probs(det)
    p, dp, i_ref = cell_model(det, mu, pn=pn)
    P, D = _prefix(p, dp)

    results = []
    if method in ('dp', 'both'):
        e, _ = maximin_partition(nbins, det, mu, P, D, i_ref,
                                 n_iter=40, step=8.0, verbose=verbose)
        results.append((e, efficiency(e, P, D, i_ref), 'dp', None))
    if method in ('closed', 'both'):
        e, _, par = fit_physical_family(det, nbins, P, D, i_ref,
                                        mu_max=mu_max, verbose=verbose)
        if e is not None:
            results.append((e, efficiency(e, P, D, i_ref), 'closed', par))
    if not results:
        raise ValueError(f'unknown method {method!r}')

    cells, eta, used, par = max(results, key=lambda t: t[1].min())
    edges = np.asarray(cells, dtype=np.float64)
    edges[-1] = np.inf                      # the top bin is open-ended
    return BinDesign(edges=edges, cell_edges=np.asarray(cells, dtype=np.int64),
                     detector=det, nbins=int(nbins), mu_grid=mu, eta=eta,
                     method=used, params=par)


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description='Optimal EMCCD histogram bin edges for flux retrieval.')
    ap.add_argument('--bias', type=float, required=True, help='bias level [ADU]')
    ap.add_argument('--gain', type=float, required=True, help='EM gain [ADU/e-]')
    ap.add_argument('--ron', type=float, required=True, help='read noise [ADU]')
    ap.add_argument('--sat', type=float, required=True, help='saturation [ADU]')
    ap.add_argument('--nbins', type=int, default=8, help='number of bins (default 8)')
    ap.add_argument('--mu-min', type=float, default=1e-3, help='min flux [e-/read]')
    ap.add_argument('--mu-max', type=float, default=8.0, help='max flux [e-/read]')
    ap.add_argument('--cic', type=float, default=0.0, help='CIC [e-/read]')
    ap.add_argument('--method', default='both', choices=('dp', 'closed', 'both'))
    ap.add_argument('--scan', action='store_true',
                    help='also report every K from 4 to 32')
    a = ap.parse_args(argv)

    if a.scan:
        log('worst-case accuracy retained versus number of bins', 'info')
        for K in (4, 6, 8, 12, 16, 24, 32):
            d = design_bins(a.bias, a.gain, a.ron, a.sat, nbins=K,
                            mu_min=a.mu_min, mu_max=a.mu_max, cic=a.cic,
                            method=a.method)
            ok = d.worst_accuracy >= 0.95
            log(f'  K={K:3d}  eta={d.worst_eta:.4f}  '
                f'sigma_full/sigma_K={d.worst_accuracy:.4f}  '
                f'[{"PASS" if ok else "fail"}]', 'value' if ok else 'skip')

    d = design_bins(a.bias, a.gain, a.ron, a.sat, nbins=a.nbins,
                    mu_min=a.mu_min, mu_max=a.mu_max, cic=a.cic,
                    method=a.method, verbose=True)
    print()
    print(d.summary())
    print()
    print('edges (embin / get_bins convention, open top bin):')
    print(f'  edges: {d.edge_list()}')
    return d


if __name__ == '__main__':
    _cli()
