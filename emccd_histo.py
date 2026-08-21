"""
emccd_histo.py
==============
Core library for EMCCD histogram-based flux estimation.

References
----------
Harpsøe et al. 2012, A&A 537, A50  https://arxiv.org/abs/1111.2066
Daigle   et al. 2009, PASP 121, 866 https://arxiv.org/abs/0908.0528

Functions
---------
load_config(yaml_path)           – load detector + histogram parameters
get_bins(config)                 – return the non-uniform bin edges (nbins+1 values)
sample_pixels(flux_e, N, config) – draw N raw ADU pixel values for a given flux
bin_pixels(pixel_values, bins)   – assign pixels to histogram bins
fit_flux(counts, bins, config)   – recover mean flux from a per-pixel histogram
"""

import math
import numpy as np
import yaml
from scipy.special import factorial, gammaincc
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Internal helpers (reused from stats_emccd.py)
# ---------------------------------------------------------------------------

def _poisson_pmf(mu, nmax):
    """Poisson probability mass function P(k|mu) for k = 0 … nmax."""
    k = np.arange(nmax + 1, dtype=np.float64)
    return np.exp(-mu + k * np.log(mu + 1e-300) - np.array([math.lgamma(ki + 1) for ki in k]))


def _psf_ron(val, ron):
    """Gaussian read-noise kernel."""
    return np.exp(-val ** 2 / (2 * ron ** 2)) / (ron * np.sqrt(2 * np.pi))


def _single_electron_pdf(vals, n, gain, bias, ron, kernel=None):
    """
    EMCCD pixel-value PDF for exactly *n* detected photo-electrons.

    For n > 0 this is a Gamma(n, gain) distribution shifted by *bias* and
    convolved with a Gaussian of std = *ron*.
    For n = 0 this is a delta at *bias* convolved with the read-noise Gaussian.

    Parameters
    ----------
    vals : 1-D float array   ADU grid (0 … 65535 recommended)
    n    : int               number of electrons (>= 0)
    gain : float             EM gain [ADU/e-]
    bias : float             bias level [ADU]
    ron  : float             read-noise std [ADU]

    Returns
    -------
    p : normalised probability array (same shape as *vals*)
    """
    if n > 0:
        safe_vals = np.where(vals > 0, vals, 0.0)
        p = safe_vals ** (n - 1) * np.exp(-safe_vals / gain) / (gain ** n * math.factorial(n - 1))
        # Shift by bias using zero-padding (np.roll would wrap high-end values
        # back below the bias level, which is unphysical).
        shift = int(round(bias))
        p = np.concatenate([np.zeros(shift), p[:-shift] if shift > 0 else p])
    else:
        p = np.zeros_like(vals, dtype=np.float64)
        idx = int(round(bias))
        if 0 <= idx < len(p):
            p[idx] = 1.0

    if kernel is None:
        kernel_x = np.arange(-5 * ron, 5 * ron + 1, 1.0)
        kernel = _psf_ron(kernel_x, ron)
    p = fftconvolve(p, kernel, mode='same')

    total = p.sum()
    if total > 0:
        p /= total
    return p


def _full_pdf(vals, mu, config):
    """
    Full EMCCD pixel-value PDF: Poisson mixture of single-electron PDFs.

    P(ADU | mu) = sum_n  P_Poisson(n|mu) * P_EMCCD(ADU|n)

    Parameters
    ----------
    vals   : 1-D float array   ADU grid
    mu     : float             mean flux [e-/frame]
    config : dict              loaded from load_config()

    Returns
    -------
    p : normalised probability array (same shape as *vals*)
    """
    det = config['detector']
    gain = det['gain']
    ron  = det['ron']
    bias = det['bias']
    nmax = det['nmax']
    cic  = det.get('cic', 0.0)   # clock-induced charges [e-/frame/pixel]

    # CIC electrons are amplified identically to photo-electrons; the two
    # independent Poisson processes sum to Poisson(mu + cic).
    pmf = _poisson_pmf(mu + cic, nmax)
    p = np.zeros_like(vals, dtype=np.float64)
    for n in range(nmax + 1):
        if pmf[n] > 1e-12:
            p += pmf[n] * _single_electron_pdf(vals, n, gain, bias, ron)

    total = p.sum()
    if total > 0:
        p /= total
    return p


def build_cache(config, bins=None):
    """
    Pre-compute per-electron bin probabilities for fast MLE.

    Returns a (nmax+1, nbins) matrix where entry [n, i] is the probability
    that a pixel with exactly *n* detected electrons falls in bin *i*.

    Build this once per (config, bins) combination and pass the result to
    fit_flux() via the `cache` keyword — the expensive 65536-pt FFT convolutions
    are then done only nmax+1 times total instead of inside every optimizer call.

    Parameters
    ----------
    config : dict  as returned by load_config()
    bins   : 1-D float array, optional  (defaults to get_bins(config))

    Returns
    -------
    bn_matrix : float array, shape (nmax+1, nbins)
    """
    if bins is None:
        bins = get_bins(config)
    det = config['detector']
    gain = det['gain']
    ron  = det['ron']
    bias = det['bias']
    nmax = det['nmax']
    full_well = det['full_well']

    vals = np.arange(0, full_well + 1, 1, dtype=np.float64)
    # Build the RON kernel once — it is the same for every n
    kernel_x = np.arange(-5 * ron, 5 * ron + 1, 1.0)
    kernel = _psf_ron(kernel_x, ron)

    nbins = len(bins) - 1
    bn_matrix = np.zeros((nmax + 1, nbins), dtype=np.float64)
    for n in range(nmax + 1):
        p = _single_electron_pdf(vals, n, gain, bias, ron, kernel=kernel)
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        bn_matrix[n] = np.clip(np.diff(np.interp(bins, vals, cdf)), 1e-300, None)

    # -------------------------------------------------------------------
    # Analytical saturation-bin correction: override the last bin with
    # P(ADU >= sat_adu | n electrons) = gammaincc(n, (sat_adu-bias)/gain).
    # This is exact (ignoring the negligible RON at the threshold) and
    # correctly tracks the saturation fraction even at high flux where the
    # Gamma distribution extends far beyond the ADU grid.  Sub-saturation
    # bins are rescaled so each row still sums to 1.
    # -------------------------------------------------------------------
    sat_adu_f = float(config['histogram'].get('saturation_adu', full_well))
    x_sat = (sat_adu_f - bias) / float(gain)
    for n in range(nmax + 1):
        p_sat = float(gammaincc(n, x_sat)) if n > 0 else 1e-100
        p_other = float(bn_matrix[n, :-1].sum())
        if p_other > 0:
            bn_matrix[n, :-1] *= (1.0 - p_sat) / p_other
        bn_matrix[n, -1] = max(p_sat, 1e-300)

    return bn_matrix

def build_nll_grid(config, bins=None, cache=None, mu_min=1e-4, mu_max=20.0, n_grid=2000):
    """
    Pre-compute the log bin-probability matrix on a dense log-spaced flux grid.

    Returns ``(mu_grid, log_bin_probs)`` where::

        log_bin_probs[j, i]  =  log P(pixel falls in bin i  |  mu = mu_grid[j])

    This is computed once per ``(config, bins)`` combination.  Passing both
    arrays to ``fit_flux`` via the ``nll_grid=`` keyword replaces the scalar
    optimizer + binary-search entirely with a **single matrix-vector product**,
    giving another ~20-50× speedup on top of the ``cache`` optimisation.

    The grid is log-spaced so that the relative step size is constant (~0.7 %
    for the default n_grid=2000 over [1e-4, 20]).  This keeps the flux
    resolution well below the statistical uncertainty at all flux levels.

    Parameters
    ----------
    config  : dict
    bins    : 1-D float array, optional  (defaults to get_bins(config))
    cache   : (nmax+1, nbins) array, optional  (from build_cache; built if None)
    mu_min  : float   Minimum flux in the grid [e-/frame]
    mu_max  : float   Maximum flux in the grid [e-/frame]
    n_grid  : int     Number of grid points (log-spaced)

    Returns
    -------
    mu_grid       : 1-D float array, shape (n_grid,)
    log_bin_probs : 2-D float array, shape (n_grid, nbins)
    """
    if bins is None:
        bins = get_bins(config)
    if cache is None:
        cache = build_cache(config, bins)
    nmax = config['detector']['nmax']
    det  = config['detector']
    hist = config['histogram']
    # Bins that lie entirely above the RON region carry the signal information.
    # The fraction of counts in these bins is a monotone function of µ and
    # serves as a fast prior estimator.
    threshold = det['bias'] + float(hist.get('ron_high_extent', hist.get('ron_extent', 3.0))) * det['ron']
    signal_mask = bins[:-1] >= threshold          # (nbins,) bool: leading edge above threshold

    cic = det.get('cic', 0.0)   # clock-induced charges [e-/frame/pixel]

    # mu_grid represents photon-only flux; model is evaluated at mu + cic so
    # the fitter returns the CIC-subtracted flux directly.
    mu_grid = np.geomspace(mu_min, mu_max, n_grid)
    log_bp    = np.zeros((n_grid, len(bins) - 1), dtype=np.float64)
    f_signal  = np.zeros(n_grid, dtype=np.float64)  # expected above-threshold fraction
    for j, mu in enumerate(mu_grid):
        pmf = _poisson_pmf(mu + cic, nmax)
        bp  = pmf @ cache
        # Poisson probability beyond nmax: all such high-n frames saturate
        p_overflow = max(0.0, 1.0 - float(pmf.sum()))
        bp[-1] = min(1.0, float(bp[-1]) + p_overflow)
        bp  = np.clip(bp, 1e-300, None)
        log_bp[j]   = np.log(bp)
        f_signal[j] = bp[signal_mask].sum()
    return mu_grid, log_bp, f_signal, signal_mask


def load_config(yaml_path):
    """
    Load EMCCD + histogram parameters from a YAML file.

    Parameters
    ----------
    yaml_path : str  Path to the YAML configuration file.

    Returns
    -------
    config : dict
    """
    with open(yaml_path) as fh:
        config = yaml.safe_load(fh)
    return config


def get_bins(config):
    """
    Build the non-uniform histogram bin edges for one EMCCD pixel.

    Bin structure
    ~~~~~~~~~~~~~
    1. A *uniform* region around the zero-electron peak:
       from  (bias - ron_low_extent * ron)
       to    (bias + ron_high_extent * ron)
       with  ron_bins_per_sigma * (ron_low_extent + ron_high_extent)  bins.
       ron_low_extent and ron_high_extent may differ (asymmetric design).

    2. A *logarithmic* region from the end of the uniform section up to
       (full_well - 1), using the remaining (nbins - uniform_bins - 1) edges.

    3. A final *saturation* bin edge at full_well + 1 (catches saturated pixels).

    Parameters
    ----------
    config : dict  as returned by load_config()

    Returns
    -------
    edges : 1-D float array of length (nbins + 1)
    """
    det = config['detector']
    hist = config['histogram']

    bias = det['bias']
    ron = det['ron']
    full_well = det['full_well']
    nbins = hist['nbins']
    ron_per_sigma = hist['ron_bins_per_sigma']
    ron_low  = float(hist.get('ron_low_extent',  hist.get('ron_extent', 3.0)))
    ron_high = float(hist.get('ron_high_extent', hist.get('ron_extent', 3.0)))
    sat_adu = float(hist.get('saturation_adu', full_well))

    # --- uniform region around zero-electron peak (may be asymmetric) ---
    low_edge = max(0.0, bias - ron_low * ron)
    high_edge = bias + ron_high * ron
    n_uniform = int(round(ron_per_sigma * (ron_low + ron_high)))
    n_uniform = min(n_uniform, nbins - 2)  # leave room for sqrt region + sat bin

    uniform_edges = np.linspace(low_edge, high_edge, n_uniform + 1)

    # --- sqrt-uniform region from RON edge to saturation threshold ---
    # Uniform spacing in sqrt(ADU) means bin widths grow as sqrt, giving more
    # resolution at low flux and coarser bins toward saturation.
    n_sqrt = nbins - n_uniform - 1  # last slot is the saturation bin
    sqrt_start = high_edge
    sqrt_end = sat_adu
    if n_sqrt > 0 and sqrt_end > sqrt_start:
        sqrt_edges = np.linspace(np.sqrt(sqrt_start), np.sqrt(sqrt_end), n_sqrt + 1) ** 2
        sqrt_edges = sqrt_edges[1:]  # drop duplicate of high_edge
    else:
        sqrt_edges = np.array([sqrt_end])

    # --- saturation bin: one open-ended edge above sat_adu ---
    # All pixels with value >= sat_adu fall into this final bin.
    sat_edge = float(full_well) + 1.0

    edges = np.concatenate([uniform_edges, sqrt_edges, [sat_edge]])

    # Ensure strictly increasing (numerical safety)
    edges = np.unique(edges)

    return edges


def sample_pixels(flux_e, N, config, rng=None):
    """
    Draw *N* raw ADU pixel values for a given mean flux using inverse-CDF sampling.

    Parameters
    ----------
    flux_e : float   Mean flux in electrons per frame per pixel.
    N      : int     Number of pixel samples to draw.
    config : dict    Loaded configuration (load_config).
    rng    : np.random.Generator or None.  If None, uses the seed in config.

    Returns
    -------
    pixel_values : 1-D float array of length N  (ADU)
    """
    if rng is None:
        seed = config.get('simulation', {}).get('seed', None)
        rng = np.random.default_rng(seed)

    det       = config['detector']
    gain      = float(det['gain'])
    bias      = float(det['bias'])
    ron       = float(det['ron'])
    nmax      = det['nmax']
    full_well = float(det['full_well'])
    cic       = float(det.get('cic', 0.0))

    # Direct Poisson + Gamma sampling: physically exact and faster than the
    # inverse-CDF / FFT-convolution approach.  Each pixel draws n electrons
    # from Poisson(flux_e + cic) then amplifies through the EM register
    # (Gamma(n, gain)) and adds read noise.  Pixels with n > nmax are set to
    # full_well — they would all saturate at these gain/sat_adu parameters.
    ns = rng.poisson(float(flux_e) + cic, size=N)
    pixel_values = np.full(N, full_well, dtype=np.float64)
    for n_val in range(nmax + 1):
        mask = ns == n_val
        if not mask.any():
            continue
        m = int(mask.sum())
        if n_val == 0:
            adu = rng.normal(bias, ron, size=m)
        else:
            adu = rng.gamma(n_val, gain, size=m) + bias + rng.normal(0.0, ron, size=m)
        pixel_values[mask] = adu

    np.clip(pixel_values, 0.0, full_well, out=pixel_values)
    return pixel_values


def bin_pixels(pixel_values, bins):
    """
    Assign raw ADU pixel values into histogram bins.

    Parameters
    ----------
    pixel_values : 1-D array   Raw ADU values (as returned by sample_pixels).
    bins         : 1-D array   Bin edges (as returned by get_bins).

    Returns
    -------
    counts : 1-D int array of length (len(bins) - 1)
    """
    counts, _ = np.histogram(pixel_values, bins=bins)
    return counts.astype(np.int64)


def fit_flux(counts, bins, config, mu_min=1e-4, mu_max=20.0, cache=None, nll_grid=None):
    """
    Recover the mean flux [e-/frame] from a per-pixel histogram via MLE.

    The likelihood is the multinomial log-likelihood:
        log L(mu) = sum_i  counts_i * log( p_i(mu) )
    where p_i(mu) is the probability that a pixel falls in bin i given flux mu.

    Parameters
    ----------
    counts   : 1-D int array    Bin counts (from bin_pixels or accumulated cube).
    bins     : 1-D float array  Bin edges (from get_bins).
    config   : dict             Detector configuration.
    mu_min   : float            Lower bound for flux search [e-/frame].
    mu_max   : float            Upper bound for flux search [e-/frame].
    cache    : (nmax+1, nbins) float array, optional
               Pre-computed per-electron bin probabilities (from build_cache).
    nll_grid : tuple (mu_grid, log_bin_probs, f_signal, signal_mask), optional
               Pre-computed dense flux grid from build_nll_grid().  When
               supplied the optimizer is replaced by a single matrix-vector
               product over all grid points followed by argmin (~0.04 ms).

    Returns
    -------
    mu_fit  : float   Best-fit mean flux [e-/frame]
    mu_lo   : float   Lower 1-sigma uncertainty (flux at delta-log-L = 0.5)
    mu_hi   : float   Upper 1-sigma uncertainty
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_total = counts.sum()

    # ------------------------------------------------------------------
    # Fast path: full-grid matmul + argmin on the pre-computed grid
    # ------------------------------------------------------------------
    if nll_grid is not None:
        mu_grid, log_bp_grid, f_signal, signal_mask = nll_grid

        # One matrix-vector product gives NLL at all 2000 grid points (~0.04 ms)
        nll_vals = -(log_bp_grid @ counts)
        j_min    = int(np.argmin(nll_vals))
        mu_fit   = float(mu_grid[j_min])
        nll_min  = nll_vals[j_min]
        target   = nll_min + 0.5

        # 1-sigma bounds via linear interpolation across the delta-log-L = 0.5 crossing
        left_nll  = nll_vals[:j_min + 1][::-1]
        left_mu   = mu_grid[:j_min + 1][::-1]
        mu_lo = float(np.interp(target, left_nll, left_mu)) if j_min > 0 else float(mu_grid[0])

        right_nll = nll_vals[j_min:]
        right_mu  = mu_grid[j_min:]
        mu_hi = float(np.interp(target, right_nll, right_mu)) if j_min < len(mu_grid) - 1 else float(mu_grid[-1])

        return mu_fit, mu_lo, mu_hi

    # ------------------------------------------------------------------
    # Standard path: scalar optimizer + binary search
    # ------------------------------------------------------------------
    if cache is None:
        cache = build_cache(config, bins)

    nmax = config['detector']['nmax']
    cic  = config['detector'].get('cic', 0.0)
    mask = counts > 0

    def neg_log_likelihood(mu):
        pmf = _poisson_pmf(mu + cic, nmax)
        bin_probs = pmf @ cache
        p_overflow = max(0.0, 1.0 - float(pmf.sum()))
        bin_probs[-1] = min(1.0, float(bin_probs[-1]) + p_overflow)
        return -np.sum(counts[mask] * np.log(np.clip(bin_probs[mask], 1e-300, None)))

    result = minimize_scalar(neg_log_likelihood, bounds=(mu_min, mu_max), method='bounded',
                             options={'xatol': 1e-5})
    mu_fit = result.x
    nll_fit = result.fun

    # --- 1-sigma uncertainty via delta-log-L = 0.5 ---
    target = nll_fit + 0.5

    def _find_bound(side):
        lo, hi = (mu_min, mu_fit) if side == 'lo' else (mu_fit, mu_max)
        # Binary search
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = neg_log_likelihood(mid)
            if side == 'lo':
                if val < target:
                    hi = mid
                else:
                    lo = mid
            else:
                if val < target:
                    lo = mid
                else:
                    hi = mid
        return 0.5 * (lo + hi)

    mu_lo = _find_bound('lo')
    mu_hi = _find_bound('hi')

    return mu_fit, mu_lo, mu_hi


def fit_flux_map(counts_map, nll_grid, chunk_size=2048):
    """
    Fit every pixel in a 2-D counts map in one vectorised pass.

    For each chunk of pixels, the negative log-likelihood is evaluated on
    the FULL mu_grid in a single matrix product (n_grid x nbins) @
    (nbins x chunk) -- exactly the quantity the scalar fit_flux()'s fast
    path computes per pixel (``nll_vals = -(log_bp_grid @ counts)``), just
    batched over many pixels at once. mu_fit is each pixel's grid argmin;
    mu_lo/mu_hi are found by walking outward from that argmin along the
    (assumed unimodal) NLL curve to the nearest grid points bracketing
    ``nll_min + 0.5`` on each side, then linearly interpolating between
    them -- again mirroring fit_flux()'s own interpolation step exactly.
    If the curve never climbs back up to that target before a grid edge
    (mu_min or mu_max), the corresponding bound is reported as that edge,
    same fallback as fit_flux().

    An earlier version of this function first narrowed the search to a
    small (<=8-point) window per pixel via hierarchical bisection, for
    speed, and only looked for the sigma bounds inside that window. That
    window is sized for how far mu_fit itself needs refining, not for how
    wide the 1-sigma interval is -- at low flux (mu << 1 e-/frame, the
    regime this whole library targets) the 1-sigma interval can span a
    large *fraction* of mu_fit (tens of percent), far wider than an 8-point
    window on a 2000-point grid (~0.5% of the grid per point). The window
    then did not contain the true +0.5 crossing, and the fallback sentinel
    silently forced a "crossing" at the window's edge instead of the grid's
    edge, extrapolating off a shallow local slope -- producing sigma bounds
    inflated by up to an order of magnitude, and occasionally an
    unphysical negative mu_lo. Evaluating the full grid removes the
    narrow-window assumption entirely; at this library's typical grid size
    (n_grid ~2000, nbins ~16-20) it costs a few seconds of extra runtime
    for a full detector's worth of pixels (see the Monte-Carlo validation
    in demo_mc.py / this README), which is a good trade for the fit no
    longer being able to silently misreport its own uncertainty by ~10x.

    Parameters
    ----------
    counts_map : array, shape (..., nbins)
        Per-pixel histogram counts.  Can be any leading shape; the last axis
        must be ``nbins``.  A 2-D detector image with shape ``(ny, nx, nbins)``
        is supported directly; the outputs will have shape ``(ny, nx)``.
    nll_grid   : tuple (mu_grid, log_bin_probs, f_signal, signal_mask)
                 from build_nll_grid()
    chunk_size : int
        Number of pixels processed per matmul. Trades peak memory
        (chunk_size x n_grid floats for the NLL matrix) against the number
        of BLAS calls; 2048 keeps that matrix a few tens of MB at the
        library's default grid size.

    Returns
    -------
    mu_fit : array, leading shape   Best-fit flux [e⁻/frame]
    mu_lo  : array, leading shape   Lower 1-σ bound
    mu_hi  : array, leading shape   Upper 1-σ bound
    """
    mu_grid, log_bp, f_signal, signal_mask = nll_grid
    n_grid = len(mu_grid)

    leading   = counts_map.shape[:-1]
    counts_2d = counts_map.reshape(-1, counts_map.shape[-1]).astype(np.float64)
    n_pix     = counts_2d.shape[0]

    mu_fit = np.empty(n_pix, dtype=np.float64)
    mu_lo  = np.empty(n_pix, dtype=np.float64)
    mu_hi  = np.empty(n_pix, dtype=np.float64)

    for start in range(0, n_pix, chunk_size):
        sl    = slice(start, min(start + chunk_size, n_pix))
        chunk = counts_2d[sl]          # (nc, nbins)
        nc    = chunk.shape[0]
        kidx  = np.arange(nc)

        # Exact NLL on the full grid for every pixel in this chunk, one matmul:
        # (n_grid, nbins) @ (nbins, nc) -> (n_grid, nc).
        nll_full = -(log_bp @ chunk.T)                # (n_grid, nc)
        j_min    = np.argmin(nll_full, axis=0)         # (nc,)
        nll_min  = nll_full[j_min, kidx]
        target   = nll_min + 0.5

        mu_fit[sl] = mu_grid[j_min]

        above = nll_full >= target[np.newaxis, :]      # (n_grid, nc)
        gi    = np.arange(n_grid)[:, np.newaxis]        # (n_grid, 1)
        jm    = j_min[np.newaxis, :]                    # (1, nc)

        # Upper bound: first grid point to the RIGHT of j_min where NLL climbs back
        # up to target. The sentinel guarantees argmax finds *something*; whether
        # that something is a genuine crossing (vs. the sentinel firing because the
        # curve never gets there) is then checked against the real nll_full value.
        right_above = above & (gi > jm)
        right_above[-1, :] = True
        j_hi   = np.argmax(right_above, axis=0)                # (nc,)
        hit_hi = nll_full[j_hi, kidx] >= target
        j_hi   = np.maximum(j_hi, 1)
        nll_b  = nll_full[j_hi - 1, kidx]
        nll_a  = nll_full[j_hi, kidx]
        frac   = (target - nll_b) / np.where(nll_a != nll_b, nll_a - nll_b, 1.0)
        mu_hi_chunk = mu_grid[j_hi - 1] + frac * (mu_grid[j_hi] - mu_grid[j_hi - 1])
        # No crossing before the grid's own upper edge (mu_max): the true bound lies
        # beyond what this grid represents, so report the edge -- same fallback as
        # fit_flux()'s "j_min < len(mu_grid) - 1 else mu_grid[-1]".
        mu_hi[sl] = np.where(hit_hi, mu_hi_chunk, mu_grid[-1])

        # Lower bound: mirror image, walking left from j_min.
        left_above       = above & (gi < jm)
        left_above[0, :] = True
        j_lo   = (n_grid - 1) - np.argmax(left_above[::-1, :], axis=0)
        hit_lo = nll_full[j_lo, kidx] >= target
        j_lo   = np.minimum(j_lo, n_grid - 2)
        nll_b  = nll_full[j_lo, kidx]
        nll_a  = nll_full[j_lo + 1, kidx]
        frac   = (target - nll_b) / np.where(nll_a != nll_b, nll_a - nll_b, 1.0)
        mu_lo_chunk = mu_grid[j_lo] + frac * (mu_grid[j_lo + 1] - mu_grid[j_lo])
        mu_lo[sl] = np.where(hit_lo, mu_lo_chunk, mu_grid[0])

    return mu_fit.reshape(leading), mu_lo.reshape(leading), mu_hi.reshape(leading)
