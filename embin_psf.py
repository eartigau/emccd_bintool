#!/usr/bin/env python3
"""
embin_psf.py -- joint Moffat PSF photometry across the chunks
=============================================================

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

WHAT THIS DOES
--------------
`run_chunks.py` already produces a light curve, by summing the flux map inside a
fixed aperture around each source. That works right up to the moment two sources
are closer together than the aperture, which is exactly the situation in this
field: TOI-1452 has a companion 3.1 arcsec away, about seven pixels, and any
aperture wide enough to hold one star holds a piece of the other.

This script replaces the aperture with a fit. It

  1. finds the stars on `astrometry_stack.fits`, the drift-corrected stack;
  2. fits ONE isotropic Moffat to all of them at once on that deep stack -- a
     single FWHM and a single beta shared by every star, each star free in
     position and flux -- which fixes the reference positions AND measures beta;
  3. then, for each chunk of `chunk_summary.fits`, fits again with beta HELD at
     the stack value: the FWHM and a rigid (dx, dy) shift of the whole field are
     free and shared, and every star gets its own flux, all solved
     simultaneously against that chunk's own FLUX_ERR map;
  4. writes a table of time, FWHM in arcsec, beta, and a flux with an error bar
     for every star.

Step 3 holds beta on purpose, and fit_bin() has the argument: beta lives in the
wings, a 64-frame chunk has no wings above the noise, and a free beta simply
rails. The deep stack does have wings, so beta is measured once, where it can
be, and reused everywhere.

    python embin_psf.py                          # everything from the YAML
    python embin_psf.py --stack other_stack.fits
    python embin_psf.py --no-plot

WHY THE BLEND IS NOT A PROBLEM HERE
-----------------------------------
Because the stars are fitted TOGETHER, not one after another. Hold the shape and
the positions fixed and the model is linear in the fluxes:

    d(p) = sum_j  F_j * M_j(p)  +  background,

so the fluxes come from a weighted linear least-squares solve whose normal
matrix is

    (M^T W M)_{jk} = sum_p  M_j(p) M_k(p) / sigma(p)^2 .

The off-diagonal element j,k IS the cross-term: the overlap integral of star j's
profile with star k's, weighted by the noise. It is not neglected, and it is not
approximated -- it is a matrix element that the solve inverts. Its consequences
come out for free:

  * the fitted fluxes are the ones that jointly explain the blended image,
    rather than each star's flux plus a share of its neighbour's wing;
  * the inverse of that same matrix is the covariance, so the error bar on
    TOI-1452 A already carries the penalty for not knowing exactly how much of
    the light belonged to B, and the A-B covariance is a number this script
    reports rather than an effect it hopes is small.

For a pair at 3.1 arcsec with 1.4 arcsec seeing, that penalty is small but real:
the log prints it, as the ratio of each blended star's error bar to what it
would have been had the neighbour not existed.

WHY THE SHAPE IS SHARED
-----------------------
One FWHM for the whole field, per chunk. The PSF is set by the atmosphere and
the optics, which do not know which star they are blurring, so letting each star
have its own width would be fitting noise -- and worse, it would let the blended
pair trade width against flux with nothing to stop it. Sharing the shape is what
makes the deblend well posed: the faint companion's profile is pinned by every
star in the field that is not blended with anything.

WHAT `flux` MEANS
-----------------
The Moffat is normalised to unit volume, so the fitted coefficient is the TOTAL
flux of the star in e-/frame, integrated to infinite radius -- not a peak height
and not the content of an aperture. That is the quantity that stays constant
when the seeing changes, which is the entire point of doing this.
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import _resolve, _timestamp, load_config, log            # noqa: E402
from embin_report import write_psf_report                            # noqa: E402


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------
def moffat_alpha(fwhm, beta):
    """The Moffat core width alpha that gives this FWHM at this beta.

    The profile is (1 + (r/alpha)^2)^(-beta), whose half-maximum radius is
    alpha * sqrt(2^(1/beta) - 1), so FWHM = 2 alpha sqrt(2^(1/beta) - 1).
    Parameterising by FWHM rather than by alpha matters for the fit: FWHM is
    close to orthogonal to beta, while alpha and beta are strongly degenerate
    and a sampler or an optimiser walking in (alpha, beta) crawls along a
    banana instead of converging.
    """
    return fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def moffat_columns(xs, ys, fwhm, beta, yy, xx, oversample=3):
    """One unit-flux Moffat per star, evaluated on the given pixel list.

    Returns an (n_star, n_pixel) array. Each row integrates to 1 over the whole
    plane, so the linear coefficient that multiplies it is the star's total
    flux.

    The profile is averaged over `oversample`^2 sub-positions inside each pixel
    rather than sampled at its centre. With a 3-pixel FWHM the centre value
    misestimates the peak pixel by several per cent, which is far larger than
    the photometric errors here and, worse, is a function of where the star
    falls inside its pixel -- so it would show up as a spurious flux wobble
    correlated with the drift.
    """
    alpha = moffat_alpha(fwhm, beta)
    a2 = alpha * alpha
    norm = (beta - 1.0) / (np.pi * a2)
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5

    cols = np.empty((len(xs), yy.size), dtype=np.float64)
    for j, (x0, y0) in enumerate(zip(xs, ys)):
        acc = np.zeros(yy.size, dtype=np.float64)
        for dy in sub:
            for dx in sub:
                r2 = (xx + dx - x0) ** 2 + (yy + dy - y0) ** 2
                acc += norm * (1.0 + r2 / a2) ** (-beta)
        cols[j] = acc / (oversample * oversample)
    return cols


def background_columns(yy, xx, kind):
    """The sky model, as extra linear columns.

    'constant' is one column of ones. 'plane' adds x and y, for a field with a
    gradient. Both are solved in the same linear stage as the fluxes, so the
    error bars already account for not knowing the sky exactly.
    """
    if kind == 'plane':
        return np.vstack([np.ones_like(xx),
                          (xx - xx.mean()) / max(xx.ptp(), 1.0),
                          (yy - yy.mean()) / max(yy.ptp(), 1.0)])
    return np.ones((1, xx.size))


# ---------------------------------------------------------------------------
# The linear stage
# ---------------------------------------------------------------------------
def solve_fluxes(cols, data, inv_sigma):
    """Weighted linear least squares for the fluxes and the background.

    `cols` is (n_col, n_pixel), `data` and `inv_sigma` are (n_pixel,).
    Returns (coefficients, covariance, weighted residual vector).

    The covariance is the inverse of the normal matrix, whose off-diagonal terms
    are the star-to-star overlap integrals: this is where the blend is handled.
    Cholesky first because the matrix is symmetric positive definite whenever
    the stars are distinguishable at all; the pseudo-inverse is the fallback for
    the degenerate case of two profiles that have collapsed onto each other.
    """
    A = (cols * inv_sigma).T                     # (n_pixel, n_col)
    b = data * inv_sigma
    ata = A.T @ A
    atb = A.T @ b
    try:
        cov = np.linalg.inv(ata)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(ata)
    coef = cov @ atb
    return coef, cov, A @ coef - b


def blend_penalty(cols, inv_sigma, i, j):
    """How much star i's error bar grows because star j is next to it.

    The ratio of the joint two-star error on i to the error i would have if j
    were not there. 1.00 means the neighbour costs nothing; 1.5 means the blend
    inflates the error bar by half. This is the cross-term, as a number.
    """
    A = (cols[[i, j]] * inv_sigma).T
    joint = np.linalg.inv(A.T @ A)[0, 0]
    alone = 1.0 / float(np.sum((cols[i] * inv_sigma) ** 2))
    return float(np.sqrt(joint / alone))


# ---------------------------------------------------------------------------
# Where to fit
# ---------------------------------------------------------------------------
def source_mask(shape, xs, ys, radius):
    """Union of square boxes around the stars.

    Fitting 436,000 pixels to place 17 stars is a waste; fitting each star in
    its own private box would break the blend, because a pixel between the two
    TOI-1452 components belongs to both. The union of the boxes is the honest
    middle: every pixel that any star reaches, fitted once, with every star that
    reaches it contributing to it.
    """
    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    r = int(np.ceil(radius))
    for x0, y0 in zip(xs, ys):
        i0, i1 = max(0, int(y0) - r), min(ny, int(y0) + r + 1)
        j0, j1 = max(0, int(x0) - r), min(nx, int(x0) + r + 1)
        mask[i0:i1, j0:j1] = True
    return mask


# ---------------------------------------------------------------------------
# Finding the stars
# ---------------------------------------------------------------------------
def find_stars(image, threshold_sigma, fwhm_guess, max_stars):
    """Detect stars on the stack with DAOStarFinder, brightest first."""
    try:
        from photutils.detection import DAOStarFinder
    except ImportError:
        log('photutils is not installed: pip install photutils', 'error')
        sys.exit(1)

    med = float(np.median(image))
    sigma = float(np.median(np.abs(image - med)) * 1.4826)
    found = DAOStarFinder(fwhm=fwhm_guess, threshold=threshold_sigma * sigma)(
        image - med)
    if found is None or not len(found):
        log(f'no source above {threshold_sigma:g} sigma on the stack', 'error')
        sys.exit(1)
    found.sort('flux')
    found.reverse()
    if max_stars and len(found) > max_stars:
        log(f'{len(found)} sources found, keeping the {max_stars} brightest',
            'warn')
        found = found[:max_stars]
    xs = np.asarray(found['xcentroid'], dtype=np.float64)
    ys = np.asarray(found['ycentroid'], dtype=np.float64)
    log(f'{len(xs)} stars on the stack above {threshold_sigma:g} sigma '
        f'(sky {med:.4f}, robust sigma {sigma:.4f} e-/frame)', 'value')
    return xs, ys, med, sigma


def find_blends(xs, ys, fwhm, factor):
    """Pairs closer than `factor` FWHM, i.e. the ones whose wings overlap."""
    pairs = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < factor * fwhm:
                pairs.append((i, j, float(d)))
    return pairs


# ---------------------------------------------------------------------------
# The two fits
# ---------------------------------------------------------------------------
def fit_stack(image, sigma_map, xs, ys, cfg_psf):
    """Shared FWHM and beta, free positions, free fluxes, on the deep stack.

    This is the only fit in which the star positions move. It exists to pin
    them: the per-chunk data is far shallower and would let a faint star's
    centroid wander off into the noise, taking its flux with it.
    """
    oversample = int(cfg_psf.get('oversample', 3))
    bkg_kind = cfg_psf.get('background', 'constant')
    fwhm0 = float(cfg_psf.get('fwhm_guess', 3.5))
    beta0 = float(cfg_psf.get('beta_guess', 3.0))
    radius = float(cfg_psf.get('fit_radius', 4.0)) * fwhm0

    mask = source_mask(image.shape, xs, ys, radius)
    iy, ix = np.nonzero(mask)
    yy, xx = iy.astype(np.float64), ix.astype(np.float64)
    data = image[iy, ix].astype(np.float64)
    inv_sigma = 1.0 / sigma_map[iy, ix].astype(np.float64)
    bkg = background_columns(yy, xx, bkg_kind)
    n = len(xs)
    log(f'stack fit: {n} stars, {data.size} pixels inside {radius:.1f} px boxes, '
        f'{2 * n + 2} shape/position parameters, {n + len(bkg)} linear')

    def residual(p):
        if not (0.5 < p[0] < 30.0 and 1.05 < p[1] < 30.0):
            return np.full(data.size, 1e6)
        cols = np.vstack([moffat_columns(p[2:2 + n], p[2 + n:2 + 2 * n],
                                         p[0], p[1], yy, xx, oversample), bkg])
        return solve_fluxes(cols, data, inv_sigma)[2]

    p0 = np.concatenate([[fwhm0, beta0], xs, ys])
    res = least_squares(residual, p0, x_scale='jac', xtol=1e-10, ftol=1e-10)
    fwhm, beta = float(res.x[0]), float(res.x[1])
    xs_fit, ys_fit = res.x[2:2 + n].copy(), res.x[2 + n:2 + 2 * n].copy()
    cols = np.vstack([moffat_columns(xs_fit, ys_fit, fwhm, beta, yy, xx,
                                     oversample), bkg])
    coef, _, r = solve_fluxes(cols, data, inv_sigma)
    chi2 = float(np.sum(r ** 2)) / (data.size - (2 * n + 2 + len(coef)))

    # Beta's error bar matters more than usual: this is the ONE place it is
    # measured, and every chunk then inherits it.
    try:
        _, s, vt = np.linalg.svd(res.jac, full_matrices=False)
        good = s > 1e-10 * s[0]
        err = np.sqrt(np.diag((vt[good].T / s[good] ** 2) @ vt[good]))
        fwhm_err, beta_err = float(err[0]), float(err[1])
    except np.linalg.LinAlgError:
        fwhm_err = beta_err = np.nan

    log(f'stack: FWHM = {fwhm:.4f} +/- {fwhm_err:.4f} px, '
        f'beta = {beta:.4f} +/- {beta_err:.4f}, chi2/dof = {chi2:.3f}', 'value')
    moved = np.hypot(xs_fit - xs, ys_fit - ys)
    log(f'positions moved {moved.mean():.3f} px on average, '
        f'{moved.max():.3f} px at most, from the detection centroids', 'value')
    return xs_fit, ys_fit, fwhm, beta, fwhm_err, beta_err, coef[:n], float(coef[n])


def fit_bin(flux, err, xs, ys, fwhm0, beta, dxy0, cfg_psf):
    """One chunk: shared FWHM and rigid shift, fixed beta; one flux per star.

    THE POSITIONS ARE NOT FREE. Over three seconds the field translates -- that
    is the drift the astrometry already measured -- but it does not rearrange
    itself, so two numbers describe the whole motion. Fitting 2N positions
    instead would add degrees of freedom the blend has no information to
    constrain, and the pair would happily slide into each other.

    BETA IS NOT FREE EITHER, and this is the one choice in this script that is
    worth arguing about. Beta controls the wings, and in a 64-frame chunk the
    wings are where the per-pixel flux estimator is at its worst: a pixel a few
    FWHM out sees well under one electron for the whole chunk, its likelihood
    barely turns over, and its fitted flux is biased low. A free beta reads that
    bias as a genuinely compact profile and runs off to the upper bound -- on
    this dataset it hits 30 in ten chunks out of fifteen, with chi2 still falling
    monotonically, which is the signature of a parameter the data cannot
    determine rather than of a wide-winged PSF.

    Beta is therefore measured ONCE on the deep stack, where fifteen times the
    frames put real signal in the wings, and held there. FWHM stays free per
    chunk, which is what actually varies with the seeing, and is the number this
    script is asked for. Set `psf.beta_free: true` to let it float per chunk
    anyway; the fit will tell you where it railed.
    """
    oversample = int(cfg_psf.get('oversample', 3))
    bkg_kind = cfg_psf.get('background', 'constant')
    radius = float(cfg_psf.get('fit_radius', 4.0)) * fwhm0

    mask = source_mask(flux.shape, xs + dxy0[0], ys + dxy0[1], radius + 3)
    iy, ix = np.nonzero(mask)
    yy, xx = iy.astype(np.float64), ix.astype(np.float64)
    data = flux[iy, ix].astype(np.float64)
    sig = err[iy, ix].astype(np.float64)

    # A pixel with a zero or non-finite error would carry infinite weight. The
    # flux fitter can produce those where the likelihood never turned over, so
    # they are given the median error rather than silently dominating the fit.
    bad = ~np.isfinite(sig) | (sig <= 0)
    if bad.any():
        sig = np.where(bad, np.median(sig[~bad]), sig)
    inv_sigma = 1.0 / sig
    bkg = background_columns(yy, xx, bkg_kind)
    n = len(xs)
    beta_free = bool(cfg_psf.get('beta_free', False))
    beta_lo, beta_hi = 1.05, 30.0

    # p is (fwhm, dx, dy) with beta held, or (fwhm, dx, dy, beta) with it free.
    def unpack(p):
        return (p[0], p[3] if beta_free else beta, p[1], p[2])

    def columns(p):
        fwhm, b, dx, dy = unpack(p)
        return np.vstack([moffat_columns(xs + dx, ys + dy, fwhm, b,
                                         yy, xx, oversample), bkg])

    def residual(p):
        fwhm, b, _, _ = unpack(p)
        if not (0.5 < fwhm < 30.0 and beta_lo < b < beta_hi):
            return np.full(data.size, 1e6)
        return solve_fluxes(columns(p), data, inv_sigma)[2]

    p0 = [fwhm0, dxy0[0], dxy0[1]] + ([beta] if beta_free else [])
    res = least_squares(residual, np.array(p0, dtype=np.float64),
                        x_scale='jac', xtol=1e-10, ftol=1e-10)
    fwhm_fit, beta_fit, dx_fit, dy_fit = unpack(res.x)

    cols = columns(res.x)
    coef, cov, r = solve_fluxes(cols, data, inv_sigma)
    n_par = len(res.x) + len(coef)
    dof = data.size - n_par
    chi2_red = float(np.sum(r ** 2)) / dof

    # Shape errors from the Gauss-Newton approximation to the Hessian. The
    # residuals are already divided by sigma, so no extra scaling is needed.
    try:
        _, s, vt = np.linalg.svd(res.jac, full_matrices=False)
        good = s > 1e-10 * s[0]
        shape_err = np.sqrt(np.diag((vt[good].T / s[good] ** 2) @ vt[good]))
    except np.linalg.LinAlgError:
        shape_err = np.full(len(res.x), np.nan)
    fwhm_err = float(shape_err[0])
    beta_err = float(shape_err[3]) if beta_free else np.nan

    # A parameter sitting on its bound has no meaningful error bar, and the
    # Jacobian there is flat enough to produce an absurdly small one. Say so
    # rather than printing 1e-30.
    railed = beta_free and (beta_fit <= beta_lo * 1.001
                            or beta_fit >= beta_hi * 0.999)
    if railed:
        beta_err = np.nan

    flux_err = np.sqrt(np.clip(np.diag(cov)[:n], 0, None))
    # The star-to-star correlation, which for the blended pair is the number
    # that says how much of one star's flux is trading against the other's.
    d = np.sqrt(np.clip(np.diag(cov)[:n], 1e-300, None))
    corr = cov[:n, :n] / np.outer(d, d)

    return {
        'fwhm': float(fwhm_fit), 'beta': float(beta_fit),
        'dx': float(dx_fit), 'dy': float(dy_fit),
        'fwhm_err': fwhm_err, 'beta_err': beta_err, 'beta_railed': railed,
        'beta_free': beta_free,
        'flux': coef[:n], 'flux_err': flux_err,
        # The error bar the flux map implies, and the same thing rescaled so the
        # fit's own chi2 comes out at 1. On this dataset chi2/dof sits near 0.78,
        # i.e. FLUX_ERR is about 12 % conservative, so the scaled column is the
        # tighter of the two. Which to believe is a judgement about the flux
        # map, not about this fit, so both are written out.
        'flux_err_scaled': flux_err * np.sqrt(chi2_red),
        'background': float(coef[n]), 'corr': corr,
        'chi2_red': chi2_red, 'dof': dof, 'n_pixels': int(data.size),
        'cols': cols, 'inv_sigma': inv_sigma,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def stack_error_map(summary_path, chunks, shape):
    """The stack's own error map, rebuilt from the per-chunk ones.

    `astrometry_stack.fits` carries the stacked image but not its uncertainty,
    and the stack fit needs weights like any other. The stack is the mean of the
    chunks after an integer shift (dx_stack, dy_stack), so its variance is the
    mean of the chunk variances put through the same shift, divided by the
    number of chunks.
    """
    err_cube = fits.getdata(summary_path, extname='FLUX_ERR').astype(np.float64)
    n = err_cube.shape[0]
    var = np.zeros(shape, dtype=np.float64)
    for k in range(n):
        shifted = np.roll(err_cube[k] ** 2,
                          (int(chunks['dy_stack'][k]), int(chunks['dx_stack'][k])),
                          axis=(0, 1))
        var += shifted
    return np.sqrt(var) / n


def star_metadata(stack_path, xs, ys, stack_flux):
    """RA/Dec from the stack WCS, and a Gaia magnitude where one matches."""
    hdr = fits.getheader(stack_path, 0)
    tab = Table({'star': np.arange(len(xs)),
                 'x': xs, 'y': ys, 'stack_flux': stack_flux})
    try:
        wcs = WCS(hdr)
        ra, dec = wcs.all_pix2world(xs, ys, 0)
        tab['ra'], tab['dec'] = ra, dec
        tab['ra'].unit = tab['dec'].unit = 'deg'
    except Exception as exc:
        log(f'no usable WCS on the stack ({exc}); no sky coordinates', 'warn')

    gmag = np.full(len(xs), np.nan)
    try:
        match = fits.getdata(stack_path, extname='GAIAMTCH')
        for i, (x0, y0) in enumerate(zip(xs, ys)):
            d = np.hypot(match['x'] - x0, match['y'] - y0)
            if d.min() < 2.0:
                gmag[i] = match['Gmag'][np.argmin(d)]
    except Exception:
        pass
    tab['Gmag'] = gmag
    return tab


def absolute_epoch(outdir, chunks):
    """UTC of the first frame, if a chunk cube still has its header table.

    `chunk_summary.fits` keeps time as seconds since the start of the sequence,
    which is what the light curve needs, but a night's worth of chunks is easier
    to place if the table also carries a wall-clock date. RETD_TOD in the cube's
    HEADERS extension is the camera's time of day for each frame.
    """
    for name in (str(chunks['file'][0]), str(chunks['file'][0]) + '.gz'):
        path = os.path.join(outdir, name)
        if not os.path.exists(path):
            continue
        try:
            rows = fits.getdata(path, extname='HEADERS')
            if 'RETD_TOD' in rows.columns.names:
                from astropy.time import Time
                return Time(str(rows['RETD_TOD'][0]).replace(' ', 'T'),
                            format='isot', scale='utc')
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def build_table(results, chunks, stars, pixscale, epoch):
    """One row per chunk: the time, the seeing, and every star's flux."""
    n = len(stars)
    tab = Table()
    tab['bin'] = np.asarray(chunks['chunk'], dtype=np.int32)
    tab['nframes'] = np.asarray(chunks['nframes'], dtype=np.int32)
    for col in ('t_start', 't_mid', 't_end'):
        tab[col] = np.asarray(chunks[col], dtype=np.float64)
        tab[col].unit = 's'
    if epoch is not None:
        from astropy.time import TimeDelta
        stamps = epoch + TimeDelta(np.asarray(tab['t_mid']), format='sec')
        tab['utc'] = stamps.isot
        tab['mjd'] = stamps.mjd

    tab['fwhm'] = np.array([r['fwhm'] for r in results]) * pixscale
    tab['fwhm_err'] = np.array([r['fwhm_err'] for r in results]) * pixscale
    tab['fwhm'].unit = tab['fwhm_err'].unit = 'arcsec'
    tab['fwhm_pix'] = np.array([r['fwhm'] for r in results])
    tab['beta'] = np.array([r['beta'] for r in results])
    tab['beta_err'] = np.array([r['beta_err'] for r in results])
    # NaN in beta_err is not a failure when beta was held: it means the value
    # came from the deep stack, and the column is constant on purpose.
    tab['beta_free'] = np.array([r['beta_free'] for r in results])
    tab['dx'] = np.array([r['dx'] for r in results])
    tab['dy'] = np.array([r['dy'] for r in results])
    tab['background'] = np.array([r['background'] for r in results])
    tab['chi2_red'] = np.array([r['chi2_red'] for r in results])

    flux = np.array([r['flux'] for r in results])
    ferr = np.array([r['flux_err'] for r in results])
    fscl = np.array([r['flux_err_scaled'] for r in results])
    for j in range(n):
        tab[f'flux_{j:02d}'] = flux[:, j]
        tab[f'flux_err_{j:02d}'] = ferr[:, j]
        tab[f'flux_err_scaled_{j:02d}'] = fscl[:, j]
        for c in (f'flux_{j:02d}', f'flux_err_{j:02d}',
                  f'flux_err_scaled_{j:02d}'):
            tab[c].unit = 'e-/frame'
    return tab


def diagnostic_figure(tab, results, stars, blends, path):
    """Seeing, beta and the light curves, on one page."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_show = min(6, len(stars))
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                           gridspec_kw={'hspace': 0.08})
    t = tab['t_mid']

    beta_free = bool(np.any(tab['beta_free']))
    ax[0].errorbar(t, tab['fwhm'], yerr=tab['fwhm_err'], fmt='o-', ms=4, lw=1,
                   color='0.2')
    ax[0].set_ylabel('FWHM [arcsec]')
    ax[0].set_title('shared Moffat shape, one fit per chunk'
                    + ('' if beta_free else
                       rf'   ($\beta = {tab["beta"][0]:.3f}$, held at the '
                       'deep-stack value)'))

    if beta_free:
        ax[1].errorbar(t, tab['beta'], yerr=tab['beta_err'], fmt='o-', ms=4,
                       lw=1, color='steelblue')
        ax[1].set_ylabel(r'Moffat $\beta$')
    else:
        # A flat line is not worth a panel. With beta held, the other thing
        # every chunk shares is the rigid shift, which is the field drifting
        # across the detector -- worth seeing, since it is what the fixed
        # positions are being corrected by.
        ax[1].plot(t, tab['dx'], 'o-', ms=4, lw=1, color='steelblue', label='dx')
        ax[1].plot(t, tab['dy'], 's-', ms=4, lw=1, color='darkorange', label='dy')
        ax[1].set_ylabel('rigid shift [px]')
        ax[1].legend(fontsize=8, frameon=False)

    blended = {i for i, j, _ in blends} | {j for i, j, _ in blends}
    for j in range(n_show):
        f = tab[f'flux_{j:02d}']
        e = tab[f'flux_err_{j:02d}']
        norm = np.median(f)
        ax[2].errorbar(t, f / norm, yerr=e / abs(norm), ls='-', ms=4, lw=1,
                       marker='s' if j in blended else 'o',
                       label=f'star {j}' + (' (blended)' if j in blended else ''))
    ax[2].set_ylabel('flux / median')
    ax[2].set_xlabel('time since the first frame [s]')
    ax[2].legend(fontsize=8, ncol=2, frameon=False)

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    log(f'diagnostic plot written to {path}')


def blend_page(image, xs, ys, fwhm, beta, fluxes, bkg, pair, pixscale,
               oversample=3):
    """The blended pair: data, model, residual, and the two profiles separated.

    This is the figure that either convinces you the deblend worked or does not.
    The residual panel is the test -- if the joint fit has mis-shared the light
    between the two components, it leaves a dipole there, bright on one star and
    dark on the other, which no amount of arguing about covariance matrices will
    explain away.

    The last panel is the same cut through the pair with each component drawn on
    its own, which is what "deblended" actually means here: two numbers, one per
    star, that add up to the thing that was measured.
    """
    import matplotlib.pyplot as plt

    i, j = pair
    cx, cy = 0.5 * (xs[i] + xs[j]), 0.5 * (ys[i] + ys[j])
    half = int(np.ceil(3.0 * fwhm + abs(ys[i] - ys[j]) + abs(xs[i] - xs[j])))
    y0, y1 = max(0, int(cy) - half), min(image.shape[0], int(cy) + half + 1)
    x0, x1 = max(0, int(cx) - half), min(image.shape[1], int(cx) + half + 1)
    cut = image[y0:y1, x0:x1]

    gy, gx = np.mgrid[y0:y1, x0:x1]
    yy, xx = gy.ravel().astype(float), gx.ravel().astype(float)
    cols = moffat_columns(xs, ys, fwhm, beta, yy, xx, oversample)
    model = (fluxes @ cols + bkg).reshape(cut.shape)
    resid = cut - model

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6),
                           gridspec_kw={'width_ratios': [1, 1, 1, 1.35]})
    vmax = float(np.nanmax(cut))
    kw = dict(origin='lower', cmap='inferno', vmin=0, vmax=vmax,
              extent=[x0, x1, y0, y1])
    for a, im, title in ((ax[0], cut, 'data'), (ax[1], model, 'model')):
        h = a.imshow(im, **kw)
        a.set_title(title, fontsize=10)
        fig.colorbar(h, ax=a, fraction=0.046)
    lim = float(np.nanmax(np.abs(resid)))
    h = ax[2].imshow(resid, origin='lower', cmap='coolwarm', vmin=-lim, vmax=lim,
                     extent=[x0, x1, y0, y1])
    ax[2].set_title('residual', fontsize=10)
    fig.colorbar(h, ax=ax[2], fraction=0.046)
    for a in ax[:3]:
        a.plot([xs[i], xs[j]], [ys[i], ys[j]], 'w+', ms=9, mew=1.4)
        a.set_xticks([])
        a.set_yticks([])

    # A cut along the line joining the two stars, with each component alone.
    d = np.hypot(xs[j] - xs[i], ys[j] - ys[i])
    ux, uy = (xs[j] - xs[i]) / d, (ys[j] - ys[i]) / d
    s = np.linspace(-2.5 * fwhm, d + 2.5 * fwhm, 400)
    px, py = xs[i] + ux * s, ys[i] + uy * s
    prof = moffat_columns(xs, ys, fwhm, beta, py, px, oversample)
    ax[3].plot(s * pixscale, fluxes @ prof + bkg, color='0.15', lw=1.6,
               label='both + sky')
    for k, lab in ((i, f'star {i}'), (j, f'star {j}')):
        ax[3].plot(s * pixscale, fluxes[k] * prof[k] + bkg, lw=1.2, ls='--',
                   label=lab)
    ax[3].axhline(bkg, color='0.6', lw=0.8)
    ax[3].set_xlabel('along the pair [arcsec]')
    ax[3].set_ylabel(r'e$^-$/frame')
    ax[3].set_title(f'separation {d * pixscale:.2f}", '
                    f'FWHM {fwhm * pixscale:.2f}"', fontsize=10)
    ax[3].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def write_region_file(path, stars, blends, fwhm, pixscale):
    """A DS9 region file labelling every fitted star by its index.

    Written in IMAGE coordinates, not sky, and deliberately so: the numbers in
    the photometry table are indices into THIS fit, and the point of the file is
    to let you load `astrometry_stack.fits` in DS9, drop the regions on top and
    see which circle is `flux_07`. Image coordinates keep that mapping exact
    even if the WCS is later re-solved.

    DS9 counts pixels from 1, numpy from 0, hence the +1 on the way out.
    """
    blended = {i for i, j, _ in blends} | {j for i, j, _ in blends}
    radius = max(2.0 * fwhm, 4.0)
    lines = [
        '# Region file format: DS9 version 4.1',
        '# fitted star positions from embin_psf.py',
        '# radius is 2 x FWHM; green = isolated, red = member of a blended pair',
        'global color=green dashlist=8 3 width=1 '
        'font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 '
        'edit=1 move=1 delete=1 include=1 source=1',
        'image',
    ]
    for k in range(len(stars)):
        colour = ' color=red' if k in blended else ''
        lines.append(f'circle({stars["x"][k] + 1:.3f},{stars["y"][k] + 1:.3f},'
                     f'{radius:.3f}) # text={{{k}}}{colour}')
    # A dashed box around each blended pair, so the one place the fit is doing
    # something non-obvious is the one place the eye is drawn to.
    for i, j, d in blends:
        cx = 0.5 * (stars['x'][i] + stars['x'][j]) + 1
        cy = 0.5 * (stars['y'][i] + stars['y'][j]) + 1
        size = d + 4 * fwhm
        lines.append(f'box({cx:.3f},{cy:.3f},{size:.2f},{size:.2f},0) # '
                     f'color=red dash=1 text={{{i}+{j}: {d * pixscale:.2f}"}}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'DS9 regions written to {path}')


def write_star_yaml(path, stars, tab, blends, meta):
    """The star list as YAML: position, sky coordinates, Gaia mag, flux.

    The FITS table is the machine-readable product and the ECSV is the readable
    one, but neither is something you can paste into a configuration or an
    e-mail. This is that: one block per star, keyed by the same index the
    photometry columns use.
    """
    import yaml

    blended = {i for i, j, _ in blends} | {j for i, j, _ in blends}
    partner = {}
    for i, j, d in blends:
        partner.setdefault(i, []).append(j)
        partner.setdefault(j, []).append(i)

    entries = {}
    for k in range(len(stars)):
        flux = np.asarray(tab[f'flux_{k:02d}'], dtype=float)
        err = np.asarray(tab[f'flux_err_{k:02d}'], dtype=float)
        med = float(np.median(flux))
        rec = {
            'x': round(float(stars['x'][k]), 3),
            'y': round(float(stars['y'][k]), 3),
            'ra': (round(float(stars['ra'][k]), 7)
                   if 'ra' in stars.colnames else None),
            'dec': (round(float(stars['dec'][k]), 7)
                    if 'dec' in stars.colnames else None),
            'gmag': (None if not np.isfinite(stars['Gmag'][k])
                     else round(float(stars['Gmag'][k]), 4)),
            'median_flux': round(med, 5),
            'median_flux_err': round(float(np.median(err)), 5),
            'scatter_percent': (round(float(np.std(flux) / abs(med) * 100), 2)
                                if med else None),
            'blended_with': partner.get(k, []),
        }
        entries[f'star_{k:02d}'] = rec

    header = (
        '# Stars fitted by embin_psf.py, in the order their fluxes appear in\n'
        '# psf_photometry.fits: star_NN here is flux_NN there, and the same\n'
        '# number labels the circle in psf_stars.reg.\n'
        '#\n'
        '#   x, y             pixel position on the deep stack, 0-indexed\n'
        '#   ra, dec          degrees, ICRS, from the stack WCS\n'
        '#   gmag             Gaia DR3 G, null where no source matched\n'
        '#                    within 2 px\n'
        '#   median_flux      e-/frame, total flux of the fitted Moffat,\n'
        '#                    median over the chunks\n'
        '#   scatter_percent  standard deviation of the light curve, as a\n'
        '#                    percentage of its own median\n'
        '#   blended_with     stars whose wings overlap this one; their\n'
        '#                    fluxes were solved jointly\n'
        f'#\n# {meta["n_star"]} stars, {meta["n_chunk"]} chunks, '
        f'generated {meta["stamp"]}\n\n')
    with open(path, 'w') as fh:
        fh.write(header)
        yaml.safe_dump(entries, fh, sort_keys=True, default_flow_style=False)
    log(f'star properties written to {path}')


def finder_page(image, stars, blends, fwhm, pixscale):
    """A finder chart: the deep stack with every fitted star circled and numbered.

    Same information as the region file, for anyone who is not going to open
    DS9. The stretch is an arcsinh on the sky-subtracted stack, which is the
    only way to show a 17th-magnitude detection and a 13th-magnitude star on the
    same greyscale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    blended = {i for i, j, _ in blends} | {j for i, j, _ in blends}
    sky = float(np.median(image))
    scale = float(np.median(np.abs(image - sky)) * 1.4826)
    shown = np.arcsinh((image - sky) / max(scale, 1e-12) / 3.0)

    ny, nx = image.shape
    fig, ax = plt.subplots(figsize=(12, 12 * ny / nx + 1.2))
    ax.imshow(shown, origin='lower', cmap='Greys_r',
              vmin=np.percentile(shown, 5), vmax=np.percentile(shown, 99.98))

    radius = max(2.5 * fwhm, 6.0)
    for k in range(len(stars)):
        x, y = float(stars['x'][k]), float(stars['y'][k])
        colour = 'crimson' if k in blended else 'springgreen'
        ax.add_patch(Circle((x, y), radius, fill=False, ec=colour, lw=1.1))
        ax.text(x + radius * 1.25, y + radius * 0.55, str(k), color=colour,
                fontsize=9, weight='bold',
                path_effects=None, ha='left', va='bottom')
    for i, j, d in blends:
        cx = 0.5 * (stars['x'][i] + stars['x'][j])
        cy = 0.5 * (stars['y'][i] + stars['y'][j])
        size = d + 6 * fwhm
        ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size,
                               fill=False, ec='crimson', lw=1.0, ls='--'))
        ax.text(cx, cy - size / 2 - 6, f'{i}+{j}: {d * pixscale:.2f}"',
                color='crimson', fontsize=9, ha='center', va='top')

    # A scale bar beats an axis label nobody reads.
    bar = 10.0 / pixscale
    ax.plot([nx * 0.02, nx * 0.02 + bar], [ny * 0.04] * 2, '-', color='white',
            lw=2.5)
    ax.text(nx * 0.02 + bar / 2, ny * 0.05, '10"', color='white', fontsize=10,
            ha='center', va='bottom')

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    ax.set_title(f'{len(stars)} fitted stars on the deep stack   '
                 f'(green isolated, red blended; numbers are the flux_NN '
                 f'columns)', fontsize=11)
    fig.tight_layout()
    return fig


def write_output(path, tab, stars, results, meta):
    """The table, plus the stars it refers to and the per-chunk correlations."""
    corr = np.array([r['corr'] for r in results], dtype=np.float32)
    primary = fits.PrimaryHDU()
    for key, value, comment in meta:
        primary.header[key] = (value, comment)
    hdus = [primary,
            fits.BinTableHDU(tab.as_array(), name='PHOT'),
            fits.BinTableHDU(stars.as_array(), name='STARS'),
            fits.ImageHDU(corr, name='FLUXCORR')]
    hdus[-1].header['COMMENT'] = ('flux-flux correlation matrix per chunk, '
                                  '(nchunk, nstar, nstar)')
    fits.HDUList(hdus).writeto(path, overwrite=True)
    log(f'photometry written to {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description='Joint isotropic-Moffat PSF photometry over the chunks, '
                    'with the blended pair deconvolved rather than avoided.')
    ap.add_argument('--config', default=os.path.join(here, 'embin_config.yaml'))
    ap.add_argument('--stack', default=None,
                    help='the drift-corrected stack (default: '
                         'astrometry_stack.fits in the output folder)')
    ap.add_argument('--summary', default=None,
                    help='the per-chunk cube (default: chunk_summary.fits in '
                         'the output folder)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    log(f'reading configuration from {os.path.abspath(args.config)}')
    cfg = load_config(args.config)
    psf = cfg.get('psf') or {}
    if not psf:
        log('no `psf:` section in the configuration, using built-in defaults',
            'warn')

    outdir = _resolve(cfg, cfg.get('output', {}).get('directory', 'data_bin'))
    stack_path = args.stack or os.path.join(outdir, 'astrometry_stack.fits')
    summary_path = args.summary or os.path.join(outdir, 'chunk_summary.fits')
    for p in (stack_path, summary_path):
        if not os.path.exists(p):
            log(f'{p} not found. Run run_chunks.py and then '
                f'pesto_astrometry.py first', 'error')
            sys.exit(1)

    # --- the stack ------------------------------------------------------
    log(f'reading the stack from {stack_path}')
    stack = fits.getdata(stack_path, 0).astype(np.float64)
    hdr = fits.getheader(stack_path, 0)
    pixscale = float(hdr.get('PIXSCALE', psf.get('pixel_scale', 1.0)))
    log(f'stack is {stack.shape[1]} x {stack.shape[0]} px at '
        f'{pixscale:.5f} arcsec/px', 'value')

    chunks = fits.getdata(summary_path, extname='CHUNKS')
    log(f'{len(chunks)} chunks in {summary_path}')

    xs, ys, _, _ = find_stars(stack,
                              float(psf.get('threshold_sigma', 8.0)),
                              float(psf.get('fwhm_guess', 3.5)),
                              int(psf.get('max_stars', 0)))

    sigma_map = stack_error_map(summary_path, chunks, stack.shape)
    xs, ys, fwhm0, beta0, fwhm0_err, beta0_err, stack_flux, stack_bkg = fit_stack(
        stack, sigma_map, xs, ys, psf)
    log(f'stack seeing: FWHM = {fwhm0 * pixscale:.4f} +/- '
        f'{fwhm0_err * pixscale:.4f} arcsec, beta = {beta0:.4f} +/- '
        f'{beta0_err:.4f}', 'value')
    if psf.get('beta_free', False):
        log('psf.beta_free is true: beta is refitted in every chunk. In a '
            '64-frame chunk it is usually not measurable and will rail; the '
            'per-chunk value is reported with a NaN error when it does', 'warn')
    else:
        log(f'beta is held at the deep-stack value {beta0:.4f} for every '
            f'chunk; only FWHM, the rigid shift and the fluxes are refitted',
            'info')

    blends = find_blends(xs, ys, fwhm0, float(psf.get('blend_factor', 3.0)))
    for i, j, d in blends:
        log(f'blend: stars {i} and {j} are {d:.2f} px = {d * pixscale:.3f} '
            f'arcsec apart, {d / fwhm0:.2f} FWHM - fitted jointly', 'value')
    if not blends:
        log('no pair closer than the blend threshold; the joint solve costs '
            'nothing and changes nothing', 'warn')

    stars = star_metadata(stack_path, xs, ys, stack_flux)

    # --- every chunk ----------------------------------------------------
    flux_cube = fits.getdata(summary_path, extname='FLUX')
    err_cube = fits.getdata(summary_path, extname='FLUX_ERR')
    results = []
    for k in range(len(chunks)):
        # Seed the shift with what the astrometry already measured, negated:
        # the stack was built as stack_pixel = chunk_pixel + shift, so a star at
        # stack position x sits at x - shift in the chunk.
        dxy0 = (-float(chunks['dx_stack'][k]), -float(chunks['dy_stack'][k]))
        r = fit_bin(flux_cube[k].astype(np.float64),
                    err_cube[k].astype(np.float64),
                    xs, ys, fwhm0, beta0, dxy0, psf)
        results.append(r)
        log(f'chunk {k:2d}  t = {chunks["t_mid"][k]:6.2f} s   '
            f'FWHM = {r["fwhm"] * pixscale:.4f}" +/- {r["fwhm_err"] * pixscale:.4f}   '
            f'beta = {r["beta"]:.3f} +/- {r["beta_err"]:.3f}   '
            f'shift = ({r["dx"]:+.2f}, {r["dy"]:+.2f}) px   '
            f'chi2/dof = {r["chi2_red"]:.3f}', 'value')

    # What the blend actually cost, now that the fits exist.
    penalties, rho_mean = (np.nan, np.nan), np.nan
    for i, j, _ in blends:
        pen_i = np.mean([blend_penalty(r['cols'], r['inv_sigma'], i, j)
                         for r in results])
        pen_j = np.mean([blend_penalty(r['cols'], r['inv_sigma'], j, i)
                         for r in results])
        rho = np.mean([r['corr'][i, j] for r in results])
        penalties, rho_mean = (pen_i, pen_j), rho
        log(f'cross-term, stars {i}/{j}: error bars inflated by '
            f'{pen_i:.5f} and {pen_j:.5f} relative to an isolated star, '
            f'flux-flux correlation {rho:+.5f}. Small is the right answer at '
            f'this separation - the point is that it is measured, not assumed',
            'value')

    epoch = absolute_epoch(outdir, chunks)
    if epoch is None:
        log('no RETD_TOD found in the chunk cubes; time stays relative to the '
            'first frame', 'warn')
    tab = build_table(results, chunks, stars, pixscale, epoch)

    out_fits = os.path.join(outdir, 'psf_photometry.fits')
    out_ecsv = os.path.join(outdir, 'psf_photometry.ecsv')
    write_output(out_fits, tab, stars, results, [
        ('ORIGIN', 'embin_psf.py', 'this script'),
        ('STACK', os.path.basename(stack_path), 'positions came from here'),
        ('SUMMARY', os.path.basename(summary_path), 'per-chunk flux maps'),
        ('NSTAR', len(xs), 'stars fitted simultaneously'),
        ('NBLEND', len(blends), 'pairs inside the blend radius'),
        ('PIXSCALE', pixscale, 'arcsec per pixel'),
        ('PSFMODEL', 'moffat-isotropic', 'shared FWHM and beta per chunk'),
        ('FLUXDEF', 'total, integrated to infinity', 'e-/frame'),
    ])
    tab.write(out_ecsv, format='ascii.ecsv', overwrite=True)
    log(f'the same table, human-readable, in {out_ecsv}')

    # The star list in the two forms that are not a FITS table: a DS9 region
    # file to drop on the stack, and a YAML block to read or paste.
    report_meta = {
        'dataset': os.path.basename(os.path.dirname(stack_path)) or 'this field',
        'stamp': _timestamp(),
        'n_star': len(xs), 'n_chunk': len(chunks),
        'nframes': int(np.median(chunks['nframes'])),
        'pixscale': pixscale,
        'stack_fwhm': fwhm0 * pixscale, 'stack_fwhm_px': fwhm0,
        'beta': beta0, 'beta_err': beta0_err,
        'oversample': int(psf.get('oversample', 3)),
        'duration': float(chunks['t_end'][-1] - chunks['t_start'][0]),
        'sky': float(np.median(chunks['sky'])),
        'penalty': penalties, 'rho': rho_mean,
        'blend_fwhm': (blends[0][2] / fwhm0) if blends else np.nan,
    }
    write_region_file(os.path.join(outdir, 'psf_stars.reg'), stars, blends,
                      fwhm0, pixscale)
    write_star_yaml(os.path.join(outdir, 'psf_stars.yaml'), stars, tab, blends,
                    report_meta)

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        figdir = os.path.join(outdir, cfg.get('output', {}).get('figures',
                                                                'figures'))
        os.makedirs(figdir, exist_ok=True)
        diagnostic_figure(tab, results, stars, blends,
                          os.path.join(figdir, 'psf_photometry.pdf'))

        figures = {'lightcurve': 'psf_photometry.pdf'}
        finder = finder_page(stack, stars, blends, fwhm0, pixscale)
        finder.savefig(os.path.join(figdir, 'psf_finder.pdf'),
                       bbox_inches='tight')
        plt.close(finder)
        figures['finder'] = 'psf_finder.pdf'
        log(f'finder chart written to {os.path.join(figdir, "psf_finder.pdf")}')

        if blends:
            fig = blend_page(stack, xs, ys, fwhm0, beta0, stack_flux, stack_bkg,
                             blends[0][:2], pixscale,
                             int(psf.get('oversample', 3)))
            fig.savefig(os.path.join(figdir, 'psf_blend.pdf'),
                        bbox_inches='tight')
            plt.close(fig)
            figures['blend'] = 'psf_blend.pdf'
            log(f'blend cutout written to '
                f'{os.path.join(figdir, "psf_blend.pdf")}')

        if write_psf_report(outdir,
                            cfg.get('output', {}).get('figures', 'figures'),
                            report_meta, tab, stars, blends, figures) is None:
            log('the numbers are all in psf_photometry.fits regardless', 'warn')

    # --- the table, on stdout -------------------------------------------
    show = ['bin', 't_mid', 'fwhm', 'fwhm_err', 'beta', 'beta_err', 'chi2_red']
    show += [c for j in range(min(3, len(xs)))
             for c in (f'flux_{j:02d}', f'flux_err_{j:02d}')]
    print()
    tab[show].pprint(max_lines=-1, max_width=-1)
    print()
    for j in range(len(xs)):
        f, e = tab[f'flux_{j:02d}'], tab[f'flux_err_{j:02d}']
        log(f'star {j:2d}  x={xs[j]:7.2f} y={ys[j]:7.2f}  '
            f'G={stars["Gmag"][j]:5.2f}  median flux {np.median(f):9.4f} '
            f'+/- {np.median(e):.4f} e-/frame  '
            f'scatter {np.std(f) / abs(np.median(f)) * 100:5.2f} %', 'value')
    return 0


if __name__ == '__main__':
    sys.exit(main())
