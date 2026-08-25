#!/usr/bin/env python3
"""
pesto_astrometry.py -- put a WCS on the binned PESTO products
=============================================================

`run_chunks.py` gives you flux maps in pixels. This puts them on the sky: it
solves the field with astrometry.net (through the pure-Python `astrometry`
package, no external solve-field), writes a WCS into the products, and adds
RA/Dec to every tracked star so you can say which of them is your target.

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

It is the PESTO sibling of `cpapir_astrometry.py` (OMM 1.6 m / CPAPIR), and
follows the same structure: detect stars, hint the solver with what is actually
known about the camera, solve, then rebuild the header the way we want it. Four
things differ, and they are the whole point of a separate file:

  1. THE PIXEL SCALE IS PESTO'S, not CPAPIR's.  0.466 arcsec/px, measured, from
     the instrument description in Cadieux et al. (2022, AJ 164, 96;
     arXiv:2208.06333, Sect. 2.3): "PESTO features a 1024x1024 pixel EMCCD
     detector with a pixel scale of 0.466", providing a field of view of
     7'.95 x 7'.95".  CPAPIR's 0.9"/px would be wrong by a factor of two and
     the solve would simply never match.

  2. NO SIP.  The CD matrix is enough here, and the code proves it rather than
     assuming it.  PESTO's usable field is a few arcminutes across, over which
     a tangent plane plus a linear CD matrix is the whole story.  On the test
     sequence a plain CD matrix fitted to Gaia lands 27 stars to 0.49 arcsec
     RMS; a cubic gets to 0.06 arcsec, but only by spending 20 free parameters
     on 54 measurements, which is centroid noise absorbed, not optics measured.
     `distortion_check` prints that table on every run so the choice is visible.

     The final CD matrix is also not the solver's.  astrometry.net's job here is
     to say WHICH catalogue star each detection is; its own WCS comes from a few
     quads.  `fit_tan_to_gaia` then fits CRVAL and CD to every matched Gaia star
     directly, which is both more accurate and what "the CD matrix is fine" is
     supposed to mean.

  3. CRPIX AT THE CENTRE OF THE FIELD.  Always, non-negotiably: CRPIX =
     ((nx+1)/2, (ny+1)/2) in the FITS 1-indexed convention, with CRVAL the sky
     position there.  A reference point in a corner (or wherever the solver
     happened to leave it) makes CRVAL meaningless as "where the telescope was
     pointing" and makes every rotation/scale diagnostic harder to read.

  4. IT READS BINNED PRODUCTS, not raw frames.  The input is a flux map in
     e-/frame from `embin.py` / `run_chunks.py`, not an ADU image.  By default
     it stacks every chunk of `chunk_summary.fits` after taking the measured
     drift out, which is the deepest image the sequence can produce and roughly
     doubles the number of solvable stars.

USAGE
-----
    python pesto_astrometry.py                     # stack, target from the YAML
    python pesto_astrometry.py --target TOI-1452   # resolve the name at SIMBAD
    python pesto_astrometry.py --ra 290.1739 --dec 73.1954
    python pesto_astrometry.py --chunk 0           # solve one chunk instead
    python pesto_astrometry.py --no-gaia           # skip the Gaia check

Settings live in the `astrometry:` section of embin_config.yaml. The command
line only overrides them for one run.

WHAT IT WRITES
--------------
    data_bin/astrometry_stack.fits   the stacked image actually solved, + WCS,
                                     + the Gaia table and the matched stars
    data_bin/chunk_summary.fits      WCS in the primary header; RA/Dec columns
                                     added to TRACKS; per-chunk field centre and
                                     measured drift added to CHUNKS
    data_bin/embin_chunkNN.fits      WCS in the primary header and in HISTCUBE
    data_bin/embin_chunkNN_flux.fits WCS in the primary header and in FLUX /
                                     FLUX_ERR: the cube and the maps fitted from
                                     it are the same sky, so both are solved
    data_bin/gaia_field.fits         the Gaia cone search, cached
    data_bin/figures/astrometry.pdf  the solved field with the catalogue on top,
                                     the residuals, and a close-up of the target

It also says which tracked light curve is the target, and warns when another
catalogue source is close enough to contaminate it.  On the PESTO test sequence
that warning fires: TOI-1452's 3.1 arcsec companion TIC 420112587 is only a
couple of PSF widths outside the aperture, so track 1 is diluted.

REQUIREMENTS BEYOND requirements.txt
------------------------------------
    pip install astrometry photutils
The first solve downloads ~1 GB of astrometry.net index files into
~/.astrometry_cache and never downloads them again.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
import urllib.parse
import urllib.request

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import _resolve, flux_path_for, load_config, log  # noqa: E402


# ---------------------------------------------------------------------------
# What is known about PESTO before any image is looked at
# ---------------------------------------------------------------------------
# Measured instrument scale, from the TOI-1452 discovery paper's instrument
# description (Cadieux et al. 2022, AJ 164, 96 = arXiv:2208.06333, Sect. 2.3):
# 1024 x 1024 EMCCD, 0.466 arcsec/px, 7.95 x 7.95 arcmin field.  This is a
# property of the camera plus the OMM 1.6 m focal reducer, not of any one
# night, so it is a far better hint than anything in a frame header -- the Nuvu
# headers carry no astrometric information at all (no RA, no DEC, no scale).
PESTO_PIXSCALE_ARCSEC = 0.466

# The bracket handed to the solver.  Tight, because the scale is fixed hardware
# and a tight bracket is what makes a small field solvable at all: it throws out
# most of the index's candidate quads before any of them is tested.  It is NOT
# centred on 0.466 by accident: the solutions on the PESTO test sequence come
# out at 0.4540 arcsec/px, 2.6 % below the published nominal value, consistently
# and with log-odds above 100, so the bracket has to reach comfortably below the
# nominal figure.  Trust the measurement written into PIXSCALE by this script
# over the round number in the paper.
PESTO_PIXSCALE_RANGE_ARCSEC = (0.430, 0.500)

# Index scales worth searching, in astrometry.net's numbering, for a field whose
# long axis is ~8 arcmin: skymarks have to fit INSIDE the field, so 5205 (11-16')
# and up can never match and only cost time.  Verified against the SCALE_L /
# SCALE_U keywords of the cached index files themselves:
#   5201 = 2.8-4.0'   5202 = 4.0-5.6'   5203 = 5.6-8.0'   5204 = 8.0-11'
# A 1024 x 426 px window is 7.95' x 3.31', so 5201 and 5202 do the work, 5203
# can still match along the long axis, and 5204 is there for full-frame
# (1024 x 1024) sequences.
PESTO_INDEX_SCALES = {1, 2, 3, 4}

# Typical stellar FWHM in pixels.  OMM seeing runs 1.5-4 arcsec, which at
# 0.466"/px is 3-8.5 px; the retry ladder below brackets that.
RETRY_FWHM = (4.0, 4.0, 6.5, 3.0)
RETRY_MAX_STARS = (40, 80, 80, 120)
RETRY_MORPH_CLIP = (5.0, 5.0, 5.0, 15.0)

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astrometry_cache')

# Everything the FITS WCS machinery may have left behind, so a re-solve cannot
# silently mix old keywords with new ones.
_WCS_PREFIXES = ('WCSAXES', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CD1_',
                 'CD2_', 'PC1_', 'PC2_', 'CROTA', 'LONPOLE', 'LATPOLE',
                 'EQUINOX', 'RADESYS', 'A_', 'B_', 'AP_', 'BP_', 'A_ORDER',
                 'B_ORDER', 'AP_ORDER', 'BP_ORDER')


def _strip_wcs(header):
    """Remove every pre-existing WCS/SIP keyword from a header, in place.

    Header.update() only overwrites the keys present in the new header; it does
    not delete the orphans.  A leftover A_2_0 from an earlier SIP solution
    would silently distort a header that no longer claims to have distortion.
    """
    for key in list(header.keys()):
        if any(key.startswith(p) for p in _WCS_PREFIXES):
            del header[key]


# ---------------------------------------------------------------------------
# Where to point the solver: resolving the target
# ---------------------------------------------------------------------------
def resolve_target(name, timeout=30.0):
    """
    Ask SIMBAD for a target's ICRS position and proper motion.

    Returns (ra_deg, dec_deg, pmra_mas_yr, pmdec_mas_yr).  pmra is the
    already-projected mu_alpha* = mu_alpha cos(delta), which is what SIMBAD
    serves and what the propagation below expects.
    """
    adql = ("SELECT b.ra, b.dec, b.pmra, b.pmdec FROM basic AS b "
            "JOIN ident AS i ON i.oidref = b.oid WHERE i.id = '%s'" % name)
    url = ('https://simbad.u-strasbg.fr/simbad/sim-tap/sync?'
           + urllib.parse.urlencode({'request': 'doQuery', 'lang': 'adql',
                                     'format': 'csv', 'query': adql}))
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        text = fh.read().decode('utf-8', 'replace')
    rows = [r for r in text.strip().splitlines()[1:] if r.strip()]
    if not rows:
        raise RuntimeError(f'SIMBAD does not resolve {name!r}')
    parts = rows[0].split(',')
    ra, dec = float(parts[0]), float(parts[1])
    pmra = float(parts[2]) if parts[2].strip() else 0.0
    pmdec = float(parts[3]) if parts[3].strip() else 0.0
    return ra, dec, pmra, pmdec


def propagate_pm(ra, dec, pmra, pmdec, epoch_from, epoch_to):
    """
    Move a catalogue position by its proper motion, in the flat approximation.

    Good to well under a milliarcsecond over the decade or two involved here,
    and the alternative (a full space-motion propagation) needs a parallax and
    a radial velocity that the position hint does not care about.  pmra is
    mu_alpha* , so it is divided by cos(dec) to become a change in alpha.
    """
    dt = epoch_to - epoch_from
    dec_new = dec + pmdec * dt / 3.6e6
    ra_new = ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec))
    return ra_new, dec_new


def epoch_of(header_table):
    """Decimal year of the sequence, from the frame headers' DATE keyword."""
    from datetime import datetime
    for key in ('DATE', 'RETD_TOD', 'SYNC_TOD'):
        if key in header_table.colnames:
            try:
                stamp = str(header_table[key][0])[:19]
                t = datetime.fromisoformat(stamp)
                start = datetime(t.year, 1, 1)
                end = datetime(t.year + 1, 1, 1)
                return t.year + (t - start).total_seconds() / (end - start).total_seconds()
            except (ValueError, TypeError):
                continue
    return None


# ---------------------------------------------------------------------------
# The image to solve
# ---------------------------------------------------------------------------
def drift_corrected_stack(flux_cube, tracks, reference_chunk=0):
    """
    Average the per-chunk flux maps after removing the field drift.

    The field moves during the sequence (0.16 px/s on the PESTO test data, so
    ~7 px over a minute).  A plain mean of the chunks therefore smears every
    star into a short streak, which both loses depth and confuses a star finder
    that is looking for round sources.  The shift of each chunk is already
    measured, for free, by the tracks: it is the median offset of every star
    that chunk shares with the reference chunk.  Shifting by whole pixels only
    is deliberate -- a sub-pixel resample would correlate neighbouring pixels
    and there is no astrometric gain to be had from it at this stage, since the
    solver is fed centroids measured afterwards.

    Returns (stacked_image, shifts, exact_shifts).  `shifts` is what was
    actually applied, in whole pixels; `exact_shifts` is the unrounded
    measurement, which is what should be used to convert anything back to the
    sky, since rounding to a pixel would quantise every derived position to
    0.45 arcsec.
    """
    n_chunk = flux_cube.shape[0]
    tr = Table(tracks) if not isinstance(tracks, Table) else tracks
    ref = {int(r['track']): (float(r['x']), float(r['y']))
           for r in tr[tr['chunk'] == reference_chunk]}

    shifts, exact, planes = [], [], []
    for c in range(n_chunk):
        here = tr[tr['chunk'] == c]
        dx = [ref[int(r['track'])][0] - float(r['x'])
              for r in here if int(r['track']) in ref]
        dy = [ref[int(r['track'])][1] - float(r['y'])
              for r in here if int(r['track']) in ref]
        fx = float(np.median(dx)) if dx else 0.0
        fy = float(np.median(dy)) if dy else 0.0
        exact.append((fx, fy))
        sx, sy = int(round(fx)), int(round(fy))
        shifts.append((sx, sy))
        planes.append(np.roll(np.roll(flux_cube[c], sy, axis=0), sx, axis=1))
    return np.mean(planes, axis=0), shifts, exact


# ---------------------------------------------------------------------------
# Star detection
# ---------------------------------------------------------------------------
def extract_stars(image, max_stars=60, fwhm_pixels=4.0, threshold_sigma=5.0,
                  morph_sigma_clip=5.0, verbose=True):
    """
    Detect the brightest round sources and return their [x, y] pixel positions.

    Same idea as the CPAPIR version: find everything, measure each detection's
    roundness and sharpness, sigma-clip those two against the field's own
    median, and only then keep the brightest survivors.  What the clip removes
    here is different, though: on a binned EMCCD flux map the contaminants are
    not detector halos but hot pixels and cosmic-ray hits that survived into
    the histogram, which are much sharper than the PSF rather than rounder.
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    mask = ~np.isfinite(image)
    _, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask, maxiters=10)
    if verbose:
        log(f'    background {median:.4f} e-/frame, noise {std:.4f} e-/frame '
            f'(robust, pixel to pixel)', 'value')
        log(f'    looking for FWHM {fwhm_pixels:.1f} px sources above '
            f'{threshold_sigma:.1f} sigma', 'value')

    finder = DAOStarFinder(fwhm=fwhm_pixels, threshold=threshold_sigma * std,
                           min_separation=fwhm_pixels)
    sources = finder(image - median, mask=mask)
    if sources is None or len(sources) == 0:
        if verbose:
            log('    -> no source at all', 'warn')
        return []

    n_before = len(sources)
    round1 = np.asarray(sources['roundness1'], dtype=float)
    sharp = np.asarray(sources['sharpness'], dtype=float)
    good = np.isfinite(round1) & np.isfinite(sharp)
    if good.sum() >= 5:
        _, r_med, r_std = sigma_clipped_stats(round1[good], sigma=morph_sigma_clip,
                                              maxiters=5)
        _, s_med, s_std = sigma_clipped_stats(sharp[good], sigma=morph_sigma_clip,
                                              maxiters=5)
        r_std, s_std = max(r_std, 1e-3), max(s_std, 1e-3)
        keep = (good
                & (np.abs(round1 - r_med) <= morph_sigma_clip * r_std)
                & (np.abs(sharp - s_med) <= morph_sigma_clip * s_std))
        sources = sources[keep]
        if verbose:
            log(f'    morphology cut (roundness + sharpness): {n_before} -> '
                f'{len(sources)} kept', 'value')
    elif verbose:
        log(f'    only {int(good.sum())} measurable source(s), morphology cut '
            f'skipped', 'warn')

    if len(sources) == 0:
        if verbose:
            log('    -> nothing survives the morphology cut', 'warn')
        return []

    sources.sort('flux', reverse=True)
    sources = sources[:max_stars]
    if verbose:
        log(f'    {len(sources)} star(s) handed to the solver, brightest at '
            f"x={sources['xcentroid'][0]:.1f} y={sources['ycentroid'][0]:.1f}",
            'value')
    return [[float(x), float(y)]
            for x, y in zip(sources['xcentroid'], sources['ycentroid'])]


def refine_centroids(image, stars_xy, box=7, max_shift=3.0):
    """
    Replace each detection's centroid with a 2D Gaussian fit in a small box.

    DAOStarFinder's centroid comes from marginal sums through a matched filter.
    It is good enough to identify a star and quite good enough to hand to the
    solver, but it is NOT good enough to be the final astrometry: on the PESTO
    stack it leaves about 0.5 arcsec of residual against Gaia, roughly a pixel,
    which is centroid error rather than anything about the sky.  A proper 2D
    Gaussian fit over a 15x15 box cuts that substantially.

    A fit that wants to move the star by more than `max_shift` pixels has
    latched onto a neighbour or onto noise, and the original centroid is kept.
    """
    from astropy.modeling import fitting, models

    ny, nx = image.shape
    fitter = fitting.LevMarLSQFitter()
    yy, xx = np.mgrid[:2 * box + 1, :2 * box + 1]
    out = []
    for x, y in stars_xy:
        xi, yi = int(round(x)), int(round(y))
        if xi < box or yi < box or xi >= nx - box or yi >= ny - box:
            out.append([x, y])
            continue
        sub = np.asarray(image[yi - box:yi + box + 1, xi - box:xi + box + 1],
                         dtype=float)
        if not np.all(np.isfinite(sub)):
            out.append([x, y])
            continue
        bg = float(np.median(sub))
        model = models.Gaussian2D(amplitude=float(sub.max()) - bg,
                                  x_mean=box, y_mean=box,
                                  x_stddev=2.0, y_stddev=2.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with np.errstate(all='ignore'):
                    fit = fitter(model, xx, yy, sub - bg)
            dx = float(fit.x_mean.value) - box
            dy = float(fit.y_mean.value) - box
            if np.isfinite(dx) and np.isfinite(dy) and max(abs(dx), abs(dy)) <= max_shift:
                out.append([xi + dx, yi + dy])
                continue
        except Exception:                                          # noqa: BLE001
            pass
        out.append([x, y])
    return out


# ---------------------------------------------------------------------------
# The header we actually want: plain TAN, CRPIX in the middle
# ---------------------------------------------------------------------------
def tan_wcs_at_centre(wcs, shape, grid_n=24):
    """
    Rebuild a solution as a pure TAN WCS with CRPIX at the centre of the field.

    Two things happen here, and they have to happen together.

    CRPIX moves to the middle.  A reference pixel left wherever the solver put
    it makes CRVAL an arbitrary sky position rather than "where the field is",
    and it puts the point of least projection error in a corner.  The new CRVAL
    is simply the solver's own sky position at the centre pixel.

    SIP is dropped.  PESTO's field is a few arcminutes; over that span the
    gnomonic projection plus a linear CD matrix is the whole story, and any
    polynomial fitted on top of it on a dozen stars is fitting noise.  Rather
    than assume that, this refits the CD matrix by least squares over a grid
    spanning the detector and RETURNS the residual, so the assumption is
    measured.  If that residual ever comes back at a substantial fraction of a
    pixel, the field is not flat and this is the function to revisit.

    Returns (header, rms_arcsec, max_arcsec).
    """
    ny, nx = shape
    crpix = np.array([(nx + 1) / 2.0, (ny + 1) / 2.0])   # FITS is 1-indexed
    crval = wcs.all_pix2world([crpix], 1)[0]

    gx, gy = np.meshgrid(np.linspace(1, nx, grid_n), np.linspace(1, ny, grid_n))
    gx, gy = gx.ravel(), gy.ravel()
    ra, dec = wcs.all_pix2world(gx, gy, 1)

    # Project the solver's sky positions onto the tangent plane centred on the
    # new CRVAL.  A helper WCS with CRPIX at the origin, unit CDELT and an
    # identity PC returns (xi, eta) in degrees directly.
    tan = WCS(naxis=2)
    tan.wcs.crval = crval
    tan.wcs.crpix = [0.0, 0.0]
    tan.wcs.cdelt = [1.0, 1.0]
    tan.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
    tan.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    xi, eta = tan.wcs_world2pix(ra, dec, 1)

    # Fit through the origin, with no constant term: at the centre pixel
    # (u, v) = (0, 0) and (xi, eta) = (0, 0) by construction, so a free
    # constant could only move the solution away from the CRVAL we just chose.
    design = np.column_stack([gx - crpix[0], gy - crpix[1]])
    cd_xi, *_ = np.linalg.lstsq(design, xi, rcond=None)
    cd_eta, *_ = np.linalg.lstsq(design, eta, rcond=None)
    cd = np.array([cd_xi, cd_eta])

    resid = np.hypot(xi - design @ cd_xi, eta - design @ cd_eta) * 3600.0

    header = _header_from_cd(crpix, crval, cd)
    header['SIPRESID'] = (float(np.sqrt(np.mean(resid ** 2))),
                          'RMS of dropping SIP over the field [arcsec]')
    header['WCSFIT'] = ('solver-linearised', 'CD linearised from the solver WCS')
    return header, float(np.sqrt(np.mean(resid ** 2))), float(resid.max())


def fit_tan_to_gaia(stars_xy, gaia, wcs_init, shape, tol_arcsec=2.0,
                   clip_sigma=3.0, n_iter=4, verbose=True):
    """
    Fit the final WCS directly to Gaia, using the solver only to identify stars.

    astrometry.net's job is to say WHICH catalogue star each detection is; it
    does that from a handful of quads and its own tweak polynomial, which on a
    field this small is fitted to too few stars to be trusted as a final
    answer.  Re-projecting that polynomial onto a linear model (see
    `tan_wcs_at_centre`) inherits its errors.  Fitting the linear model to the
    matched Gaia positions instead throws the polynomial away entirely and uses
    every matched star, which is both more accurate and exactly what "the CD
    matrix is fine" is supposed to mean.

    The model is the four numbers of a TAN WCS with CRPIX pinned to the centre:

        xi  = CD1_1 u + CD1_2 v
        eta = CD2_1 u + CD2_2 v,      u = x - CRPIX1,  v = y - CRPIX2

    with (xi, eta) the Gaia positions projected onto the tangent plane at CRVAL.
    A free constant in that fit would mean the tangent point is not at CRPIX, so
    instead of keeping one, the constant is used to MOVE CRVAL and the fit is
    repeated: after two or three passes it is at the milliarcsecond level and
    CRVAL is, exactly, the sky position of the central pixel.  Outliers (bad
    matches, blends, the odd cosmic ray) are sigma-clipped between passes.

    Returns (header, matched_table, rms_arcsec).
    """
    ny, nx = shape
    crpix = np.array([(nx + 1) / 2.0, (ny + 1) / 2.0])
    xy = np.asarray(stars_xy, dtype=float)
    ra_g = np.asarray(gaia['RAJ2000'], dtype=float)
    dec_g = np.asarray(gaia['DEJ2000'], dtype=float)

    wcs = wcs_init
    matched = None
    for it in range(n_iter):
        # Identify: nearest Gaia source to each detection under the current WCS.
        ra_s, dec_s = wcs.all_pix2world(xy[:, 0] + 1, xy[:, 1] + 1, 1)
        cosd = np.cos(np.radians(np.median(dec_g)))
        idx, sep = [], []
        for r, d in zip(ra_s, dec_s):
            sp = np.hypot((ra_g - r) * cosd, dec_g - d) * 3600.0
            j = int(np.argmin(sp))
            idx.append(j)
            sep.append(sp[j])
        idx, sep = np.array(idx), np.array(sep)
        keep = sep <= (tol_arcsec if it == 0 else max(3 * rms, 0.3))
        # A Gaia source claimed by two detections is a blend or a bad match;
        # keep neither, rather than letting one of them pull the fit.
        for j in np.unique(idx[keep]):
            if (idx[keep] == j).sum() > 1:
                keep &= idx != j
        if keep.sum() < 3:
            raise RuntimeError(f'only {keep.sum()} usable Gaia match(es): '
                               f'cannot fit a CD matrix')

        crval = wcs.all_pix2world([crpix], 1)[0]
        for _ in range(3):        # move CRVAL onto CRPIX, then re-fit
            tan = WCS(naxis=2)
            tan.wcs.crval = crval
            tan.wcs.crpix = [0.0, 0.0]
            tan.wcs.cdelt = [1.0, 1.0]
            tan.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
            tan.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            xi, eta = tan.wcs_world2pix(ra_g[idx[keep]], dec_g[idx[keep]], 1)

            u = xy[keep, 0] + 1 - crpix[0]
            v = xy[keep, 1] + 1 - crpix[1]
            design = np.column_stack([np.ones_like(u), u, v])
            c_xi, *_ = np.linalg.lstsq(design, xi, rcond=None)
            c_eta, *_ = np.linalg.lstsq(design, eta, rcond=None)
            # The constant is the tangent point's offset from CRPIX; walk CRVAL
            # over by it and the next pass returns a constant near zero.
            crval = tan.wcs_pix2world([[c_xi[0], c_eta[0]]], 1)[0]

        cd = np.array([[c_xi[1], c_xi[2]], [c_eta[1], c_eta[2]]])
        res_xi = (xi - design @ c_xi) * 3600.0
        res_eta = (eta - design @ c_eta) * 3600.0
        resid = np.hypot(res_xi, res_eta)
        rms = float(np.sqrt(np.mean(resid ** 2)))

        header = fits.Header()
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC--TAN'
        header['CRPIX1'], header['CRPIX2'] = crpix
        header['CRVAL1'], header['CRVAL2'] = crval
        header['CD1_1'], header['CD1_2'] = cd[0]
        header['CD2_1'], header['CD2_2'] = cd[1]
        header['CUNIT1'] = header['CUNIT2'] = 'deg'
        wcs = WCS(header)

        matched = Table({'x': xy[keep, 0], 'y': xy[keep, 1],
                         'gaia_ra': ra_g[idx[keep]], 'gaia_dec': dec_g[idx[keep]],
                         'Gmag': np.asarray(gaia['Gmag'], dtype=float)[idx[keep]],
                         'd_ra': res_xi, 'd_dec': res_eta, 'sep': resid})
        if verbose:
            log(f'    pass {it + 1}: {keep.sum()} star(s) matched, '
                f'RMS {rms:.3f} arcsec', 'value')
        # Drop anything beyond clip_sigma of the current fit and go round again.
        if it < n_iter - 1 and resid.max() <= clip_sigma * rms:
            break

    full = _header_from_cd(crpix, crval, cd)
    full['GAIANSTR'] = (len(matched), 'Stars used in the Gaia CD fit')
    full['GAIARMS'] = (rms, 'RMS of the CD fit against Gaia [arcsec]')
    full['GAIARMSA'] = (float(np.std(matched['d_ra'])),
                        'RMS vs Gaia, RA axis [arcsec]')
    full['GAIARMSD'] = (float(np.std(matched['d_dec'])),
                        'RMS vs Gaia, Dec axis [arcsec]')
    full['WCSFIT'] = ('gaia-dr3', 'CD fitted to Gaia, solver used only to match')
    return full, matched, rms


def distortion_check(matched, crpix, max_degree=3):
    """
    Say out loud what a distortion polynomial would and would not buy.

    Not applying SIP is a decision, so it should be reported rather than
    assumed.  This refits the matched residuals with polynomials of increasing
    degree and prints the RMS each one reaches next to the number of free
    parameters it spends.  A degree that only wins by spending nearly as many
    parameters as there are measurements has not found distortion, it has
    absorbed the centroid noise, and its "improvement" would not survive on the
    next night's data.

    Returns [(degree, n_parameters, rms_arcsec), ...].
    """
    u = np.asarray(matched['x'], dtype=float) + 1 - crpix[0]
    v = np.asarray(matched['y'], dtype=float) + 1 - crpix[1]
    out = []
    for deg in range(1, max_degree + 1):
        terms = [(i, j) for d in range(deg + 1) for i in range(d + 1)
                 for j in [d - i]]
        design = np.column_stack([u ** i * v ** j for i, j in terms])
        res = []
        for comp in ('d_ra', 'd_dec'):
            y = np.asarray(matched[comp], dtype=float)
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            res.append(y - design @ coef)
        out.append((deg, 2 * len(terms),
                    float(np.sqrt(np.mean(np.hypot(*res) ** 2)))))
    return out


def _header_from_cd(crpix, crval, cd):
    """Assemble the FITS header for a plain TAN WCS from CRPIX, CRVAL and CD."""
    header = fits.Header()
    header['WCSAXES'] = 2
    header['CTYPE1'] = ('RA---TAN', 'Gnomonic projection, no SIP')
    header['CTYPE2'] = ('DEC--TAN', 'Gnomonic projection, no SIP')
    header['CRPIX1'] = (float(crpix[0]), 'Reference pixel: CENTRE of the field')
    header['CRPIX2'] = (float(crpix[1]), 'Reference pixel: CENTRE of the field')
    header['CRVAL1'] = (float(crval[0]), 'RA at the centre of the field [deg]')
    header['CRVAL2'] = (float(crval[1]), 'Dec at the centre of the field [deg]')
    header['CD1_1'], header['CD1_2'] = float(cd[0][0]), float(cd[0][1])
    header['CD2_1'], header['CD2_2'] = float(cd[1][0]), float(cd[1][1])
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['RADESYS'] = 'ICRS'
    scale_x, scale_y, theta, shear = decompose_cd(np.asarray(cd))
    header['PIXSCALE'] = (3600.0 * np.sqrt(abs(scale_x * scale_y)),
                          'Pixel scale, geometric mean of both axes [arcsec]')
    header['PXSCAL1'] = (3600.0 * abs(scale_x), 'Pixel scale, x axis [arcsec]')
    header['PXSCAL2'] = (3600.0 * abs(scale_y), 'Pixel scale, y axis [arcsec]')
    header['ROTATION'] = (np.degrees(theta), 'Rotation of pixel x from RA [deg]')
    header['SHEAR'] = (shear, 'CD matrix shear, dimensionless')
    return header


def decompose_cd(cd):
    """
    Split a CD matrix into scale on each axis, rotation and shear.

    Standard QR-like decomposition, CD = R(theta) @ [[sx, sx*shear], [0, sy]].
    Zero shear with sx = sy means square, unrotated-relative-to-each-other
    pixels, which is what a healthy solution looks like.
    """
    a, b, c, d = cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]
    scale_x = np.hypot(a, c)
    theta = np.arctan2(c, a)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    shear = (b * cos_t + d * sin_t) / scale_x
    scale_y = -b * sin_t + d * cos_t
    return scale_x, scale_y, theta, shear


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------
def solve_field(image, ra_hint=None, dec_hint=None, radius_hint_deg=0.3,
                cache_directory=DEFAULT_CACHE_DIR, scales=PESTO_INDEX_SCALES,
                fallback_radius_deg=2.0, verbose=True):
    """
    Solve one image and return the astrometry.net solution and its WCS.

    The scale is always hinted, because it is known hardware (see
    PESTO_PIXSCALE_RANGE_ARCSEC) and a small field is close to unsolvable
    without it.  The position is hinted when a target is known, and the same
    star lists are retried against a much wider radius before giving up.  A
    truly unconstrained whole-sky search is the last resort only: on a field
    this small it can run for many minutes, and if the truth is not within a
    couple of degrees of the target that is a pointing problem, not something
    a wider search will fix.
    """
    import astrometry

    lo, hi = PESTO_PIXSCALE_RANGE_ARCSEC
    size_hint = astrometry.SizeHint(lower_arcsec_per_pixel=lo,
                                    upper_arcsec_per_pixel=hi)
    if verbose:
        log(f'  scale hint (PESTO hardware): {lo:.3f} to {hi:.3f} arcsec/px '
            f'(nominal {PESTO_PIXSCALE_ARCSEC})', 'value')

    position_candidates = []
    if ra_hint is not None and dec_hint is not None:
        position_candidates.append((
            f'target ({ra_hint:.4f}, {dec_hint:.4f}) +/-{radius_hint_deg * 60:.0f}\'',
            astrometry.PositionHint(ra_deg=ra_hint, dec_deg=dec_hint,
                                    radius_deg=radius_hint_deg)))
        position_candidates.append((
            f'target widened +/-{fallback_radius_deg:.1f} deg',
            astrometry.PositionHint(ra_deg=ra_hint, dec_deg=dec_hint,
                                    radius_deg=fallback_radius_deg)))
    else:
        position_candidates.append(('whole sky', None))

    if verbose:
        log(f'  loading astrometry.net index files, scales {sorted(scales)} '
            f'(cache: {cache_directory})', 'info')
    index_files = astrometry.series_5200.index_files(
        cache_directory=cache_directory, scales=set(scales))
    if verbose:
        log(f'  {len(index_files)} index file(s) ready', 'value')
    solver = astrometry.Solver(index_files)

    t0 = time.time()
    solution = None
    try:
        for attempt, (n_stars, fwhm, clip) in enumerate(
                zip(RETRY_MAX_STARS, RETRY_FWHM, RETRY_MORPH_CLIP), start=1):
            if verbose:
                log(f'  attempt {attempt}/{len(RETRY_FWHM)}: max {n_stars} stars, '
                    f'FWHM {fwhm:.1f} px, morphology clip {clip:.1f}', 'info')
            stars = extract_stars(image, max_stars=n_stars, fwhm_pixels=fwhm,
                                  morph_sigma_clip=clip, verbose=verbose)
            if len(stars) < 4:
                if verbose:
                    log(f'    -> {len(stars)} star(s): fewer than the 4 a quad '
                        f'needs, next attempt', 'warn')
                continue
            for label, hint in position_candidates:
                if verbose:
                    log(f'    solving against {label} ...', 'info')
                result = solver.solve(
                    stars=stars, size_hint=size_hint, position_hint=hint,
                    solution_parameters=astrometry.SolutionParameters())
                if result.has_match():
                    solution = result
                    break
                if verbose:
                    log(f'    -> no match against {label}', 'warn')
            if solution is not None:
                break
    finally:
        solver.close()

    elapsed = time.time() - t0
    if solution is None:
        log(f'no astrometric solution after {elapsed:.1f} s', 'error')
        raise RuntimeError('PESTO field not solved')

    match = solution.best_match()
    if verbose:
        log(f'  SOLVED in {elapsed:.1f} s', 'info')
        log(f'  field centre  : {match.center_ra_deg:.6f}, '
            f'{match.center_dec_deg:.6f} deg', 'value')
        log(f'  scale         : {match.scale_arcsec_per_pixel:.4f} arcsec/px '
            f'(hardware value {PESTO_PIXSCALE_ARCSEC})', 'value')
        log(f'  log-odds      : {match.logodds:.1f} '
            f'(anything above ~30 is a certain match)', 'value')
        log(f'  index stars in field: {len(match.stars)}', 'value')
    return solution, match.astropy_wcs()


# ---------------------------------------------------------------------------
# Independent check: does the solution land on Gaia?
# ---------------------------------------------------------------------------
def _tap_csv(url, params, timeout, tries=3, pause=(4, 12)):
    """
    GET one TAP sync query and return the CSV text.

    TAP services answer 503 "too busy" under load often enough that a single
    attempt is not a reliable way to ask them anything, so this retries with a
    growing pause before giving up on a service.
    """
    full = url + '?' + urllib.parse.urlencode(params)
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(full, timeout=timeout) as fh:
                return fh.read().decode('utf-8', 'replace')
        except Exception as exc:                                  # noqa: BLE001
            last = exc
            if attempt < tries - 1:
                wait = pause[min(attempt, len(pause) - 1)]
                log(f'    {urllib.parse.urlparse(url).netloc} busy ({exc}); '
                    f'retrying in {wait} s', 'warn')
                time.sleep(wait)
    raise RuntimeError(f'{urllib.parse.urlparse(url).netloc}: {last}')


def _gaia_from_vizier(ra, dec, radius_deg, gmag_limit, timeout):
    """Gaia DR3 through VizieR (I/355/gaiadr3).  Positions come back at J2000."""
    adql = (f'SELECT RAJ2000, DEJ2000, pmRA, pmDE, Gmag FROM "I/355/gaiadr3" '
            f"WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) AND Gmag < {gmag_limit}")
    text = _tap_csv('https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync',
                    {'request': 'doQuery', 'lang': 'adql', 'format': 'csv',
                     'query': adql}, timeout)
    return Table.read(text.splitlines(), format='ascii.csv'), 2000.0


def _gaia_from_esa(ra, dec, radius_deg, gmag_limit, timeout):
    """
    Gaia DR3 straight from the ESA archive, as a fallback when VizieR is down.

    Note the epoch: gaiadr3.gaia_source serves positions at J2016.0, the
    catalogue's own reference epoch, whereas VizieR's RAJ2000/DEJ2000 columns
    have already been walked back to J2000.  Getting that wrong would put every
    star a decade and a half of proper motion out of place, so the epoch travels
    with the table rather than being assumed downstream.
    """
    adql = (f'SELECT ra, dec, pmra, pmdec, phot_g_mean_mag '
            f'FROM gaiadr3.gaia_source '
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
            f'AND phot_g_mean_mag < {gmag_limit}')
    text = _tap_csv('https://gea.esac.esa.int/tap-server/tap/sync',
                    {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                     'QUERY': adql}, timeout)
    t = Table.read(text.splitlines(), format='ascii.csv')
    t.rename_columns(['ra', 'dec', 'pmra', 'pmdec', 'phot_g_mean_mag'],
                     ['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE', 'Gmag'])
    return t, 2016.0


def query_gaia(ra, dec, radius_deg, gmag_limit=19.0, timeout=60.0,
               epoch_to=None, cache_path=None):
    """
    Cone-search Gaia DR3 around a position, and bring it to the right epoch.

    VizieR first because it answers in a second or two for a field this size;
    the ESA archive as a fallback, since VizieR's sync endpoint does go down.
    Whichever answers, the positions are propagated by their own proper motions
    from that service's reference epoch to `epoch_to`: at 0.45 arcsec/px a
    decade of proper motion is a real offset (TOI-1452 itself moves 2.0 arcsec
    between J2000 and 2026, more than four pixels), so skipping it would put a
    floor on the residuals well above the centroid noise.

    `cache_path`, when given, is a FITS file the answer is written to and read
    back from, so that re-solving the same field does not query anything.
    """
    if cache_path and os.path.exists(cache_path):
        t = Table.read(cache_path)
        log(f'    Gaia from cache: {cache_path}', 'info')
        return t

    errors = []
    for name, fetch in (('VizieR', _gaia_from_vizier), ('ESA', _gaia_from_esa)):
        try:
            t, epoch_from = fetch(ra, dec, radius_deg, gmag_limit, timeout)
            log(f'    Gaia DR3 from {name}: {len(t)} source(s), positions at '
                f'J{epoch_from:.1f}', 'value')
            break
        except Exception as exc:                                  # noqa: BLE001
            errors.append(f'{name}: {exc}')
            t = None
    if t is None:
        raise RuntimeError('no Gaia service answered -- ' + '; '.join(errors))

    if epoch_to is not None and len(t):
        pmra = np.nan_to_num(np.asarray(t['pmRA'], dtype=float))
        pmdec = np.nan_to_num(np.asarray(t['pmDE'], dtype=float))
        ra_new, dec_new = propagate_pm(np.asarray(t['RAJ2000'], dtype=float),
                                       np.asarray(t['DEJ2000'], dtype=float),
                                       pmra, pmdec, epoch_from, epoch_to)
        t['RAJ2000'], t['DEJ2000'] = ra_new, dec_new
        t.meta['EPOCH'] = epoch_to
    if cache_path:
        t.write(cache_path, overwrite=True)
        log(f'    Gaia cached in {cache_path}', 'info')
    return t


def match_to_gaia(stars_xy, wcs, gaia, tol_arcsec=2.0):
    """
    Match detected stars to Gaia and measure the residuals.

    Nearest neighbour within `tol_arcsec`, which is generous at 4 px: the point
    is to measure how good the solution is, not to build a catalogue, and a
    tight tolerance would quietly hide exactly the failure worth seeing.

    Returns (matched_table, rms_ra, rms_dec, rms_total) in arcsec, all robust
    (median absolute deviation scaled to a Gaussian sigma) so that one bad
    match cannot set the answer.
    """
    if not len(gaia) or not len(stars_xy):
        return Table(), np.nan, np.nan, np.nan
    xy = np.asarray(stars_xy, dtype=float)
    ra_s, dec_s = wcs.all_pix2world(xy[:, 0] + 1, xy[:, 1] + 1, 1)
    ra_g = np.asarray(gaia['RAJ2000'], dtype=float)
    dec_g = np.asarray(gaia['DEJ2000'], dtype=float)
    cosd = np.cos(np.radians(dec_g.mean()))

    rows = []
    for i, (r, d) in enumerate(zip(ra_s, dec_s)):
        dra = (ra_g - r) * cosd * 3600.0
        ddec = (dec_g - d) * 3600.0
        sep = np.hypot(dra, ddec)
        j = int(np.argmin(sep))
        if sep[j] <= tol_arcsec:
            rows.append({'x': xy[i, 0], 'y': xy[i, 1], 'ra': r, 'dec': d,
                         'gaia_ra': ra_g[j], 'gaia_dec': dec_g[j],
                         'Gmag': float(gaia['Gmag'][j]),
                         'd_ra': -dra[j], 'd_dec': -ddec[j], 'sep': sep[j]})
    if not rows:
        return Table(), np.nan, np.nan, np.nan
    t = Table(rows)

    def robust(v):
        return 1.4826 * float(np.median(np.abs(np.asarray(v) - np.median(v))))

    return (t, robust(t['d_ra']), robust(t['d_dec']),
            1.4826 * float(np.median(np.abs(t['sep'] - np.median(t['sep'])))))


# ---------------------------------------------------------------------------
# Writing the solution back into the binning products
# ---------------------------------------------------------------------------
# Which extensions of an embin product are images on the sky, and therefore
# want a WCS. HISTCUBE is three-dimensional, so it gets the two spatial axes
# plus a third axis that is the histogram bin and is explicitly NOT a sky axis.
# FLUX / FLUX_ERR live in the flux file and HISTCUBE in the cube file, so no
# single product carries all three; the lists are the union, and each file is
# solved on whichever of them it happens to hold.
_IMAGE_EXTS = ('FLUX', 'FLUX_ERR', 'MU_LO', 'MU_HI')
_CUBE_EXTS = ('HISTCUBE',)


def wcs_for_shift(header, shift):
    """
    The same solution as seen by a frame whose field sat `shift` pixels away.

    Every chunk of a drifting sequence looks at slightly different sky through
    the same pixels. The scale, the rotation and the reference PIXEL are
    properties of the optics and do not change; only CRVAL does. So the chunk
    header is the stack header with CRPIX still nailed to the centre of the
    detector and CRVAL moved to whatever sky is at that centre during that
    chunk. Doing it the other way round, by shifting CRPIX, would be
    numerically equivalent and would throw away the one thing that makes a set
    of headers easy to compare.
    """
    out = header.copy()
    if shift is None or (shift[0] == 0 and shift[1] == 0):
        return out
    w = WCS(header)
    ra, dec = w.all_pix2world([header['CRPIX1'] + shift[0]],
                              [header['CRPIX2'] + shift[1]], 1)
    out['CRVAL1'] = float(ra[0])
    out['CRVAL2'] = float(dec[0])
    out['DXDRIFT'] = (float(shift[0]), 'Field drift from the reference chunk [pix]')
    out['DYDRIFT'] = (float(shift[1]), 'Field drift from the reference chunk [pix]')
    return out


def write_wcs_into(path, header, label=''):
    """
    Put a WCS into every image extension of one embin product.

    The primary header carries it too, so that a reader that only looks there
    still learns where the field is, even though the primary holds no pixels.

    A gzipped product cannot be edited in place -- astropy has to rewrite the
    whole stream to change one card -- so it is read, edited in memory and
    written back over itself through a temporary file, which also means an
    interrupted run leaves the original intact rather than a half-written file.
    A plain .fits is still updated in place, which is much cheaper.
    """
    zipped = path.endswith('.gz')
    hdul = fits.open(path, mode='readonly' if zipped else 'update')
    try:
        targets = [0]
        for i, hdu in enumerate(hdul[1:], start=1):
            name = hdu.header.get('EXTNAME', '')
            if name in _IMAGE_EXTS + _CUBE_EXTS:
                targets.append(i)
        for i in targets:
            hdr = hdul[i].header
            _strip_wcs(hdr)
            # .copy(): Header.update() inserts the SAME Card objects rather
            # than copies, so editing this header afterwards (as the cube
            # branch below does) would silently edit the solution header too,
            # and every product written after this one would inherit it.
            hdr.update(header.copy())
            if hdul[i].header.get('EXTNAME', '') in _CUBE_EXTS:
                # A histogram cube's third axis is a bin index, not a sky or
                # spectral coordinate. Saying so explicitly stops any WCS-aware
                # reader from inventing a meaning for it.
                hdr['WCSAXES'] = 3
                hdr['CTYPE3'] = ('BIN', 'Histogram bin index, not a sky axis')
                hdr['CRPIX3'] = (1.0, 'First histogram bin')
                hdr['CRVAL3'] = (0.0, 'Bin numbering starts at zero')
                hdr['CD3_3'] = (1.0, 'One bin per plane')
                for k in ('CD1_3', 'CD2_3', 'CD3_1', 'CD3_2'):
                    hdr[k] = (0.0, 'No coupling between sky and bin axes')
            hdr['ASTROMSR'] = ('pesto_astrometry.py', 'WCS added after binning')
        if zipped:
            tmp = path + '.tmp'
            hdul.writeto(tmp, overwrite=True)
            os.replace(tmp, path)
        else:
            hdul.flush()
    finally:
        hdul.close()
    log(f'    WCS written into {os.path.basename(path)}'
        + (f' ({label})' if label else '') + f', {len(targets)} extension(s)',
        'value')


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(path, image, stars_xy, wcs, gaia, matched, target=None):
    """The solved field, with Gaia drawn on top: the check you can actually see."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(12, 7.0))
    ax = fig.add_subplot(2, 1, 1)
    vmax = float(np.nanpercentile(image, 99.9))
    ax.imshow(image, origin='lower', vmin=0, vmax=vmax, cmap='inferno')
    if len(gaia):
        gx, gy = wcs.all_world2pix(np.asarray(gaia['RAJ2000'], dtype=float),
                                   np.asarray(gaia['DEJ2000'], dtype=float), 0)
        inside = ((gx > -20) & (gx < image.shape[1] + 20)
                  & (gy > -20) & (gy < image.shape[0] + 20))
        ax.scatter(gx[inside], gy[inside], s=90, facecolors='none',
                   edgecolors='#39A0ED', lw=1.0, label=f'Gaia DR3 ({inside.sum()})')
    xy = np.asarray(stars_xy, dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], marker='+', s=70, color='#7BE495', lw=1.0,
               label=f'detected ({len(xy)})')
    if target is not None:
        tx, ty = wcs.all_world2pix([target[0]], [target[1]], 0)
        ax.scatter(tx, ty, s=260, facecolors='none', edgecolors='#F5F749',
                   lw=1.6, label=target[2])
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.4)
    ax.set_title('Solved field: detections against Gaia DR3', fontsize=10)

    ax = fig.add_subplot(2, 3, 4)
    if len(matched):
        ax.scatter(matched['d_ra'], matched['d_dec'], s=28, color='#2E6FDB')
        lim = max(1.0, 1.2 * float(np.max(np.abs(
            np.concatenate([matched['d_ra'], matched['d_dec']])))))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(0, color='0.6', lw=0.8)
        ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta$RA [arcsec]')
    ax.set_ylabel(r'$\Delta$Dec [arcsec]')
    ax.set_title(f'Residuals against Gaia ({len(matched)} stars)', fontsize=10)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(2, 3, 5)
    if len(matched):
        ax.scatter(matched['Gmag'], matched['sep'], s=28, color='#2E6FDB')
    ax.set_xlabel('Gaia G [mag]')
    ax.set_ylabel('separation [arcsec]')
    ax.set_title('Residual against brightness', fontsize=10)
    ax.grid(alpha=0.25)

    # A close-up on the target.  On this field it is the panel that matters:
    # what looks like one tracked star at full scale is two, and the light
    # curve of a blend is not the light curve of the planet host.
    ax = fig.add_subplot(2, 3, 6)
    if target is not None:
        tx, ty = wcs.all_world2pix([target[0]], [target[1]], 0)
        cx, cy, half = int(tx[0]), int(ty[0]), 20
        y0, y1 = max(0, cy - half), min(image.shape[0], cy + half + 1)
        x0, x1 = max(0, cx - half), min(image.shape[1], cx + half + 1)
        sub = image[y0:y1, x0:x1]
        ax.imshow(sub, origin='lower', cmap='inferno',
                  extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
                  vmax=float(np.nanpercentile(sub, 99.5)))
        if len(gaia):
            gx, gy = wcs.all_world2pix(np.asarray(gaia['RAJ2000'], dtype=float),
                                       np.asarray(gaia['DEJ2000'], dtype=float), 0)
            near = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
            ax.scatter(gx[near], gy[near], s=140, facecolors='none',
                       edgecolors='#39A0ED', lw=1.1)
            for xx, yy, gm in zip(gx[near], gy[near],
                                  np.asarray(gaia['Gmag'], dtype=float)[near]):
                ax.annotate(f'G={gm:.2f}', (xx, yy), textcoords='offset points',
                            xytext=(8, 6), fontsize=7, color='#CFE9FF')
        ax.scatter(tx, ty, s=260, facecolors='none', edgecolors='#F5F749', lw=1.6)
        ax.set_xlabel('x [pix]')
    ax.set_title(f'{target[2] if target else "target"}, close up',
                 fontsize=10)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f'figure: {path}', 'value')


# ---------------------------------------------------------------------------
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description='Astrometric solution for the binned PESTO products.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', default=os.path.join(here, 'embin_config.yaml'))
    ap.add_argument('--target', default=None,
                    help='target name to resolve at SIMBAD (default: from the YAML)')
    ap.add_argument('--ra', type=float, default=None, help='position hint, RA [deg]')
    ap.add_argument('--dec', type=float, default=None, help='position hint, Dec [deg]')
    ap.add_argument('--chunk', type=int, default=None,
                    help='solve this chunk alone instead of the drift-corrected stack')
    ap.add_argument('--radius', type=float, default=None,
                    help='position hint radius [deg]')
    ap.add_argument('--no-gaia', action='store_true', help='skip the Gaia check')
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--no-write', action='store_true',
                    help='solve and report, but do not touch any file')
    ap.add_argument('--no-chunks', action='store_true',
                    help='update only the summary, not each embin_chunkNN.fits')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    acfg = cfg.get('astrometry', {}) or {}
    out_cfg = cfg.get('output', {})
    outdir = _resolve(cfg, out_cfg.get('directory', 'data_bin'))
    summary = os.path.join(outdir, out_cfg.get('summary',
                                               'chunk_summary.fits'))
    # run_chunks.py gzips its products by default, so the name in the config is
    # the uncompressed one and what is on disk usually ends in .gz. Take
    # whichever exists, so neither setting has to be kept in step with the other.
    if not os.path.exists(summary) and os.path.exists(summary + '.gz'):
        summary += '.gz'
    if not os.path.exists(summary):
        log(f'{summary} does not exist: run run_chunks.py first', 'error')
        return 1

    log('PESTO astrometry', 'info')
    log(f'  input: {summary}', 'info')
    hdul = fits.open(summary)
    flux_cube = hdul['FLUX'].data
    tracks = Table(hdul['TRACKS'].data) if 'TRACKS' in hdul else Table()
    chunk_tab = Table(hdul['CHUNKS'].data)

    # --- the epoch of the observation, for proper motions -----------------
    epoch = None
    chunk0 = os.path.join(outdir, str(chunk_tab['file'][0]))
    if os.path.exists(chunk0):
        with fits.open(chunk0) as ch:
            if 'HEADERS' in ch:
                epoch = epoch_of(Table(ch['HEADERS'].data))
    if epoch is not None:
        log(f'  epoch of observation: {epoch:.3f}', 'value')
    else:
        log('  no usable date in the frame headers: proper motions not applied',
            'warn')

    # --- where to point the solver ----------------------------------------
    target_name = args.target or acfg.get('target')
    ra_hint, dec_hint = args.ra, args.dec
    target_radec = None
    if ra_hint is None and dec_hint is None:
        if target_name:
            log(f'  resolving {target_name} at SIMBAD ...', 'info')
            ra0, dec0, pmra, pmdec = resolve_target(target_name)
            log(f'  {target_name}: RA {ra0:.6f}, Dec {dec0:.6f} deg (ICRS), '
                f'pm ({pmra:.1f}, {pmdec:.1f}) mas/yr', 'value')
            if epoch is not None:
                # SIMBAD's basic.ra/dec are ICRS at epoch J2000, verified
                # against VizieR I/355/gaiadr3's RAJ2000/DEJ2000 for this very
                # target: the two agree to 1e-8 deg, so both are J2000 and the
                # proper motion has to be run over the full 2000 -> now.
                ra_hint, dec_hint = propagate_pm(ra0, dec0, pmra, pmdec,
                                                 2000.0, epoch)
                log(f'  propagated to {epoch:.2f}: RA {ra_hint:.6f}, '
                    f'Dec {dec_hint:.6f} deg '
                    f'({3600 * np.hypot((ra_hint - ra0) * np.cos(np.radians(dec0)), dec_hint - dec0):.2f} '
                    f'arcsec of proper motion)', 'value')
            else:
                ra_hint, dec_hint = ra0, dec0
            target_radec = (ra_hint, dec_hint, target_name)
        else:
            log('  no target and no --ra/--dec: the search will be blind, which '
                'on an 8 arcmin field is slow and may not converge', 'warn')
    else:
        target_radec = (ra_hint, dec_hint, 'position hint')

    # --- the image to solve -----------------------------------------------
    if args.chunk is not None:
        image = flux_cube[args.chunk]
        label = f'chunk {args.chunk}'
        shifts = exact = None
    else:
        image, shifts, exact = drift_corrected_stack(flux_cube, tracks)
        total = np.hypot(*np.subtract(exact[-1], exact[0]))
        label = f'drift-corrected stack of {flux_cube.shape[0]} chunks'
        log(f'  stacking {flux_cube.shape[0]} chunks, drift taken out '
            f'({total:.1f} px end to end)', 'value')
    log(f'  solving on the {label}, {image.shape[1]} x {image.shape[0]} px '
        f'= {image.shape[1] * PESTO_PIXSCALE_ARCSEC / 60:.2f} x '
        f'{image.shape[0] * PESTO_PIXSCALE_ARCSEC / 60:.2f} arcmin', 'value')

    radius = args.radius if args.radius is not None else float(
        acfg.get('search_radius_deg', 0.3))
    solution, wcs = solve_field(image, ra_hint=ra_hint, dec_hint=dec_hint,
                                radius_hint_deg=radius)

    # --- Gaia, then the header we want -------------------------------------
    # The solver has done its one irreplaceable job: it has told us which
    # catalogue star each detection is.  Its own WCS is fitted to a handful of
    # quads and is not the answer we want to keep, so from here on Gaia does
    # the work.
    stars_xy = extract_stars(image, max_stars=200, fwhm_pixels=4.0, verbose=False)
    log(f'  refining {len(stars_xy)} centroid(s) with a 2D Gaussian fit', 'info')
    stars_xy = refine_centroids(image, stars_xy)
    gaia, matched = Table(), Table()
    header = None

    if not (args.no_gaia or not acfg.get('gaia', True)):
        fov = 0.5 * np.hypot(*image.shape) * PESTO_PIXSCALE_ARCSEC / 3600.0
        log(f'  fetching Gaia DR3 over {fov * 66:.1f} arcmin ...', 'info')
        try:
            centre = wcs.all_pix2world(
                [[(image.shape[1] + 1) / 2, (image.shape[0] + 1) / 2]], 1)[0]
            gaia = query_gaia(centre[0], centre[1], fov * 1.1, epoch_to=epoch,
                              cache_path=os.path.join(outdir, 'gaia_field.fits'))
            log('  fitting CRVAL and the CD matrix to Gaia, CRPIX pinned to the '
                'field centre:', 'info')
            header, matched, rms_gaia = fit_tan_to_gaia(stars_xy, gaia, wcs,
                                                        image.shape)
        except Exception as exc:
            log(f'  Gaia fit unavailable ({exc}); falling back to linearising '
                f'the solver WCS, which is less accurate', 'warn')
            header = None

    if header is None:
        header, rms_sip, max_sip = tan_wcs_at_centre(wcs, image.shape)
        level = 'value' if rms_sip < 0.1 else 'warn'
        log(f'  cost of linearising the solver WCS: {rms_sip:.3f} arcsec RMS, '
            f'{max_sip:.3f} arcsec worst corner', level)

    wcs_out = WCS(header)
    log('  header: plain TAN, no SIP, CRPIX at the CENTRE of the field', 'info')
    log(f'  CRPIX = ({header["CRPIX1"]:.1f}, {header["CRPIX2"]:.1f})  '
        f'[centre of {image.shape[1]} x {image.shape[0]}]', 'value')
    log(f'  CRVAL = ({header["CRVAL1"]:.6f}, {header["CRVAL2"]:.6f}) deg', 'value')
    log(f'  pixel scale = {header["PIXSCALE"]:.4f} arcsec/px '
        f'(x {header["PXSCAL1"]:.4f}, y {header["PXSCAL2"]:.4f}); PESTO nominal '
        f'{PESTO_PIXSCALE_ARCSEC}, i.e. '
        f'{100 * (header["PIXSCALE"] / PESTO_PIXSCALE_ARCSEC - 1):+.1f} %', 'value')
    log(f'  rotation = {header["ROTATION"]:.4f} deg, shear = {header["SHEAR"]:.2e}',
        'value')
    if len(matched):
        log(f'  fit against Gaia: {len(matched)} star(s), RMS '
            f'{header["GAIARMS"]:.3f} arcsec = '
            f'{header["GAIARMS"] / header["PIXSCALE"]:.2f} px '
            f'(RA {header["GAIARMSA"]:.3f}, Dec {header["GAIARMSD"]:.3f})',
            'value' if header['GAIARMS'] < 0.5 else 'warn')
        log('  would a distortion polynomial do better?  (2 measurements per '
            f'star, so {2 * len(matched)} in total)', 'info')
        for deg, npar, r in distortion_check(
                matched, (header['CRPIX1'], header['CRPIX2'])):
            note = ('  <- the CD matrix, what is written'
                    if deg == 1 else
                    f'  <- {npar} free parameters for {2 * len(matched)} '
                    f'measurements')
            log(f'    degree {deg}: RMS {r:.3f} arcsec{note}', 'value')
        log('  the high degrees win by spending parameters, not by finding '
            'optics: a cubic has almost as many free numbers as there are '
            'measurements, so its residual is centroid noise absorbed, not '
            'distortion measured. The CD matrix stays, as it should.', 'info')

    # --- where is the target? ---------------------------------------------
    if target_radec is not None:
        tx, ty = wcs_out.all_world2pix([target_radec[0]], [target_radec[1]], 0)
        log(f'  {target_radec[2]} lands at x = {tx[0]:.1f}, '
            f'y = {ty[0]:.1f} on the solved image', 'value')

        # Which of the light curves is the target, and is it clean?  A tracked
        # source is a detection, not a star: if two catalogue sources sit
        # inside the aperture, that light curve is their sum, and saying so
        # here is worth more than any header keyword.
        if len(tracks):
            d = np.hypot(np.asarray(tracks['x'], dtype=float) - tx[0],
                         np.asarray(tracks['y'], dtype=float) - ty[0])
            j = int(np.argmin(d))
            t_id = int(tracks['track'][j])
            log(f'  -> that is track {t_id}, {d[j] * header["PIXSCALE"]:.2f} '
                f'arcsec away: this is the light curve of {target_radec[2]}',
                'value')
            if len(gaia):
                sep = 3600.0 * np.hypot(
                    (np.asarray(gaia['RAJ2000'], dtype=float) - target_radec[0])
                    * np.cos(np.radians(target_radec[1])),
                    np.asarray(gaia['DEJ2000'], dtype=float) - target_radec[1])
                aper = float(cfg.get('chunks', {}).get('aperture', 3.0)) * \
                    header['PIXSCALE']
                close = np.sort(sep[(sep > 0.05) & (sep < 3 * aper)])
                if len(close):
                    inside = close[0] <= aper
                    log(f'  WARNING: {len(close)} other Gaia source(s) within '
                        f'{3 * aper:.1f} arcsec of the target, nearest at '
                        f'{close[0]:.2f} arcsec, against an aperture radius of '
                        f'{aper:.2f} arcsec. '
                        + ('It is INSIDE the aperture: track '
                           f'{t_id} is the sum of both stars.'
                           if inside else
                           'It is outside the aperture, but only by a couple '
                           'of PSF widths, so some of its light is in track '
                           f'{t_id} and the transit depth will be diluted.'),
                        'warn')

    if args.no_write:
        log('--no-write: nothing written', 'warn')
        return 0

    # --- write the solved stack -------------------------------------------
    stack_path = os.path.join(outdir, 'astrometry_stack.fits')
    hdr = header.copy()
    hdr['ORIGIN'] = ('pesto_astrometry.py', 'emccd_bintool')
    hdr['BUNIT'] = 'e-/frame'
    hdr['IMGTYPE'] = (label, 'What was solved')
    hdr['PIXSNOM'] = (PESTO_PIXSCALE_ARCSEC, 'PESTO nominal scale [arcsec/px]')
    if epoch is not None:
        hdr['EPOCH'] = (epoch, 'Decimal year of the sequence')
    hdus = [fits.PrimaryHDU(data=image.astype(np.float32), header=hdr)]
    if len(matched):
        hdus.append(fits.BinTableHDU(matched, name='GAIAMTCH'))
    if len(gaia):
        hdus.append(fits.BinTableHDU(gaia, name='GAIA'))
    fits.HDUList(hdus).writeto(stack_path, overwrite=True)
    log(f'wrote {stack_path}', 'value')

    # --- fold the solution back into chunk_summary.fits --------------------
    _strip_wcs(hdul[0].header)
    hdul[0].header.update(header.copy())        # see write_wcs_into on .copy()
    hdul[0].header['ASTROMSR'] = ('pesto_astrometry.py',
                                  'WCS added after binning')

    if len(tracks):
        ra_t, dec_t = wcs_out.all_pix2world(np.asarray(tracks['x'], dtype=float),
                                            np.asarray(tracks['y'], dtype=float), 0)
        # Each chunk sees the field at a different place, so a single WCS is
        # only right for the reference chunk.  Undo each chunk's measured shift
        # before converting, so every row's sky position is the star's, not the
        # telescope's.
        if exact is not None:
            sx = np.array([exact[int(c)][0] for c in tracks['chunk']], dtype=float)
            sy = np.array([exact[int(c)][1] for c in tracks['chunk']], dtype=float)
            ra_t, dec_t = wcs_out.all_pix2world(
                np.asarray(tracks['x'], dtype=float) + sx,
                np.asarray(tracks['y'], dtype=float) + sy, 0)
        tracks['ra'] = ra_t
        tracks['dec'] = dec_t
        tracks['ra'].unit = tracks['dec'].unit = 'deg'
        hdul['TRACKS'] = fits.BinTableHDU(tracks, name='TRACKS')

    if shifts is not None:
        cx = (image.shape[1] - 1) / 2.0
        cy = (image.shape[0] - 1) / 2.0
        # The stack was built as stack_pixel = chunk_pixel + shift, so the
        # sky at chunk c's own centre pixel is the WCS evaluated at that pixel
        # PLUS c's shift.  This is where the telescope was actually pointing
        # during that chunk, which is the drift, measured on the sky.
        ra_c, dec_c = wcs_out.all_pix2world(
            [cx + s[0] for s in exact], [cy + s[1] for s in exact], 0)
        chunk_tab['ra_centre'] = ra_c
        chunk_tab['dec_centre'] = dec_c
        chunk_tab['dx_drift'] = np.array([s[0] for s in exact])
        chunk_tab['dy_drift'] = np.array([s[1] for s in exact])
        chunk_tab['ra_centre'].unit = chunk_tab['dec_centre'].unit = 'deg'
        hdul['CHUNKS'] = fits.BinTableHDU(chunk_tab, name='CHUNKS')

    hdul.writeto(summary, overwrite=True)
    log(f'wrote the WCS, and sky coordinates for every track, into {summary}',
        'value')
    hdul.close()

    # --- and into every per-chunk binning product --------------------------
    # The binning step knows nothing about the sky and is left that way; this
    # is where the sky is attached to it. Each chunk gets the SAME CD matrix
    # and the SAME central CRPIX, and its own CRVAL, because that is the only
    # thing the drift actually changes.
    if not args.no_chunks:
        log('attaching the solution to each binning product:', 'info')
        n_done = 0
        for row, shift in zip(chunk_tab, exact if exact is not None
                              else [(0.0, 0.0)] * len(chunk_tab)):
            # Each chunk is TWO files -- the histogram cube and the flux maps
            # fitted from it -- and both are images of the same sky, so both get
            # the same solution. The flux file's name is in the CHUNKS table when
            # the summary was written by a current run_chunks.py, and derivable
            # from the cube's name when it was not.
            names = [str(row['file'])]
            if 'flux_file' in chunk_tab.colnames and str(row['flux_file']):
                names.append(str(row['flux_file']))
            else:
                names.append(os.path.basename(flux_path_for(str(row['file']))))
            solution = wcs_for_shift(header, shift)
            wrote_any = False
            for name in names:
                path = os.path.join(outdir, name)
                if not os.path.exists(path) and os.path.exists(path + '.gz'):
                    path += '.gz'
                if not os.path.exists(path):
                    log(f'    {name}: not on disk, skipped', 'warn')
                    continue
                write_wcs_into(path, solution,
                               label=f'drift {shift[0]:+.2f}, {shift[1]:+.2f} px')
                wrote_any = True
            n_done += int(wrote_any)
        log(f'  {n_done} chunk product(s) now carry a WCS', 'value')

    if not args.no_figure:
        figdir = os.path.join(outdir, str(cfg.get('output', {}).get(
            'figures', 'figures')))
        make_figure(os.path.join(figdir, 'astrometry.pdf'), image, stars_xy,
                    wcs_out, gaia, matched, target_radec)

    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
