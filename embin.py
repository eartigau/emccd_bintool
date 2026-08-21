#!/usr/bin/env python3
"""
embin.py -- histogram-bin cubes and mean-flux maps from individual EMCCD frames
===============================================================================

WHAT THIS DOES
--------------
Given a folder of individual EMCCD frames (one raw readout per file) and a set
of histogram bin edges defined in a YAML configuration file, this script:

  1. Accumulates, for EVERY pixel independently, a histogram of that pixel's
     raw ADU values across all the frames. The result is a cube with the x/y
     size of the input images and `len(edges) - 1` planes along z; plane b of
     pixel (y, x) counts how many of the N frames had that pixel's value inside
     bin b.

  2. Fits the mean flux of every pixel, in electrons per frame, from that
     pixel's own histogram, using the maximum-likelihood estimator of the
     maximum-likelihood estimator in emccd_histo.py. The result is a
     2-D image with the same x/y size as the input frames.

  3. Detects the sources that stand more than `detect_sigma` sigma above the
     sky in that flux map -- the significance being the filtered flux divided by
     its own propagated error, so the cut means the same thing everywhere on the
     detector -- and cuts a small box around each one,
     saving the FULL, ORIGINAL, unbinned ADU data of every frame inside that
     box -- so the raw pixel-level time series of each star survives the
     histogram compression untouched.

  4. Writes everything to one multi-extension FITS file:

        HDU 0  PRIMARY    no data; the header documents the bin edges and
                          every parameter of the run
        HDU 1  FLUX       2-D image, best-fit mean flux [e-/frame] per pixel
        HDU 2  FLUX_ERR   2-D image, its 1-sigma statistical error [e-/frame]
        HDU 3  HISTCUBE   3-D cube (Nbin, ny, nx) of per-pixel bin counts
        HDU 4  HEADERS    table, one row per input file, one column per FITS
                          keyword found across all of them
        HDU 5+ STAMPnn    3-D cube (Nframes, box, box) of raw ADU for one
                          detected source; its header gives the source's pixel
                          position, its detection significance and the stamp's
                          corner in frame coordinates (the asymmetric MU_LO /
                          MU_HI bounds are inserted before the stamps when
                          fit.asymmetric_bounds is on)

WHY A HISTOGRAM CUBE
--------------------
An EMCCD read is not "the flux, plus noise". A pixel that saw n photo-electrons
this frame reads out at bias + Gamma(n, gain) + N(0, RON): the EM register's
avalanche gain is stochastic, so the SAME n gives wildly different ADU values
frame to frame. Averaging the ADU values therefore throws away most of the
information at the sub-electron-per-frame fluxes these detectors are used at.
The full SHAPE of a pixel's ADU distribution is what carries the flux, and a
handful of well-placed histogram bins captures essentially all of it (see
`histogram.edges` in the YAML for the information efficiency of the default
binning) at a tiny fraction of the storage of the raw cube.

USAGE
-----
    python embin.py                       # uses embin_config.yaml, as shipped
    python embin.py --first 640 --n-files 64   # frames 640 to 703 instead
    python embin.py --output somewhere/else.fits

Everything else -- which folder the frames are in, the bin edges, the detector
constants, the detection threshold -- is set in embin_config.yaml, which is
commented line by line. You should not have to edit this file to use it.

To process a WHOLE sequence rather than one set of frames, use run_chunks.py.
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import center_of_mass, correlate1d, maximum_filter

# The physical EMCCD model (Poisson -> Gamma -> Gaussian -> bias) and the
# maximum-likelihood flux fitter live in emccd_histo.py, right next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emccd_histo import build_cache, build_nll_grid, fit_flux_map  # noqa: E402


# ---------------------------------------------------------------------------
# Timestamped, colour-coded logging
# ---------------------------------------------------------------------------
# Colour per log() level - see that function's docstring for what each means.
_LOG_COLORS = {
    'info': '\033[92m',        # green - general info / progress narration
    'value': '\033[94m',       # blue - a computed/measured numeric result
    'warn': '\033[38;5;208m',  # orange - something skipped, non-critical issue
    'error': '\033[91m',       # red - explains why the run is stopping
}
_LOG_RESET = '\033[0m'


def _timestamp():
    """'YYMMDD HH:MM:SS.SS' for the current instant."""
    now = datetime.now()
    return now.strftime('%y%m%d %H:%M:%S') + f'.{now.microsecond // 10000:02d}'


def log(message, level='info'):
    """Print one status line as 'YYMMDD HH:MM:SS.SS | message', colour-coded by level:
      'info'  (green)  - general progress narration (what step is happening now).
      'value' (blue)   - a computed/measured numeric result being reported.
      'warn'  (orange) - something skipped, or a non-critical issue.
      'error' (red)    - explains why the run is stopping entirely.
    """
    color = _LOG_COLORS.get(level, _LOG_COLORS['info'])
    print(f"{color}{_timestamp()} | {message}{_LOG_RESET}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path):
    """Read the YAML config and resolve every relative path against ITS OWN
    directory (not the current working directory), so the script can be run
    from anywhere and still find the data the config points at."""
    path = os.path.abspath(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg['_config_path'] = path
    cfg['_config_dir'] = os.path.dirname(path)
    return cfg


def _resolve(cfg, p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg['_config_dir'], p))


def _numeric_key(path):
    """Sort key: the LAST run of digits in the file name, as an integer.

    Frame files here are named nc-image_<i>.fits, and an alphabetical sort puts
    nc-image_10.fits before nc-image_2.fits. Sorting on the embedded integer is
    what "the N first files" means. Files with no digits sort last, by name.
    """
    name = os.path.basename(path)
    digits = re.findall(r'\d+', name)
    return (0, int(digits[-1]), name) if digits else (1, 0, name)


def all_frames(cfg):
    """Every frame the configuration points at, in numeric filename order.

    No selection of any kind is applied here: `input.first_file` / `n_files` are
    the business of find_files(), and run_chunks.py deliberately ignores them.
    """
    inp = cfg['input']
    directory = _resolve(cfg, inp['directory'])
    pattern = inp.get('pattern', '*.fits')
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=_numeric_key)
    if not files:
        log(f'no file matches {os.path.join(directory, pattern)} - nothing to do. '
            f'Is input.directory in the configuration pointing at your frames?', 'error')
        sys.exit(1)
    return files


def find_files(cfg):
    """The list of input frames, in numeric filename order, cut to the requested chunk."""
    inp = cfg['input']
    directory = _resolve(cfg, inp['directory'])
    files = all_frames(cfg)
    n_avail = len(files)
    # `first_file` is a 0-based offset into that numeric order, so a long
    # sequence can be processed one contiguous chunk of frames at a time
    # (see run_chunks.py) without the frames of one chunk leaking into another.
    first = int(inp.get('first_file') or 0)
    if first:
        if first >= n_avail:
            log(f'first_file = {first} is past the last of {n_avail} files - nothing to do', 'error')
            sys.exit(1)
        files = files[first:]
    n_files = inp.get('n_files')
    if n_files:
        if n_files > len(files):
            log(f'only {len(files)} files available from offset {first}, '
                f'{n_files} requested - using all of them', 'warn')
        files = files[:n_files]
    log(f'{len(files)} input frames from {directory} '
        f'(of {n_avail} matching, starting at index {first})', 'value')
    log(f'first: {os.path.basename(files[0])}   last: {os.path.basename(files[-1])}')
    return files


def get_edges(cfg):
    """Bin edges as a float array, validated to be strictly increasing.

    Convention (shared with the physical model, see the YAML): the last bin is
    OPEN-ENDED -- every value >= edges[-2] falls in it -- so edges[-1] is only a
    formal sentinel and edges[-2] is the effective saturation/overflow
    threshold. The first bin absorbs anything below edges[0].
    """
    edges = np.asarray(cfg['histogram']['edges'], dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3:
        log('histogram.edges must be a list of at least 3 increasing values', 'error')
        sys.exit(1)
    if np.any(np.diff(edges) <= 0):
        log('histogram.edges must be strictly increasing', 'error')
        sys.exit(1)
    return edges


def output_path(cfg, override=None, name=None):
    """Where a result file goes: <output.directory>/<name>, folder created.

    EVERY file this toolkit writes lands under `output.directory` (`data_bin` by
    default), so nothing is ever dropped next to the raw frames. A path given on
    the command line with --output wins over the configuration and is used
    exactly as typed.
    """
    out_cfg = cfg.get('output', {})
    if override:
        return override
    directory = _resolve(cfg, out_cfg.get('directory', 'data_bin'))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name or out_cfg.get('file', 'embin_output.fits'))


# ---------------------------------------------------------------------------
# Step 1 -- the per-pixel histogram cube
# ---------------------------------------------------------------------------
def build_histogram_cube(files, edges, hdu_index=0):
    """Accumulate one ADU histogram per pixel over all frames.

    Returns
    -------
    cube      : int32 array (nbins, ny, nx) -- counts per bin per pixel
    n_under   : int  values below edges[0]  (counted into bin 0)
    n_over    : int  values >= edges[-2]    (counted into the open last bin)
    """
    nbins = len(edges) - 1
    bin_ids = np.arange(nbins, dtype=np.int64)[:, None, None]

    cube = None
    n_under = 0
    n_over = 0
    for i, path in enumerate(files):
        img = fits.getdata(path, ext=hdu_index)
        if img is None or img.ndim != 2:
            log(f'{os.path.basename(path)}: HDU {hdu_index} is not a 2-D image - stopping', 'error')
            sys.exit(1)
        img = np.asarray(img, dtype=np.float64)
        if cube is None:
            ny, nx = img.shape
            cube = np.zeros((nbins, ny, nx), dtype=np.int32)
            log(f'frames are {nx} x {ny} pixels, binning into {nbins} bins '
                f'-> cube ({nbins}, {ny}, {nx})', 'value')
        elif img.shape != cube.shape[1:]:
            log(f'{os.path.basename(path)} is {img.shape}, expected {cube.shape[1:]} '
                f'- all frames must share one shape', 'error')
            sys.exit(1)

        # searchsorted(..., 'right') - 1 is exactly numpy.histogram's rule: a value
        # equal to an edge belongs to the bin STARTING at that edge. The clip then
        # implements the two out-of-range conventions documented in get_edges().
        n_under += int(np.count_nonzero(img < edges[0]))
        n_over += int(np.count_nonzero(img >= edges[-2]))
        idx = np.searchsorted(edges, img, side='right') - 1
        np.clip(idx, 0, nbins - 1, out=idx)

        # One vectorised pass over the bin axis: (nbins, ny, nx) of booleans,
        # ~7 MB at this frame size, far cheaper than a per-pixel loop.
        cube += (idx[None, :, :] == bin_ids)

        if (i + 1) % 25 == 0 or i + 1 == len(files):
            log(f'  binned {i + 1}/{len(files)} frames')

    return cube, n_under, n_over


# ---------------------------------------------------------------------------
# Step 2 -- the per-pixel mean-flux map
# ---------------------------------------------------------------------------
def make_model_config(cfg, edges):
    """The emccd_histo.py configuration dictionary for this detector + binning.

    saturation_adu is set to edges[-2], NOT edges[-1]: emccd_histo's build_cache
    overrides the last bin's probability with the analytic P(ADU >= saturation_adu),
    which is only correct if the last histogram bin really is the open-ended
    "everything above" bin -- which is exactly how build_histogram_cube fills it.
    """
    det = cfg['detector']
    return {
        'detector': {
            'gain': float(det['gain']),
            'ron': float(det['ron']),
            'bias': float(det['bias']),
            'full_well': int(det['full_well']),
            'nmax': int(det['nmax']),
            'cic': float(det.get('cic', 0.0)),
        },
        'histogram': {
            'nbins': len(edges) - 1,
            'saturation_adu': float(edges[-2]),
        },
    }


def build_flux_grid(cfg, edges):
    """The tabulated negative log-likelihood of every flux on the search grid.

    It depends only on the detector constants and the bin edges, never on the
    data, so a caller that fits several frame sets with one binning (run_chunks.py)
    builds it once and passes it back into fit_flux_image().
    """
    fit_cfg = cfg.get('fit', {})
    model_cfg = make_model_config(cfg, edges)

    log('building the per-electron bin-probability cache ...')
    cache = build_cache(model_cfg, edges)

    mu_min = float(fit_cfg.get('mu_min', 1e-4))
    mu_max = float(fit_cfg.get('mu_max', 10.0))
    n_grid = int(fit_cfg.get('n_grid', 2000))
    log(f'building the flux grid: {n_grid} log-spaced points over '
        f'{mu_min:g} - {mu_max:g} e-/frame ...')
    return build_nll_grid(model_cfg, edges, cache=cache,
                          mu_min=mu_min, mu_max=mu_max, n_grid=n_grid)


def fit_flux_image(cube, edges, cfg, nll_grid=None):
    """Maximum-likelihood mean flux [e-/frame] for every pixel of the cube.

    Returns (mu_fit, mu_err, mu_lo, mu_hi), each a 2-D array shaped like the
    frames: the best-fit flux, its symmetrised 1-sigma error, and the two
    (generally asymmetric) delta-log-likelihood = 0.5 bounds it is built from.

    `nll_grid` is the output of build_flux_grid(); it is rebuilt here when not
    supplied, so a single call needs nothing extra.
    """
    fit_cfg = cfg.get('fit', {})
    if nll_grid is None:
        nll_grid = build_flux_grid(cfg, edges)

    # fit_flux_map wants the bin axis LAST; the cube stores it first so that the
    # FITS file reads naturally as a stack of images, one per bin.
    counts_map = np.moveaxis(cube, 0, -1)
    log(f'fitting {counts_map.shape[0] * counts_map.shape[1]:,} pixels (vectorised MLE) ...')
    mu_fit, mu_lo, mu_hi = fit_flux_map(counts_map, nll_grid,
                                        chunk_size=int(fit_cfg.get('chunk_size', 2048)))
    # The likelihood interval is genuinely asymmetric at low flux (mu is bounded
    # below by zero), so mu_lo/mu_hi are the honest answer and are kept as their
    # own extensions on request. FLUX_ERR is their half-width: the one number
    # that "the error on the flux" usually means, and what the detection
    # significance divides by.
    mu_err = 0.5 * (mu_hi - mu_lo)
    return mu_fit, mu_err, mu_lo, mu_hi


# ---------------------------------------------------------------------------
# Step 3 -- source detection and raw postage stamps
# ---------------------------------------------------------------------------
def significance_map(flux, flux_err, cfg):
    """The detection statistic: how many sigma above the sky each pixel sits.

    The flux map is first matched-filtered with a small Gaussian, a crude stand-in
    for the PSF: a star spreads over a few pixels, so summing them with PSF-shaped
    weights beats looking at any one of them, and it stops a single noisy pixel of
    the fitted flux map from registering as a source. `smooth_sigma` = 0 disables it.

    The denominator can be built two ways (`sources.significance` in the YAML):

      'error_map'   -- from the per-pixel MLE uncertainty in FLUX_ERR, propagated
                       through the very same filter. If the filter has normalised
                       weights w (sum w = 1), then var(sum w_i F_i) = sum w_i^2
                       var(F_i), so the numerator is filtered with w and the
                       variance with w^2 -- which is why the kernel is built
                       explicitly here instead of calling gaussian_filter twice.
                       This is the statistically meaningful "N sigma": it is each
                       pixel's OWN Poisson/Gamma counting uncertainty from its own
                       frames, so the threshold means the same thing everywhere on
                       the detector and for any number of input frames.

      'map_scatter' -- from the robust scatter of the filtered map itself
                       (1.4826 x its median absolute deviation), i.e. an empirical
                       "how much does this map wobble" estimate. It needs no error
                       map and it absorbs any systematic that the per-pixel model
                       does not describe, but the sigma it measures is a single
                       number for the whole frame, so it is only as good as the
                       assumption that the noise is uniform across it.

    In both cases the sky level subtracted off is the median of the filtered map,
    which a handful of bright stars cannot drag around.

    Returns (significance, filtered_flux, sky_level).
    """
    src_cfg = cfg.get('sources', {})
    smooth = float(src_cfg.get('smooth_sigma', 1.0))
    mode = str(src_cfg.get('significance', 'error_map'))

    if smooth > 0:
        # Normalised 1-D Gaussian weights, truncated at 4 sigma like
        # scipy's own gaussian_filter. The 2-D filter is the separable
        # product of two of these, so filtering with w^2 is just as separable.
        radius = max(1, int(4.0 * smooth + 0.5))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        w = np.exp(-0.5 * (x / smooth) ** 2)
        w /= w.sum()
        def _filter(img, weights):
            out = correlate1d(img, weights, axis=0, mode='nearest')
            return correlate1d(out, weights, axis=1, mode='nearest')
        filtered = _filter(flux, w)
        filtered_var = _filter(flux_err ** 2, w ** 2) if flux_err is not None else None
    else:
        filtered = flux.copy()
        filtered_var = flux_err ** 2 if flux_err is not None else None

    sky = float(np.median(filtered))

    if mode == 'error_map':
        if filtered_var is None:
            log('significance: error_map requested but no flux error available '
                '- falling back to map_scatter', 'warn')
        else:
            noise = np.sqrt(np.maximum(filtered_var, 0.0))
            typical = float(np.median(noise))
            log(f'significance from FLUX_ERR: sky {sky:.4f} e-/frame, '
                f'median filtered noise {typical:.4f} e-/frame', 'value')
            with np.errstate(divide='ignore', invalid='ignore'):
                signif = np.where(noise > 0, (filtered - sky) / noise, 0.0)
            return signif, filtered, sky

    mad = float(np.median(np.abs(filtered - sky)))
    noise = 1.4826 * mad
    if noise <= 0:
        log('filtered flux map has zero robust scatter - cannot detect sources', 'warn')
        return None, filtered, sky
    log(f'significance from the map scatter: sky {sky:.4f} e-/frame, '
        f'robust sigma {noise:.4f} e-/frame', 'value')
    return (filtered - sky) / noise, filtered, sky


def detect_sources(flux, flux_err, cfg):
    """Find the point sources that stand more than `detect_sigma` sigma above the sky.

    A plain local-maximum finder is all this needs: the flux map is already the
    optimal per-pixel statistic, so a source is simply a pixel that is both
    significant (see significance_map) and the most significant thing within
    `min_separation` pixels. Positions and fluxes are always reported from the
    UNfiltered map.

    Returns a list of dicts, most significant first, each with the peak pixel, the
    sub-pixel centroid, the stamp corner, the peak flux and its significance.
    """
    src_cfg = cfg.get('sources', {})
    box = int(src_cfg.get('box_size', 16))
    nsig = float(src_cfg.get('detect_sigma', 8.0))
    min_sep = int(src_cfg.get('min_separation', 8))
    max_src = src_cfg.get('max_sources')

    ny, nx = flux.shape
    half = box // 2

    signif, filtered, sky = significance_map(flux, flux_err, cfg)
    if signif is None:
        return []

    peaks = (signif == maximum_filter(signif, size=2 * min_sep + 1)) & (signif > nsig)
    ys, xs = np.nonzero(peaks)
    log(f'{len(ys)} local maxima above {nsig:g} sigma', 'value')

    order = np.argsort(signif[ys, xs])[::-1]
    sources = []
    n_edge = 0
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        # "Forget stars too close to the edge": the whole box must fit inside
        # the frame, otherwise the stamp would be incomplete/padded.
        x0, y0 = x - half, y - half
        if x0 < 0 or y0 < 0 or x0 + box > nx or y0 + box > ny:
            n_edge += 1
            continue

        # Sub-pixel centroid: flux-weighted centre of mass of the sky-subtracted
        # box, which is where the star actually sits between pixel centres.
        cut = flux[y0:y0 + box, x0:x0 + box] - sky
        np.clip(cut, 0.0, None, out=cut)
        if cut.sum() > 0:
            cy, cx = center_of_mass(cut)
            xcen, ycen = x0 + float(cx), y0 + float(cy)
        else:
            xcen, ycen = float(x), float(y)

        sources.append({'xpeak': x, 'ypeak': y, 'xcen': xcen, 'ycen': ycen,
                        'x0': x0, 'y0': y0, 'box': box,
                        'peak_flux': float(flux[y, x]),
                        'peak_err': float(flux_err[y, x]) if flux_err is not None else np.nan,
                        'signif': float(signif[y, x])})
        if max_src and len(sources) >= int(max_src):
            log(f'reached max_sources = {max_src}, ignoring less significant detections', 'warn')
            break

    if n_edge:
        log(f'{n_edge} detection(s) dropped: a {box}x{box} box would fall off the frame', 'warn')
    log(f'{len(sources)} source(s) kept above {nsig:g} sigma', 'value')
    return sources


def extract_stamps(files, sources, hdu_index=0):
    """Raw, unbinned ADU data for every source: one (n_frames, box, box) cube each.

    This is a second pass over the input files, on purpose: holding every frame
    in memory to avoid it would cost ~1 GB for a full 1000-frame sequence, while
    re-reading a few hundred MB of FITS costs seconds.
    """
    stamps = [np.empty((len(files), s['box'], s['box']), dtype=np.int32) for s in sources]
    for i, path in enumerate(files):
        img = fits.getdata(path, ext=hdu_index)
        for j, s in enumerate(sources):
            b, x0, y0 = s['box'], s['x0'], s['y0']
            stamps[j][i] = img[y0:y0 + b, x0:x0 + b]
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            log(f'  cut stamps from {i + 1}/{len(files)} frames')
    return stamps


# ---------------------------------------------------------------------------
# Step 4 -- the table of every input file's keywords
# ---------------------------------------------------------------------------
def header_table(files, hdu_index=0):
    """One row per input frame, one column per FITS keyword seen in ANY of them.

    Columns appear in the order the keywords are first encountered. Structural
    keywords (SIMPLE, BITPIX, NAXIS*, EXTEND, BSCALE, BZERO) and the free-form
    COMMENT / HISTORY / blank cards are left out: they describe the file format,
    not the observation, and COMMENT/HISTORY are multi-valued so they have no
    single-cell representation. A keyword missing from one file's header gets a
    blank ('' for text, NaN for numbers) in that row.
    """
    skip = {'SIMPLE', 'BITPIX', 'EXTEND', 'BSCALE', 'BZERO', 'COMMENT', 'HISTORY', ''}
    keys = []
    comments = {}
    rows = []
    for path in files:
        hdr = fits.getheader(path, ext=hdu_index)
        row = {'FILENAME': os.path.basename(path)}
        for key in hdr:
            if key in skip or key.startswith('NAXIS'):
                continue
            if key not in row:
                row[key] = hdr[key]
            if key not in keys:
                keys.append(key)
                comments[key] = hdr.comments[key]
        rows.append(row)

    columns = {'FILENAME': [r['FILENAME'] for r in rows]}
    for key in keys:
        values = [r.get(key) for r in rows]
        present = [v for v in values if v is not None]
        # One column per keyword, typed by what the headers actually hold, so
        # numeric keywords stay numeric (and usable) instead of becoming strings.
        if present and all(isinstance(v, bool) for v in present):
            columns[key] = np.array([bool(v) if v is not None else False for v in values])
        elif present and all(isinstance(v, (int, np.integer)) for v in present):
            columns[key] = np.array([int(v) if v is not None else -999999 for v in values])
        elif present and all(isinstance(v, (int, float, np.floating, np.integer)) for v in present):
            columns[key] = np.array([float(v) if v is not None else np.nan for v in values])
        else:
            columns[key] = np.array([str(v) if v is not None else '' for v in values])

    table = Table(columns)
    for key in keys:
        # Carry each keyword's own FITS comment across as the column description.
        if comments.get(key):
            table[key].description = comments[key]
    log(f'header table: {len(table)} rows x {len(table.colnames)} keyword columns', 'value')
    return table


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _kw(hdr, key, value, comment=''):
    """Append one keyword card at the CURRENT end of the header.

    ``bottom=True`` is what makes the ordering hold: astropy's default is to
    insert a new value card BEFORE any trailing commentary cards, which would
    hoist every keyword above the COMMENT section headings written below and
    scramble the layout these headers are built to have.
    """
    hdr.append((key, value, comment), bottom=True)


def _comment(hdr, text):
    """Append one COMMENT card at the CURRENT end of the header.

    Plain ``hdr['COMMENT'] = text`` does not do this: astropy keeps commentary
    keywords together, so every COMMENT ends up bunched at the bottom of the
    header instead of sitting where it was written. These COMMENT lines are
    section headings and inline explanations, so their position is the point.
    """
    hdr.append(('COMMENT', text), bottom=True)


def primary_header(cfg, files, edges, n_under, n_over, n_sources):
    """The PRIMARY header: the bin edges, plus a complete record of the run."""
    hdr = fits.Header()
    _kw(hdr, 'ORIGIN', 'embin.py', 'emccd_bintool')
    _kw(hdr, 'DATE', datetime.now().isoformat(timespec='seconds'), 'File creation date (local)')
    _kw(hdr, 'CONFIG', os.path.basename(cfg['_config_path']), 'YAML configuration used')

    _comment(hdr, '--- input frames -------------------------------------------')
    # A long path fills the card on its own (astropy spills it over CONTINUE
    # cards), so it gets no inline comment -- the COMMENT line above says what
    # this block is.
    _kw(hdr, 'INPUTDIR', _resolve(cfg, cfg['input']['directory']))
    _kw(hdr, 'INPUTPAT', cfg['input'].get('pattern', '*.fits'), 'Filename pattern')
    _kw(hdr, 'NFRAMES', len(files), 'Number of frames binned')
    _kw(hdr, 'FIRSTIDX', int(cfg['input'].get('first_file') or 0),
        '0-based index of the first frame used')
    _kw(hdr, 'FIRSTIMG', os.path.basename(files[0]), 'First frame (numeric filename order)')
    _kw(hdr, 'LASTIMG', os.path.basename(files[-1]), 'Last frame (numeric filename order)')

    _comment(hdr, '--- histogram bin edges [raw ADU] -------------------------')
    _comment(hdr, 'Bin b spans [BINEDG<b>, BINEDG<b+1>), b = 0 .. NBIN-1.')
    _comment(hdr, 'The LAST bin is open-ended: every value >= BINEDG<NBIN-1>')
    _comment(hdr, 'is counted in it, so the final edge is a formal sentinel.')
    _comment(hdr, 'The FIRST bin absorbs anything below BINEDG00.')
    _kw(hdr, 'NBIN', len(edges) - 1, 'Number of histogram bins (= NAXIS3 of HISTCUBE)')
    _kw(hdr, 'NEDGE', len(edges), 'Number of bin edges (= NBIN + 1)')
    for i, e in enumerate(edges):
        _kw(hdr, f'BINEDG{i:02d}', float(e), f'Bin edge {i} [ADU]')
    _kw(hdr, 'SATADU', float(edges[-2]), 'Open last bin threshold, model saturation [ADU]')
    _kw(hdr, 'NUNDER', n_under, 'Pixel values below BINEDG00 (folded into bin 0)')
    _kw(hdr, 'NOVER', n_over, 'Pixel values >= SATADU (in the open last bin)')

    det = cfg['detector']
    _comment(hdr, '--- detector constants (from pesto_stats calibration) -----')
    _kw(hdr, 'BIAS', float(det['bias']), 'Electronics bias level [ADU]')
    _kw(hdr, 'RON', float(det['ron']), 'Read-out noise, 1 sigma [ADU]')
    _kw(hdr, 'GAIN', float(det['gain']), 'EM gain [ADU/e-]')
    _kw(hdr, 'FULLWELL', int(det['full_well']), 'Top of the model ADU grid [ADU]')
    _kw(hdr, 'NMAX', int(det['nmax']), 'Max electrons in the Poisson sum')
    _kw(hdr, 'CIC', float(det.get('cic', 0.0)), 'Clock-induced charge [e-/frame]')

    fit_cfg = cfg.get('fit', {})
    _comment(hdr, '--- flux fit ----------------------------------------------')
    _kw(hdr, 'FITDONE', bool(fit_cfg.get('enabled', True)), 'Was the per-pixel flux fit run?')
    if fit_cfg.get('enabled', True):
        _kw(hdr, 'MUMIN', float(fit_cfg.get('mu_min', 1e-4)), 'Flux grid lower bound [e-/frame]')
        _kw(hdr, 'MUMAX', float(fit_cfg.get('mu_max', 10.0)), 'Flux grid upper bound [e-/frame]')
        _kw(hdr, 'MUNGRID', int(fit_cfg.get('n_grid', 2000)), 'Flux grid points (log-spaced)')
        _kw(hdr, 'MUBOUNDS', bool(fit_cfg.get('asymmetric_bounds', False)),
            'Are the MU_LO / MU_HI extensions present?')

    src_cfg = cfg.get('sources', {})
    _comment(hdr, '--- sources and raw postage stamps ------------------------')
    _kw(hdr, 'NSTAMP', n_sources, 'Number of STAMPnn extensions in this file')
    if n_sources:
        _kw(hdr, 'BOXSIZE', int(src_cfg.get('box_size', 16)), 'Stamp size [pixels, square]')
        _kw(hdr, 'DETSIG', float(src_cfg.get('detect_sigma', 8.0)), 'Detection threshold [sigma]')
        _kw(hdr, 'DETMODE', str(src_cfg.get('significance', 'error_map')), 'How the detection sigma is measured')
        _kw(hdr, 'SMOOTHSG', float(src_cfg.get('smooth_sigma', 1.0)), 'Matched-filter width before detection [pix]')
        _kw(hdr, 'MINSEP', int(src_cfg.get('min_separation', 8)), 'Minimum separation between sources [pix]')
    return hdr


def stamp_header(src, index, n_frames):
    """The header of one STAMPnn extension: where this star is, and what the axes are."""
    hdr = fits.Header()
    _kw(hdr, 'EXTNAME', f'STAMP{index:02d}', 'Raw ADU cube for one detected source')
    _comment(hdr, 'Raw, unbinned ADU values, one plane per input frame, in the')
    _comment(hdr, 'same order as the HEADERS table rows. Axes: (frame, y, x).')
    _comment(hdr, 'XCEN/YCEN/XPEAK/YPEAK/X0/Y0 are 0-indexed numpy coordinates')
    _comment(hdr, 'in the ORIGINAL frame; XDS9/YDS9 are the same centroid in')
    _comment(hdr, '1-indexed FITS/DS9 physical convention (numpy + 1).')
    _kw(hdr, 'SRCID', index, 'Source number, 1 = brightest')
    _kw(hdr, 'XCEN', src['xcen'], 'Source x centroid in frame [pix, 0-indexed]')
    _kw(hdr, 'YCEN', src['ycen'], 'Source y centroid in frame [pix, 0-indexed]')
    _kw(hdr, 'XPEAK', src['xpeak'], 'Peak pixel x in frame [pix, 0-indexed]')
    _kw(hdr, 'YPEAK', src['ypeak'], 'Peak pixel y in frame [pix, 0-indexed]')
    _kw(hdr, 'XDS9', src['xcen'] + 1.0, 'Source x centroid [pix, FITS 1-indexed]')
    _kw(hdr, 'YDS9', src['ycen'] + 1.0, 'Source y centroid [pix, FITS 1-indexed]')
    _kw(hdr, 'X0', src['x0'], 'Stamp corner x in frame [pix, 0-indexed]')
    _kw(hdr, 'Y0', src['y0'], 'Stamp corner y in frame [pix, 0-indexed]')
    _kw(hdr, 'BOXSIZE', src['box'], 'Stamp size [pixels, square]')
    _kw(hdr, 'NFRAMES', n_frames, 'Number of frames (= NAXIS3)')
    _kw(hdr, 'PEAKFLUX', src['peak_flux'], 'Fitted flux at peak pixel [e-/frame]')
    _kw(hdr, 'PEAKERR', src['peak_err'], 'Error on PEAKFLUX [e-/frame]')
    _kw(hdr, 'SIGNIF', src['signif'], 'Detection significance [sigma]')
    return hdr


def write_mef(path, cfg, files, edges, cube, flux, flux_err, mu_lo, mu_hi,
              sources, stamps, n_under, n_over, overwrite=True):
    """Assemble and write the multi-extension FITS file."""
    hdus = [fits.PrimaryHDU(header=primary_header(cfg, files, edges,
                                                  n_under, n_over, len(sources)))]

    flux_hdr = fits.Header()
    _kw(flux_hdr, 'EXTNAME', 'FLUX', 'Per-pixel mean flux')
    _kw(flux_hdr, 'BUNIT', 'e-/frame', 'Electrons per frame per pixel')
    _comment(flux_hdr, "Maximum-likelihood mean flux, fitted from each pixel's")
    _comment(flux_hdr, 'own histogram in HISTCUBE, using emccd_histo.py.')
    if flux is None:
        flux_data = None
        _comment(flux_hdr, 'Flux fit was disabled in the configuration (fit.enabled=false).')
    else:
        flux_data = flux.astype(np.float32)
    hdus.append(fits.ImageHDU(data=flux_data, header=flux_hdr))

    err_hdr = fits.Header()
    _kw(err_hdr, 'EXTNAME', 'FLUX_ERR', 'Error on the per-pixel mean flux')
    _kw(err_hdr, 'BUNIT', 'e-/frame', 'Electrons per frame per pixel')
    _comment(err_hdr, 'Symmetrised 1-sigma statistical error on FLUX, i.e. half')
    _comment(err_hdr, 'the width of the delta-log-likelihood = 0.5 interval,')
    _comment(err_hdr, '(MU_HI - MU_LO) / 2. It is a COUNTING error only: it says')
    _comment(err_hdr, 'how well NFRAMES frames pin down this pixel own flux under')
    _comment(err_hdr, 'the model, and knows nothing of flat-field or PSF errors.')
    hdus.append(fits.ImageHDU(data=None if flux_err is None else flux_err.astype(np.float32),
                              header=err_hdr))

    cube_hdr = fits.Header()
    _kw(cube_hdr, 'EXTNAME', 'HISTCUBE', 'Per-pixel histogram of raw ADU values')
    _kw(cube_hdr, 'BUNIT', 'count', 'Number of frames in this bin')
    _comment(cube_hdr, 'Axes: (bin, y, x). Plane b counts, for every pixel, how')
    _comment(cube_hdr, 'many frames had that pixel inside bin b. Bin edges are')
    _comment(cube_hdr, 'the BINEDGnn keywords of the primary header.')
    _kw(cube_hdr, 'NBIN', cube.shape[0], 'Number of histogram bins')
    for i, e in enumerate(edges):
        _kw(cube_hdr, f'BINEDG{i:02d}', float(e), f'Bin edge {i} [ADU]')
    hdus.append(fits.ImageHDU(data=cube, header=cube_hdr))

    tab_hdu = fits.BinTableHDU(header_table(files, cfg['input'].get('hdu', 0)), name='HEADERS')
    _comment(tab_hdu.header, 'One row per input frame, in the same order as the')
    _comment(tab_hdu.header, 'planes of the STAMPnn cubes. One column per FITS keyword')
    _comment(tab_hdu.header, 'found in any of the input files.')
    hdus.append(tab_hdu)

    if mu_lo is not None:
        for name, data, what in (('MU_LO', mu_lo, 'Lower'), ('MU_HI', mu_hi, 'Upper')):
            hdr = fits.Header()
            _kw(hdr, 'EXTNAME', name, f'{what} 1-sigma flux bound')
            _kw(hdr, 'BUNIT', 'e-/frame', 'Electrons per frame per pixel')
            _comment(hdr, f'{what} bound of the delta-log-likelihood = 0.5 interval,')
            _comment(hdr, 'before it is symmetrised into FLUX_ERR. The interval is')
            _comment(hdr, 'genuinely asymmetric at low flux, where mu is bounded by 0.')
            hdus.append(fits.ImageHDU(data=data.astype(np.float32), header=hdr))

    for i, (src, stamp) in enumerate(zip(sources, stamps), start=1):
        hdus.append(fits.ImageHDU(data=stamp, header=stamp_header(src, i, stamp.shape[0])))

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    log(f'wrote {path} ({len(hdus)} HDUs, {os.path.getsize(path) / 1e6:.1f} MB)', 'value')


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embin_config.yaml')
    parser.add_argument('--config', '-c', default=default_cfg, help='YAML configuration file')
    parser.add_argument('--output', '-o', default=None, help='Override output.file from the config')
    parser.add_argument('--first', type=int, default=None,
                        help='Override input.first_file: 0-based index of the first frame to use')
    parser.add_argument('--n-files', type=int, default=None,
                        help='Override input.n_files: how many frames to use from that index on')
    args = parser.parse_args()

    log(f'reading configuration from {args.config}')
    cfg = load_config(args.config)
    if args.first is not None:
        cfg['input']['first_file'] = args.first
    if args.n_files is not None:
        cfg['input']['n_files'] = args.n_files
    edges = get_edges(cfg)
    hdu_index = cfg['input'].get('hdu', 0)
    files = find_files(cfg)

    log(f'{len(edges) - 1} histogram bins from {edges[0]:g} to {edges[-2]:g} ADU '
        f'(last bin open-ended)', 'value')

    # --- 1. histogram cube -------------------------------------------------
    log('[1/4] accumulating the per-pixel histogram cube ...')
    cube, n_under, n_over = build_histogram_cube(files, edges, hdu_index)
    n_samples = int(cube.sum())
    log(f'{n_samples:,} pixel values binned '
        f'({n_under:,} below the first edge, {n_over:,} in the open last bin)', 'value')

    # --- 2. flux map and its error ----------------------------------------
    flux = flux_err = mu_lo = mu_hi = None
    if cfg.get('fit', {}).get('enabled', True):
        log('[2/4] fitting the mean flux of every pixel ...')
        flux, flux_err, mu_lo, mu_hi = fit_flux_image(cube, edges, cfg)
        log(f'flux map: median {np.median(flux):.4f}, '
            f'99.9th pct {np.percentile(flux, 99.9):.3f}, '
            f'max {flux.max():.3f} e-/frame', 'value')
        log(f'flux error: median {np.median(flux_err):.4f} e-/frame '
            f'over {len(files)} frames', 'value')
        if not cfg.get('fit', {}).get('asymmetric_bounds', False):
            mu_lo = mu_hi = None
    else:
        log('[2/4] flux fit disabled in the configuration - skipping', 'warn')

    # --- 3. sources and raw stamps ----------------------------------------
    sources, stamps = [], []
    if flux is not None and cfg.get('sources', {}).get('enabled', True):
        log('[3/4] detecting sources in the flux map ...')
        sources = detect_sources(flux, flux_err, cfg)
        for i, s in enumerate(sources, start=1):
            log(f'  source {i:2d}: (x, y) = ({s["xcen"]:7.2f}, {s["ycen"]:7.2f})  '
                f'peak {s["peak_flux"]:.3f} +/- {s["peak_err"]:.3f} e-/frame  '
                f'{s["signif"]:6.1f} sigma', 'value')
        if sources:
            log(f'cutting {len(sources)} raw {sources[0]["box"]}x{sources[0]["box"]} '
                f'stamps from every frame ...')
            stamps = extract_stamps(files, sources, hdu_index)
    else:
        log('[3/4] source detection skipped (no flux map, or disabled)', 'warn')

    # --- 4. write ----------------------------------------------------------
    log('[4/4] writing the output MEF ...')
    out_path = _resolve(cfg, output_path(cfg, args.output))
    write_mef(out_path, cfg, files, edges, cube, flux, flux_err, mu_lo, mu_hi,
              sources, stamps, n_under, n_over,
              overwrite=bool(cfg.get('output', {}).get('overwrite', True)))
    log('done')


if __name__ == '__main__':
    main()
