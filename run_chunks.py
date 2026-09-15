#!/usr/bin/env python3
"""
run_chunks.py -- walk a long EMCCD sequence in contiguous chunks of frames
==========================================================================

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

WHAT THIS DOES
--------------
`embin.py` turns ONE set of frames into one per-pixel histogram cube and one
fitted mean-flux map. That is the right product for a static field, but a real
acquisition drifts: on the PESTO test sequence the field moves at
about 0.16 pix/s, so a per-pixel histogram of all 1000 frames smears every star
over ~7 pixels and the "mean flux of pixel (y, x)" stops meaning anything about
a star.

This script therefore cuts the sequence into contiguous CHUNKS of
`chunks.size` frames, runs the full embin analysis on each chunk on its own, and
then stitches the chunk results into one time-resolved product. Everything lands
in the folder named by `output.directory` in the YAML (`data_bin` by default):

  * embin_chunk00.fits, embin_chunk01.fits, ... -- one CUBE file per chunk:
    the histogram cube, the header table and the raw star stamps, and nothing
    fitted;
  * embin_chunk00_flux.fits, ... -- one FLUX file per chunk: that chunk's mean
    flux map and its error, in two extensions. These are fits to the cube
    beside them, so they need not be archived: embin.flux_maps_from_cube()
    rebuilds them from the cube file alone;
  * chunk_summary.fits -- the flux maps of every chunk as one (n_chunk, ny, nx)
    cube, the matching error cube, a table describing each chunk (which frames,
    which times), and the LIGHT CURVES: every star tracked from chunk to chunk,
    with an aperture flux measured on each chunk's flux map;
  * chunk_lightcurves.csv, the photometry as one table: one row per chunk,
    with flux_starK and eflux_starK for every star, the chunk's times and UTC
    dates, the sky, the drift and the raw-frame keywords; and chunk_stars.csv,
    one row per star, with its position and its variability statistics;
  * figures/*.pdf -- the light curves, the measured drift of the field, and a
    montage of the per-chunk flux maps.

The chunk length is the usual exposure-time trade-off in a new guise. Long
chunks give each pixel more reads, so a finer flux measurement, but they smear
the star further; short chunks freeze the field but leave few reads per pixel.
64 frames on the PESTO test sequence is 3.1 s, about half a pixel of drift.

The frames left over at the end (the remainder of N / chunk size) are DROPPED,
and the program says so on screen. It never pretends to have used them.

USAGE
-----
    python run_chunks.py                    # settings from embin_config.yaml
    python run_chunks.py --n-chunks 2       # a quick trial on two chunks first
    python run_chunks.py --chunk-size 128 --no-stamps
    python run_chunks.py --csv-only         # rewrite the two tables, no binning

Every parameter -- which folder the frames are in, the bin edges, the detector
constants, the chunk size, where the output goes -- lives in embin_config.yaml,
the same file embin.py reads, so the two programs can never disagree about the
binning. The command-line flags above only override it for one run.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import UnitsWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import (_resolve, all_frames, build_flux_grid,  # noqa: E402
                   build_histogram_cube, detect_sources, extract_stamps,
                   fit_flux_image, flux_path_for, get_edges, gz_path,
                   load_config, log, write_cube_mef, write_flux_mef)


# ---------------------------------------------------------------------------
# The frame list, and how it is cut up
# ---------------------------------------------------------------------------
def frame_time(path, hdu_index=0):
    """Acquisition time of one frame, in seconds.

    HOSTTIME is the camera's own free-running clock in MILLISECONDS (consecutive
    PESTO frames differ by 48.97, matching the 48.971 ms EXPOSURE), and it is
    the only high-resolution timestamp in these headers -- DATE is truncated to
    the millisecond. Frames without it fall back to NaN rather than to a
    fabricated cadence.
    """
    hdr = fits.getheader(path, ext=hdu_index)
    if 'HOSTTIME' in hdr:
        return float(hdr['HOSTTIME']) / 1000.0
    return np.nan


# ---------------------------------------------------------------------------
# Photometry on a chunk's flux map
# ---------------------------------------------------------------------------
def aperture_flux(flux, flux_err, sky, xcen, ycen, radius):
    """Sky-subtracted flux [e-/frame] inside a circular aperture, and its error.

    The flux map is already the per-pixel mean flux, so a star's total flux is
    the plain sum of its pixels once the sky level has been removed from each of
    them; no PSF model enters. Pixel errors are added in quadrature, which is
    right because each pixel was fitted from its own frames independently.
    """
    ny, nx = flux.shape
    r = int(np.ceil(radius))
    x0, x1 = max(0, int(round(xcen)) - r), min(nx, int(round(xcen)) + r + 1)
    y0, y1 = max(0, int(round(ycen)) - r), min(ny, int(round(ycen)) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan, np.nan

    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - xcen) ** 2 + (yy - ycen) ** 2 <= radius ** 2
    if not inside.any():
        return np.nan, np.nan

    cut = flux[y0:y1, x0:x1][inside] - sky
    err = flux_err[y0:y1, x0:x1][inside] if flux_err is not None else None
    total = float(cut.sum())
    total_err = float(np.sqrt((err ** 2).sum())) if err is not None else np.nan
    return total, total_err


def sky_level(flux, reject_pct=99.0):
    """The background flux [e-/frame] of one chunk's flux map.

    NOT the median: over 64 frames a background pixel usually collects zero
    electrons, so the per-pixel MLE piles up against the bottom of the flux grid
    and the median of the map is that floor, not the sky. The MEAN of the
    source-free pixels is the honest estimate, because the MLE is unbiased in
    the mean even when any single pixel of it is uninformative. `reject_pct`
    drops the brightest percent of pixels (the stars, the hot pixels) before
    averaging; stars cover far less than that here.
    """
    cut = np.percentile(flux, reject_pct)
    return float(np.mean(flux[flux < cut]))


def link_tracks(per_chunk_sources, tol):
    """Chain each chunk's detections into tracks that follow one star in time.

    Consecutive chunks are matched nearest-neighbour within `tol` pixels. The
    field drifts by well under a pixel per chunk here, so the match is never
    ambiguous; chaining (rather than matching everything back to chunk 0) is
    what lets the total drift be many pixels without widening the tolerance.

    Returns a list of tracks, each a dict {chunk_index: source dict}.
    """
    tracks = []
    for c, sources in enumerate(per_chunk_sources):
        # Where every open track was last seen, and how bright it was there.
        taken = set()
        for tr in tracks:
            last = tr[max(tr)]
            best, best_d = None, tol
            for j, s in enumerate(sources):
                if j in taken:
                    continue
                d = np.hypot(s['xcen'] - last['xcen'], s['ycen'] - last['ycen'])
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                tr[c] = sources[best]
                taken.add(best)
        for j, s in enumerate(sources):
            if j not in taken:
                tracks.append({c: s})
    # Longest, then brightest first.
    tracks.sort(key=lambda tr: (-len(tr), -max(s['peak_flux'] for s in tr.values())))
    return tracks


# ---------------------------------------------------------------------------
# Is the star actually variable?
# ---------------------------------------------------------------------------
def variability_stats(rows):
    """
    Test each light curve against the hypothesis "this star did not vary".

    The null model is a single constant flux, fitted as the inverse-variance
    weighted mean of the chunk fluxes.  Under that model

        chi2 = sum_c (f_c - fbar)^2 / sigma_c^2

    follows a chi-square law with n-1 degrees of freedom, so chi2/dof near 1
    means the scatter is fully explained by the quoted photon errors and there
    is nothing left to call variability.  A large chi2 means either the star
    varied or the errors are underestimated, which is why the excess scatter is
    reported alongside: it says HOW BIG the unexplained part is, in percent,
    while the p-value says how sure we are that it is there at all.

    Returns an astropy Table with one row per track.
    """
    from scipy.stats import chi2 as chi2_dist
    from scipy.special import ndtri

    tr = Table(rows) if not isinstance(rows, Table) else rows
    out = []
    for i in np.unique(tr['track']):
        m = tr['track'] == i
        f = np.asarray(tr['flux'][m], dtype=float)
        e = np.asarray(tr['flux_err'][m], dtype=float)
        good = np.isfinite(f) & np.isfinite(e) & (e > 0)
        f, e = f[good], e[good]
        n = f.size
        if n < 2:
            continue
        w = 1.0 / e ** 2
        fbar = float(np.sum(w * f) / np.sum(w))
        chi2 = float(np.sum(((f - fbar) / e) ** 2))
        dof = n - 1
        red = chi2 / dof
        # Survival function of the chi-square: the probability that a
        # non-variable star of this brightness would scatter at least this much.
        pval = float(chi2_dist.sf(chi2, dof))
        # The same thing as a one-sided Gaussian significance, which is the
        # number people actually want to quote.  Clipped so that a p-value that
        # underflows to zero does not become an infinity.
        sig = float(-ndtri(max(pval, 1e-300)))
        rms = float(np.std(f, ddof=1))
        # Scatter left over once the photon noise is taken out.  Negative under
        # the root means the curve is quieter than its own error bars; report 0.
        excess = float(np.sqrt(max(rms ** 2 - np.mean(e ** 2), 0.0)))
        out.append({'track': int(i), 'n': n, 'flux_mean': fbar,
                    'flux_rms': rms, 'median_err': float(np.median(e)),
                    'chi2': chi2, 'dof': dof, 'chi2_red': red,
                    'p_value': pval, 'sigma': sig,
                    'rms_pct': 100 * rms / fbar if fbar > 0 else np.nan,
                    'excess_rms_pct': 100 * excess / fbar if fbar > 0 else np.nan})
    t = Table(out) if out else Table(
        names=('track', 'n', 'flux_mean', 'flux_rms', 'median_err', 'chi2',
               'dof', 'chi2_red', 'p_value', 'sigma', 'rms_pct',
               'excess_rms_pct'),
        dtype=(int, int, float, float, float, float, int, float, float, float,
               float, float))
    for col in ('flux_mean', 'flux_rms', 'median_err'):
        if col in t.colnames:
            t[col].unit = 'e-/frame'
    return t


# A light curve is called variable when a constant flux is rejected at this
# many sigma. It decides the verdict in each panel title of
# chunk_lightcurves.pdf, the colour of each track's log line and the
# lc_verdict column of the light-curve CSV, so the three always agree.
VARIABLE_SIGMA = 5.0


def figure_tracks(tracks, n_chunks, max_panels=8):
    """The tracks that get a panel in chunk_lightcurves.pdf.

    Those seen in at least half the chunks, and in no fewer than three, up to
    `max_panels` of them. Track numbers already run from the longest and
    brightest track down, so the ones kept are the best-measured stars.
    """
    ids = np.unique(tracks['track'])
    keep = [i for i in ids if (tracks['track'] == i).sum() >= max(3, n_chunks // 2)]
    return keep[:max_panels]


# ---------------------------------------------------------------------------
# The light curves as tables
# ---------------------------------------------------------------------------
# A run writes two CSVs, and between them they hold every number behind
# figures/chunk_lightcurves.pdf:
#
#   chunk_lightcurves.csv   one row per chunk: when it was and which frames it
#                           is, then flux_starK and eflux_starK for every star,
#                           then the sky, the drift and the keywords of the raw
#                           frames. A star not detected in a chunk (out of the
#                           field, or under the detection threshold) is NaN
#                           there, so every row carries the same columns.
#   chunk_stars.csv         one row per star, star1 first: where it sits on the
#                           sky and on the detector, in how many chunks it was
#                           measured, and its variability statistics.
#
# Both are ECSV: a plain comma-separated table under '#' lines that carry each
# column's meaning and every keyword of the run.

# Keywords of the raw frames carried into the per-chunk table. One that holds
# the same value in every frame used is written once, in the file's header; one
# that changes becomes a column, averaged over each chunk's frames (for text,
# its distinct values in the chunk joined by '|'). Extend the list freely: a
# keyword the frames do not have is skipped.
CSV_FRAME_KEYWORDS = (
    'OBJECT', 'PROGRAM', 'OBSERVER', 'OPERATOR', 'TELESCOP', 'INSTRUME',
    'ORIGIN', 'FILTER', 'RA', 'DEC', 'EPOCH', 'SERIAL', 'SOFT_VER', 'EXPOSURE',
    'EFF_EXP', 'WAIT_TIM', 'EM_CGAIN', 'KGAIN_01', 'ANA_GAIN', 'ANA_OFF',
    'H_FREQ', 'V_FREQ', 'BIN_X', 'BIN_Y', 'SHUTTER', 'SET_TEMP', 'TEMP_CCD',
    'AIRMASS', 'HA', 'UT', 'ST', 'FOCUS', 'ROTATOR', 'PESTOROT', 'PESTOMIR',
    'DOME', 'TEMPOUT', 'HUMOUT', 'TEMPIN', 'HUMIN', 'TEMPM')

# What each column means, written into the file's header. Units sit in the
# brackets rather than in astropy's unit slot, which has no 'e-/frame' and
# would warn on every read. A column not listed here still goes out.
CHUNK_COLUMNS = {
    'date_mid': 'middle of the chunk [UTC], from the DATE keyword of its first '
                'and last frame',
    'mjd_mid': 'date_mid as a modified Julian date [d, UTC]',
    't_mid': 'middle of the chunk since the first frame, camera clock '
             '(HOSTTIME) [s]; the x axis of the figure',
    'first_file': 'first frame of the chunk',
    'last_file': 'last frame of the chunk',
    'chunk': 'chunk index, in file order',
    'first_index': "position of the chunk's first frame in the file list",
    'nframes': 'frames in the chunk',
    't_start': 'first frame of the chunk, since the first frame, camera clock [s]',
    't_end': 'last frame of the chunk, since the first frame, camera clock [s]',
    'date_start': "DATE keyword of the chunk's first frame [UTC]",
    'date_end': "DATE keyword of the chunk's last frame [UTC]",
    'time_ordered': 'false if a frame of the chunk is earlier than the one '
                    'before it: the chunk straddles a jump back in time and '
                    'its flux mixes two moments',
    'max_gap': 'longest interval between two consecutive frames of the chunk, '
               'camera clock [s]; a jump back in time counts by its size, so '
               'this is large wherever time_ordered is false',
    'sky': "sky level of the chunk's flux map, already subtracted from every "
           'flux in this row [e-/frame]',
    'map_median_err': "median per-pixel error of the chunk's flux map [e-/frame]",
    'n_sources': 'sources detected in the chunk',
    'file': "the chunk's histogram-cube file",
    'flux_file': "the chunk's flux-map file",
    'ra_centre': 'right ascension of the field centre in this chunk [deg]',
    'dec_centre': 'declination of the field centre in this chunk [deg]',
    'dx_drift': 'shift of this chunk onto the astrometric stack, x '
                '(stack pixel = chunk pixel + shift) [pix]',
    'dy_drift': 'shift of this chunk onto the astrometric stack, y [pix]',
    # The telescope control system writes these in hours, not degrees: PESTO's
    # RA x 15 is the RA of the field centre, and its ST minus RA is its HA.
    'RA': "telescope right ascension, mean over the chunk's frames [h]",
    'DEC': "telescope declination, mean over the chunk's frames [deg]",
    'HA': "hour angle, mean over the chunk's frames [h]",
    'UT': "UT of the telescope control system, mean over the chunk's frames [h]",
    'ST': "sidereal time, mean over the chunk's frames [h]",
}

STAR_COLUMNS = {
    'star': 'star label: the flux_ and eflux_ columns of the per-chunk table '
            'carry it',
    'track': 'track number in the summary (star1 is track 0)',
    'ra': 'right ascension, mean over the chunks it was seen in [deg]',
    'dec': 'declination, mean over the chunks it was seen in [deg]',
    'x_mean': 'mean x on the detector [pix], as in the panel title of the figure',
    'y_mean': 'mean y on the detector [pix], as in the panel title of the figure',
    'x_rms': 'rms of x over the chunks [pix]: the field drift, mostly',
    'y_rms': 'rms of y over the chunks [pix]',
    'n_valid': 'chunks with a flux measurement; the star is NaN in the other '
               'rows of the per-chunk table',
    'n_chunks': 'chunks in the run',
    'first_chunk': 'first chunk the star was detected in',
    'last_chunk': 'last chunk the star was detected in',
    'flux_mean': 'best constant flux, inverse-variance mean [e-/frame]; the '
                 'dashed line of the figure',
    'flux_rms': 'rms of the light curve [e-/frame]',
    'median_err': 'median flux error of the light curve [e-/frame]',
    'rms_pct': 'flux_rms in percent of flux_mean [%]',
    'excess_rms_pct': 'scatter left once the photon errors are removed, in '
                      'percent of flux_mean [%]',
    'chi2': 'chi2 of the light curve against flux_mean',
    'dof': 'degrees of freedom of chi2',
    'chi2_red': 'chi2 / dof',
    'p_value': 'probability that a constant star scatters at least this much',
    'sigma': 'p_value as a one-sided Gaussian significance [sigma]',
    'verdict': f'variable if sigma >= {VARIABLE_SIGMA:g}, as the panel title '
               f'says; not tested when one chunk is all there is',
    'in_figure': 'true if the star has a panel in figures/chunk_lightcurves.pdf',
    'peak_flux_median': 'median peak flux of the detection [e-/frame]',
    'signif_median': 'median detection significance [sigma]',
}

STAR_COLUMN_ORDER = ('star', 'track', 'ra', 'dec', 'x_mean', 'y_mean',
                     'n_valid', 'n_chunks', 'first_chunk', 'last_chunk',
                     'flux_mean', 'flux_rms', 'median_err', 'rms_pct',
                     'excess_rms_pct', 'chi2', 'dof', 'chi2_red', 'p_value',
                     'sigma', 'verdict', 'in_figure', 'peak_flux_median',
                     'signif_median', 'x_rms', 'y_rms')


def _frame_columns(chunks, frames, hdu_index):
    """Dates, time order and raw-frame keywords of every chunk, from the headers.

    Returns (columns, constant): per-chunk columns as name -> list, one entry
    per row of `chunks`, and the CSV_FRAME_KEYWORDS that hold one value over
    every frame used, as name -> (value, comment). (None, None) when `frames`
    is not the file list the chunks were cut from.
    """
    headers = []
    for row in chunks:
        first, n = int(row['first_index']), int(row['nframes'])
        sub = frames[first:first + n]
        if (len(sub) != n or os.path.basename(sub[0]) != row['first_file']
                or os.path.basename(sub[-1]) != row['last_file']):
            log(f"chunk {row['chunk']} was cut from {row['first_file']} .. "
                f"{row['last_file']}, which are not the frames found now; the "
                f"tables go out without dates or frame keywords", 'warn')
            return None, None
        headers.append([fits.getheader(p, ext=hdu_index) for p in sub])
    every = [h for chunk in headers for h in chunk]

    constant, varying = {}, []
    for key in CSV_FRAME_KEYWORDS:
        values = [h.get(key) for h in every]
        if all(v is None for v in values):
            continue
        if len({repr(v) for v in values}) == 1:
            constant[key] = (values[0], every[0].comments[key])
        else:
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                          for v in values)
            varying.append((key, numeric))

    columns = {}
    if all('DATE' in chunk[0] and 'DATE' in chunk[-1] for chunk in headers):
        t0 = Time([chunk[0]['DATE'] for chunk in headers], scale='utc')
        t1 = Time([chunk[-1]['DATE'] for chunk in headers], scale='utc')
        mid = t0 + 0.5 * (t1 - t0)
        columns.update(date_start=t0.isot, date_mid=mid.isot, date_end=t1.isot,
                       mjd_mid=mid.mjd)
    if all('HOSTTIME' in h for h in every):
        # The clock t_start and t_end come from: HOSTTIME, in milliseconds.
        steps = [np.diff([h['HOSTTIME'] for h in chunk]) / 1000.0
                 for chunk in headers]
        columns['time_ordered'] = [bool(np.all(s > 0)) for s in steps]
        columns['max_gap'] = [float(np.abs(s).max()) if s.size else 0.0
                              for s in steps]
    for key, numeric in varying:
        if numeric:
            columns[key] = [float(np.mean([h[key] for h in chunk]))
                            for chunk in headers]
        else:
            columns[key] = ['|'.join(dict.fromkeys(str(h.get(key)) for h in chunk))
                            for chunk in headers]
    return columns, constant


def _describe(name, known, labels, dtype):
    """What to say about one column in the file's header."""
    if name in known:
        return known[name]
    for prefix, what, unit in (
            ('flux_', 'sky-subtracted aperture flux', '[e-/frame]'),
            ('eflux_', '1-sigma error on the flux', '[e-/frame]'),
            ('x_', 'centroid on the detector, x', '[pix]'),
            ('y_', 'centroid on the detector, y', '[pix]')):
        if name.startswith(prefix) and name[len(prefix):] in labels:
            return (f'{what} of {name[len(prefix):]} {unit}, NaN in the chunks '
                    f'where it was not detected')
    if name in CSV_FRAME_KEYWORDS:
        return ("raw-frame keyword, mean over the chunk's frames"
                if dtype.kind in 'iuf' else
                "raw-frame keyword, the chunk's values joined by '|'")
    return ''


def photometry_tables(summary_path, frames=None, hdu_index=0):
    """The two light-curve tables, from a chunk summary already on disk.

    Returns (per_chunk, per_star) as astropy Tables, each with its column
    descriptions and the keywords of the run in `meta`, or (None, None) when
    the summary holds no light curves at all.

    `frames` is the file list the chunks were cut from; without it the UTC
    dates and the raw-frame keywords are left out.
    """
    with fits.open(summary_path) as hdul:
        names = {h.name for h in hdul}
        header = hdul[0].header.copy()
    if not {'CHUNKS', 'TRACKS'} <= names:
        log(f'{os.path.basename(summary_path)} holds no light curves, so there '
            f'are no photometry tables to write', 'warn')
        return None, None
    with warnings.catch_warnings():
        # 'e-/frame' is no astropy unit; the units go into the descriptions.
        warnings.simplefilter('ignore', UnitsWarning)
        chunks = Table.read(summary_path, hdu='CHUNKS')
        tracks = Table.read(summary_path, hdu='TRACKS')
        var = (Table.read(summary_path, hdu='VARSTAT')
               if 'VARSTAT' in names else None)

    ids = [int(i) for i in np.unique(tracks['track'])]
    label = {i: f'star{k}' for k, i in enumerate(ids, start=1)}
    row_of = {int(c): k for k, c in enumerate(chunks['chunk'])}
    shown = {int(i) for i in figure_tracks(tracks, len(chunks))}

    # --- the photometry, two columns per star ------------------------------
    # A star is measured in the chunks it was detected in and nowhere else, so
    # each column starts as NaN and only the chunks the star appears in are
    # filled. A star that drifts out of the field, or falls under the
    # detection threshold for a while, leaves NaN there rather than a hole in
    # the table: every row has the same columns whatever was in the field.
    per_star_arrays = {}
    for i in ids:
        cols = {name: np.full(len(chunks), np.nan)
                for name in ('flux', 'eflux', 'x', 'y')}
        for row in tracks[tracks['track'] == i]:
            k = row_of.get(int(row['chunk']))
            if k is None:
                continue
            cols['flux'][k], cols['eflux'][k] = row['flux'], row['flux_err']
            cols['x'][k], cols['y'][k] = row['x'], row['y']
        per_star_arrays[i] = cols

    # --- one row per chunk -------------------------------------------------
    base = Table(chunks, copy=True)
    if 'median_err' in base.colnames:
        base.rename_column('median_err', 'map_median_err')
    constant = {}
    if frames is not None:
        log(f'reading the headers of the frames behind the {len(chunks)} '
            f'chunks ...')
        columns, constant = _frame_columns(base, frames, hdu_index)
        for name, values in (columns or {}).items():
            base[name] = values
        constant = constant or {}

    per_chunk = Table()
    for name in ('date_mid', 'mjd_mid', 't_mid', 'first_file', 'last_file',
                 'chunk'):
        if name in base.colnames:
            per_chunk[name] = base[name]
    for i in ids:                      # the photometry, star by star
        per_chunk[f'flux_{label[i]}'] = per_star_arrays[i]['flux']
        per_chunk[f'eflux_{label[i]}'] = per_star_arrays[i]['eflux']
    for name in base.colnames:         # everything else the chunk knows
        if name not in per_chunk.colnames:
            per_chunk[name] = base[name]
    for i in ids:                      # where each star was, chunk by chunk
        per_chunk[f'x_{label[i]}'] = per_star_arrays[i]['x']
        per_chunk[f'y_{label[i]}'] = per_star_arrays[i]['y']

    # --- one row per star --------------------------------------------------
    stats = {int(r['track']): r for r in var} if var is not None else {}
    rows = []
    for i in ids:
        t = tracks[tracks['track'] == i]
        v = stats.get(i)
        row = {'star': label[i], 'track': i,
               'x_mean': float(np.mean(t['x'])), 'y_mean': float(np.mean(t['y'])),
               'x_rms': float(np.std(t['x'])), 'y_rms': float(np.std(t['y'])),
               'n_valid': int(np.isfinite(np.asarray(t['flux'], float)).sum()),
               'n_chunks': len(chunks),
               'first_chunk': int(np.min(t['chunk'])),
               'last_chunk': int(np.max(t['chunk'])),
               'peak_flux_median': float(np.median(t['peak_flux'])),
               'signif_median': float(np.median(t['signif'])),
               'in_figure': i in shown}
        if 'ra' in t.colnames:
            row['ra'] = float(np.mean(t['ra']))
            row['dec'] = float(np.mean(t['dec']))
        for name in ('flux_mean', 'flux_rms', 'median_err', 'chi2', 'dof',
                     'chi2_red', 'p_value', 'sigma', 'rms_pct',
                     'excess_rms_pct'):
            row[name] = float(v[name]) if v is not None else np.nan
        # A star seen in a single chunk has nothing to test: no scatter to
        # compare with its error bar, so it gets no verdict rather than a
        # reassuring one.
        row['verdict'] = ('not tested' if v is None else
                          'variable' if v['sigma'] >= VARIABLE_SIGMA else
                          'not significant')
        rows.append(row)
    per_star = Table(rows)
    per_star = per_star[[c for c in STAR_COLUMN_ORDER if c in per_star.colnames]]

    # --- the headers of both files -----------------------------------------
    def plain(value):
        return value.item() if isinstance(value, np.generic) else value

    skip = {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'COMMENT', 'HISTORY', ''}
    meta = {
        'summary_file': os.path.basename(summary_path),
        'figure': 'figures/chunk_lightcurves.pdf',
        'summary_keywords': {k: {'value': header[k], 'comment': header.comments[k]}
                             for k in header if k not in skip},
        'frame_keywords': {k: {'value': v, 'comment': c}
                           for k, (v, c) in constant.items()},
    }
    named = [c for c in ('track', 'ra', 'dec', 'x_mean', 'y_mean', 'n_valid')
             if c in per_star.colnames]
    per_chunk.meta = dict(
        meta, description=(
            'Aperture photometry from run_chunks.py, one row per chunk: the '
            'numbers behind figures/chunk_lightcurves.pdf. flux_starK and '
            'eflux_starK are the flux of star K and its error; NaN means the '
            'star was not detected in that chunk. Rows are in chunk (file) '
            'order, which is not time order when one folder holds more than '
            'one acquisition: sort on mjd_mid for that.'),
        stars={str(r['star']): {c: plain(r[c]) for c in named} for r in per_star})
    per_star.meta = dict(
        meta, description=(
            'One row per star tracked by run_chunks.py, star1 first. The star '
            'column names the flux_ and eflux_ columns of the per-chunk '
            'table, chunk_lightcurves.csv.'))

    labels = set(label.values())
    for table, known in ((per_chunk, CHUNK_COLUMNS), (per_star, STAR_COLUMNS)):
        for col in table.itercols():
            col.unit = None
            described = _describe(col.name, known, labels, col.dtype)
            if described:
                col.description = described
    return per_chunk, per_star


def write_photometry_csv(summary_path, csv_path, star_csv_path, frames=None,
                         hdu_index=0, overwrite=True):
    """Both light-curve tables, written where they are asked for."""
    per_chunk, per_star = photometry_tables(summary_path, frames, hdu_index)
    if per_chunk is None:
        return None, None
    per_chunk.write(csv_path, format='ascii.ecsv', delimiter=',',
                    overwrite=overwrite)
    per_star.write(star_csv_path, format='ascii.ecsv', delimiter=',',
                   overwrite=overwrite)
    log(f'wrote {csv_path}: {len(per_chunk)} chunks x '
        f'{len(per_chunk.colnames)} columns, {len(per_star)} stars in it', 'value')
    log(f'wrote {star_csv_path}: {len(per_star)} stars x '
        f'{len(per_star.colnames)} columns', 'value')
    return csv_path, star_csv_path


def photometry_csv_from_config(cfg, summary_path):
    """Write both tables beside a summary that is already on disk.

    What `--csv-only` does, and what pesto_astrometry.py calls once it has put
    sky coordinates into the summary, so the tables never lag behind it. The
    raw frames only add dates and keywords: if they are no longer on disk, the
    tables go out without those rather than the run stopping.
    """
    inp, out_cfg = cfg['input'], cfg.get('output') or {}
    pattern = os.path.join(_resolve(cfg, inp['directory']),
                           inp.get('pattern', '*.fits'))
    frames = all_frames(cfg) if glob.glob(pattern) else None
    if frames is None:
        log(f'no frame matches {pattern}; the photometry tables go out without '
            f'their dates and keywords', 'warn')
    outdir = os.path.dirname(summary_path)
    return write_photometry_csv(
        summary_path,
        os.path.join(outdir, str(out_cfg.get('lightcurve_csv',
                                             'chunk_lightcurves.csv'))),
        os.path.join(outdir, str(out_cfg.get('stars_csv', 'chunk_stars.csv'))),
        frames, int(inp.get('hdu', 0)),
        overwrite=bool(out_cfg.get('overwrite', True)))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_summary(path, cfg, edges, chunks, flux_cube, err_cube, tracks,
                  radius, overwrite=True):
    """The stitched, time-resolved product: flux cubes, chunk table, light curves."""
    hdr = fits.Header()
    hdr['ORIGIN'] = ('run_chunks.py', 'emccd_bintool')
    hdr['CONFIG'] = (os.path.basename(cfg['_config_path']), 'YAML configuration used')
    hdr['NCHUNK'] = (len(chunks), 'Number of frame chunks')
    hdr['CHUNKSZ'] = (chunks[0]['nframes'], 'Frames per chunk')
    hdr['NFRAMES'] = (sum(c['nframes'] for c in chunks), 'Frames used in total')
    hdr['NDROP'] = (chunks[0]['n_dropped'], 'Frames dropped at the end of the sequence')
    hdr['NBIN'] = (len(edges) - 1, 'Histogram bins per pixel per chunk')
    hdr['SATADU'] = (float(edges[-2]), 'Open last bin threshold [ADU]')
    hdr['APRAD'] = (float(radius), 'Light-curve aperture radius [pix]')
    hdr['NTRACK'] = (len(tracks), 'Rows in the TRACKS table')
    for key, value, comment in (
            ('BIAS', float(cfg['detector']['bias']), 'Electronics bias level [ADU]'),
            ('RON', float(cfg['detector']['ron']), 'Read-out noise, 1 sigma [ADU]'),
            ('GAIN', float(cfg['detector']['gain']), 'EM gain [ADU/e-]')):
        hdr[key] = (value, comment)
    hdr.append(('COMMENT', 'FLUX / FLUX_ERR: one fitted flux map per chunk,'), bottom=True)
    hdr.append(('COMMENT', 'stacked along axis 3 in chunk (time) order.'), bottom=True)
    hdr.append(('COMMENT', 'CHUNKS: which frames and which times each plane is.'), bottom=True)
    hdr.append(('COMMENT', 'TRACKS: one row per source per chunk - the light'), bottom=True)
    hdr.append(('COMMENT', 'curves, with positions, so drift is measurable too.'), bottom=True)
    hdr.append(('COMMENT', 'VARSTAT: one row per track - chi2 of the light curve'), bottom=True)
    hdr.append(('COMMENT', 'against a constant flux, and how significant it is.'), bottom=True)

    hdus = [fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data=flux_cube.astype(np.float32), name='FLUX'),
            fits.ImageHDU(data=err_cube.astype(np.float32), name='FLUX_ERR')]
    hdus[1].header['BUNIT'] = 'e-/frame'
    hdus[2].header['BUNIT'] = 'e-/frame'

    ctab = Table({
        'chunk': np.array([c['index'] for c in chunks], dtype=np.int32),
        'first_index': np.array([c['first'] for c in chunks], dtype=np.int32),
        'nframes': np.array([c['nframes'] for c in chunks], dtype=np.int32),
        'first_file': [c['first_file'] for c in chunks],
        'last_file': [c['last_file'] for c in chunks],
        't_start': np.array([c['t_start'] for c in chunks]),
        't_mid': np.array([c['t_mid'] for c in chunks]),
        't_end': np.array([c['t_end'] for c in chunks]),
        'sky': np.array([c['sky'] for c in chunks]),
        'median_err': np.array([c['median_err'] for c in chunks]),
        'n_sources': np.array([c['n_sources'] for c in chunks], dtype=np.int32),
        'file': [c['file'] for c in chunks],
        'flux_file': [c['flux_file'] for c in chunks],
    })
    ctab['t_start'].unit = ctab['t_mid'].unit = ctab['t_end'].unit = 's'
    hdus.append(fits.BinTableHDU(ctab, name='CHUNKS'))

    if tracks:
        hdus.append(fits.BinTableHDU(Table(tracks), name='TRACKS'))
        hdus.append(fits.BinTableHDU(variability_stats(tracks), name='VARSTAT'))

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    log(f'wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)', 'value')


def make_figures(figdir, chunks, tracks, flux_cube):
    """Light curves and measured drift, as PDFs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    t = np.array([c['t_mid'] for c in chunks])
    if not np.all(np.isfinite(t)):
        t = np.array([c['index'] for c in chunks], dtype=float)

    tr = Table(tracks) if tracks else None
    var = variability_stats(tr) if tr is not None else None
    paths = []

    # --- light curves ------------------------------------------------------
    if tr is not None:
        keep = figure_tracks(tr, len(chunks))
        if keep:
            fig, axes = plt.subplots(len(keep), 1, sharex=True,
                                     figsize=(7.5, 1.8 * len(keep) + 1.0))
            axes = np.atleast_1d(axes)
            for ax, i in zip(axes, keep):
                m = tr['track'] == i
                # Points only, no line between them: a line drawn through
                # independent chunk measurements invents a trend the data does
                # not contain, and it is exactly the trend the eye then reads
                # as variability.  The chi2 below is what decides that.
                ax.errorbar(tr['t_mid'][m], tr['flux'][m], yerr=tr['flux_err'][m],
                            fmt='o', ms=3.5, lw=0, elinewidth=1, capsize=2,
                            color='C0')
                v = var[var['track'] == i]
                if len(v):
                    v = v[0]
                    ax.axhline(v['flux_mean'], color='C3', lw=1.0, ls='--',
                               alpha=0.8)
                    verdict = ('variable' if v['sigma'] >= VARIABLE_SIGMA
                               else 'not significant')
                    note = (f"track {i}  (x, y) = ({np.mean(tr['x'][m]):.0f}, "
                            f"{np.mean(tr['y'][m]):.0f})   "
                            f"rms {v['rms_pct']:.1f} %   "
                            f"$\\chi^2$/dof {v['chi2_red']:.2f} "
                            f"({v['chi2']:.1f}/{v['dof']:d})   "
                            f"{v['sigma']:.1f}$\\sigma$: {verdict}")
                else:
                    note = (f"track {i}  (x, y) = ({np.mean(tr['x'][m]):.0f}, "
                            f"{np.mean(tr['y'][m]):.0f})")
                ax.set_ylabel('e$^-$/frame')
                # Above the axes, not inside them: with no connecting line the
                # points spread over the full height and a floating label lands
                # on top of them.
                ax.set_title(note, fontsize=8, loc='left', pad=3)
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel('time since the first frame [s]')
            fig.suptitle(
                f'Aperture light curves, {chunks[0]["nframes"]}-frame chunks'
                '\n(dashed line: best constant flux; $\\chi^2$ is measured '
                'against it)', fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            p = os.path.join(figdir, 'chunk_lightcurves.pdf')
            fig.savefig(p)
            plt.close(fig)
            paths.append(p)

        # --- drift ---------------------------------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for i in keep:
            m = tr['track'] == i
            ax[0].plot(tr['t_mid'][m], tr['x'][m] - tr['x'][m][0], 'o-', ms=3, lw=1)
            ax[1].plot(tr['t_mid'][m], tr['y'][m] - tr['y'][m][0], 'o-', ms=3, lw=1)
        for a, lab in zip(ax, ('x', 'y')):
            a.set_xlabel('time since the first frame [s]')
            a.set_ylabel(f'{lab} - {lab}(first chunk) [pix]')
            a.grid(alpha=0.25)
        ax[0].set_title('Measured field drift, chunk by chunk')
        fig.tight_layout()
        p = os.path.join(figdir, 'chunk_drift.pdf')
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # --- flux-map montage --------------------------------------------------
    n = len(chunks)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 1.6 * nrow))
    vmax = float(np.nanpercentile(flux_cube, 99.98))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        if k >= n:
            ax.axis('off')
            continue
        ax.imshow(flux_cube[k], origin='lower', vmin=0, vmax=vmax, cmap='inferno')
        ax.set_title(f'chunk {k}   t = {chunks[k]["t_mid"]:.1f} s', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    p = os.path.join(figdir, 'chunk_flux_maps.pdf')
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    for p in paths:
        log(f'figure: {p}', 'value')


# ---------------------------------------------------------------------------
def main(argv=None):
    """Bin a whole sequence. `argv` lets a wrapper drive this without a shell."""
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', default=os.path.join(here, 'embin_config.yaml'),
                    help='YAML configuration (the same one embin.py reads)')
    # Every default below is None on purpose: a flag that is not typed leaves
    # the value from the `chunks:` section of the YAML alone. The command line
    # is for trying something once, the YAML is where settings live.
    ap.add_argument('--chunk-size', type=int, default=None,
                    help='frames per chunk (default: chunks.size in the YAML)')
    ap.add_argument('--n-chunks', type=int, default=None,
                    help='how many chunks to run (default: chunks.n_chunks, '
                         'or as many complete ones as fit)')
    ap.add_argument('--outdir', default=None,
                    help='where the results go (default: output.directory in the YAML)')
    ap.add_argument('--aperture', type=float, default=None,
                    help='light-curve aperture radius [pix] (default: chunks.aperture)')
    ap.add_argument('--match-radius', type=float, default=None,
                    help='chunk-to-chunk source matching tolerance [pix] '
                         '(default: chunks.match_radius)')
    ap.add_argument('--no-stamps', action='store_true',
                    help='skip the raw postage stamps: much smaller output, and the '
                         'frames are then read only once')
    ap.add_argument('--csv-only', action='store_true',
                    help='bin nothing: write the photometry tables from the '
                         'chunk summary of an earlier run, and stop')
    args = ap.parse_args(argv)

    log(f'reading configuration from {args.config}')
    cfg = load_config(args.config)
    edges = get_edges(cfg)
    hdu_index = cfg['input'].get('hdu', 0)

    ch = cfg.get('chunks', {})
    out_cfg = cfg.get('output', {})
    size = int(args.chunk_size if args.chunk_size is not None else ch.get('size', 64))
    want = args.n_chunks if args.n_chunks is not None else ch.get('n_chunks')
    aperture = float(args.aperture if args.aperture is not None else ch.get('aperture', 3.0))
    match_radius = float(args.match_radius if args.match_radius is not None
                         else ch.get('match_radius', 4.0))
    want_stamps = bool(ch.get('stamps', True)) and not args.no_stamps
    want_figures = bool(ch.get('figures', True))

    outdir = args.outdir or _resolve(cfg, out_cfg.get('directory', 'data_bin'))
    figdir = os.path.join(outdir, str(out_cfg.get('figures', 'figures')))
    prefix = str(out_cfg.get('chunk_prefix', 'embin_chunk'))
    overwrite = bool(out_cfg.get('overwrite', True))
    # Every product is gzipped unless the config says otherwise: these files are
    # mostly small counts and empty sky, and compress by an order of magnitude.
    compress = bool(out_cfg.get('compress', True))
    summary_name = gz_path(str(out_cfg.get('summary', 'chunk_summary.fits')), compress)
    # Never gzipped, these two: they are meant to be opened in a spreadsheet.
    csv_name = str(out_cfg.get('lightcurve_csv', 'chunk_lightcurves.csv'))
    stars_name = str(out_cfg.get('stars_csv', 'chunk_stars.csv'))

    # Rebuilding the tables needs the summary and the frame headers, not the
    # binning: after pesto_astrometry.py has added sky coordinates, or after a
    # change to what goes in them, that is seconds rather than the whole run.
    if args.csv_only:
        summary = os.path.join(outdir, summary_name)
        if not os.path.exists(summary):
            log(f'--csv-only needs {summary}, which is not there yet: run '
                f'without it first', 'error')
            sys.exit(1)
        photometry_csv_from_config(cfg, summary)
        return 0

    if size < 2:
        log(f'chunks.size = {size} makes no sense; it must be at least 2', 'error')
        sys.exit(1)

    files = all_frames(cfg)
    log(f'{len(files)} frames found', 'value')
    n_full = len(files) // size
    if n_full == 0:
        log(f'only {len(files)} frame(s) found, which is fewer than one chunk of '
            f'{size} - lower chunks.size or point input.directory somewhere else', 'error')
        sys.exit(1)
    n_chunks = n_full if want in (None, 0) else min(int(want), n_full)
    n_used = n_chunks * size
    n_dropped = len(files) - n_used
    log(f'{n_chunks} chunks of {size} frames = {n_used} frames used', 'value')
    if n_dropped:
        log(f'{n_dropped} frame(s) at the end of the sequence dropped: '
            f'{os.path.basename(files[n_used])} .. {os.path.basename(files[-1])}', 'warn')

    log(f'{len(edges) - 1} histogram bins from {edges[0]:g} to {edges[-2]:g} ADU '
        f'(last bin open-ended)', 'value')

    os.makedirs(outdir, exist_ok=True)
    log(f'results will be written to {outdir}', 'value')
    t0 = frame_time(files[0], hdu_index)

    # The flux grid depends only on the detector and the binning, so it is built
    # once here and handed to every chunk.
    nll_grid = build_flux_grid(cfg, edges) if cfg.get('fit', {}).get('enabled', True) else None

    chunks, per_chunk_sources = [], []
    flux_cube = err_cube = None

    for c in range(n_chunks):
        first = c * size
        sub = files[first:first + size]
        log(f'=== chunk {c + 1}/{n_chunks}: frames {first}..{first + size - 1} '
            f'({os.path.basename(sub[0])} .. {os.path.basename(sub[-1])}) ===')

        cube, n_under, n_over = build_histogram_cube(sub, edges, hdu_index)

        flux = flux_err = mu_lo = mu_hi = None
        sky = np.nan
        if nll_grid is not None:
            flux, flux_err, mu_lo, mu_hi = fit_flux_image(cube, edges, cfg, nll_grid=nll_grid)
            log(f'flux map: median {np.median(flux):.4f}, '
                f'99.9th pct {np.percentile(flux, 99.9):.3f}, '
                f'max {flux.max():.3f} e-/frame', 'value')
            if not cfg.get('fit', {}).get('asymmetric_bounds', False):
                mu_lo = mu_hi = None
            if flux_cube is None:
                flux_cube = np.empty((n_chunks,) + flux.shape, dtype=np.float32)
                err_cube = np.empty_like(flux_cube)
            flux_cube[c] = flux
            err_cube[c] = flux_err
            sky = sky_level(flux)
            log(f'sky level (mean of the source-free pixels): {sky:.4f} e-/frame', 'value')

        sources, stamps = [], []
        if flux is not None and cfg.get('sources', {}).get('enabled', True):
            sources = detect_sources(flux, flux_err, cfg)
            for i, s in enumerate(sources, start=1):
                log(f'  source {i:2d}: (x, y) = ({s["xcen"]:7.2f}, {s["ycen"]:7.2f})  '
                    f'peak {s["peak_flux"]:.3f} +/- {s["peak_err"]:.3f} e-/frame  '
                    f'{s["signif"]:6.1f} sigma', 'value')
            if sources and want_stamps:
                stamps = extract_stamps(sub, sources, hdu_index)
        per_chunk_sources.append(sources)

        out = gz_path(os.path.join(outdir, f'{prefix}{c:02d}.fits'), compress)
        flux_out = flux_path_for(out)
        cfg['input']['first_file'] = first          # so the MEF header records the chunk
        cfg['input']['n_files'] = size
        write_cube_mef(out, cfg, sub, edges, cube, sources, stamps, n_under, n_over,
                       flux_file=flux_out if flux is not None else None,
                       overwrite=overwrite)
        if flux is not None:
            write_flux_mef(flux_out, flux, flux_err, mu_lo, mu_hi,
                           cube_header=fits.getheader(out, 0), cube_file=out,
                           overwrite=overwrite)

        times = np.array([frame_time(p, hdu_index) for p in (sub[0], sub[-1])]) - t0
        chunks.append({
            'index': c, 'first': first, 'nframes': len(sub), 'n_dropped': n_dropped,
            'first_file': os.path.basename(sub[0]), 'last_file': os.path.basename(sub[-1]),
            't_start': times[0], 't_end': times[1], 't_mid': 0.5 * (times[0] + times[1]),
            'sky': sky,
            'median_err': float(np.median(flux_err)) if flux_err is not None else np.nan,
            'n_sources': len(sources),
            'file': os.path.basename(out),
            'flux_file': os.path.basename(flux_out) if flux is not None else '',
        })

    # --- light curves ------------------------------------------------------
    rows = []
    tracks = link_tracks(per_chunk_sources, match_radius)
    log(f'{len(tracks)} track(s) linked across the chunks '
        f'({sum(len(t) for t in tracks)} detections)', 'value')
    for t_id, tr in enumerate(tracks):
        for c, s in sorted(tr.items()):
            f, fe = aperture_flux(flux_cube[c], err_cube[c], chunks[c]['sky'],
                                  s['xcen'], s['ycen'], aperture)
            rows.append({'track': t_id, 'chunk': c, 't_mid': chunks[c]['t_mid'],
                         'x': s['xcen'], 'y': s['ycen'],
                         'flux': f, 'flux_err': fe,
                         'peak_flux': s['peak_flux'], 'peak_err': s['peak_err'],
                         'signif': s['signif']})

    # Which of these light curves is actually saying something?
    var = variability_stats(rows)
    if len(var):
        log('variability of each track, against a constant flux:', 'info')
        for v in var:
            level = 'value' if v['sigma'] >= VARIABLE_SIGMA else 'warn'
            log(f"  track {v['track']:2d}  <f> = {v['flux_mean']:8.4f} e-/frame  "
                f"rms {v['rms_pct']:5.2f} %  chi2/dof = {v['chi2_red']:6.2f} "
                f"({v['chi2']:.1f}/{v['dof']:d})  p = {v['p_value']:.2e}  "
                f"{v['sigma']:5.1f} sigma"
                + ('' if v['sigma'] >= VARIABLE_SIGMA
                   else '  -> consistent with constant'),
                level)

    summary = os.path.join(outdir, summary_name)
    write_summary(summary, cfg, edges, chunks, flux_cube, err_cube, rows, aperture,
                  overwrite=overwrite)
    write_photometry_csv(summary, os.path.join(outdir, csv_name),
                         os.path.join(outdir, stars_name), files, hdu_index,
                         overwrite=overwrite)

    if flux_cube is not None and want_figures:
        make_figures(figdir, chunks, rows, flux_cube)
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
