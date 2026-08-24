#!/usr/bin/env python3
"""
run_chunks.py -- walk a long EMCCD sequence in contiguous chunks of frames
==========================================================================

Authors: Etienne Artigau, Galina Sherren, Rene Doyon, Jonathan St-Antoine
         Universite de Montreal / Observatoire du Mont-Megantic

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

  * embin_chunk00.fits, embin_chunk01.fits, ... -- one complete embin result per
    chunk: histogram cube, flux map, error map, header table, raw star stamps;
  * chunk_summary.fits -- the flux maps of every chunk as one (n_chunk, ny, nx)
    cube, the matching error cube, a table describing each chunk (which frames,
    which times), and the LIGHT CURVES: every star tracked from chunk to chunk,
    with an aperture flux measured on each chunk's flux map;
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

Every parameter -- which folder the frames are in, the bin edges, the detector
constants, the chunk size, where the output goes -- lives in embin_config.yaml,
the same file embin.py reads, so the two programs can never disagree about the
binning. The command-line flags above only override it for one run.
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import (_resolve, all_frames, build_flux_grid,  # noqa: E402
                   build_histogram_cube, detect_sources, extract_stamps,
                   fit_flux_image, get_edges, load_config, log, write_mef)


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
        ids = np.unique(tr['track'])
        keep = [i for i in ids if (tr['track'] == i).sum() >= max(3, len(chunks) // 2)]
        keep = keep[:8]
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
                    verdict = ('variable' if v['sigma'] >= 5
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
    summary_name = str(out_cfg.get('summary', 'chunk_summary.fits'))
    overwrite = bool(out_cfg.get('overwrite', True))

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

        out = os.path.join(outdir, f'{prefix}{c:02d}.fits')
        cfg['input']['first_file'] = first          # so the MEF header records the chunk
        cfg['input']['n_files'] = size
        write_mef(out, cfg, sub, edges, cube, flux, flux_err, mu_lo, mu_hi,
                  sources, stamps, n_under, n_over, overwrite=overwrite)

        times = np.array([frame_time(p, hdu_index) for p in (sub[0], sub[-1])]) - t0
        chunks.append({
            'index': c, 'first': first, 'nframes': len(sub), 'n_dropped': n_dropped,
            'first_file': os.path.basename(sub[0]), 'last_file': os.path.basename(sub[-1]),
            't_start': times[0], 't_end': times[1], 't_mid': 0.5 * (times[0] + times[1]),
            'sky': sky,
            'median_err': float(np.median(flux_err)) if flux_err is not None else np.nan,
            'n_sources': len(sources),
            'file': os.path.basename(out),
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
            level = 'value' if v['sigma'] >= 5 else 'warn'
            log(f"  track {v['track']:2d}  <f> = {v['flux_mean']:8.4f} e-/frame  "
                f"rms {v['rms_pct']:5.2f} %  chi2/dof = {v['chi2_red']:6.2f} "
                f"({v['chi2']:.1f}/{v['dof']:d})  p = {v['p_value']:.2e}  "
                f"{v['sigma']:5.1f} sigma"
                + ('' if v['sigma'] >= 5 else '  -> consistent with constant'),
                level)

    summary = os.path.join(outdir, summary_name)
    write_summary(summary, cfg, edges, chunks, flux_cube, err_cube, rows, aperture,
                  overwrite=overwrite)

    if flux_cube is not None and want_figures:
        make_figures(figdir, chunks, rows, flux_cube)
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
