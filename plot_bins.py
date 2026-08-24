#!/usr/bin/env python3
"""
plot_bins.py -- see where your bin edges land on your own data
==============================================================

Reads a handful of your raw frames, builds the histogram of ALL their pixel
values at 1 ADU resolution, and draws the bin edges of `histogram.edges` on top
of it. This is the picture to look at before trusting a set of bin edges, and
the picture to show when explaining what the binning does.

Authors: Etienne Artigau, Galina Sherren, Rene Doyon, Jonathan St-Antoine
         Universite de Montreal / Observatoire du Mont-Megantic

What you should see on a healthy EMCCD:

  * a tall, narrow peak at the bias level -- the read noise. This is where the
    overwhelming majority of pixels sit, because at these fluxes most pixels see
    no electron at all in a given frame;
  * a long tail stretching to the right -- the EM-gain amplified electrons. Its
    height at a given ADU is what carries the flux;
  * several bin edges INSIDE the peak (the tool spends bins where the counts
    are), a few wide bins climbing the tail, and the last edge placed where the
    tail has effectively died out.

If instead your edges sit mostly in an empty region, the detector constants in
the configuration do not match this data, and the flux fit will be wrong.
Re-derive them, re-run bin_optimizer.py, and paste the new edges in.

USAGE
-----
    python plot_bins.py                 # uses embin_config.yaml
    python plot_bins.py --n-files 50    # average over more frames
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import all_frames, get_edges, load_config, log, output_path  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', default=os.path.join(here, 'embin_config.yaml'),
                    help='YAML configuration (the same one embin.py reads)')
    ap.add_argument('--n-files', type=int, default=20,
                    help='how many frames to histogram (default 20; more is smoother, '
                         'not more correct)')
    ap.add_argument('--output', '-o', default=None, help='where to write the PDF')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    edges = get_edges(cfg)
    det = cfg['detector']
    bias, gain, ron = float(det['bias']), float(det['gain']), float(det['ron'])

    files = all_frames(cfg)[:max(1, args.n_files)]
    log(f'histogramming {len(files)} frame(s) at 1 ADU resolution ...')

    top = int(max(edges[-2] * 1.6, bias + 12 * gain))
    counts = np.zeros(top + 1, dtype=np.int64)
    for path in files:
        img = np.asarray(fits.getdata(path, ext=cfg['input'].get('hdu', 0)))
        counts += np.bincount(np.clip(img.astype(np.int64).ravel(), 0, top),
                              minlength=top + 1)
    total = counts.sum()
    log(f'{total:,} pixel values', 'value')

    # What fraction of the reads each bin actually collects. A bin holding a
    # vanishing fraction is not necessarily wasted (the top bin is meant to be
    # nearly empty), but a bin holding zero certainly is.
    adu = np.arange(top + 1)
    log('fraction of all reads landing in each bin:', 'value')
    last = len(edges) - 2                      # index of the open, "everything above" bin
    for i in range(len(edges) - 1):
        lo = edges[i]
        top_label = 'inf   ' if i == last else f'{edges[i + 1]:6.0f}'
        sel = (adu >= lo) if i == last else ((adu >= lo) & (adu < edges[i + 1]))
        frac = counts[sel].sum() / total
        log(f'  bin {i:2d}  [{lo:6.0f}, {top_label})   {100 * frac:9.5f} %'
            + ('   <-- nothing lands here' if frac == 0 else ''),
            'warn' if frac == 0 else 'value')

    # Two panels, because the two things worth seeing live at very different
    # scales: the whole range on the left, and the read-noise peak (where
    # several edges sit within a few ADU of one another) on the right.
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={'width_ratios': [1.55, 1]})

    for a in (ax, zoom):
        a.step(adu, np.maximum(counts, 0.5), where='mid', lw=0.9, color='#1f4e8c')
        for i, e in enumerate(edges[:-1]):
            a.axvline(e, color='#d1495b', lw=1.0, ls='--', zorder=3,
                      label='bin edges' if i == 0 else None)
        a.axvline(bias, color='#2a9d8f', lw=1.4, zorder=3,
                  label=f'bias = {bias:.1f} ADU')
        a.set_yscale('log')
        a.set_xlabel('pixel value [ADU]')
        a.grid(alpha=0.2, axis='y')   # no vertical grid: it would look like bin edges

    ax.axvline(bias + gain, color='#e07b39', lw=1.4, zorder=3,
               label=f'bias + 1 electron = {bias + gain:.0f} ADU')
    ax.set_xlim(0, top)
    ax.set_ylim(0.5, counts.max() * 3)
    ax.set_ylabel('number of reads per ADU')
    ax.set_title(f'All {len(files)} frames, every pixel', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.annotate('read-noise peak:\nmost pixels saw no\nelectron at all',
                xy=(bias, counts.max()), xytext=(0.03, 0.20), fontsize=8,
                textcoords='axes fraction', ha='left', va='center', color='#2a9d8f',
                arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=1,
                                connectionstyle='arc3,rad=-0.2'))
    ax.annotate('EM-gain tail:\nthe electrons are in here', fontsize=8,
                xy=(bias + 4 * gain, max(counts[int(bias + 4 * gain)], 1)),
                xytext=(0.42, 0.30), textcoords='axes fraction', ha='left',
                color='#1f4e8c',
                arrowprops=dict(arrowstyle='->', color='#1f4e8c', lw=1,
                                connectionstyle='arc3,rad=0.2'))

    lo, hi = int(bias - 6 * ron), int(bias + 10 * ron)
    zoom.set_xlim(lo, hi)
    zoom.set_ylim(0.5, counts[lo:hi].max() * 3)
    zoom.set_title('The read-noise peak, close up', fontsize=10)

    fig.suptitle('Where the bin edges land on the real data', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or output_path(cfg, name=os.path.join(
        str(cfg.get('output', {}).get('figures', 'figures')), 'bins_on_data.pdf'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    log(f'figure: {out}', 'value')


if __name__ == '__main__':
    main()
