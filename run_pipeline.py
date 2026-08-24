#!/usr/bin/env python3
"""
run_pipeline.py -- binning, then astrometry, in one command
===========================================================

This is a WRAPPER. It contains no science of its own: it runs

    1. run_chunks.py       the binning, which knows nothing about the sky
    2. pesto_astrometry.py the astrometry, which adds a WCS to what step 1 made

and that separation is deliberate, not incidental.

WHY TWO STEPS AND NOT ONE
-------------------------
The binning step is the part that is genuinely general. It takes frames, builds
one histogram per pixel, and fits a flux; nothing in it assumes the frames are
images of a star field, or images of anything at all. The same code will bin a
spectrograph's detector, where the "sources" are echelle orders and there is no
astrometric solution to be had. Putting a call to a star matcher inside it
would make it useless for that, and would make it fail on any field too sparse
to solve.

So `embin.py` and `run_chunks.py` stay pure, and everything that needs to know
about the sky lives in `pesto_astrometry.py`, which reads the binning products
and writes a WCS back into them. If the astrometry fails, or you do not want it,
the binning products are still complete and correct on their own.

USAGE
-----
    python run_pipeline.py                    # bin the night, then solve it
    python run_pipeline.py --n-chunks 2       # a quick trial on two chunks
    python run_pipeline.py --skip-binning     # solve products that already exist
    python run_pipeline.py --skip-astrometry  # exactly what run_chunks.py does

Every other option is passed straight through to the step it belongs to; run
`python run_chunks.py --help` or `python pesto_astrometry.py --help` to see
them. The settings live, as always, in `embin_config.yaml`.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import log  # noqa: E402


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description='Bin a sequence, then put the result on the sky.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c',
                    default=os.path.join(here, 'embin_config.yaml'))
    ap.add_argument('--skip-binning', action='store_true',
                    help='do not re-bin: solve the products already in the '
                         'output directory')
    ap.add_argument('--skip-astrometry', action='store_true',
                    help='bin only, leaving the products without a WCS')
    # passed through to run_chunks.py
    ap.add_argument('--chunk-size', type=int, default=None)
    ap.add_argument('--n-chunks', type=int, default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--aperture', type=float, default=None)
    ap.add_argument('--match-radius', type=float, default=None)
    ap.add_argument('--no-stamps', action='store_true')
    # passed through to pesto_astrometry.py
    ap.add_argument('--target', default=None)
    ap.add_argument('--ra', type=float, default=None)
    ap.add_argument('--dec', type=float, default=None)
    ap.add_argument('--radius', type=float, default=None)
    ap.add_argument('--no-gaia', action='store_true')
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--no-chunks', action='store_true',
                    help='update only the summary, not each embin_chunkNN.fits')
    args = ap.parse_args(argv)

    t0 = time.time()

    # --- step 1: the pure binning -----------------------------------------
    if args.skip_binning:
        log('step 1/2: binning SKIPPED, using the products already on disk',
            'warn')
    else:
        log('step 1/2: binning (run_chunks.py, no astrometry involved)', 'info')
        import run_chunks
        argv1 = ['--config', args.config]
        for flag, value in (('--chunk-size', args.chunk_size),
                            ('--n-chunks', args.n_chunks),
                            ('--outdir', args.outdir),
                            ('--aperture', args.aperture),
                            ('--match-radius', args.match_radius)):
            if value is not None:
                argv1 += [flag, str(value)]
        if args.no_stamps:
            argv1.append('--no-stamps')
        rc = run_chunks.main(argv1)
        if rc:
            log('binning failed; not attempting the astrometry', 'error')
            return rc

    # --- step 2: the astrometry -------------------------------------------
    if args.skip_astrometry:
        log('step 2/2: astrometry SKIPPED, the products carry no WCS', 'warn')
    else:
        log('step 2/2: astrometry (pesto_astrometry.py, reads what step 1 wrote)',
            'info')
        import pesto_astrometry
        argv2 = ['--config', args.config]
        for flag, value in (('--target', args.target), ('--ra', args.ra),
                            ('--dec', args.dec), ('--radius', args.radius)):
            if value is not None:
                argv2 += [flag, str(value)]
        for flag, on in (('--no-gaia', args.no_gaia),
                         ('--no-figure', args.no_figure),
                         ('--no-chunks', args.no_chunks)):
            if on:
                argv2.append(flag)
        rc = pesto_astrometry.main(argv2)
        if rc:
            log('astrometry failed. The binning products are still complete '
                'and usable; they simply have no WCS.', 'warn')
            return rc

    log(f'pipeline finished in {time.time() - t0:.1f} s', 'info')
    return 0


if __name__ == '__main__':
    sys.exit(main())
