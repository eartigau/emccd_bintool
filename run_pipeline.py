#!/usr/bin/env python3
"""
run_pipeline.py -- the whole night, one command, four explicit steps
====================================================================

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

This is a WRAPPER. It contains no science of its own: it runs the four programs
of this repository in the order they depend on each other, and nothing else.

    step          program               what it turns into what
    ------------  --------------------  ------------------------------------
    calibrate     embin_mcmc.py         frames         -> detector constants
                                                          and optimal bins
    bin           run_chunks.py         frames         -> histogram cubes and
                                                          flux maps, per chunk
    astrometry    pesto_astrometry.py   flux maps      -> the same, with a WCS
    photometry    embin_psf.py          flux maps      -> a flux per star per
                                                          chunk, deblended

    python run_pipeline.py                     # bin, astrometry, photometry
    python run_pipeline.py --list              # print the steps and stop
    python run_pipeline.py --only photometry   # just re-do the photometry
    python run_pipeline.py --skip astrometry   # ... and everything else
    python run_pipeline.py --with-calibration  # measure the detector first
    python run_pipeline.py --n-chunks 2        # a quick trial

Every other option is passed straight through to the step it belongs to; run
`python run_chunks.py --help`, `python pesto_astrometry.py --help` or
`python embin_psf.py --help` to see them. The settings live, as always, in
`embin_config.yaml`.

WHY SEPARATE STEPS AND NOT ONE
------------------------------
Because each one fails, and is useful, independently.

The BINNING is the part that is genuinely general. It takes frames, builds one
histogram per pixel, and fits a flux; nothing in it assumes the frames are
images of a star field, or images of anything at all. The same code will bin a
spectrograph's detector, where the "sources" are echelle orders and there is no
astrometric solution to be had. Putting a star matcher inside it would make it
useless for that, and would make it fail on any field too sparse to solve.

The ASTROMETRY needs the sky. It reads what the binning wrote and writes a WCS
back into it. If it fails -- too few stars, no internet for the Gaia query, a
pointing that is not where the header claims -- the binning products are still
complete and correct. They simply have no WCS.

The PHOTOMETRY needs both: the flux maps from the binning, and the drift-
corrected stack the astrometry produced, to know where the stars are and which
of them overlap. It is the only step that deblends, and the only one whose
answer is a light curve rather than an image.

The CALIBRATION is deliberately NOT in the default chain, because its output is
not a file the other steps read: it is four numbers you paste into section 3 of
`embin_config.yaml`, and a set of bin edges you paste into section 2. Run it
when the detector or the EM gain setting changes, read its report, edit the
configuration, and then run the other three. Wiring it to edit the config
behind your back would mean a silent change to the calibration underneath every
result you have already produced.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import log  # noqa: E402


# The pipeline, as data. Each entry is
#   (name, module, one-line description, in the default chain?)
# and `--only` / `--skip` / `--list` all read from this and nothing else, so
# adding a step means adding a row here and a builder below.
STEPS = [
    ('calibrate', 'embin_mcmc',
     'measure bias, RON, gain and the sky flux from the frames, and propose '
     'the optimal bin edges', False),
    ('bin', 'run_chunks',
     'histogram every pixel of every chunk and fit its flux', True),
    ('astrometry', 'pesto_astrometry',
     'stack the chunks, solve against Gaia, write a WCS into the products',
     True),
    ('photometry', 'embin_psf',
     'joint Moffat fit: one flux per star per chunk, blends deconvolved', True),
]
STEP_NAMES = [s[0] for s in STEPS]


def _argv_for(name, args):
    """The command line each step gets, built from this wrapper's options.

    Only the options that belong to a step are forwarded to it, so a typo in a
    flag is an error here rather than a silently ignored argument three modules
    down.
    """
    argv = ['--config', args.config]
    if name == 'calibrate':
        for flag, value in (('--nframes', args.nframes),
                            ('--steps', args.mcmc_steps)):
            if value is not None:
                argv += [flag, str(value)]
        if args.no_bin_draws:
            argv.append('--no-bin-draws')
    elif name == 'bin':
        for flag, value in (('--chunk-size', args.chunk_size),
                            ('--n-chunks', args.n_chunks),
                            ('--outdir', args.outdir),
                            ('--aperture', args.aperture),
                            ('--match-radius', args.match_radius)):
            if value is not None:
                argv += [flag, str(value)]
        if args.no_stamps:
            argv.append('--no-stamps')
    elif name == 'astrometry':
        for flag, value in (('--target', args.target), ('--ra', args.ra),
                            ('--dec', args.dec), ('--radius', args.radius)):
            if value is not None:
                argv += [flag, str(value)]
        for flag, on in (('--no-gaia', args.no_gaia),
                         ('--no-figure', args.no_figure),
                         ('--no-chunks', args.no_chunks)):
            if on:
                argv.append(flag)
    elif name == 'photometry':
        for flag, value in (('--stack', args.stack), ('--summary', args.summary)):
            if value is not None:
                argv += [flag, str(value)]
        if args.no_plot:
            argv.append('--no-plot')
    return argv


def _selected(args):
    """Which steps to run, after --only / --skip / --with-calibration.

    Legacy --skip-binning and --skip-astrometry still work; they are just the
    old spelling of --skip bin and --skip astrometry.
    """
    if args.only:
        chosen = list(args.only)
    else:
        chosen = [n for n, _, _, default in STEPS if default]
        if args.with_calibration:
            chosen.insert(0, 'calibrate')
    skip = set(args.skip or [])
    if args.skip_binning:
        skip.add('bin')
    if args.skip_astrometry:
        skip.add('astrometry')
    return [n for n in STEP_NAMES if n in chosen and n not in skip]


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description='Bin a sequence, put it on the sky, and measure every '
                    'star in it.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='steps, in order:\n' + '\n'.join(
            f'  {n:<11s} {d}{"" if default else "   [not run by default]"}'
            for n, _, d, default in STEPS))
    ap.add_argument('--config', '-c',
                    default=os.path.join(here, 'embin_config.yaml'))
    ap.add_argument('--list', action='store_true',
                    help='print the steps that would run, and stop')
    ap.add_argument('--only', action='append', choices=STEP_NAMES,
                    help='run only this step; repeatable')
    ap.add_argument('--skip', action='append', choices=STEP_NAMES,
                    help='do not run this step; repeatable')
    ap.add_argument('--with-calibration', action='store_true',
                    help='also run the detector calibration first. Its numbers '
                         'are NOT applied automatically: read its report and '
                         'edit sections 2 and 3 of the YAML yourself')
    ap.add_argument('--skip-binning', action='store_true',
                    help=argparse.SUPPRESS)          # old spelling of --skip bin
    ap.add_argument('--skip-astrometry', action='store_true',
                    help=argparse.SUPPRESS)          # old spelling
    # --- passed through to embin_mcmc.py ---------------------------------
    ap.add_argument('--nframes', type=int, default=None)
    ap.add_argument('--mcmc-steps', type=int, default=None)
    ap.add_argument('--no-bin-draws', action='store_true')
    # --- passed through to run_chunks.py ---------------------------------
    ap.add_argument('--chunk-size', type=int, default=None)
    ap.add_argument('--n-chunks', type=int, default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--aperture', type=float, default=None)
    ap.add_argument('--match-radius', type=float, default=None)
    ap.add_argument('--no-stamps', action='store_true')
    # --- passed through to pesto_astrometry.py ---------------------------
    ap.add_argument('--target', default=None)
    ap.add_argument('--ra', type=float, default=None)
    ap.add_argument('--dec', type=float, default=None)
    ap.add_argument('--radius', type=float, default=None)
    ap.add_argument('--no-gaia', action='store_true')
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--no-chunks', action='store_true',
                    help='update only the summary, not each embin_chunkNN.fits.gz')
    # --- passed through to embin_psf.py ----------------------------------
    ap.add_argument('--stack', default=None,
                    help='the drift-corrected stack the photometry reads')
    ap.add_argument('--summary', default=None,
                    help='the per-chunk cube the photometry reads')
    ap.add_argument('--no-plot', action='store_true',
                    help='photometry: numbers only, no figures and no report')
    args = ap.parse_args(argv)

    run = _selected(args)
    if not run:
        log('every step was skipped; nothing to do', 'warn')
        return 0

    descriptions = {n: d for n, _, d, _ in STEPS}
    if args.list:
        log(f'{len(run)} step(s) would run, in this order:')
        for i, name in enumerate(run, 1):
            log(f'  {i}. {name:<11s} {descriptions[name]}', 'value')
        for name in STEP_NAMES:
            if name not in run:
                log(f'     {name:<11s} not run', 'warn')
        return 0

    t0 = time.time()
    for i, name in enumerate(run, 1):
        module_name = dict((n, m) for n, m, _, _ in STEPS)[name]
        log(f'step {i}/{len(run)}: {name} ({module_name}.py) -- '
            f'{descriptions[name]}', 'info')
        t_step = time.time()
        module = __import__(module_name)
        rc = module.main(_argv_for(name, args))
        if rc:
            # Say what survives the failure, because in this pipeline that
            # varies: a failed astrometry leaves usable products, a failed
            # binning leaves the later steps with nothing to read.
            if name == 'bin':
                log('binning failed; the later steps have nothing to read',
                    'error')
            elif name == 'astrometry':
                log('astrometry failed. The binning products are still '
                    'complete and usable; they simply have no WCS, and the '
                    'photometry has no stack to find stars on', 'warn')
            elif name == 'photometry':
                log('photometry failed. The flux maps and the WCS are '
                    'unaffected', 'warn')
            else:
                log(f'{name} failed', 'error')
            return rc
        log(f'step {i}/{len(run)}: {name} done in {time.time() - t_step:.1f} s',
            'info')

    if 'calibrate' in run:
        log('the calibration numbers above are NOT applied automatically: '
            'paste them into sections 2 and 3 of the configuration, then '
            're-run the binning', 'warn')
    log(f'pipeline finished in {time.time() - t0:.1f} s', 'info')
    return 0


if __name__ == '__main__':
    sys.exit(main())
