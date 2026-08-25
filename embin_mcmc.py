#!/usr/bin/env python3
"""
embin_mcmc.py -- measure the detector constants from the frames themselves
==========================================================================

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

WHAT THIS DOES
--------------
Section 3 of `embin_config.yaml` (the `detector:` block) is a set of numbers
that everything else in this toolkit depends on: the bias, the read-out noise,
the EM gain. Section 2 (`histogram: edges:`) is a set of bin edges whose optimal
placement is a function of exactly those three numbers. Until now both had to
come from somewhere else -- a separate calibration run, or a previous paper.

This script derives them from the science frames themselves. It takes the first
N frames the configuration points at, builds ONE histogram of raw ADU values
pooled over every (unmasked) pixel and every frame, and samples the posterior of

    bias   [ADU]      the electronics pedestal
    ron    [ADU]      read-out noise, 1 sigma
    gain   [ADU/e-]   mean EM gain
    mu     [e-/frame] the mean flux per pixel

with emcee. Feeding the fitted detector through `bin_optimizer.design_bins`, it
then prints the optimal histogram bin edges that detector implies, ready to
paste into the configuration, and writes a three-page report,
`data_bin/figures/mcmc_report.pdf`:

    page 1   the triangle plot -- every parameter against every other, with its
             own marginal posterior along the diagonal
    page 2   the pooled histogram against the fitted model, with the pulls and
             the proposed bin edges drawn on it
    page 3   per-pixel uncertainty against per-pixel flux, hexagonally binned,
             for EVERY pixel of the frame -- stars included, whether or not they
             were used to fit the detector -- against the noiseless Poisson
             ideal and the EM excess-noise limit

Page 3 is the one that says whether any of this worked, and it is deliberately
run on pixels the fit never saw: see flux_sigma_map() for why that is the point
rather than a leak.

    python embin_mcmc.py                       # everything from the YAML
    python embin_mcmc.py --nframes 128         # override the frame count
    python embin_mcmc.py --steps 6000          # longer chain
    python embin_mcmc.py --no-bin-draws        # skip the bin-edge error bars
    python embin_mcmc.py --no-flux-map         # skip page 3, and its runtime

WHAT `mu` IS, AND IS NOT
------------------------
`mu` here is ONE number: the mean flux of a typical (sky) pixel, in electrons
per frame. It is not a per-pixel flux map -- that is embin.py's job, and it
needs the bin edges this script proposes. The two are complementary: this script
measures the constants that are common to the whole detector, embin.py measures
the one quantity that varies from pixel to pixel.

`mu` is also inseparable from the clock-induced charge. CIC electrons enter the
EM register the same way photo-electrons do, so the two Poisson processes add
and only their SUM is observable in a pixel-value histogram. What comes out of
this fit is therefore sky + CIC, which is the same convention `detector: cic:`
already uses in the configuration ("whatever CIC exists is reported as part of
the sky").

THE MODEL
---------
The probability that a pixel exposed to a mean flux mu reads out at a value
`u` ADU above the bias, before read noise, is the classical EMCCD law (Basden
et al. 2003; Harpsoe et al. 2012):

    P(u) = exp(-mu) delta(u)                                        [no electron]
         + sqrt(mu / (G u)) I_1(2 sqrt(mu u / G))
           exp(-(sqrt(u/G) - sqrt(mu))^2)              for u > 0    [n >= 1]

which is the Poisson sum over electron number n of the Gamma(n, G) output of
the EM register, done in closed form -- `I_1` is the modified Bessel function.
Note what is NOT in it: there is no `nmax` truncation. The Poisson sum is
summed exactly, so the high-flux tail is right by construction, and one of the
config's fiddlier knobs simply does not enter here.

That law is then convolved with the read noise and integrated over each 1-ADU
digitisation step, giving the probability of each integer ADU value. The
likelihood of the observed histogram is multinomial:

    ln L = sum_k  n_k ln p_k(bias, ron, gain, mu)

with p_k renormalised over the fitted ADU range, so the range can be cut
wherever the data stops without biasing anything.

This implementation was checked against a 2e8-sample Monte Carlo of the exact
signal chain (Poisson -> Gamma -> Gaussian -> digitise): the model and the
simulated histogram agree to within the Monte Carlo noise over the whole range
where the simulation has counts.

ABOUT THE ERROR BARS -- READ THIS
---------------------------------
64 frames of a 1024 x 426 detector is 2.8e7 pixel values. A four-parameter fit
to that many samples has formal uncertainties of order 1e-4 of each parameter,
and NO real detector is described by four numbers to that precision: fixed
pattern in the bias, pixel-to-pixel gain variation, a sky that is not uniform
and faint stars that the mask did not catch all push chi2/dof well above 1.

So this script reports BOTH:

  * the formal posterior width, which is what the MCMC actually sampled, and
  * that width inflated by sqrt(chi2/dof), which is the usual way of saying
    "the model does not fit this well, so do not trust the formal error".

The inflated numbers are the honest ones to quote. The central values are
robust -- they reproduce an independent calibration of this detector to well
inside the inflated errors -- but the formal error bars are not uncertainties
on the detector, they are uncertainties on a four-parameter caricature of it.
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import erf, ive

# embin.py holds the configuration loader, the path resolution rules and the
# timestamped logger the whole toolkit prints through; bin_optimizer.py turns a
# detector into the bin edges that retrieve flux best from it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bin_optimizer import design_bins                                # noqa: E402
from embin import (_numeric_key, _resolve, _timestamp,               # noqa: E402
                   build_flux_grid, build_histogram_cube, fit_flux_image,
                   load_config, log)
from embin_report import write_report                                # noqa: E402

_SQRT2 = np.sqrt(2.0)

# The four sampled parameters, in chain order. `mu` is sampled as its natural
# logarithm (a flux is a positive scale, and its prior is log-uniform), so the
# chain carries ln(mu) and every report converts back.
PARAM_NAMES = ['bias', 'ron', 'gain', 'mu']
PARAM_LABELS = [r'bias  [ADU]', r'RON  [ADU]', r'gain  [ADU/e$^-$]',
                r'$\mu$  [e$^-$/frame]']


# ---------------------------------------------------------------------------
# The model: P(ADU = k) for the EMCCD signal chain
# ---------------------------------------------------------------------------
def emccd_pmf(x, bias, ron, gain, mu, du=0.25):
    """Probability of each integer ADU value in `x`, for one set of constants.

    Parameters
    ----------
    x     : 1-D float array   the integer ADU values to evaluate, ascending
    bias  : float             pedestal [ADU]
    ron   : float             read-out noise, 1 sigma [ADU]
    gain  : float             mean EM gain [ADU/e-]
    mu    : float             mean flux [e-/frame], sky + CIC
    du    : float             step of the internal grid the EM continuum is
                              tabulated on, in ADU, before it is convolved with
                              the read noise. 0.25 keeps the discretisation
                              error below 1e-4 of the model value everywhere;
                              0.5 is twice as fast and still below 3e-4.

    Returns
    -------
    p : float array, same shape as `x`, summing to 1 over the FULL ADU axis
        (so over a truncated `x` it sums to slightly less, and the caller
        renormalises).

    The two terms are handled differently on purpose. The no-electron term is a
    delta function at the bias, so read noise plus digitisation turns it into an
    exact difference of error functions -- no grid, no interpolation, which
    matters because that term IS the peak and carries most of the counts. Only
    the smooth n >= 1 continuum goes through the numerical convolution.
    """
    # --- the n >= 1 continuum, on a fine grid of u = (ADU - bias) ------------
    umax = x[-1] - bias + 8.0 * ron
    u = np.arange(0.0, umax, du)
    cont = np.zeros_like(u)
    pos = u > 0
    # ive(1, y) is I_1(y) exp(-|y|), and the exponent below is what is left of
    # exp(-mu - u/G) exp(+2 sqrt(mu u / G)) once that exp(-y) is folded in:
    # a perfect square, so nothing ever overflows however bright the pixel.
    cont[pos] = (np.sqrt(mu / (gain * u[pos]))
                 * ive(1, 2.0 * np.sqrt(mu * u[pos] / gain))
                 * np.exp(-(np.sqrt(u[pos] / gain) - np.sqrt(mu)) ** 2))
    cont[0] = (mu / gain) * np.exp(-mu)      # the finite u -> 0+ limit

    # Trapezoid weights: the u = 0 sample sits on the edge of the integration
    # domain and counts half. Without this the model normalisation drifts by
    # ~du/1000 and the fitted gain drifts with the grid step.
    w = np.full_like(cont, du)
    w[0] = 0.5 * du

    # --- read noise and digitisation, in one kernel -------------------------
    # K(v) is the probability that a true value u lands in the 1-ADU-wide
    # integer bin centred u + v: a Gaussian of width `ron` integrated over the
    # bin, not sampled at its centre. At 2.8e7 samples the difference between
    # the two is larger than the Poisson noise, so it has to be the integral.
    half = int(np.ceil(6.0 * ron / du))
    v = np.arange(-half, half + 1) * du
    kernel = 0.5 * (erf((v + 0.5) / (ron * _SQRT2))
                    - erf((v - 0.5) / (ron * _SQRT2)))

    conv = fftconvolve(cont * w, kernel, mode='full')
    conv_grid = np.arange(len(conv)) * du + (u[0] + v[0])

    # --- evaluate at the data's ADU values ----------------------------------
    # Interpolating in (x - bias) rather than shifting the grid is what makes
    # `bias` a continuous parameter. Rounding it to the nearest integer ADU, as
    # a shift-the-array implementation has to, puts steps in the likelihood
    # that an MCMC walks straight into.
    xx = x - bias
    p_zero = 0.5 * (erf((xx + 0.5) / (ron * _SQRT2))
                    - erf((xx - 0.5) / (ron * _SQRT2)))
    return (np.exp(-mu) * p_zero
            + np.interp(xx, conv_grid, conv, left=0.0, right=0.0))


# ---------------------------------------------------------------------------
# The data: one pooled ADU histogram
# ---------------------------------------------------------------------------
def frame_list(cfg, n_frames):
    """The first `n_frames` frames the configuration points at, in numeric order.

    `n_frames` None means every frame in the folder.
    """
    import glob
    inp = cfg['input']
    directory = _resolve(cfg, inp['directory'])
    pattern = inp.get('pattern', '*.fits')
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=_numeric_key)
    if not files:
        log(f'no file matches {os.path.join(directory, pattern)} - nothing to do. '
            f'Is input.directory in the configuration pointing at your frames?', 'error')
        sys.exit(1)
    if n_frames is None:
        log(f'{len(files)} frames in {directory}, taking all of them')
        return files
    if len(files) < n_frames:
        log(f'only {len(files)} frames available, using all of them '
            f'instead of the {n_frames} requested', 'warn')
        n_frames = len(files)
    log(f'{len(files)} frames in {directory}, taking the first {n_frames}')
    return files[:n_frames]


def _read(path, hdu_index):
    """One frame, as float32, with a readable message if the file is not there.

    The empty-file case is worth its own message: a frame sitting in a folder
    synced by OneDrive, iCloud Drive or Dropbox can be a placeholder with the
    full size in `ls` and no bytes on disk, and astropy's complaint about a
    missing SIMPLE card does not point anywhere near the actual problem.
    """
    try:
        img = fits.getdata(path, ext=hdu_index)
    except OSError as exc:
        log(f'{os.path.basename(path)} could not be read as FITS ({exc}). '
            f'If it is in a cloud-synced folder, check that it is really on '
            f'disk: find <folder> -name "*.fits" -flags +dataless | wc -l', 'error')
        sys.exit(1)
    if img is None or img.ndim != 2:
        log(f'{os.path.basename(path)}: HDU {hdu_index} is not a 2-D image', 'error')
        sys.exit(1)
    return np.asarray(img, dtype=np.float32)


def pooled_histogram(files, hdu_index, clip_sigma):
    """Histogram of raw ADU pooled over frames and over the surviving pixels.

    Two passes over the frames, so peak memory is one frame plus one accumulator
    rather than the whole stack:

      pass 1  the mean of each pixel across the frames, used only to decide
              which pixels to keep;
      pass 2  the histogram itself, over the kept pixels.

    WHY MASK AT ALL. The model has ONE flux for every pixel. A star is a pixel
    with a hundred times the sky flux, and a handful of them pollute exactly the
    high-ADU tail that the gain is measured from. `clip_sigma` cuts pixels whose
    mean sits more than that many robust sigmas off the median mean, in either
    direction -- stars and hot pixels above, dead pixels below.

    WHAT MASKING COSTS. The surviving pixels still do not all share one flux
    (the sky is not perfectly flat, and faint sources hide under the cut), and
    cutting the top of a distribution biases the fitted mu slightly low. Set
    `clip_sigma` to null to fit every pixel and see how much it matters: on the
    PESTO night it moves mu by a few per cent and the gain by under one.
    """
    log('pass 1/2: mean of each pixel across the frames')
    total = None
    for path in files:
        img = _read(path, hdu_index)
        total = img.astype(np.float64) if total is None else total + img
    mean_img = total / len(files)

    if clip_sigma is None:
        keep = np.ones(mean_img.shape, dtype=bool)
        log('no bright-pixel clipping, every pixel is fitted', 'warn')
    else:
        med = np.median(mean_img)
        sig = np.median(np.abs(mean_img - med)) * 1.4826
        keep = np.abs(mean_img - med) < clip_sigma * sig
        log(f'per-pixel mean: median {med:.4f} ADU, robust sigma {sig:.4f} ADU', 'value')
        log(f'keeping {keep.sum()} of {keep.size} pixels '
            f'({100.0 * keep.mean():.2f} %) within {clip_sigma:g} sigma; '
            f'{keep.size - keep.sum()} rejected as stars, hot or dead', 'value')

    log('pass 2/2: pooled ADU histogram over the kept pixels')
    hist = np.zeros(65536, dtype=np.int64)
    for path in files:
        img = _read(path, hdu_index)
        vals = np.rint(img[keep]).astype(np.int64)
        np.clip(vals, 0, 65535, out=vals)
        hist += np.bincount(vals, minlength=65536)
    return hist, keep, mean_img


def fit_range(hist, adu_min, adu_max):
    """Cut the histogram to the ADU range that is actually fitted.

    Defaults to the full support of the data -- lowest and highest ADU with any
    count at all. Empty bins inside the range are harmless: a bin with zero
    counts contributes zero to sum(n_k ln p_k) whatever the model says about it.
    """
    nz = np.nonzero(hist)[0]
    lo = int(nz.min()) if adu_min is None else int(adu_min)
    hi = int(nz.max()) if adu_max is None else int(adu_max)
    x = np.arange(lo, hi + 1, dtype=np.float64)
    n = hist[lo:hi + 1].astype(np.float64)
    dropped = hist.sum() - n.sum()
    log(f'fitting ADU {lo} to {hi} ({len(x)} bins, {int(n.sum())} pixel values, '
        f'{dropped} outside the range)', 'value')
    return x, n


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------
class Posterior:
    """The multinomial log-posterior, as a picklable callable.

    theta = (bias, ron, gain, ln_mu). Flat priors on the first three within
    their configured ranges, flat in ln_mu -- i.e. log-uniform on the flux,
    which is the scale-invariant choice for a positive quantity whose order of
    magnitude is not known in advance.
    """

    def __init__(self, x, counts, priors, du):
        self.x = x
        self.counts = counts
        self.du = du
        self.bias_lo, self.bias_hi = priors['bias']
        self.ron_lo, self.ron_hi = priors['ron']
        self.gain_lo, self.gain_hi = priors['gain']
        self.lnmu_lo = np.log(priors['mu'][0])
        self.lnmu_hi = np.log(priors['mu'][1])

    def __call__(self, theta):
        bias, ron, gain, ln_mu = theta
        if not (self.bias_lo < bias < self.bias_hi
                and self.ron_lo < ron < self.ron_hi
                and self.gain_lo < gain < self.gain_hi
                and self.lnmu_lo < ln_mu < self.lnmu_hi):
            return -np.inf
        p = emccd_pmf(self.x, bias, ron, gain, np.exp(ln_mu), du=self.du)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return -np.inf
        # Renormalise over the fitted range: the likelihood is then the
        # multinomial one for "given that the value landed in [lo, hi], which
        # bin did it land in", which is exactly what the truncated data says.
        return float(np.dot(self.counts, np.log(np.maximum(p / total, 1e-300))))


def max_likelihood(post, start):
    """Nelder-Mead maximum of the posterior, used to seed the walkers.

    Starting the chain in a ball around the maximum rather than around the
    configuration's guessed values is what keeps the burn-in to a few hundred
    steps instead of a few thousand.
    """
    res = minimize(lambda t: -post(t), start, method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-2,
                            'maxfev': 20000, 'maxiter': 20000})
    bias, ron, gain, ln_mu = res.x
    log(f'maximum likelihood: bias={bias:.4f} ADU, ron={ron:.4f} ADU, '
        f'gain={gain:.4f} ADU/e-, mu={np.exp(ln_mu):.6f} e-/frame '
        f'({res.nfev} evaluations)', 'value')
    return res.x


def run_mcmc(post, start, walkers, steps, burn, thin, seed):
    """Sample the posterior with emcee and return the flattened chain.

    The chain columns are (bias, ron, gain, mu) -- ln_mu is exponentiated on the
    way out, so everything downstream sees the flux itself.
    """
    try:
        import emcee
    except ImportError:
        log('emcee is not installed: pip install emcee corner', 'error')
        sys.exit(1)

    # A burn-in longer than the chain leaves nothing behind, and the failure
    # surfaces far downstream as an empty array. Catch it here instead.
    if burn >= steps:
        burn = max(0, steps // 2)
        log(f'mcmc.burn is not shorter than mcmc.steps; discarding the first '
            f'{burn} steps instead', 'warn')

    rng = np.random.default_rng(seed)
    ndim = len(start)
    # A small ball, scaled to each parameter, so no walker starts outside the
    # prior or in a region where the model is numerically unhappy.
    scale = np.array([0.01, 0.005, 0.05, 0.01])
    p0 = start + scale * rng.standard_normal((walkers, ndim))

    sampler = emcee.EnsembleSampler(walkers, ndim, post)
    log(f'sampling: {walkers} walkers x {steps} steps '
        f'(burn {burn}, thin {thin}) ...')
    sampler.run_mcmc(p0, steps, progress=False)

    acc = float(np.mean(sampler.acceptance_fraction))
    log(f'mean acceptance fraction {acc:.3f}', 'value')
    if acc < 0.15 or acc > 0.7:
        log(f'acceptance fraction {acc:.3f} is outside the healthy 0.2-0.5 band; '
            f'the chain may be poorly mixed', 'warn')
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        log('autocorrelation time: ' + ', '.join(
            f'{nm}={t:.0f}' for nm, t in zip(PARAM_NAMES, tau)), 'value')
        if np.any(steps < 50 * tau):
            log(f'chain is shorter than 50 autocorrelation times; '
                f'raise mcmc.steps for a cleaner posterior', 'warn')
    except Exception as exc:                      # emcee raises on short chains
        log(f'autocorrelation time not measurable: {exc}', 'warn')

    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    chain = np.column_stack([chain[:, 0], chain[:, 1], chain[:, 2],
                             np.exp(chain[:, 3])])
    log(f'{len(chain)} posterior samples kept', 'value')
    return chain


def summarise(chain, chi2_dof):
    """Median and 16/84 percentiles per parameter, formal and inflated.

    Returns a dict name -> (median, minus, plus, minus_inflated, plus_inflated).
    """
    infl = np.sqrt(max(chi2_dof, 1.0))
    out = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = np.percentile(chain[:, i], [15.865, 50.0, 84.135])
        out[name] = (med, med - lo, hi - med, (med - lo) * infl, (hi - med) * infl)
    return out


def goodness_of_fit(x, counts, theta_med, du):
    """chi2 per degree of freedom of the posterior-median model.

    Only bins whose expected count exceeds 10 are used, so the statistic stays
    a chi2 rather than a Poisson-in-the-tail curiosity.
    """
    p = emccd_pmf(x, theta_med[0], theta_med[1], theta_med[2], theta_med[3], du=du)
    expect = p / p.sum() * counts.sum()
    ok = expect > 10
    pull = (counts - expect) / np.sqrt(np.maximum(expect, 1.0))
    chi2 = float(np.sum(pull[ok] ** 2))
    dof = int(ok.sum()) - len(theta_med)
    return expect, pull, chi2 / dof, dof


# ---------------------------------------------------------------------------
# Report pages
# ---------------------------------------------------------------------------
# Every page below BUILDS and RETURNS a figure rather than writing it; main()
# collects them into one multi-page PDF. One file is easier to page through than
# three, and the three pages only mean anything read together: the posterior,
# whether the model that posterior came from actually fits, and what the whole
# thing delivers on the pixels it was never fitted on.
def corner_page(chain):
    """The triangle plot: every parameter against every other, plus its own
    marginal posterior along the diagonal."""
    try:
        import corner
    except ImportError:
        log('corner is not installed, skipping the triangle plot: '
            'pip install corner', 'warn')
        return None

    fig = corner.corner(chain, labels=PARAM_LABELS,
                        quantiles=[0.15865, 0.5, 0.84135],
                        show_titles=True, title_fmt='.4g',
                        title_kwargs={'fontsize': 9},
                        label_kwargs={'fontsize': 10})
    # No suptitle: corner puts a title over every diagonal panel already, and a
    # figure-level one lands on top of the first of them. The LaTeX caption says
    # what needs saying instead.
    return fig


def fit_page(x, counts, expect, pull, design=None):
    """Observed histogram against the fitted model, and the pulls underneath."""
    import matplotlib.pyplot as plt

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=[3, 1],
        gridspec_kw={'hspace': 0.05})

    ax.step(x, np.maximum(counts, 0.1), where='mid', lw=0.8,
            color='0.35', label='observed')
    ax.plot(x, expect, lw=1.4, color='crimson', label='posterior-median model')
    ax.set_yscale('log')
    ax.set_ylim(0.3, counts.max() * 3)
    ax.set_ylabel('pixel values per ADU')
    if design is not None:
        for e in np.asarray(design.edges)[1:-1]:
            ax.axvline(e, color='steelblue', lw=0.7, ls=':')
        ax.plot([], [], color='steelblue', lw=0.7, ls=':',
                label=f'proposed {design.nbins}-bin edges')
    ax.legend(frameon=False)
    ax.set_title('pooled ADU histogram and the fitted EMCCD model')

    axr.axhline(0, color='0.6', lw=0.8)
    axr.step(x, pull, where='mid', lw=0.7, color='0.25')
    axr.set_ylabel('pull')
    axr.set_xlabel('raw pixel value [ADU]')
    lim = np.percentile(np.abs(pull), 99.5)
    axr.set_ylim(-1.2 * lim, 1.2 * lim)
    return fig


def flux_sigma_map(cfg, files, design, summary, hdu_index):
    """Per-pixel flux and its 1-sigma error, for EVERY pixel of the frame.

    This deliberately does NOT reuse the mask from the histogram fit. The four
    detector constants were measured on sky pixels only, because a star breaks
    the one-flux-for-every-pixel assumption the pooled histogram rests on. The
    per-pixel fit has no such assumption -- each pixel gets its own flux -- so
    the stars are exactly the pixels worth putting back in: they are the only
    place in the field where the high-flux behaviour of the estimator can be
    seen at all.

    The route is the toolkit's own: bin the frames into the K optimal bins this
    script just proposed, then run embin.py's vectorised maximum-likelihood
    fitter over them. So the flux and the error plotted on the next page are
    what a real embin.py run on this night would produce, with the detector
    measured here and the bins proposed here -- not a separate calculation that
    happens to agree.

    HOW MANY FRAMES THIS NEEDS, which is NOT the number the detector fit needs.
    The pooled histogram pools 2.8e7 pixel values whatever N is, so 64 frames
    already over-determine four constants. A per-pixel fit has only N values per
    pixel, and at a sky flux of 0.017 e-/frame, 64 frames means about one
    electron per pixel for the whole sequence: a third of the pixels see zero
    events, their likelihood is monotonic, and the estimator pins them to the
    bottom of the flux grid. The median fitted flux then lands a factor of three
    under the truth and the page says nothing about the estimator, only about
    small-number statistics. Feed it the whole sequence (`mcmc.flux_map_frames:
    null`) and each pixel gets of order twenty events, which is the regime the
    plot is worth reading in.
    """
    # As an array, not the plain list edge_list() hands back: emccd_histo's
    # grid builder compares the edges to a threshold element-wise.
    edges = np.asarray(design.edge_list(), dtype=np.float64)
    log(f'binning {len(files)} frames into the {design.nbins} proposed bins, '
        f'all pixels this time ...')
    cube, _, _ = build_histogram_cube(files, edges, hdu_index)

    # embin's fitter reads the detector out of a config dict; hand it the one we
    # just measured rather than whatever the YAML still says.
    fit_cfg = dict(cfg)
    fit_cfg['detector'] = dict(cfg.get('detector', {}))
    fit_cfg['detector'].update(bias=float(summary['bias'][0]),
                               ron=float(summary['ron'][0]),
                               gain=float(summary['gain'][0]),
                               cic=0.0)
    nll_grid = build_flux_grid(fit_cfg, edges)
    mu, mu_err, _, _ = fit_flux_image(cube, edges, fit_cfg, nll_grid=nll_grid)
    return mu, mu_err


def flux_sigma_page(mu, sigma, n_frames, summary, det_note):
    """Per-pixel uncertainty against per-pixel flux, for all pixels at once.

    This is the real-data twin of a Monte-Carlo validation. A simulation checks
    the estimator by generating many trials at a range of KNOWN fluxes; a field
    of view hands you the same thing for free, because its pixels already span
    two decades of flux, from blank sky to the core of a star. What the two
    reference curves mean:

        sigma = sqrt(mu / N)        the noiseless photon-counting ideal, the
                                    Cramer-Rao floor nothing can beat
        sigma = sqrt(2 mu / N)      the same floor with the EM register's
                                    excess-noise factor F = sqrt(2)

    Sky pixels should sit a little above the Poisson line -- the read-noise
    penalty of thresholding at finite RON/gain -- and climb toward and past the
    excess-noise line as multi-electron events become common at high flux.

    Hexagonal bins, not a scatter: 436,000 points overplot into a black blob,
    and what matters here is where the pixels are DENSE, which is the sky clump.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Two cuts, both of them about not drawing pixels whose value means nothing.
    #
    # In x: a pixel whose fitted flux is below 1/N saw less than one electron in
    # the whole sequence. There is no measurement there to plot -- the estimator
    # is running on the shape of a likelihood that never turned over -- and
    # those pixels otherwise form a long spurious tail that dominates the axis.
    #
    # In y: the ratio is bounded at 2, i.e. twice the noiseless Poisson floor.
    # Anything above that is a pixel whose fit failed, not a pixel with a
    # genuinely bad error bar, and stretching the axis to hold them squashes the
    # part of the plot that carries the result.
    valid = np.isfinite(mu) & np.isfinite(sigma) & (mu > 0) & (sigma > 0)
    mu_floor = 1.0 / n_frames
    ratio_all = np.where(valid, sigma / np.sqrt(np.maximum(mu, 1e-30) / n_frames),
                         np.inf)
    keep = valid & (mu > mu_floor) & (ratio_all <= 2.0)
    log(f'flux-sigma page: {int(keep.sum()):,} of {int(valid.sum()):,} pixels '
        f'kept (mu > 1/N = {mu_floor:.5f} e-/frame and sigma/sigma_Poisson <= 2)',
        'value')

    mu_v, sig_v = mu[keep], sigma[keep]
    ratio = sig_v / np.sqrt(mu_v / n_frames)

    grid = np.geomspace(max(mu_v.min(), 1e-4), mu_v.max(), 200)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    hb1 = ax1.hexbin(mu_v, sig_v, gridsize=70, xscale='log', yscale='log',
                     mincnt=1, cmap='inferno', norm=LogNorm())
    ax1.plot(grid, np.sqrt(grid / n_frames), '--', color='cyan', lw=1.8,
             label=r'noiseless Poisson ideal  $\sigma=\sqrt{\mu/N}$')
    ax1.plot(grid, np.sqrt(2 * grid / n_frames), ':', color='cyan', lw=1.8,
             label=r'EM excess-noise limit  $\sigma=\sqrt{2\mu/N}$')
    ax1.axvline(summary['mu'][0], color='springgreen', lw=1.4, ls='-.',
                label=r'$\mu$ from the pooled-histogram MCMC')
    ax1.set_ylabel(r'statistical uncertainty per pixel  [e$^-$/frame]')
    ax1.set_title(f'flux against uncertainty, {mu_v.size:,} pixels '
                  rf'($\mu > 1/N$, $\sigma/\sigma_\mathrm{{P}} \leq 2$, '
                  f'N = {n_frames} frames)\n{det_note}', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    fig.colorbar(hb1, ax=ax1, label='pixels per bin')

    hb2 = ax2.hexbin(mu_v, ratio, gridsize=70, xscale='log', yscale='linear',
                     mincnt=1, cmap='inferno', norm=LogNorm())
    ax2.axhline(1.0, ls='--', color='cyan', lw=1.8,
                label='Poisson ideal (ratio = 1)')
    ax2.axhline(np.sqrt(2), ls=':', color='cyan', lw=1.8,
                label=r'EM excess-noise limit (ratio = $\sqrt{2}$)')
    ax2.axvline(summary['mu'][0], color='springgreen', lw=1.4, ls='-.')
    ax2.set_xscale('log')
    ax2.set_ylim(0.85, 2.0)
    ax2.set_xlabel(r'mean flux per pixel  [e$^-$/frame]')
    ax2.set_ylabel(r'$\sigma_\mathrm{measured}\,/\,\sigma_\mathrm{Poisson}(\mu)$')
    ax2.legend(loc='upper right', fontsize=9)
    fig.colorbar(hb2, ax=ax2, label='pixels per bin')

    # The same numeric cross-check pesto_stats/flux_vs_sigma.py prints: how much
    # of the unbeatable ideal survives, at the flux most pixels actually have.
    sky = float(np.median(mu_v))
    near = np.abs(np.log10(mu_v / sky)) < 0.05
    med_sigma = float(np.median(sig_v[near]))
    ideal = np.sqrt(sky / n_frames)
    throughput = float((ideal / med_sigma) ** 2)
    log(f'at the bulk sky flux (median mu = {sky:.5f} e-/frame, '
        f'{int(near.sum())} pixels within 12 %): median sigma = {med_sigma:.6f}, '
        f'Poisson ideal = {ideal:.6f}, throughput = {throughput:.3f}', 'value')
    return fig, {'map_median_mu': sky, 'throughput': throughput,
                 'map_median_sigma': med_sigma}


def save_figures(pages, figdir):
    """Each page to its own vector PDF, for LaTeX to include.

    Returns the role -> file name mapping embin_report wants. The figure objects
    are deliberately left open: if LaTeX turns out to be missing, stitch_figures
    still needs them.
    """
    names = {}
    for role, fig in pages.items():
        if fig is None:
            continue
        name = f'mcmc_{role}.pdf'
        fig.savefig(os.path.join(figdir, name), bbox_inches='tight')
        names[role] = name
    log(f'{len(names)} figures written to {figdir}')
    return names


def stitch_figures(pages, path):
    """Fallback when there is no working LaTeX: the figures, one per page.

    Worse than the typeset report -- no tables, no copy-ready block, no words
    around the plots -- but it means a missing TeX install costs you the prose,
    not the pictures. The numbers are on stdout either way.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    figs = [f for f in pages.values() if f is not None]
    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
    log(f'no typeset report: {len(figs)} raw figures stitched into {path}', 'warn')
    return path


# ---------------------------------------------------------------------------
# Bin design
# ---------------------------------------------------------------------------
def propose_bins(summary, cfg, bin_cfg, chain, n_draws):
    """Optimal bin edges for the detector that was just measured.

    The design itself comes from the posterior MEDIAN detector. `n_draws`
    posterior samples are then designed as well, purely to put an uncertainty on
    each edge: the spread tells you which cuts are pinned down by the data and
    which are floating. It costs about two seconds per draw, so it is the one
    part of this script worth switching off when iterating.
    """
    det_cfg = cfg.get('detector', {})
    saturation = float(det_cfg.get('full_well', 5000))
    nbins = int(bin_cfg.get('nbins', 8))
    mu_min = float(bin_cfg.get('mu_min', 1.0e-3))
    mu_max = float(bin_cfg.get('mu_max', 8.0))
    method = bin_cfg.get('method', 'both')

    bias, ron, gain = (summary['bias'][0], summary['ron'][0], summary['gain'][0])
    log(f'designing {nbins} optimal bins for bias={bias:.4f}, gain={gain:.4f}, '
        f'ron={ron:.4f}, saturation={saturation:g} ADU, over '
        f'mu = {mu_min:g} to {mu_max:g} e-/frame ...')
    design = design_bins(bias, gain, ron, saturation, nbins=nbins,
                         mu_min=mu_min, mu_max=mu_max, method=method)
    print(design.summary())

    spread = None
    if n_draws > 0:
        log(f'propagating the posterior into the edges with {n_draws} draws ...')
        idx = np.linspace(0, len(chain) - 1, n_draws).astype(int)
        drawn = []
        for j, k in enumerate(idx, 1):
            b, r, g = chain[k, 0], chain[k, 1], chain[k, 2]
            try:
                d = design_bins(b, g, r, saturation, nbins=nbins, mu_min=mu_min,
                                mu_max=mu_max, method=method)
            except Exception as exc:
                log(f'draw {j}/{n_draws} failed ({exc}), skipped', 'warn')
                continue
            drawn.append(d.edge_list())
        if drawn:
            spread = np.std(np.array(drawn, dtype=float), axis=0)
            log('edge scatter over the posterior [ADU]: '
                + ', '.join(f'{s:.1f}' for s in spread[1:-1]), 'value')
    return design, spread


def yaml_block(summary, design, chi2_dof, cfg):
    """The two configuration blocks to paste back into embin_config.yaml."""
    det_cfg = cfg.get('detector', {})
    infl = np.sqrt(max(chi2_dof, 1.0))
    lines = ['',
             '# ---- paste into embin_config.yaml '
             '-------------------------------------',
             'histogram:',
             '  edges: ' + str(design.edge_list()),
             '',
             'detector:']
    for name, unit in (('bias', 'ADU'), ('ron', 'ADU'), ('gain', 'ADU/e-')):
        med, m, p, mi, pi = summary[name]
        lines.append(f'  {name}: {med:.4f}'.ljust(24)
                     + f'# {unit}, -{mi:.4f} +{pi:.4f} '
                       f'(formal -{m:.4f} +{p:.4f}, inflated by {infl:.1f})')
    lines += [f'  full_well: {det_cfg.get("full_well", 5000)}'.ljust(24)
              + '# NOT fitted: this night never reaches saturation',
              f'  nmax: {det_cfg.get("nmax", 40)}'.ljust(24)
              + '# NOT fitted: a truncation of embin.py\'s Poisson sum',
              '  cic: 0.0'.ljust(24)
              + '# indistinguishable from the sky flux, kept there']
    med, m, p, mi, pi = summary['mu']
    lines += ['',
              f'# mean flux per pixel: {med:.6f} -{mi:.6f} +{pi:.6f} e-/frame '
              f'(sky + CIC)',
              f'# chi2/dof of this model = {chi2_dof:.1f}; the errors above are '
              f'the formal',
              f'# posterior widths inflated by sqrt(chi2/dof) = {infl:.1f}.',
              '# ---------------------------------'
              '-------------------------------------']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description='MCMC calibration of the EMCCD detector constants and the '
                    'mean flux per pixel, straight from the frames, plus the '
                    'optimal histogram bins that follow from them.')
    ap.add_argument('--config', default=os.path.join(here, 'embin_config.yaml'),
                    help='the YAML configuration (default: embin_config.yaml '
                         'beside this script)')
    ap.add_argument('--nframes', type=int, default=None,
                    help='override mcmc.n_frames')
    ap.add_argument('--steps', type=int, default=None, help='override mcmc.steps')
    ap.add_argument('--walkers', type=int, default=None,
                    help='override mcmc.walkers')
    ap.add_argument('--no-bin-draws', action='store_true',
                    help='skip the per-edge uncertainty')
    ap.add_argument('--no-flux-map', action='store_true',
                    help='skip the whole-frame per-pixel fit, and with it the '
                         'flux-against-uncertainty page of the report')
    ap.add_argument('--map-frames', type=int, default=None,
                    help='override mcmc.flux_map_frames: how many frames the '
                         'per-pixel fit on page 3 uses (default: all of them)')
    ap.add_argument('--no-plots', action='store_true',
                    help='numbers only, no PDF written')
    args = ap.parse_args(argv)

    log(f'reading configuration from {os.path.abspath(args.config)}')
    cfg = load_config(args.config)
    mc = cfg.get('mcmc') or {}
    if not mc:
        log('no `mcmc:` section in the configuration, using built-in defaults', 'warn')

    n_frames = args.nframes or int(mc.get('n_frames', 64))
    hdu_index = int(cfg.get('input', {}).get('hdu', 0))
    clip = mc.get('bright_pixel_clip', 5.0)
    clip = None if clip is None else float(clip)
    du = float(mc.get('grid_step', 0.25))

    priors = {'bias': [250.0, 400.0], 'ron': [0.5, 40.0],
              'gain': [5.0, 500.0], 'mu': [1.0e-6, 10.0]}
    priors.update({k: [float(v[0]), float(v[1])]
                   for k, v in (mc.get('priors') or {}).items()})

    # --- data -----------------------------------------------------------
    files = frame_list(cfg, n_frames)
    hist, keep, _ = pooled_histogram(files, hdu_index, clip)
    x, counts = fit_range(hist, mc.get('adu_min'), mc.get('adu_max'))

    # --- posterior ------------------------------------------------------
    post = Posterior(x, counts, priors, du)
    det_cfg = cfg.get('detector', {})
    start = np.array([float(det_cfg.get('bias', np.median(x))),
                      float(det_cfg.get('ron', 7.0)),
                      float(det_cfg.get('gain', 70.0)),
                      np.log(float(mc.get('mu_start', 0.02)))])
    log('starting from the configuration\'s own detector values: '
        f'bias={start[0]:g}, ron={start[1]:g}, gain={start[2]:g} ADU/e-')
    start = max_likelihood(post, start)

    chain = run_mcmc(post,
                     start,
                     walkers=args.walkers or int(mc.get('walkers', 32)),
                     steps=args.steps or int(mc.get('steps', 3000)),
                     burn=int(mc.get('burn', 600)),
                     thin=int(mc.get('thin', 10)),
                     seed=int(mc.get('seed', 42)))

    theta_med = np.median(chain, axis=0)
    expect, pull, chi2_dof, dof = goodness_of_fit(x, counts, theta_med, du)
    log(f'chi2/dof = {chi2_dof:.2f} over {dof} degrees of freedom', 'value')
    if chi2_dof > 2:
        log(f'chi2/dof = {chi2_dof:.1f}: a four-parameter detector does not '
            f'describe {int(counts.sum())} pixel values to their Poisson '
            f'precision. Quote the inflated errors below, not the formal ones', 'warn')

    summary = summarise(chain, chi2_dof)
    for name, unit in (('bias', 'ADU'), ('ron', 'ADU'),
                       ('gain', 'ADU/e-'), ('mu', 'e-/frame')):
        med, m, p, mi, pi = summary[name]
        log(f'{name:>5s} = {med:.5f} -{mi:.5f} +{pi:.5f} {unit}   '
            f'(formal -{m:.5f} +{p:.5f})', 'value')

    # --- outputs --------------------------------------------------------
    out_cfg = cfg.get('output', {})
    outdir = _resolve(cfg, out_cfg.get('directory', 'data_bin'))
    figdir = os.path.join(outdir, out_cfg.get('figures', 'figures'))
    os.makedirs(figdir, exist_ok=True)

    np.savez_compressed(os.path.join(outdir, 'mcmc_chain.npz'),
                        chain=chain, names=np.array(PARAM_NAMES),
                        adu=x, counts=counts, expect=expect,
                        chi2_dof=chi2_dof, n_frames=len(files),
                        n_pixels=int(keep.sum()))
    log(f'posterior samples written to {os.path.join(outdir, "mcmc_chain.npz")}')

    bin_cfg = mc.get('bins') or {}
    n_draws = 0 if args.no_bin_draws else int(bin_cfg.get('n_draws', 24))
    design, spread = propose_bins(summary, cfg, bin_cfg, chain, n_draws)

    # Everything the report needs to say what this run was, gathered in one
    # place so the template only ever reads from a dictionary.
    det_cfg = cfg.get('detector', {})
    meta = {
        'dataset': os.path.basename(_resolve(cfg, cfg['input']['directory'])),
        'directory': _resolve(cfg, cfg['input']['directory']),
        'stamp': _timestamp(),
        'n_frames': len(files), 'n_kept': int(keep.sum()),
        'n_total': int(keep.size), 'n_samples': int(counts.sum()),
        'clip': (f'{clip:g} robust sigma on the per-pixel mean'
                 if clip is not None else 'none, every pixel fitted'),
        'adu_lo': int(x[0]), 'adu_hi': int(x[-1]), 'n_bins': len(x),
        'walkers': args.walkers or int(mc.get('walkers', 32)),
        'steps': args.steps or int(mc.get('steps', 3000)),
        'n_post': len(chain), 'dof': dof,
        'bin_mu_min': float(bin_cfg.get('mu_min', 1.0e-3)),
        'bin_mu_max': float(bin_cfg.get('mu_max', 8.0)),
        'n_draws': n_draws,
        'full_well': det_cfg.get('full_well', 5000),
        'nmax': det_cfg.get('nmax', 40),
    }

    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')

        pages = {'corner': corner_page(chain),
                 'fit': fit_page(x, counts, expect, pull, design)}

        # The third page needs a per-pixel fit over the whole frame, which costs
        # roughly as much as the MCMC did. It is the page that shows whether any
        # of this works, so it is on by default, but --no-flux-map skips it.
        if not args.no_flux_map:
            # Its own frame count, deliberately: see flux_sigma_map's docstring
            # for why the per-pixel fit needs far more frames than the pooled
            # one does. null (the default) means the whole sequence.
            n_map = args.map_frames or mc.get('flux_map_frames', None)
            n_map = None if n_map in (None, 0, 'all') else int(n_map)
            map_files = (files if n_map == len(files)
                         else frame_list(cfg, n_map))
            mu_map, sig_map = flux_sigma_map(cfg, map_files, design,
                                             summary, hdu_index)
            det_note = (f'bias = {summary["bias"][0]:.2f} ADU, '
                        f'RON = {summary["ron"][0]:.2f} ADU, '
                        f'gain = {summary["gain"][0]:.1f} ADU/e-, '
                        f'{design.nbins} optimal bins, all fitted here')
            pages['flux_sigma'], stats = flux_sigma_page(
                mu_map, sig_map, len(map_files), summary, det_note)
            meta.update(stats, map_frames=len(map_files))
            fits.HDUList([fits.PrimaryHDU(mu_map.astype(np.float32)),
                          fits.ImageHDU(sig_map.astype(np.float32),
                                        name='FLUX_ERR')]).writeto(
                os.path.join(outdir, 'mcmc_flux_map.fits'), overwrite=True)
            log(f'per-pixel flux and error written to '
                f'{os.path.join(outdir, "mcmc_flux_map.fits")}')

        names = save_figures(pages, figdir)
        report = write_report(outdir, out_cfg.get('figures', 'figures'), meta,
                              summary, design, spread, chi2_dof, names)
        if report is None:
            # Deliberately NOT called mcmc_report.pdf: that name belongs to the
            # typeset one in the folder above, and two files a directory apart
            # sharing a name is how you end up reading last week's run.
            stitch_figures(pages,
                           os.path.join(figdir, 'mcmc_figures_only.pdf'))
        import matplotlib.pyplot as plt
        plt.close('all')

    print(yaml_block(summary, design, chi2_dof, cfg))
    return 0


if __name__ == '__main__':
    sys.exit(main())
