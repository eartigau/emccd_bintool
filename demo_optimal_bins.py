#!/usr/bin/env python3
"""
demo_optimal_bins.py
====================
Demonstration and Monte-Carlo proof for ``bin_optimizer.py``.

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

It answers, end to end and with simulated data rather than theory alone:

  1. How many histogram bins does an EMCCD flux retrieval actually need?
  2. Where exactly do those bin edges go?
  3. How does that compare with classical thresholding (photon counting), given
     the best threshold the method could possibly use?
  4. Does a real maximum-likelihood fit to those bins reach the accuracy the
     Fisher-information calculation promises?

Everything is driven by the detector constants at the top of ``main()`` (or by
``emccd_config.yaml`` when it is present), so the same script re-runs for any
other EMCCD.

Outputs
-------
figures/bins/fig1_bin_count.pdf     accuracy retained versus number of bins
figures/bins/fig2_where_the_cuts_go.pdf  the pixel-value PDF with the 8 cuts
figures/bins/fig3_information.pdf   where the flux information sits in ADU
figures/bins/fig4_efficiency.pdf    efficiency versus flux, several designs
figures/bins/fig5_montecarlo.pdf    simulated sigma versus the Cramer-Rao bound
figures/bins/fig6_thresholding.pdf  optimal thresholding, on its own best terms
figures/bins/bin_design.json        the numbers, for the write-up
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bin_optimizer as bo
from bin_optimizer import log

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'bins')

# A palette that stays legible in print and in both light and dark viewers.
CDESIGN, CFULL, CNAIVE, CGREY = '#2E6FDB', '#111111', '#D1495B', '#8A8A8A'


# ---------------------------------------------------------------------------
# Reference: a histogram so fine it is indistinguishable from "keep everything"
# ---------------------------------------------------------------------------
def reference_edges(det, n_tail=400):
    """
    A very fine binning used as the practical stand-in for the full 1-ADU
    histogram in the Monte Carlo: every single ADU through the read-noise peak,
    then n_tail sqrt-spaced steps up to saturation.  Its efficiency is checked
    against the exact 1-ADU information and is >0.9999 in practice, so any
    accuracy it fails to reach is a property of the estimator, not of the bins.
    """
    sat = int(round(det.sat_adu))
    fine = np.arange(max(0, int(det.bias - 8 * det.ron)),
                     int(np.ceil(det.bias + 8 * det.ron)) + 1)
    u0 = (fine[-1] - det.bias) / det.gain
    tail = det.bias + det.gain * np.linspace(np.sqrt(max(u0, 1e-6)),
                                             np.sqrt(det.S), n_tail) ** 2
    e = np.concatenate([[0], fine, np.round(tail), [sat + 1]])
    return np.unique(np.clip(e.astype(np.int64), 0, sat + 1))


def uniform_edges(det, K):
    """The obvious, thoughtless alternative: K equal-width bins bias -> sat."""
    sat = int(round(det.sat_adu))
    e = np.linspace(max(0.0, det.bias - 3 * det.ron), sat, K)
    return np.unique(np.concatenate([[0], np.round(e).astype(np.int64),
                                     [sat + 1]]))[:K + 1]


# ---------------------------------------------------------------------------
# Thresholding: the two-bin competitor
# ---------------------------------------------------------------------------
def threshold_edges(det, cut):
    """The two-bin design "below / at-or-above a single cut", in cell indices."""
    return np.array([0, int(cut), int(round(det.sat_adu)) + 1], dtype=np.int64)


def threshold_eta(det, P, D, i_ref, cuts=None):
    """
    eta(cut, mu) for every single-threshold design.

    Row `a` of the returned array is the efficiency curve of a detector read
    reduced to one bit: "did this pixel exceed cuts[a]?".  That bit is exactly
    what classical EMCCD photon counting keeps, so the whole method lives in
    this one table.
    """
    if cuts is None:
        cuts = np.arange(max(1, int(det.bias - 8 * det.ron)),
                         int(round(det.sat_adu)) + 1)
    eta = np.array([bo.efficiency(threshold_edges(det, c), P, D, i_ref)
                    for c in cuts])
    return np.asarray(cuts), eta


def threshold_study(det, P, D, i_ref, c_pc=5.0, cuts=None):
    """
    The three thresholding regimes worth putting next to a real bin design.

    ``photon_counting``
        one fixed cut at bias + ``c_pc`` sigma, placed to keep read noise out
        of the counted population and for no other reason.  This is what
        photon counting normally means in practice.
    ``best_fixed``
        the single cut that maximises the *worst* efficiency over the flux
        range, i.e. thresholding tuned by the same max-min criterion the bin
        design is tuned by.  The fair champion of the method.
    ``envelope``
        the best cut re-chosen at every flux.  Unreachable in practice (it
        would need the answer in advance), so it is a ceiling on thresholding
        itself rather than on any one implementation of it.

    Returns a dict; efficiencies are eta, not sqrt(eta).
    """
    cuts, eta = threshold_eta(det, P, D, i_ref, cuts)
    worst = eta.min(axis=1)
    i_fix = int(np.argmax(worst))
    i_env = np.argmax(eta, axis=0)
    cut_pc = int(np.ceil(det.bias + c_pc * det.ron))
    j_pc = int(np.searchsorted(cuts, cut_pc))
    return {
        'cuts': cuts,
        'photon_counting': {'cut': cut_pc, 'eta': eta[j_pc],
                            'sigma': c_pc},
        'best_fixed': {'cut': int(cuts[i_fix]), 'eta': eta[i_fix],
                       'sigma': float((cuts[i_fix] - det.bias) / det.ron),
                       'u': float((cuts[i_fix] - det.bias) / det.gain)},
        'envelope': {'eta': eta[i_env, np.arange(eta.shape[1])],
                     'cut': cuts[i_env]},
    }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
TRIM = 0.005          # fraction of the estimates trimmed from each tail


def _trim_factor(f):
    """
    Ratio std(trimmed Gaussian) / sigma when a fraction ``f`` is cut from each
    tail.  For a normal truncated at +/- a, var = 1 - 2 a phi(a) / (1 - 2f).
    """
    from scipy.stats import norm
    a = norm.isf(f)
    return float(np.sqrt(1.0 - 2.0 * a * norm.pdf(a) / (1.0 - 2.0 * f)))


def mc_scatter(det, designs, tables, mu_true, n_reads, n_trials, rng, fit_grid,
               block=40):
    """
    Simulate ``n_trials`` independent pixels, each read ``n_reads`` times, at a
    true flux ``mu_true``.  Every design bins the SAME simulated reads, so the
    comparison between designs carries no Monte-Carlo noise of its own.

    Trials are processed in blocks so that the raw ADU array never has to be
    held in full: at the lowest fluxes ``n_reads`` runs into the hundreds of
    thousands.

    Returns {name: (median_estimate, robust_sigma)}.
    """
    est = {name: [] for name in designs}
    done = 0
    while done < n_trials:
        nb = min(block, n_trials - done)
        x = bo.simulate_adu(det, mu_true, n_reads * nb, rng).reshape(nb, n_reads)
        for name, edges in designs.items():
            nbin = len(edges) - 1
            idx = np.searchsorted(np.asarray(edges[1:-1], dtype=np.float64),
                                  x, side='right')
            counts = np.empty((nb, nbin))
            for t in range(nb):
                counts[t] = np.bincount(idx[t], minlength=nbin)
            est[name].append(bo.mle_from_counts(counts, fit_grid, tables[name]))
        done += nb

    out = {}
    for name in designs:
        e = np.sort(np.concatenate(est[name]))
        # Scatter is measured as a lightly trimmed standard deviation.  A plain
        # MAD (or any quantile) lands ON the log-spaced fit grid and is
        # therefore quantised at the grid step, which at high flux is a
        # noticeable fraction of sigma itself; a standard deviation averages
        # over the whole sample and does not suffer from that.  The 0.5 % trim
        # protects against a trial that happens to see no electron at all and
        # sends the MLE to the edge of the grid.
        cut = max(1, int(round(TRIM * e.size)))
        core = e[cut:e.size - cut]
        # Trimming a Gaussian and taking the plain standard deviation of what
        # is left biases sigma low by a known factor; divide it back out so the
        # points can be compared with the Cramer-Rao bound directly.
        out[name] = (float(np.median(e)),
                     float(np.std(core, ddof=1)) / _trim_factor(TRIM))
    return out


def reads_for_flux(mu, target_electrons=400.0, lo=2000, hi=300000):
    """
    Number of reads to simulate at flux ``mu``.

    Kept inversely proportional to the flux so that every point of the Monte
    Carlo collects a comparable number of detected electrons.  With a fixed
    read count the low-flux estimates would rest on a handful of electrons,
    the estimator would be visibly discrete, and the measured scatter would
    stop meaning what the Cramer-Rao bound predicts.
    """
    return int(np.clip(target_electrons / mu, lo, hi))


# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    np.seterr(all='ignore')

    # --- detector under study -------------------------------------------
    BIAS, GAIN, RON, SAT, CIC = 1000.0, 5000.0, 30.0, 60000.0, 0.001
    MU_MIN, MU_MAX = 1e-3, 8.0
    C_PC = 5.0            # the conventional photon-counting cut, in RON sigmas

    log('EMCCD optimal-binning demonstration', 'info')
    log(f'detector: bias={BIAS:g} ADU, G={GAIN:g} ADU/e-, RON={RON:g} ADU, '
        f'sat={SAT:g} ADU, CIC={CIC:g} e-/read', 'value')
    log(f'flux range of interest: {MU_MIN:g} to {MU_MAX:g} e-/read', 'value')

    det = bo.Detector(bias=BIAS, gain=GAIN, ron=RON, sat_adu=SAT, cic=CIC,
                      nmax=60)
    mu = np.geomspace(MU_MIN, MU_MAX, 21)
    pn = bo.per_electron_cell_probs(det)
    p, dp, i_ref = bo.cell_model(det, mu, pn=pn)
    P, D = bo._prefix(p, dp)
    thr = threshold_study(det, P, D, i_ref, c_pc=C_PC)
    log(f'thresholding, {C_PC:g} sigma photon counting: cut '
        f'{thr["photon_counting"]["cut"]} ADU, worst-case accuracy '
        f'{100 * np.sqrt(thr["photon_counting"]["eta"].min()):.1f} %', 'value')
    log(f'thresholding, best fixed cut: {thr["best_fixed"]["cut"]} ADU '
        f'(bias + {thr["best_fixed"]["sigma"]:.1f} sigma = '
        f'{thr["best_fixed"]["u"]:.2f} G), worst-case accuracy '
        f'{100 * np.sqrt(thr["best_fixed"]["eta"].min()):.1f} %', 'value')
    log(f'thresholding, best cut re-chosen at every flux: worst-case accuracy '
        f'{100 * np.sqrt(thr["envelope"]["eta"].min()):.1f} % -- no single '
        f'threshold can do better than this', 'value')
    log(f'reduced parameters: r = RON/G = {det.r:.4f}, '
        f'S = (sat-bias)/G = {det.S:.2f} e-', 'value')

    # sanity: the fine reference really is as good as keeping every ADU
    ref = reference_edges(det)
    eta_ref = bo.efficiency(ref, P, D, i_ref)
    log(f'fine reference histogram: {len(ref) - 1} bins, '
        f'worst-case efficiency {eta_ref.min():.6f} (1 = full 1-ADU histogram)',
        'value')

    # ================================================================
    # 1.  How many bins?
    # ================================================================
    log('step 1: scanning the number of bins', 'info')
    Ks = [4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
    designs, worst_eta = {}, {}
    for K in Ks:
        d = bo.design_bins(BIAS, GAIN, RON, SAT, nbins=K, mu_min=MU_MIN,
                           mu_max=MU_MAX, cic=CIC, method='both')
        designs[K] = d
        worst_eta[K] = d.worst_eta
        flag = 'PASS' if d.worst_accuracy >= 0.95 else 'fail'
        log(f'  K={K:3d}  eta={d.worst_eta:.4f}  '
            f'sigma_full/sigma_K={d.worst_accuracy:.4f}  [{flag}]',
            'value' if flag == 'PASS' else 'skip')

    pow2 = [K for K in Ks if K & (K - 1) == 0]
    kmin = min(K for K in pow2 if np.sqrt(worst_eta[K]) >= 0.95)
    log(f'smallest power-of-two bin count meeting the 95 % accuracy target: '
        f'K = {kmin}', 'value')

    best = designs[8]
    print()
    print(best.summary())
    print()

    # --- fig 1: accuracy versus bin count -------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    acc = np.array([np.sqrt(worst_eta[K]) for K in Ks])
    ax.plot(Ks, 100 * acc, 'o-', color=CDESIGN, lw=1.8, ms=5,
            label=r'optimal $K$-bin design')
    unif = []
    for K in Ks:
        e = uniform_edges(det, K)
        unif.append(np.sqrt(bo.efficiency(e, P, D, i_ref).min()))
    ax.plot(Ks, 100 * np.array(unif), 's--', color=CNAIVE, lw=1.4, ms=4,
            label=r'$K$ equal-width bins')
    ax.axhline(95, color=CGREY, ls=':', lw=1.2)
    ax.annotate('95 % target', xy=(Ks[-1], 95), xytext=(-6, 4),
                textcoords='offset points', ha='right', color=CGREY, fontsize=9)
    ax.axvline(kmin, color=CDESIGN, ls=':', lw=1.0, alpha=0.6)
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel('number of histogram bins $K$')
    ax.set_ylabel(r'worst-case accuracy $\sigma_{\rm full}/\sigma_K$  [%]')
    ax.set_ylim(0, 104)
    ax.set_title(r'8 bins retain %.1f %% of the full-histogram accuracy'
                 % (100 * np.sqrt(worst_eta[8])))
    ax.legend(frameon=False, loc='lower right')
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig1_bin_count.pdf'))
    plt.close(fig)
    log('wrote fig1_bin_count.pdf', 'info')

    # ================================================================
    # 2.  Where do the cuts go?
    # ================================================================
    log('step 2: drawing the pixel-value PDF and the cuts', 'info')
    edges8 = best.cell_edges
    xs = np.arange(len(p[0]))
    x_sig = (xs - det.bias) / det.ron          # pixel value in RON sigmas
    x_u = (xs - det.bias) / det.gain           # pixel value in gain units
    cuts_sig = (edges8[1:-1] - det.bias) / det.ron
    cuts_u = (edges8[1:-1] - det.bias) / det.gain
    SHOW = [(0, r'$\mu=10^{-3}$', '#3B7DD8'), (10, r'$\mu=0.09$', '#E08A1E'),
            (16, r'$\mu=1.3$', '#1F9D55'), (20, r'$\mu=8$', '#C43C4E')]

    def draw_cuts(ax, coords, lim):
        for c in coords:
            if lim[0] <= c <= lim[1]:
                ax.axvline(c, color=CDESIGN, lw=1.0, alpha=0.75)

    LIM_SIG, LIM_U = (-6.0, 14.0), (-0.3, det.S + 0.3)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharey='row')
    for col, (xv, lim, xlab, cuts) in enumerate([
            (x_sig, LIM_SIG, r'$(x-\mathrm{bias})/\sigma_{\rm RON}$', cuts_sig),
            (x_u, LIM_U, r'$u=(x-\mathrm{bias})/G$', cuts_u)]):
        ax = axes[0, col]
        for j, lab, c in SHOW:
            ax.semilogy(xv, np.clip(p[j], 1e-12, None), lw=1.3, color=c,
                        label=lab + r' e$^-$/read')
        draw_cuts(ax, cuts, lim)
        ax.set_xlim(*lim)
        ax.set_ylim(1e-10, 1)
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_title('the read-noise peak' if col == 0 else 'the EM tail',
                     fontsize=10)
        if col == 0:
            ax.set_ylabel('P(pixel value) per ADU')

        ax = axes[1, col]
        for j, lab, c in SHOW:
            with np.errstate(all='ignore'):
                info = np.where(p[j] > 0,
                                dp[j] ** 2 / np.where(p[j] > 0, p[j], 1), 0)
            ax.semilogy(xv, np.clip(info, 1e-14, None), lw=1.1, color=c)
        draw_cuts(ax, cuts, lim)
        ax.set_xlim(*lim)
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.2, lw=0.5)
        if col == 0:
            ax.set_ylabel('Fisher information per ADU')
    fig.suptitle('Where the 8 cuts go (vertical lines), and why', fontsize=12)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=9, ncol=4,
               loc='upper center', bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(FIGDIR, 'fig2_where_the_cuts_go.pdf'))
    plt.close(fig)
    log('wrote fig2_where_the_cuts_go.pdf', 'info')

    # --- fig 3: cumulative information versus pixel value ---------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for col, (xv, lim, xlab, cuts) in enumerate([
            (x_sig, LIM_SIG, r'$(x-\mathrm{bias})/\sigma_{\rm RON}$', cuts_sig),
            (x_u, LIM_U, r'$u=(x-\mathrm{bias})/G$', cuts_u)]):
        ax = axes[col]
        for j, lab, c in SHOW:
            with np.errstate(all='ignore'):
                info = np.where(p[j] > 0,
                                dp[j] ** 2 / np.where(p[j] > 0, p[j], 1), 0)
            ax.plot(xv, np.cumsum(info) / info.sum(), lw=1.4, color=c,
                    label=lab + r' e$^-$/read')
        draw_cuts(ax, cuts, lim)
        ax.set_xlim(*lim)
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title('the read-noise peak' if col == 0 else 'the EM tail',
                     fontsize=10)
        if col == 0:
            ax.set_ylabel('cumulative fraction of\nthe flux information')
            ax.legend(frameon=False, fontsize=8, loc='center right')
    fig.suptitle('The cuts follow the information, not the counts', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGDIR, 'fig3_information.pdf'))
    plt.close(fig)
    log('wrote fig3_information.pdf', 'info')

    # ================================================================
    # 3.  Efficiency versus flux
    # ================================================================
    log('step 3: efficiency curves', 'info')
    mu_fine = np.geomspace(MU_MIN, MU_MAX, 61)
    pf, dpf, iref_f = bo.cell_model(det, mu_fine, pn=pn)
    Pf, Df = bo._prefix(pf, dpf)

    curves = {
        '1 threshold, best fixed cut': (threshold_edges(det, thr['best_fixed']['cut']), '#8E44AD', '-.'),
        '8 bins, optimal': (designs[8].cell_edges, CDESIGN, '-'),
        '16 bins, optimal': (designs[16].cell_edges, '#1F9D55', '-'),
        '4 bins, optimal': (designs[4].cell_edges, '#B07D2B', '--'),
        '8 bins, equal width': (uniform_edges(det, 8), CNAIVE, '--'),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, (e, c, ls) in curves.items():
        ax.semilogx(mu_fine, 100 * np.sqrt(bo.efficiency(e, Pf, Df, iref_f)),
                    ls, color=c, lw=1.7, label=name)
    ax.axhline(95, color=CGREY, ls=':', lw=1.2)
    ax.set_xlabel(r'true flux $\mu$  [e$^-$/read]')
    ax.set_ylabel(r'$\sigma_{\rm full}/\sigma_K$  [%]')
    ax.set_ylim(0, 104)
    ax.set_title('accuracy retained, across the whole flux range')
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig4_efficiency.pdf'))
    plt.close(fig)
    log('wrote fig4_efficiency.pdf', 'info')

    # ================================================================
    # 3b.  Against thresholding, on its own best terms
    # ================================================================
    log('step 3b: the same comparison against optimal thresholding', 'info')
    thr_f = threshold_study(det, Pf, Df, iref_f, c_pc=C_PC)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    ax.semilogx(mu_fine,
                100 * np.sqrt(bo.efficiency(designs[8].cell_edges, Pf, Df,
                                            iref_f)),
                '-', color=CDESIGN, lw=2.0, label='8 bins, optimal')
    ax.fill_between(mu_fine, 100 * np.sqrt(thr_f['envelope']['eta']), 104,
                    color='#8E44AD', alpha=0.07, lw=0)
    ax.semilogx(mu_fine, 100 * np.sqrt(thr_f['envelope']['eta']), '-',
                color='#8E44AD', lw=1.6,
                label='1 threshold, best cut at each flux (ceiling)')
    ax.semilogx(mu_fine, 100 * np.sqrt(thr_f['best_fixed']['eta']), '-.',
                color='#8E44AD', lw=1.6,
                label=f'1 threshold, best fixed cut '
                      f'({thr_f["best_fixed"]["cut"]} ADU)')
    ax.semilogx(mu_fine, 100 * np.sqrt(thr_f['photon_counting']['eta']), '--',
                color='#B07D2B', lw=1.6,
                label=rf'photon counting, ${C_PC:g}\sigma$ cut '
                      f'({thr_f["photon_counting"]["cut"]} ADU)')
    ax.axhline(95, color=CGREY, ls=':', lw=1.2)
    ax.set_xlabel(r'true flux $\mu$  [e$^-$/read]')
    ax.set_ylabel(r'$\sigma_{\rm full}/\sigma$  [%]')
    ax.set_ylim(0, 104)
    ax.set_title('one threshold versus eight bins', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    ax.loglog(mu_fine, (thr_f['envelope']['cut'] - det.bias) / det.gain, '-',
              color='#8E44AD', lw=1.6, label='best cut at each flux')
    ax.axhline((thr_f['best_fixed']['cut'] - det.bias) / det.gain, ls='-.',
               color='#8E44AD', lw=1.2, label='best fixed cut')
    ax.axhline((thr_f['photon_counting']['cut'] - det.bias) / det.gain, ls='--',
               color='#B07D2B', lw=1.2, label=rf'${C_PC:g}\sigma$ cut')
    ax.set_xlabel(r'true flux $\mu$  [e$^-$/read]')
    ax.set_ylabel(r'cut position  $u=(x-\mathrm{bias})/G$')
    ax.set_title('where the best threshold wants to sit', fontsize=10)
    ax.annotate('counts single electrons', xy=(2.2e-3, 0.021), fontsize=8,
                color='#8E44AD', va='top')
    ax.annotate('gives up on them and\nmeasures the tail instead',
                xy=(1.9, 2.6), fontsize=8, color='#8E44AD', va='bottom')
    ax.legend(frameon=False, fontsize=8, loc='center left')
    ax.grid(alpha=0.25, lw=0.5, which='both')

    fig.suptitle('Thresholding compared with an optimal histogram', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGDIR, 'fig6_thresholding.pdf'))
    plt.close(fig)
    log('wrote fig6_thresholding.pdf', 'info')

    # ================================================================
    # 4.  Monte Carlo: does a real fit deliver it?
    # ================================================================
    log('step 4: Monte-Carlo validation of the Cramer-Rao prediction', 'info')
    N_TRIALS = 1200
    mu_mc = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 8.0])
    n_reads_mc = np.array([reads_for_flux(m) for m in mu_mc])
    # The grid step must stay well below sigma_mu at the HIGHEST flux, or the
    # maximum-likelihood estimates quantise onto the grid and the measured
    # scatter collapses.  12000 log-spaced points give a ~0.1 % step, i.e.
    # roughly 10 points per sigma even at mu = 8 e-/read.
    fit_grid = np.geomspace(1e-4, 25.0, 12000)
    rng = np.random.default_rng(20260821)

    mc_designs = {
        'reference': ref,
        '8 bins, optimal': designs[8].cell_edges,
        '8 bins, equal width': uniform_edges(det, 8),
    }
    log(f'{N_TRIALS} independent pixels per flux level; reads per pixel set to '
        f'hold ~400 detected electrons ({n_reads_mc.min()} to '
        f'{n_reads_mc.max()})', 'value')
    tables = {k: bo.bin_probability_table(det, e, fit_grid, pn=pn)
              for k, e in mc_designs.items()}

    results = {k: {'sig': [], 'med': []} for k in mc_designs}
    for m, nr in zip(mu_mc, n_reads_mc):
        r = mc_scatter(det, mc_designs, tables, m, int(nr), N_TRIALS, rng,
                       fit_grid)
        for k in mc_designs:
            results[k]['med'].append(r[k][0])
            results[k]['sig'].append(r[k][1])
        log(f'  mu={m:7.3f}  N={nr:7d}: sigma  ref={r["reference"][1]:.6f}  '
            f'optimal-8={r["8 bins, optimal"][1]:.6f}  '
            f'uniform-8={r["8 bins, equal width"][1]:.6f}', 'value')

    # Cramer-Rao predictions on the same flux points, per single read
    pm, dpm, iref_m = bo.cell_model(det, mu_mc, pn=pn)
    Pm, Dm = bo._prefix(pm, dpm)
    cr = {k: 1.0 / np.sqrt(bo.fisher_of_edges(e, Pm, Dm))
          for k, e in mc_designs.items()}
    # simulated scatter renormalised to one read, so both are functions of mu
    sig1 = {k: np.array(results[k]['sig']) * np.sqrt(n_reads_mc)
            for k in mc_designs}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    style = {'reference': (CFULL, 'o', 11, 'none'),
             '8 bins, optimal': (CDESIGN, 'o', 5, CDESIGN),
             '8 bins, equal width': (CNAIVE, 's', 5, CNAIVE)}
    for k, (c, mk, ms, mfc) in style.items():
        ax.loglog(mu_mc, cr[k], '-', color=c, lw=1.4, alpha=0.75)
        ax.loglog(mu_mc, sig1[k], mk, color=c, ms=ms, mfc=mfc, mew=1.3,
                  label=k)
    ax.set_xlabel(r'true flux $\mu$  [e$^-$/read]')
    ax.set_ylabel(r'$\sigma_{\hat\mu}\,\sqrt{N}$  [e$^-$ per single read]')
    ax.set_title(f'lines: Cramer-Rao bound;  points: simulation\n'
                 f'({N_TRIALS} pixels per flux level)', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, which='both')

    ax = axes[1]
    for k, c in [('8 bins, optimal', CDESIGN), ('8 bins, equal width', CNAIVE)]:
        ratio = np.array(results['reference']['sig']) / np.array(results[k]['sig'])
        ax.semilogx(mu_mc, 100 * ratio, 'o-', color=c, lw=1.5, ms=5,
                    label=k + ' (simulated)')
        ax.semilogx(mu_mc, 100 * cr['reference'] / cr[k], ':', color=c, lw=1.4,
                    label=k + ' (predicted)')
    ax.axhline(95, color=CGREY, ls=':', lw=1.2)
    ax.set_xlabel(r'true flux $\mu$  [e$^-$/read]')
    ax.set_ylabel(r'$\sigma_{\rm reference}/\sigma_K$  [%]')
    ax.set_ylim(0, 115)
    ax.set_title('accuracy retained: simulated versus predicted', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig5_montecarlo.pdf'))
    plt.close(fig)
    log('wrote fig5_montecarlo.pdf', 'info')

    # ================================================================
    # 5.  Verdict + machine-readable dump
    # ================================================================
    ratio8 = sig1['reference'] / sig1['8 bins, optimal']
    log('verdict', 'info')
    log(f'  predicted worst-case accuracy of the 8-bin design : '
        f'{100 * designs[8].worst_accuracy:.1f} %', 'value')
    log(f'  simulated  worst-case accuracy of the 8-bin design : '
        f'{100 * ratio8.min():.1f} %', 'value')
    log(f'  8-bin edges [ADU]: {best.edge_list()}', 'value')
    log(f'  best any threshold can do, worst case over the range : '
        f'{100 * np.sqrt(thr_f["envelope"]["eta"].min()):.1f} %', 'value')

    payload = {
        'detector': {'bias': BIAS, 'gain': GAIN, 'ron': RON, 'saturation': SAT,
                     'cic': CIC, 'r': det.r, 'S': det.S},
        'flux_range': [MU_MIN, MU_MAX],
        'kmin_power_of_two': int(kmin),
        'reference_bins': int(len(ref) - 1),
        'reference_efficiency': float(eta_ref.min()),
        'scan': {str(K): {'worst_eta': float(worst_eta[K]),
                          'worst_accuracy': float(np.sqrt(worst_eta[K])),
                          'edges': designs[K].edge_list(),
                          'method': designs[K].method}
                 for K in Ks},
        'uniform_scan': {str(K): float(u) for K, u in zip(Ks, unif)},
        'thresholding': {
            'mu_grid': [float(v) for v in mu_fine],
            'photon_counting': {
                'sigma': C_PC,
                'cut': int(thr_f['photon_counting']['cut']),
                'eta': [float(v) for v in thr_f['photon_counting']['eta']]},
            'best_fixed': {
                'cut': int(thr_f['best_fixed']['cut']),
                'sigma': float(thr_f['best_fixed']['sigma']),
                'u': float(thr_f['best_fixed']['u']),
                'eta': [float(v) for v in thr_f['best_fixed']['eta']]},
            'envelope': {
                'cut': [int(v) for v in thr_f['envelope']['cut']],
                'eta': [float(v) for v in thr_f['envelope']['eta']]},
        },
        'design8': {'edges': best.edge_list(),
                    'u_edges': [float(v) for v in best.u_edges[:-1]],
                    'sigma_cuts': [float((e - BIAS) / RON)
                                   for e in best.edges[1:-1]],
                    'eta': [float(v) for v in best.eta],
                    'mu_grid': [float(v) for v in best.mu_grid],
                    'worst_accuracy': float(best.worst_accuracy),
                    'method': best.method,
                    'params': best.params},
        'monte_carlo': {'n_reads': [int(v) for v in n_reads_mc],
                        'n_trials': N_TRIALS,
                        'mu': [float(v) for v in mu_mc],
                        'sigma_per_read': {k: [float(v) for v in sig1[k]]
                                           for k in sig1},
                        'sigma': {k: [float(v) for v in results[k]['sig']]
                                  for k in results},
                        'median': {k: [float(v) for v in results[k]['med']]
                                   for k in results},
                        'cramer_rao': {k: [float(v) for v in cr[k]] for k in cr},
                        'ratio8_min': float(ratio8.min())},
    }
    with open(os.path.join(FIGDIR, 'bin_design.json'), 'w') as fh:
        json.dump(payload, fh, indent=2)
    log(f'wrote {os.path.join(FIGDIR, "bin_design.json")}', 'info')
    return payload


if __name__ == '__main__':
    main()
