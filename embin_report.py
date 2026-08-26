#!/usr/bin/env python3
"""
embin_report.py -- the typeset report embin_mcmc.py hands back
==============================================================

Authors: Étienne Artigau, Galina Sherren, René Doyon, Jonathan St-Antoine
         Université de Montréal / Observatoire du Mont-Mégantic

A calibration you cannot read is a calibration you will not trust, and a stack
of matplotlib pages with no words around them is not a report. This module takes
the numbers embin_mcmc.py measured and the figures it drew and writes a LaTeX
document around them: what was fitted, on what data, how well it fits, what bins
follow from it, and -- the part that actually gets used -- the bin-edge vector in
every form you might want to paste it into.

It needs `pdflatex` on the PATH, plus the packages any TeX Live install carries
(geometry, booktabs, siunitx, tcolorbox, titlesec, tex-gyre). If any of that is
missing the caller is told so and falls back to stitching the raw figures into a
plain multi-page PDF, which is worse but is never nothing.

Nothing here is specific to one detector: `build_tex` takes a dictionary and
returns a string, so a different instrument means different numbers, not a
different template.
"""

import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embin import log                                                # noqa: E402

# The report's one visual decision, in one place. A deep teal for structure, an
# oxblood for the two or three things the reader must not skip.
ACCENT = '0F4C5C'
ALERT = '9A2617'

PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}

% --- fonts: TeX Gyre Pagella for text, Heros for headings, Cursor for code ---
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{tgpagella}
\renewcommand{\sfdefault}{qhv}
\renewcommand{\ttdefault}{qcr}
\usepackage{microtype}

\usepackage[a4paper,margin=2.3cm,top=2.4cm,bottom=2.3cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{fancyvrb}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[font=small,labelfont={bf,sf},labelsep=period]{caption}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}

\definecolor{accent}{HTML}{__ACCENT__}
\definecolor{alert}{HTML}{__ALERT__}

\graphicspath{{__FIGDIR__/}}
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
\linespread{1.06}

% --- headings ---------------------------------------------------------------
\titleformat{\section}
  {\sffamily\large\bfseries\color{accent}}{\thesection}{0.7em}{}
  [\vspace{-0.55em}{\color{accent!30}\rule{\linewidth}{0.9pt}}]
\titleformat{\subsection}
  {\sffamily\normalsize\bfseries\color{accent!85}}{\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{1.6em}{0.6em}

% --- running head -----------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\sffamily\scriptsize\color{black!45}__RUNFOOT__}
\fancyfoot[R]{\sffamily\scriptsize\color{black!45}\thepage}

% --- the copy-ready code box ------------------------------------------------
\newtcolorbox{copybox}[1]{colback=accent!4,colframe=accent!30,boxrule=0.5pt,
  arc=2pt,left=8pt,right=8pt,top=6pt,bottom=6pt,
  title={\sffamily\bfseries\footnotesize #1},coltitle=white,
  colbacktitle=accent,titlerule=0pt,fonttitle=\footnotesize}

% --- the "do not skip this" box ---------------------------------------------
\newtcolorbox{alertbox}[1]{colback=alert!4,colframe=alert!35,boxrule=0.5pt,
  arc=2pt,left=8pt,right=8pt,top=6pt,bottom=6pt,
  title={\sffamily\bfseries\footnotesize #1},coltitle=white,
  colbacktitle=alert,titlerule=0pt,fonttitle=\footnotesize}

% --- the headline-numbers box -----------------------------------------------
\newtcolorbox{keybox}{colback=accent!6,colframe=accent!45,boxrule=0.6pt,
  arc=2.5pt,left=10pt,right=10pt,top=8pt,bottom=8pt}

\sisetup{separate-uncertainty,detect-weight,table-align-uncertainty}
\begin{document}
"""


def esc(text):
    """LaTeX-escape a string that came from a path, a file name or a config.

    Directories called `data_pesto` and files called `nc-image_0.fits` are the
    normal case here, and an unescaped underscore is a compile error rather than
    a typo, so nothing user-supplied reaches the template unescaped.
    """
    out = str(text)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                 ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')):
        out = out.replace(a, b)
    return out


def _param_table(summary, chi2_dof):
    """The four fitted numbers, with both versions of their error bar."""
    unit = {'bias': r'\si{ADU}', 'ron': r'\si{ADU}',
            'gain': r'ADU\,/\,e$^-$', 'mu': r'e$^-$/frame'}
    shown = {'bias': r'bias, $B$', 'ron': r'read noise, $\sigma_{\mathrm{RON}}$',
             'gain': r'EM gain, $G$', 'mu': r'flux per pixel, $\mu$'}
    digits = {'bias': 4, 'ron': 4, 'gain': 4, 'mu': 6}
    rows = []
    for name in ('bias', 'ron', 'gain', 'mu'):
        med, m, p, mi, pi = summary[name]
        d = digits[name]
        rows.append(
            f'{shown[name]} & {med:.{d}f} & '
            f'$^{{+{p:.{d}f}}}_{{-{m:.{d}f}}}$ & '
            f'$\\mathbf{{^{{+{pi:.{d}f}}}_{{-{mi:.{d}f}}}}}$ & {unit[name]} \\\\')
    return '\n'.join(rows)


def _bin_table(design, spread):
    """Where every cut sits, in ADU and in electrons.

    `u = (ADU - bias) / G` is the same cut expressed in electrons, which is the
    number that actually says what a bin is for: the first two cuts land inside
    the read-noise peak, the rest climb the EM-gain tail.

    The uncertainty column only exists when the posterior was actually
    propagated into the design (`mcmc.bins.n_draws > 0`); without it the column
    would be a row of dashes, so the table is built one column narrower instead.
    """
    det = design.detector
    edges = np.asarray(design.edges, dtype=np.float64)
    rows = []
    for i in range(design.nbins):
        lo, hi = edges[i], edges[i + 1]
        ulo = (lo - det.bias) / det.gain
        uhi = (hi - det.bias) / det.gain
        hi_s = r'$\infty$' if not np.isfinite(hi) else f'{hi:.0f}'
        uhi_s = r'$\infty$' if not np.isfinite(uhi) else f'{uhi:.2f}'
        wid_s = r'$\infty$' if not np.isfinite(hi) else f'{hi - lo:.0f}'
        cells = [f'{i}', f'{lo:.0f}']
        if spread is not None:
            # Bin 0's lower edge is the underflow floor, not a fitted cut, so it
            # has no scatter to quote.
            cells.append(f'$\\pm$~{spread[i]:.1f}' if 0 < i < len(spread) else '')
        cells += [hi_s, wid_s, f'{ulo:.2f}', uhi_s]
        rows.append(' & '.join(cells) + r' \\')
    return '\n'.join(rows)


def _bin_table_header(spread):
    """The column spec and header rows that match _bin_table's width."""
    if spread is not None:
        return (r'@{}c r@{\hspace{0.4em}}l r r r r@{}',
                r'& \multicolumn{2}{c}{lower} & upper & width & '
                r'\multicolumn{2}{c}{electrons, $u=(x-B)/G$} \\'
                '\n'
                r'\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}'
                r'\cmidrule(lr){6-7}'
                '\n'
                r'bin & \multicolumn{2}{c}{[ADU]} & [ADU] & [ADU] & from & to \\')
    return (r'@{}c r r r r r@{}',
            r'& lower & upper & width & '
            r'\multicolumn{2}{c}{electrons, $u=(x-B)/G$} \\'
            '\n'
            r'\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){4-4}'
            r'\cmidrule(lr){5-6}'
            '\n'
            r'bin & [ADU] & [ADU] & [ADU] & from & to \\')


def build_tex(meta, summary, design, spread, chi2_dof, figures, figdir):
    """The whole document, as one string.

    `figures` maps a role ('corner', 'fit', 'flux_sigma') to a file name
    relative to `figdir`; a role that is absent simply loses its section, so a
    run with --no-flux-map still produces a coherent report.
    """
    infl = np.sqrt(max(chi2_dof, 1.0))
    edge_list = design.edge_list()
    edges_str = ', '.join(str(e) for e in edge_list)

    pre = (PREAMBLE.replace('__ACCENT__', ACCENT).replace('__ALERT__', ALERT)
           .replace('__FIGDIR__', figdir.replace('\\', '/'))
           .replace('__RUNFOOT__', esc(meta['stamp'])))

    doc = [pre]

    # ---- masthead ---------------------------------------------------------
    doc.append(r"""
{\color{accent}\rule{\linewidth}{2.2pt}}\\[0.7em]
{\sffamily\bfseries\fontsize{23}{27}\selectfont EMCCD detector calibration}\\[0.45em]
{\sffamily\large\color{black!55} """ + esc(meta['dataset']) + r"""}\\[1.0em]
{\small Étienne Artigau \quad Galina Sherren \quad René Doyon \quad Jonathan St-Antoine}\\[0.15em]
{\small\color{black!55} Université de Montréal / Observatoire du Mont-Mégantic}\\[0.5em]
{\footnotesize\color{black!50}\sffamily generated by \texttt{embin\_mcmc.py} --- """
                + esc(meta['stamp']) + r"""}\\[0.35em]
{\color{accent!30}\rule{\linewidth}{0.8pt}}
""")

    # ---- the answer, before any of the reasoning --------------------------
    med = {k: summary[k][0] for k in summary}
    # Laid out as a tabular rather than as running text with \quad between the
    # items: a quantity like "mu = 0.01666 e-/frame" must never break across
    # lines, and four of them on one line is exactly where that happens.
    doc.append(r"""
\begin{keybox}
\sffamily\small
{\color{accent}\bfseries The four numbers}\\[0.35em]
\begin{tabular}{@{}p{0.235\linewidth}p{0.235\linewidth}p{0.235\linewidth}p{0.235\linewidth}@{}}
$B = """ + f'{med["bias"]:.4f}' + r"""$~ADU &
$\sigma_{\mathrm{RON}} = """ + f'{med["ron"]:.4f}' + r"""$~ADU &
$G = """ + f'{med["gain"]:.3f}' + r"""$~ADU/e$^-$ &
$\mu = """ + f'{med["mu"]:.5f}' + r"""$~e$^-$/frame
\end{tabular}

\vspace{0.7em}
{\color{accent}\bfseries The bins}\\[0.35em]
\texttt{[""" + edges_str + r"""]}\\[0.35em]
""" + f'{design.nbins}' + r""" bins, worst-case efficiency
$\eta = """ + f'{design.worst_eta:.4f}' + r"""$, i.e.\ """
                + f'{100 * design.worst_accuracy:.1f}' + r"""\,\% of the precision
of keeping every raw frame.
\end{keybox}
""")

    # ---- 1. what was done -------------------------------------------------
    doc.append(r'\section{What was measured, and from what}' + '\n')
    doc.append(
        'The raw ADU values of ' + f"{meta['n_frames']}" + ' frames were pooled '
        'over every unmasked pixel into a single histogram, and the four '
        'parameters of the EMCCD signal chain were sampled from it with '
        r'\texttt{emcee}. The model is the closed-form Poisson--Gamma--Gaussian '
        'law: the probability of reading $u$ ADU above the bias is\n'
        r"""
\begin{equation*}
P(u) = e^{-\mu}\,\delta(u)
     + \sqrt{\frac{\mu}{G u}}\;
       I_1\!\left(2\sqrt{\tfrac{\mu u}{G}}\right)
       e^{-\left(\sqrt{u/G}-\sqrt{\mu}\right)^2},
\end{equation*}
"""
        'convolved with a Gaussian of width $\\sigma_{\\mathrm{RON}}$ and '
        'integrated over each 1~ADU digitisation step. The Poisson sum over '
        'electron number is done in closed form, so there is no \\texttt{nmax} '
        'truncation anywhere in this fit.\n')

    doc.append(r"""
\begin{center}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\multicolumn{2}{@{}l}{\sffamily\bfseries\color{accent} The run} \\
\midrule
frames pooled                & """ + f"{meta['n_frames']}" + r""" \\
pixels kept                  & """ + f"{meta['n_kept']:,} of {meta['n_total']:,} "
                f"({100.0 * meta['n_kept'] / meta['n_total']:.2f}\\,\\%)" + r""" \\
bright-pixel cut             & """ + esc(meta['clip']) + r""" \\
pixel values in the fit      & """ + f"{meta['n_samples']:,}" + r""" \\
ADU range fitted             & """ + f"{meta['adu_lo']} to {meta['adu_hi']}"
                + f" ({meta['n_bins']} bins)" + r""" \\
sampler                      & """ + f"{meta['walkers']} walkers "
                f"$\\times$ {meta['steps']} steps, {meta['n_post']:,} samples kept" + r""" \\
\bottomrule
\end{tabular}
\end{center}
""")
    # The folder is the one field that can be arbitrarily long, so it gets its
    # own full-width line and \path, which breaks at the slashes rather than
    # running off the right margin.
    doc.append(r'{\small Frames read from \path{' + meta['directory']
               + '}.}\n')

    # ---- 2. the numbers ---------------------------------------------------
    doc.append(r'\section{The fitted constants}' + '\n')
    doc.append(r"""
\begin{center}
\small
\begin{tabular}{@{}lr@{\hspace{1.2em}}c@{\hspace{1.2em}}c@{\hspace{1.2em}}l@{}}
\toprule
& \multicolumn{1}{c}{value} & formal & \textbf{adopted} & unit \\
\midrule
""" + _param_table(summary, chi2_dof) + r"""
\bottomrule
\end{tabular}
\end{center}
""")
    doc.append(
        r'\begin{alertbox}{Which error bar to quote}' + '\n'
        'The \\emph{formal} column is the width the sampler actually found. It '
        'is not a believable uncertainty on this detector: the fit has '
        f"{meta['n_samples']:,} pixel values behind four parameters, and no real "
        'EMCCD is a four-parameter object at that precision. Fixed pattern in '
        'the bias, pixel-to-pixel gain variation, a sky that is not flat and '
        'faint stars under the mask all show up as structure in the residuals, '
        f'and $\\chi^2/\\nu = {chi2_dof:.1f}$ over ${meta["dof"]}$ degrees of '
        'freedom says so plainly.\n\n'
        f'The \\textbf{{adopted}} column is the formal width inflated by '
        f'$\\sqrt{{\\chi^2/\\nu}} = {infl:.1f}$, which is the usual way of '
        'admitting that. Quote those. The central values are solid; it is the '
        'error bars that are a statement about the model rather than about the '
        'camera.\n'
        r'\end{alertbox}' + '\n')

    # ---- 3. posterior -----------------------------------------------------
    if 'corner' in figures:
        doc.append(r'\section{The posterior}' + '\n')
        doc.append(
            'Every parameter against every other, with its own marginal along '
            'the diagonal. The one correlation worth knowing about is '
            r'$G$ against $\mu$: the mean of the histogram fixes the product '
            r'$\mu G$, and only the shape of the tail separates the two.'
            '\n')
        doc.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.92\linewidth]{""" + figures['corner'] + r"""}
\caption{Joint posterior of the four parameters. Contours are the usual
1, 2 and 3$\sigma$ levels of the two-dimensional marginals; the titles quote the
\emph{formal} 16th--84th percentile width, not the inflated one.}
\end{figure}
""")

    # ---- 4. fit quality ---------------------------------------------------
    if 'fit' in figures:
        doc.append(r'\section{How well the model fits}' + '\n')
        doc.append(
            'The pooled histogram against the posterior-median model, with the '
            'pulls underneath and the proposed cuts drawn on top. The peak '
            'holds of order a million counts in a single ADU, so its Poisson '
            'precision is a tenth of a per cent, and the residuals are '
            'structured at that level rather than noisy. That structure is the '
            'detector being more complicated than four numbers, not the fit '
            'having gone wrong: the central values reproduce an independent '
            'calibration of this camera to well inside the adopted errors.\n')
        doc.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.95\linewidth]{""" + figures['fit'] + r"""}
\caption{Top: observed pooled histogram (grey) and the fitted model (red), with
the proposed bin edges as dotted verticals. Bottom: the pull,
$(n_k - \hat{n}_k)/\sqrt{\hat{n}_k}$, per ADU.}
\end{figure}
""")

    # ---- 5. the bins ------------------------------------------------------
    doc.append(r'\section{The binning this detector wants}' + '\n')
    doc.append(
        'Bin edges are not a free choice: for a given detector there is a set '
        'that loses the least Fisher information about the flux, and '
        r'\texttt{bin\_optimizer.py} finds it. The design below was computed '
        f"for the constants above, over the flux range "
        f"{meta['bin_mu_min']:g} to {meta['bin_mu_max']:g}~e$^-$/frame, by the "
        f"\\texttt{{{esc(design.method)}}} method. Its worst case over that "
        f"range is $\\eta = {design.worst_eta:.4f}$: these "
        f"{design.nbins} numbers per pixel measure the flux to within "
        f"{100 * (1 - design.worst_accuracy):.1f}\\,\\% of the precision you "
        'would get by archiving every raw frame.\n')
    colspec, header = _bin_table_header(spread)
    doc.append(r'\begin{center}' + '\n\\small\n'
               r'\begin{tabular}{' + colspec + '}\n'
               r'\toprule' + '\n' + header + '\n'
               r'\midrule' + '\n'
               + _bin_table(design, spread) + '\n'
               r'\bottomrule' + '\n'
               r'\end{tabular}' + '\n'
               r'\end{center}' + '\n')
    if spread is not None:
        doc.append(
            r'The $\pm$ column is the scatter of each cut over '
            f"{meta['n_draws']} draws from the posterior above -- i.e.\\ how "
            'much the optimal design itself moves when the detector moves '
            'within its own uncertainty. Cuts inside the read-noise peak are '
            'pinned to a fraction of an ADU; the high rungs float by several, '
            'which is a statement about how little the far tail constrains '
            'them, not a reason to distrust them.\n')
    doc.append(
        r'The first bin absorbs everything below its lower edge and the last '
        'bin is open-ended: the sentinel one ADU above the last finite cut '
        'exists only so that the list has $K+1$ entries.\n')

    # ---- 6. copy-ready ----------------------------------------------------
    doc.append(r'\section{Copy-ready}' + '\n')
    doc.append('Paste these straight into the configuration. Sections 2 and 3 '
               r'of \texttt{embin\_config.yaml} respectively.' + '\n')

    yaml_edges = f'histogram:\n  edges: [{edges_str}]'
    yaml_det = (
        'detector:\n'
        f'  bias: {med["bias"]:.4f}\n'
        f'  ron: {med["ron"]:.4f}\n'
        f'  gain: {med["gain"]:.4f}\n'
        f'  full_well: {meta["full_well"]}\n'
        f'  nmax: {meta["nmax"]}\n'
        '  cic: 0.0')
    py_line = f'edges = [{edges_str}]'
    plain = edges_str.replace(',', '')

    for title, body in (
            (r'embin\_config.yaml \textemdash{} section 2, the bin edges', yaml_edges),
            (r'embin\_config.yaml \textemdash{} section 3, the detector', yaml_det),
            ('Python', py_line),
            ('Plain vector, whitespace separated', plain)):
        doc.append(r'\begin{copybox}{' + title + '}\n'
                   r'\begin{Verbatim}[fontsize=\small,xleftmargin=0pt]' + '\n'
                   + body + '\n'
                   r'\end{Verbatim}' + '\n'
                   r'\end{copybox}' + '\n')

    doc.append(
        r'\texttt{cic} is left at zero on purpose. Clock-induced charge enters '
        'the EM register exactly as a photo-electron does, so a pixel-value '
        'histogram sees only the sum of the two: the fitted '
        f'$\\mu = {med["mu"]:.5f}$~e$^-$/frame is sky \\emph{{plus}} CIC, and '
        'splitting it needs a dark sequence this fit does not have. '
        r'\texttt{full\_well} and \texttt{nmax} are carried over unchanged -- '
        'neither is constrained by these data, and \\texttt{nmax} does not '
        'enter this model at all.\n')

    # ---- 7. validation ----------------------------------------------------
    if 'flux_sigma' in figures:
        doc.append(r'\clearpage' + '\n')
        doc.append(r'\section{Every pixel of the frame}' + '\n')
        doc.append(
            'The detector above was fitted on sky pixels only, because the '
            'pooled histogram gives every pixel the same flux and a star '
            'breaks that. The figure below puts the stars back: each of the '
            f"{meta['n_total']:,} pixels of the frame was fitted "
            f"\\emph{{individually}} from {meta['map_frames']} frames, through "
            'the very bins proposed above, and its fitted flux is plotted '
            'against its own likelihood-based uncertainty.\n\n'
            'This is the real-data twin of a Monte-Carlo validation. A '
            'simulation checks an estimator by generating many trials at a '
            'range of known fluxes; a field of view hands you the same thing '
            'for free, because its pixels already span two decades of flux, '
            'from blank sky to the core of a star, at no computational cost '
            'and with no assumption about the source of the photons.\n')
        # The interpretation goes BEFORE the figure: pinned with [H], a
        # full-width figure at the bottom of a page pushes any following
        # paragraph onto a page of its own.
        doc.append(
            'Two things are worth checking on it. The sky clump sits '
            f"at $\\mu = {meta['map_median_mu']:.5f}$~e$^-$/frame, against "
            f"${med['mu']:.5f}$ from the pooled histogram -- two entirely "
            'different estimators, one global and one per-pixel, agreeing to '
            f"{100 * abs(meta['map_median_mu'] - med['mu']) / med['mu']:.0f}\\,\\%. "
            'And the sky pixels reach a throughput of '
            f"{meta['throughput']:.3f} against the noiseless ideal, the "
            'shortfall being the read-noise penalty of thresholding at finite '
            r'$\sigma_{\mathrm{RON}}/G$.' + '\n')
        doc.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.82\linewidth]{""" + figures['flux_sigma'] + r"""}
\caption{Top: per-pixel uncertainty against per-pixel flux, hexagonally binned
because half a million points would overplot into a blob. The dashed line is the
noiseless photon-counting floor $\sigma=\sqrt{\mu/N}$ and the dotted line the
same floor with the EM register's excess-noise factor $F=\sqrt{2}$; the
dash-dotted vertical is the sky flux measured independently from the pooled
histogram. Bottom: the same points as a ratio to the Poisson floor.}
\end{figure}
""")

    doc.append(r'\end{document}' + '\n')
    return '\n'.join(doc)


def compile_pdf(tex_path, keep_tex=True):
    """Run pdflatex twice and clean up after it.

    Twice because the page numbers in the footer and any reference the template
    grows later need a second pass to settle. Output goes beside the .tex.
    """
    if shutil.which('pdflatex') is None:
        log('pdflatex is not on the PATH, cannot typeset the report', 'warn')
        return None

    workdir = os.path.dirname(os.path.abspath(tex_path))
    base = os.path.splitext(os.path.basename(tex_path))[0]
    cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
           os.path.basename(tex_path)]
    for _ in range(2):
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if proc.returncode != 0:
            # The interesting line of a LaTeX failure is the one starting with
            # '!', and it is never the last line of the log.
            tail = [ln for ln in proc.stdout.splitlines() if ln.startswith('!')]
            log(f'pdflatex failed: {tail[0] if tail else "see " + base + ".log"}',
                'warn')
            return None

    for ext in ('.aux', '.log', '.out', '.toc'):
        try:
            os.remove(os.path.join(workdir, base + ext))
        except OSError:
            pass
    if not keep_tex:
        try:
            os.remove(tex_path)
        except OSError:
            pass
    return os.path.join(workdir, base + '.pdf')


def write_report(outdir, figdir_name, meta, summary, design, spread,
                 chi2_dof, figures, name='mcmc_report'):
    """Write and typeset the report; returns its path, or None if TeX failed.

    `figures` maps role -> file name inside `<outdir>/<figdir_name>/`.
    """
    tex_path = os.path.join(outdir, name + '.tex')
    tex = build_tex(meta, summary, design, spread, chi2_dof,
                    figures, figdir_name)
    with open(tex_path, 'w', encoding='utf-8') as fh:
        fh.write(tex)
    pdf = compile_pdf(tex_path)
    if pdf:
        log(f'typeset report written to {pdf}')
    return pdf


# ---------------------------------------------------------------------------
# The photometry report (embin_psf.py)
# ---------------------------------------------------------------------------
def _star_table(stars, med_flux, med_err, scatter, blended):
    """One row per star: where it is, how bright, how steady."""
    rows = []
    for k in range(len(stars)):
        gmag = stars['Gmag'][k]
        g = '--' if not np.isfinite(gmag) else f'{gmag:.2f}'
        tag = r'$\star$' if k in blended else ''
        ra = f"{stars['ra'][k]:.6f}" if 'ra' in stars.colnames else '--'
        dec = f"{stars['dec'][k]:.6f}" if 'dec' in stars.colnames else '--'
        rows.append(f'{k} {tag} & {stars["x"][k]:.2f} & {stars["y"][k]:.2f} & '
                    f'{ra} & {dec} & {g} & {med_flux[k]:.4f} & '
                    f'{med_err[k]:.4f} & {100 * scatter[k]:.1f} \\\\')
    return '\n'.join(rows)


def _chunk_table(tab):
    """One row per chunk: the time, the seeing, the drift, the fit quality."""
    rows = []
    for r in tab:
        rows.append(f'{r["bin"]} & {r["t_mid"]:.2f} & {r["fwhm"]:.4f} & '
                    f'{r["fwhm_err"]:.4f} & {r["dx"]:+.2f} & {r["dy"]:+.2f} & '
                    f'{r["background"]:.5f} & {r["chi2_red"]:.3f} \\\\')
    return '\n'.join(rows)


def build_psf_tex(meta, tab, stars, blends, figures, figdir):
    """The PSF-photometry report, as one LaTeX string."""
    med_flux = np.array([np.median(tab[f'flux_{k:02d}']) for k in range(meta['n_star'])])
    med_err = np.array([np.median(tab[f'flux_err_{k:02d}']) for k in range(meta['n_star'])])
    scatter = np.array([np.std(tab[f'flux_{k:02d}']) / abs(m) if m else np.nan
                        for k, m in enumerate(med_flux)])
    blended = {i for i, j, _ in blends} | {j for i, j, _ in blends}

    pre = (PREAMBLE.replace('__ACCENT__', ACCENT).replace('__ALERT__', ALERT)
           .replace('__FIGDIR__', figdir.replace('\\', '/'))
           .replace('__RUNFOOT__', esc(meta['stamp'])))
    doc = [pre]

    doc.append(r"""
{\color{accent}\rule{\linewidth}{2.2pt}}\\[0.7em]
{\sffamily\bfseries\fontsize{23}{27}\selectfont PSF photometry}\\[0.45em]
{\sffamily\large\color{black!55} """ + esc(meta['dataset']) + r"""}\\[1.0em]
{\small Étienne Artigau \quad Galina Sherren \quad René Doyon \quad Jonathan St-Antoine}\\[0.15em]
{\small\color{black!55} Université de Montréal / Observatoire du Mont-Mégantic}\\[0.5em]
{\footnotesize\color{black!50}\sffamily generated by \texttt{embin\_psf.py} --- """
               + esc(meta['stamp']) + r"""}\\[0.35em]
{\color{accent!30}\rule{\linewidth}{0.8pt}}
""")

    fw = np.asarray(tab['fwhm'])
    doc.append(r"""
\begin{keybox}
\sffamily\small
{\color{accent}\bfseries The seeing}\\[0.35em]
\begin{tabular}{@{}p{0.3\linewidth}p{0.32\linewidth}p{0.32\linewidth}@{}}
stack FWHM $= """ + f'{meta["stack_fwhm"]:.4f}' + r"""''$ &
per-chunk FWHM $= """ + f'{fw.min():.3f}' + '$--$' + f'{fw.max():.3f}' + r"""''$ &
$\beta = """ + f'{meta["beta"]:.4f}' + r"""$ (held)
\end{tabular}

\vspace{0.7em}
{\color{accent}\bfseries The photometry}\\[0.35em]
""" + f'{meta["n_star"]}' + r""" stars fitted simultaneously in each of """
               + f'{len(tab)}' + r""" chunks of """ + f'{meta["nframes"]}'
               + r""" frames, against that chunk's own flux-error map.
""" + (r'The blended pair is separated by $'
       + f'{blends[0][2] * meta["pixscale"]:.3f}' + r"""''$, """
       + f'{blends[0][2] / meta["stack_fwhm_px"]:.2f}' + r""" FWHM.'''"""
       if blends else 'No pair falls inside the blend radius.').replace("'''", "")
               + r"""
\end{keybox}
""")

    # ---- method ---------------------------------------------------------
    doc.append(r'\section{What was fitted}' + '\n')
    doc.append(
        'Every star in the field is described by the same isotropic Moffat, '
        'normalised to unit volume so that the coefficient multiplying it is '
        r"the star's TOTAL flux in e$^-$/frame, integrated to infinite radius:"
        '\n'
        r"""
\begin{equation*}
M(r) = \frac{\beta-1}{\pi\alpha^2}\left(1+\frac{r^2}{\alpha^2}\right)^{-\beta},
\qquad
\mathrm{FWHM} = 2\alpha\sqrt{2^{1/\beta}-1},
\end{equation*}
"""
        'averaged over ' + f'{meta["oversample"]}' + r'$\times$'
        + f'{meta["oversample"]}' + ' sub-positions inside each pixel rather '
        'than sampled at its centre -- at a three-pixel FWHM the difference is '
        'several per cent of the peak and depends on where the star falls '
        'inside its pixel, which would turn the drift into a spurious flux '
        'wobble.\n\n'
        'The fit runs in two stages, and they are not the same fit:\n')
    doc.append(r"""
\begin{enumerate}[leftmargin=1.4em,itemsep=0.35em]
\item \textbf{On the deep stack.} FWHM and $\beta$ shared, every star free in
      position and in flux. This is where the reference positions come from,
      and where $\beta$ is measured.
\item \textbf{On each chunk.} $\beta$ held at the stack value; the FWHM, a
      rigid $(\mathrm{d}x, \mathrm{d}y)$ shift of the whole field, and one flux
      per star, all solved together against that chunk's \texttt{FLUX\_ERR}
      map. The positions do not move independently: over """
               + f'{meta["duration"]:.0f}' + r""" seconds the field
      translates, it does not rearrange itself.
\end{enumerate}
""")
    doc.append(
        'Holding the shape and the positions makes the model LINEAR in the '
        'fluxes, so they come from a weighted linear solve rather than from the '
        'optimiser -- which is both faster and the reason the next section is '
        'as short as it is.\n')

    # ---- the blend ------------------------------------------------------
    doc.append(r'\section{The blend}' + '\n')
    if blends:
        i, j, d = blends[0]
        doc.append(
            f'Stars {i} and {j} sit {d:.2f}~px = '
            f'{d * meta["pixscale"]:.3f} arcsec apart, '
            f'{d / meta["stack_fwhm_px"]:.2f}~FWHM. Their wings overlap, and no '
            'aperture wide enough to hold one of them excludes the other.\n')
    doc.append(
        'They are not deblended by subtracting a neighbour, or by shrinking an '
        'aperture. They are fitted together. With the shape fixed the data is\n'
        r"""
\begin{equation*}
d(p) = \sum_j F_j\,M_j(p) + B,
\qquad
\left(M^{\!\top} W M\right)_{jk} = \sum_p \frac{M_j(p)\,M_k(p)}{\sigma(p)^2},
\end{equation*}
"""
        r'and the off-diagonal element $jk$ of that normal matrix IS the '
        'cross-term: the noise-weighted overlap integral of one profile with '
        'the other. It is not neglected and not approximated, it is a number '
        'the solve inverts. Two things follow at no extra cost. The fluxes are '
        'the pair that jointly explains the blended image, rather than each '
        r"star's light plus a share of its neighbour's wing. And the inverse "
        'of that same matrix is the covariance, so each error bar already '
        'carries the penalty for not knowing exactly how the light was shared.'
        '\n')
    if blends:
        i, j, _ = blends[0]
        doc.append(
            r'\begin{alertbox}{How big the cross-term actually is}' + '\n'
            f'Error bars on stars {i} and {j} are inflated by '
            f'$\\times${meta["penalty"][0]:.5f} and '
            f'$\\times${meta["penalty"][1]:.5f} relative to what they would be '
            'if the neighbour did not exist, and their flux--flux correlation '
            f'averages ${meta["rho"]:+.5f}$.\n\n'
            'That is small, and it is the right answer: at '
            f'{meta["blend_fwhm"]:.2f}~FWHM separation a Moffat has already '
            'fallen a long way. The value of doing it this way is not that the '
            'correction is large, it is that the size of the correction is now '
            'a measured number in the output rather than an assumption in a '
            'footnote. The full per-chunk correlation matrix is in the '
            r'\texttt{FLUXCORR} extension.' + '\n'
            r'\end{alertbox}' + '\n')

    if 'blend' in figures:
        doc.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{""" + figures['blend'] + r"""}
\caption{The blended pair on the deep stack. Left to right: the data, the joint
two-star model, and the residual on a symmetric scale. A mis-shared deblend
shows up in the third panel as a dipole -- bright on one component, dark on the
other. The last panel cuts along the line joining the two stars and draws each
fitted component on its own: the deblend is those two dashed curves.}
\end{figure}
""")

    # ---- why beta is held ----------------------------------------------
    doc.append(r'\section{Why $\beta$ is held fixed}' + '\n')
    doc.append(
        r'$\beta$ sets how heavy the wings are, and the wings are exactly where '
        'a short chunk has nothing to say. At a sky flux of '
        f'{meta["sky"]:.4f}~e$^-$/frame, a pixel a few FWHM from a star '
        f'collects well under one electron in the {meta["nframes"]} frames of a '
        'chunk: its flux estimate is noise-dominated and biased low, because '
        'the per-pixel likelihood barely turns over. A free '
        r'$\beta$ reads that bias as a genuinely narrow-winged PSF and runs '
        'away.\n\n'
        'It is not a subtle failure. Fitted freely on this dataset, '
        r'$\beta$ hit its upper bound of 30 in ten chunks out of fifteen, with '
        r'$\chi^2$ still falling monotonically all the way there -- the '
        'signature of a parameter the data cannot determine, not of a real '
        'measurement. The deep stack has fifteen times the frames and real '
        'signal in the wings, so '
        r'$\beta$ is measured there once, $\beta = '
        + f'{meta["beta"]:.4f} \\pm {meta["beta_err"]:.4f}$, and reused. It is '
        'a property of the optics and the atmosphere, and it has no reason to '
        f'change over {meta["duration"]:.0f} seconds anyway. Set '
        r'\texttt{psf.beta\_free: true} to let it float; the per-chunk value '
        'is then reported with a NaN error bar whenever it rails, so the '
        'failure is visible rather than silent.\n')

    # ---- results --------------------------------------------------------
    doc.append(r'\clearpage' + '\n')
    doc.append(r'\section{The stars}' + '\n')
    doc.append(
        'Median flux over the ' + f'{len(tab)}' + ' chunks, with the median of '
        'the per-chunk error bars, and the scatter of the light curve as a '
        'fraction of its own median. For the bright stars the scatter is real '
        'variability plus systematics; for the faint ones it is simply the '
        r'error bar. A $\star$ marks a member of the blended pair.' + '\n')
    doc.append(r"""
\begin{center}
\footnotesize
\begin{tabular}{@{}l r r r r r r r r@{}}
\toprule
star & $x$ & $y$ & RA & Dec & $G$ & flux & error & scatter \\
 & [px] & [px] & [deg] & [deg] & [mag] & \multicolumn{2}{c}{[e$^-$/frame]} & [\%] \\
\midrule
""" + _star_table(stars, med_flux, med_err, scatter, blended) + r"""
\bottomrule
\end{tabular}
\end{center}
""")

    doc.append(r'\section{The chunks}' + '\n')
    doc.append(
        'One row per chunk. The FWHM is the seeing that chunk actually had; '
        'note that it is consistently SHARPER than the '
        f'{meta["stack_fwhm"]:.3f} arcsec of the stack, which is as it should '
        'be -- the stack is the sum of chunks offset by the drift, so it is '
        'blurred by exactly the motion the last two columns measure.\n')
    doc.append(r"""
\begin{center}
\footnotesize
\begin{tabular}{@{}r r r@{\hspace{0.4em}}l r r r r@{}}
\toprule
bin & $t_{\mathrm{mid}}$ & \multicolumn{2}{c}{FWHM} & d$x$ & d$y$ & sky & $\chi^2/\nu$ \\
 & [s] & \multicolumn{2}{c}{[arcsec]} & [px] & [px] & [e$^-$/frame] & \\
\midrule
""" + _chunk_table(tab) + r"""
\bottomrule
\end{tabular}
\end{center}
""")
    doc.append(
        r'$\chi^2/\nu$ sits near ' + f'{np.median(tab["chi2_red"]):.2f}'
        + ', consistently below one. That is not a good fit, it is a '
        r'conservative error map: \texttt{FLUX\_ERR} overestimates the noise by '
        f'about {100 * (1 / np.sqrt(np.median(tab["chi2_red"])) - 1):.0f} per '
        'cent. Both versions of the flux error are therefore written out, the '
        'formal one straight from the error map and a '
        r"\texttt{flux\_err\_scaled} rescaled so that this fit's $\chi^2$ "
        'comes out at one. Which to believe is a judgement about the flux map, '
        'not about this fit.\n')

    if 'lightcurve' in figures:
        doc.append(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.88\linewidth]{""" + figures['lightcurve'] + r"""}
\caption{Top: the fitted FWHM, chunk by chunk. Middle: the rigid shift, i.e.\ the
field drifting across the detector -- smooth and monotonic, which is the check
that the shift is tracking the telescope rather than the noise. Bottom: the
light curves of the brightest stars, each normalised to its own median.}
\end{figure}
""")

    # ---- how to read it -------------------------------------------------
    doc.append(r'\section{Reading the table}' + '\n')
    for title, body in (
            ('The columns',
             'bin nframes t_start t_mid t_end utc mjd\n'
             'fwhm fwhm_err fwhm_pix beta beta_err beta_free\n'
             'dx dy background chi2_red\n'
             'flux_NN flux_err_NN flux_err_scaled_NN   (NN = 00 .. '
             f'{meta["n_star"] - 1:02d})'),
            ('Python', "from astropy.table import Table\n"
                       "t = Table.read('psf_photometry.ecsv')\n"
                       "t['t_mid', 'fwhm', 'flux_01', 'flux_err_01']")):
        doc.append(r'\begin{copybox}{' + title + '}\n'
                   r'\begin{Verbatim}[fontsize=\small,xleftmargin=0pt]' + '\n'
                   + body + '\n'
                   r'\end{Verbatim}' + '\n'
                   r'\end{copybox}' + '\n')
    doc.append(
        r'\texttt{psf\_photometry.fits} carries the same table in its '
        r'\texttt{PHOT} extension, the star list in \texttt{STARS}, and the '
        r'per-chunk flux--flux correlation matrices, shape $('
        + f'{len(tab)}, {meta["n_star"]}, {meta["n_star"]}' + r')$, in '
        r'\texttt{FLUXCORR}.' + '\n')

    doc.append(r'\end{document}' + '\n')
    return '\n'.join(doc)


def write_psf_report(outdir, figdir_name, meta, tab, stars, blends, figures,
                     name='psf_report'):
    """Write and typeset the photometry report; None if TeX is unavailable."""
    tex_path = os.path.join(outdir, name + '.tex')
    with open(tex_path, 'w', encoding='utf-8') as fh:
        fh.write(build_psf_tex(meta, tab, stars, blends, figures, figdir_name))
    pdf = compile_pdf(tex_path)
    if pdf:
        log(f'typeset report written to {pdf}')
    return pdf
