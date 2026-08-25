# emccd_bintool

[Étienne Artigau](https://orcid.org/0000-0003-3506-5667), [Galina Sherren](https://orcid.org/0009-0006-5677-3944), [René Doyon](https://orcid.org/0000-0001-5485-4675), Jonathan St-Antoine
(Université de Montréal / Observatoire du Mont-Mégantic)

**Three things live in this repository.**

1. **`bin_optimizer.py`** answers the question *"how few numbers per pixel can I
   keep and still measure the flux properly?"* For an EMCCD it computes where
   the histogram bin edges belong, and how much precision a given number of bins
   costs you. The full reasoning, with every derivation and the Monte-Carlo
   validation, is the paper: *Optimal Histogram Binning for EMCCD
   Photon-Counting Flux Retrieval*, whose source and figure
   scripts live in the `papier_histofit` directory of
   [emccd_histo_fit](https://github.com/eartigau/emccd_histo_fit).

2. **`embin.py` and `run_chunks.py`** apply that to a real night of PESTO data.
   They read a folder of raw frames and produce, for every pixel, a histogram of
   the values that pixel took, then a fitted mean flux in electrons per frame,
   then the stars found in that flux map, then light curves.

3. **`pesto_astrometry.py`** puts those flux maps on the sky: it solves the
   field with astrometry.net, writes a plain TAN WCS (CRPIX at the centre of
   the field, no SIP), and gives every tracked star an RA and a Dec, so you can
   say which light curve is your target and whether anything is blended with
   it. This step is optional and needs two extra packages.

Everything is driven by one settings file, **`embin_config.yaml`**. You do not
need to edit any Python file to use this toolkit.

**Project page:** <https://eartigau.github.io/emccd_bintool/> gives the same
material as this README, plus the figures, in three pages: how to run it, how to
read the results, and why the bins are where they are. It asks for a password
(`omm4ever`) so it stays a working document for the group rather than a
published one. That gate is a door, not a safe: it hides the pages from casual
visitors and from search engines, and nothing on them is confidential anyway.

---

## The short version

```
your frames  ->  per-pixel histogram (16 numbers per pixel)  ->  flux map  ->  stars  ->  light curves
```

An EMCCD read is not "the flux plus a bit of noise". A pixel that collected *n*
photo-electrons in one frame reads out at

```
bias  +  Gamma(n, gain)  +  Gaussian(0, read noise)
```

The EM register multiplies each electron by a *random* amount, so the same *n*
gives wildly different ADU values from one frame to the next. Averaging the ADU
values throws away most of the information at the sub-electron fluxes these
cameras are used at. The **shape** of a pixel's ADU distribution is what carries
the flux, and 16 well-placed histogram bins capture 99.4 % of it, in 16 numbers
per pixel instead of one number per pixel per frame.

Classical photon counting, one threshold per read, is the 2-bin case of the same
picture. It is nearly optimal below one electron per frame (98.5 % at the PESTO
sky level) and then hits a hard ceiling: above a couple of electrons per frame no
threshold, however well placed, retains more than 80 % of the precision, because
one bit cannot tell one electron from two. The comparison, with numbers, is on
the project page and in the PDF.

---

## Installing it (do this once)

You need Python 3.9 or newer. Open a terminal, then:

```bash
# 1. Get the code.
git clone https://github.com/eartigau/emccd_bintool.git
cd emccd_bintool

# 2. Make a virtual environment. This keeps the packages used here separate
#    from the rest of your machine, so nothing you install can break another
#    project. The folder .venv is created inside emccd_bintool.
python3 -m venv .venv

# 3. Activate it. You must do this EVERY time you open a new terminal to work
#    on this project. Your prompt gains a "(.venv)" prefix when it worked.
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 4. Install the four packages this code needs.
pip install -r requirements.txt
```

Check that it worked:

```bash
python bin_optimizer.py --bias 303.6178 --gain 71.0598 --ron 6.8164 --sat 5000 \
                        --nbins 16 --mu-min 1e-3 --mu-max 4
```

You should get a table of 16 bins ending with a line saying the worst-case
efficiency is about 0.989. If instead you get `ModuleNotFoundError: No module
named 'numpy'`, step 3 or step 4 did not happen: activate the environment and
run the `pip install` again.

---

## Running it on a night of data

### Step 1 -- put your frames somewhere

Create a folder called `data_night` inside `emccd_bintool` and copy your raw
frames into it. One FITS file per readout, which is how PESTO writes them
(`nc-image_0.fits`, `nc-image_1.fits`, ...).

```bash
mkdir data_night
cp /wherever/your/frames/are/*.fits data_night/
```

If your frames are large and you would rather not copy them, open
`embin_config.yaml` and change the line

```yaml
  directory: "data_night"
```

to the real path, for example `directory: "/Users/you/data/TOI1452Science"`.

### Step 2 -- check the detector constants

Open `embin_config.yaml` and look at section 3. Those four numbers describe the
camera:

```yaml
detector:
  bias: 303.6178      # ADU
  ron: 6.8164         # ADU
  gain: 71.0598       # ADU/e-
  full_well: 5000     # ADU
```

They are correct **for the PESTO EMCCD at the gain setting used in the test
sequence**. If your data comes from a different camera, or from PESTO at a
different EM gain, these numbers are wrong for you and everything downstream
will be wrong with them. Get them from a calibration fit first, put them here,
and then **re-run the bin optimiser** (see "Designing your own bins" below),
because the bin edges depend on them.

### Step 3 -- try two chunks before you launch the whole night

```bash
python run_chunks.py --n-chunks 2
```

This takes about half a minute per chunk and tells you, in colour, exactly what
it is doing: green for progress, blue for measured numbers, orange for anything
it skipped, red if it stops. Look at the output and check three things:

- it found your frames ("`1001 frames in ...`");
- the sky level it reports is sensible (`0.0160 e-/frame` on the PESTO test
  sequence);
- it found some stars.

### Step 4 -- run the whole thing

```bash
python run_chunks.py
```

On the 1001-frame PESTO test sequence this takes about 3.5 minutes and produces
15 chunks. Everything is written into the folder **`data_bin/`**.

---

## What you get, file by file

Everything lands in `data_bin/` (change `output.directory` in the YAML if you
want it elsewhere).

| File | What it is |
|---|---|
| `embin_chunk00.fits.gz` ... | one cube file per chunk of frames, described below |
| `embin_chunk00_flux.fits.gz` ... | the mean flux and its error for that chunk, in two extensions |
| `chunk_summary.fits.gz` | all the chunks stitched together, plus the light curves |
| `figures/chunk_lightcurves.pdf` | flux of every tracked star versus time |
| `figures/chunk_drift.pdf` | how far the field moved during the sequence |
| `figures/chunk_flux_maps.pdf` | the flux map of each chunk, side by side |
| `astrometry_stack.fits` | only after `pesto_astrometry.py`: the stacked, solved image with its WCS |
| `figures/astrometry.pdf` | only after `pesto_astrometry.py`: the solved field against Gaia |

Each chunk comes as two files, and the split is deliberate: the histogram cube
is the measurement, the flux map is one reduction of it.

`embin_chunkNN.fits.gz` is the **cube file**, the one to archive. Everything is
written gzipped: astropy compresses and decompresses on the file name alone, so
nothing you do with these files changes, and `gunzip` gives an ordinary FITS.
It pays here, because a histogram cube is small counts and empty sky: 7.0 MB of
data becomes 0.75 MB on disk. Set `output.compress: false` for plain `.fits`.
Open one with

```bash
python -c "from astropy.io import fits; fits.open('data_bin/embin_chunk00.fits.gz').info()"
```

and you will see:

| Extension | Shape | Contents |
|---|---|---|
| `PRIMARY` | no data | a header recording every setting of the run: the bin edges, the detector constants, the flux grid, which frames went in |
| `HISTCUBE` | (16, 426, 1024) | the histograms: plane *b* holds, for each pixel, how many frames fell in bin *b*. Stored in the narrowest integer type a count out of *N* frames can need: `uint8` below 256 frames, `uint16` below 65536, `int32` above |
| `HEADERS` | table | one row per input frame, one column per FITS keyword, so the timestamps survive |
| `STAMP01`, `STAMP02`, ... | (64, 16, 16) | the raw, unbinned ADU values of every frame in a small box around each detected star |

Nothing fitted is in there. `embin_chunkNN_flux.fits` is the **flux file**:

| Extension | Shape | Contents |
|---|---|---|
| `PRIMARY` | no data | provenance: `CUBEFILE` names the cube above, and the bin edges, detector constants and flux grid are copied from its header |
| `FLUX` | (426, 1024) | the fitted mean flux of each pixel, in electrons per frame |
| `FLUX_ERR` | (426, 1024) | the 1-sigma error on that flux, same units |

Both maps are fits to the cube beside them and hold nothing it does not, so a
flux file is never the only copy of anything. Delete one and rebuild it from the
cube alone, with no frames and no configuration file:

```bash
python embin.py --from-cube data_bin/embin_chunk00.fits.gz
```

```python
from embin import flux_maps_from_cube
flux, flux_err, mu_lo, mu_hi = flux_maps_from_cube('data_bin/embin_chunk00.fits.gz')
```

That reader is also how anything else should get a mean out of these files: the
cube's own header carries the bias, the read noise, the gain and the bin edges,
so `flux_maps_from_cube` needs nothing but the path.

`chunk_summary.fits` holds the same flux maps stacked in time, `(15, 426, 1024)`,
plus three tables: `CHUNKS` (which frames and which times each plane covers),
`TRACKS` (one row per star per chunk: position, aperture flux, error) and
`VARSTAT` (one row per star: the chi-square of its light curve against a
constant flux, the p-value, and the equivalent Gaussian significance, so you can
tell a real variable from scintillation). Reading the light curves in Python:

```python
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

t = Table(fits.getdata('data_bin/chunk_summary.fits', 'TRACKS'))
star = t[t['track'] == 0]                      # track 0 is the brightest star
plt.errorbar(star['t_mid'], star['flux'], yerr=star['flux_err'], fmt='o')
plt.xlabel('time since the first frame [s]')
plt.ylabel('flux [e-/frame]')
plt.show()

# Is it really variable, or is that just noise?
v = Table(fits.getdata('data_bin/chunk_summary.fits', 'VARSTAT'))
print(v['track', 'chi2', 'dof', 'chi2_red', 'p_value', 'sigma'])
```

The points are deliberately not joined by a line, in the figures and here: the
chunks are independent measurements, and a line between them draws a trend the
data does not contain. `chi2_red` near 1 means the scatter is entirely explained
by the quoted photon errors, and there is nothing to report. On the PESTO test
sequence the highest is 3.76 (4.6 sigma), which is scintillation, not a variable
star: the stars do not vary in step, and none reaches 5 sigma.

---

## Why chunks, and how to choose their size

A histogram is built **per pixel**. It only means something if the star stayed
on that pixel for the whole set of frames. Real fields drift: on the PESTO test
sequence, about 0.16 pixel per second. Over 1000 frames (49 s) that is 7 pixels,
so a histogram of the whole night smears every star across 7 pixels and tells
you nothing about any of them.

So the sequence is cut into consecutive chunks and each chunk is analysed on its
own. `chunks.size` in the YAML sets how many frames go in a chunk:

- **more frames per chunk** = more reads per pixel = a more precise flux, but
  the star drifts further during the chunk and gets smeared;
- **fewer frames per chunk** = the star stays put, but each flux is noisier.

64 frames on the PESTO sequence is 3.1 s and about half a pixel of drift, which
is a good compromise. Frames left over at the end are dropped, and the program
says exactly how many: with 1001 frames and chunks of 64, you get 15 chunks
using 960 frames, and 41 frames are dropped.

---

## Putting it on the sky (optional)

`run_chunks.py` measures everything in pixels. To turn that into RA and Dec:

```bash
pip install astrometry photutils      # one extra install, once
python pesto_astrometry.py            # target name comes from the YAML
```

or, to do both steps at once:

```bash
python run_pipeline.py                # bin the night, then solve it
python run_pipeline.py --skip-binning # solve products that already exist
```

**The two steps are deliberately separate.** The binning knows nothing about
the sky: it takes frames, builds one histogram per pixel and fits a flux, and
none of that assumes the frames are pictures of a star field. The same code is
meant to bin a spectrograph's detector, where the sources are echelle orders
and there is no astrometric solution to be had at all. So `embin.py` and
`run_chunks.py` stay pure, everything that needs the sky lives in
`pesto_astrometry.py`, and `run_pipeline.py` is a wrapper that runs one after
the other. If the astrometry fails, the binning products are still complete.

The astrometry writes its solution into **every** product, not just the summary:
each `embin_chunkNN.fits.gz` gets the WCS in its primary header and in
`HISTCUBE`, and each `embin_chunkNN_flux.fits.gz` in its primary header and in
`FLUX` and `FLUX_ERR`, so the cube and the maps fitted from it look at the
same sky. Every chunk keeps the same CD matrix and the same
central `CRPIX`; only `CRVAL` moves, by that chunk's measured drift, because
that is the only thing the drift actually changes. `HISTCUBE`'s third axis is
labelled `CTYPE3 = 'BIN'`, so no WCS-aware reader invents a sky meaning for it.

It stacks all the chunks after taking the measured drift out (the deepest image
the sequence can make), solves that with astrometry.net, and then **refits the
WCS to Gaia DR3** using the solver only to identify which star is which. What it
writes is deliberately plain:

- **`CTYPE = RA---TAN` / `DEC--TAN`, no SIP.** Over an 8-arcminute field a
  linear CD matrix is the whole story. The program does not take that on faith:
  it prints what a quadratic and a cubic would achieve, next to how many free
  parameters each one spends. On the test sequence the CD matrix reaches
  0.49 arcsec RMS on 27 stars and a cubic reaches 0.06, but with 20 free
  parameters for 54 measurements, which is fitting the centroid noise.
- **`CRPIX` at the exact centre of the field**, `((nx+1)/2, (ny+1)/2)`, with
  `CRVAL` the sky position there. Not wherever the solver happened to leave it.
- **The pixel scale is measured, and it is not the catalogue value.** PESTO is
  published at 0.466 arcsec/px (Cadieux et al. 2022); this data solves at
  0.4547, consistently and with log-odds above 100. Believe `PIXSCALE` in your
  own header over the round number.

Afterwards, `chunk_summary.fits` gains `ra`/`dec` columns on every row of
`TRACKS`, and the program tells you in plain words which track is your target:

```
TOI-1452 lands at x = 645.6, y = 146.9 on the solved image
-> that is track 1, 1.14 arcsec away: this is the light curve of TOI-1452
WARNING: 1 other Gaia source within 4.1 arcsec of the target, nearest at
3.10 arcsec, against an aperture radius of 1.36 arcsec. It is outside the
aperture, but only by a couple of PSF widths, so some of its light is in
track 1 and the transit depth will be diluted.
```

That warning is the point of the whole step. The 3.1-arcsec neighbour is
TIC 420112587, TOI-1452's known companion, and any transit depth measured from
track 1 without correcting for it is too shallow. `figures/astrometry.pdf` shows
the solved field, the residuals against Gaia, and a close-up in which the pair
is resolved.

If your target is not TOI-1452, change `astrometry.target` in the YAML, or pass
`--target NAME` / `--ra --dec`. The name is resolved at SIMBAD and moved to the
epoch of your frames by its own proper motion; without a target the search is
blind, which on a field this small is slow and often fails, since the Nuvu
headers contain no pointing information at all.

---

## Designing your own bins

The bin edges shipped in `embin_config.yaml` are the optimum **for the PESTO
constants above**. For any other detector or gain setting, compute your own:

```bash
python bin_optimizer.py --bias YOUR_BIAS --gain YOUR_GAIN --ron YOUR_RON \
                        --sat YOUR_SATURATION --nbins 16 \
                        --mu-min 1e-3 --mu-max 4 --scan
```

`--scan` also prints how good 4, 8, 12, 16, 24 and 32 bins would be, so you can
see what a smaller histogram would cost you. Copy the printed edge list into
`histogram.edges` in the YAML and you are done.

The same thing from Python:

```python
from bin_optimizer import design_bins

d = design_bins(bias=303.6178, gain=71.0598, ron=6.8164, saturation=5000,
                nbins=16, mu_min=1e-3, mu_max=4.0)
print(d.summary())        # every cut, in ADU, in electrons, in read-noise sigmas
print(d.edge_list())      # the list to paste into the YAML
print(d.worst_accuracy)   # 0.9943 -> these bins keep 99.4 % of the precision
```

Two rules that matter more than they look:

- **Set the flux range to what your science needs, and no wider.** The design
  protects the *worst* flux in the range you ask for, so asking for a range you
  do not need makes every flux worse.
- **Bin edges are whole ADU, and cuts through the read-noise peak are rounded
  UP, never down.** Rounding down lets read noise leak into the bin that is
  supposed to count electrons. For a detector with read noise below one ADU,
  that single rounding decision can take the efficiency from 0.99 to 0.51.

---

## Command-line options

Everything below has a home in `embin_config.yaml`, and that is where a setting
you always want belongs. The flags are for the one-off: a quick test, a rerun
with a different chunk size, a file written somewhere else.

`run_chunks.py`, the program you normally run:

| Flag | What it does |
|---|---|
| `-c`, `--config FILE` | read a different YAML than `embin_config.yaml` |
| `--chunk-size N` | frames per chunk, overriding `chunks.size` |
| `--n-chunks N` | stop after N chunks; this is the flag for a two-chunk test |
| `--outdir DIR` | write the results somewhere other than `output.directory` |
| `--aperture R` | light-curve aperture radius in pixels |
| `--match-radius R` | how far a star may move between chunks and still be the same star |
| `--no-stamps` | skip the raw postage stamps. Much smaller files, and the frames are read only once |

`embin.py`, one set of frames on its own:

| Flag | What it does |
|---|---|
| `-c`, `--config FILE` | as above |
| `-o`, `--output FILE` | override `output.cube_file` |
| `--flux-output FILE` | override the flux file name, which otherwise follows the cube's |
| `--from-cube CUBE` | do not read any frames: re-fit an existing cube file from its own header, and write the flux file next to it |
| `--first N` | 0-based index of the first frame to use |
| `--n-files N` | how many frames to use from there on |

`--from-cube` is the one worth remembering. It needs neither the frames nor a
configuration file, because the cube's header already carries the bin edges, the
detector constants and the flux grid, so it is how you rebuild a flux map you
deleted, or refit an archived cube years later with a better model.

---

## When something goes wrong

| What you see | What it means |
|---|---|
| `no file matches .../data_night/*.fits` | the folder is empty or `input.directory` points at the wrong place |
| `only 12 frame(s) found, which is fewer than one chunk of 64` | you have fewer frames than `chunks.size`; lower it |
| `histogram.edges must be strictly increasing` | you edited the edge list and left two equal or out-of-order numbers |
| `ModuleNotFoundError` | the virtual environment is not activated: run `source .venv/bin/activate` |
| the flux map is all `0.0001` | that is `fit.mu_min`, the bottom of the search grid: the pixels really did see nothing, which is normal for background pixels over few frames |
| every star sits at `fit.mu_max` | your flux grid stops below the real fluxes; raise `mu_max` |
| stars look smeared or doubled in `chunk_flux_maps.pdf` | the field drifted too much inside one chunk: lower `chunks.size` |

---

## Repository map

| File | What it is |
|---|---|
| `embin_config.yaml` | **the only file you edit.** Every setting, commented line by line |
| `run_chunks.py` | a whole sequence, cut into chunks: the program you normally run |
| `embin.py` | one set of frames -> one histogram cube + one flux map |
| `bin_optimizer.py` | where the bin edges belong, and what they cost |
| `emccd_histo.py` | the physical model and the maximum-likelihood flux fitter |
| `run_pipeline.py` | the wrapper: binning, then astrometry, in one command |
| `pesto_astrometry.py` | optional: solves the field and puts a WCS on the results |
| `demo_optimal_bins.py` | a standalone demonstration of the whole argument, including a Monte Carlo check |
| `plot_bins.py` | draws the bin edges on top of a real pixel-value histogram |
| `docs/` | the project web page (password protected) |

---

## Authors and credits

[Étienne Artigau](https://orcid.org/0000-0003-3506-5667), [Galina Sherren](https://orcid.org/0009-0006-5677-3944), [René Doyon](https://orcid.org/0000-0001-5485-4675), Jonathan St-Antoine
(Université de Montréal and Observatoire du Mont-Mégantic).

The detector constants come from the `pesto_stats` calibration of the PESTO
EMCCD at the Observatoire du Mont-Mégantic: a full MCMC fit of the physical
model to a source-free sky region.

Raw frames are **not** in this repository, and should not be committed to it. A
single sequence is close to a gigabyte, and it is observing data. `.gitignore`
keeps `data_night/` and `data_bin/` out of git for exactly that reason.
