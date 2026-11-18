# Nykteris

A night report for Ekos/KStars astrophotography sessions. Point it at a night's lights folder, the matching
Ekos `.analyze` file and, optionally, the guide log. It writes one HTML page that flags bad subs and gives a likely
cause for each: clouds, guiding, seeing, focus drift, low altitude, trails or the meridian flip.

It does not stack, weight or delete anything. Culling tools already exist (Siril, ASTAP, PixInsight); Nykteris is
about explaining *why* subs went bad, using what Ekos already records.

## Install

Python 3.10 or newer, on Linux, macOS or Windows.

```sh
pipx install git+https://github.com/xenotheles/nykteris-night-report
# or, from a clone:
pip install -e ".[test]"
```

## Use

```sh
nykteris report ~/Pictures/M_33/Light
```

By default it looks for `*.analyze` and `guide_log-*.txt` / `PHD2_GuideLog*.txt` next to the lights folder and up
to two levels above it. Name them explicitly when they live elsewhere:

```sh
nykteris report ~/Pictures/M_33/Light \
    --analyze ~/.local/share/kstars/analyze/ekos-2026-11-18T20-45-57.analyze \
    --guide-log ~/.local/share/kstars/guidelogs/guide_log-2026-11-18T20-48-27.txt \
    -o ~/reports/2026-11-18.html
```

Outputs, next to the report:

- `<name>.html`: the report (self-contained, open in any browser)
- `<name>.csv`: every sub with its metrics, verdict and causes
- `<name>-rejects.txt`: rejected subs, one per line with the main cause

`--move-rejects` moves rejected subs into a `rejected/` folder beside them. Nothing is ever deleted.
Measurements are cached in `.nykteris-cache.json`, so a rerun with different thresholds is fast.

Where Ekos keeps its files on Linux (StellarMate and Astroberry are the same):

| File | Location |
|---|---|
| `.analyze` session | `~/.local/share/kstars/analyze/` |
| Internal guider log | `~/.local/share/kstars/guidelogs/` (enable "Save guide log" in the Guide options) |
| PHD2 guide log | `~/PHD2/` |

## What it measures and how it decides

Every sub is measured with [sep](https://github.com/kbarbary/sep): star count, FWHM (twice the half-flux radius),
eccentricity and how consistent the elongation angle is, background level and noise, and the longest line-like
object. Each sub is compared with the median of its group (same filter, exposure and binning).

The `.analyze` file and guide log add context for each exposure window: guide RMS during the exposure, guide star
losses, mount altitude and pier side, focuser temperature, autofocus runs and meridian flips. Ekos' own HFR per sub
is shown next to the Nykteris value as a cross-check. Without the `.analyze` file the report falls back to FITS
headers (altitude computed from coordinates, site and time; flips from `PIERSIDE`; focus from `FOCUSPOS` and
`FOCUSTEM`).

| Cause | Rule (defaults) |
|---|---|
| Meridian flip | Sub degraded in any way and started within 10 min after a completed flip, or first sub on a new pier side |
| Low altitude | Blurred, starved or bright sub below 30 deg, where the airmass explains the blur (FWHM ~ airmass^0.6) |
| Clouds or haze | Star count below 60% of the median while FWHM stays normal; higher confidence with background up or guide star lost |
| Guiding error | Elongated stars with a consistent angle plus guide RMS above 1.5" or 2x the night's median |
| Elongation | Elongated stars with normal guiding: flexure, wind, a loose focuser |
| Focus drift | FWHM above 1.25x the median with round stars, and the temperature changed by 1 C since the last refocus or FWHM rose steadily over 3+ subs |
| Seeing or wind | FWHM above 1.25x with round stars and normal guiding, without a focus pattern |
| Trail | A detected object at least 40 px long with an axis ratio of 6 or more (verdict "check", not "reject") |

Eccentricity is only judged when FWHM is at least 2 px. Undersampled stars look elongated on perfectly good nights,
which is the most common complaint about existing culling tools.

### Thresholds

All thresholds are educated guesses until they have been checked against real nights. Print the defaults, edit,
and pass the file back:

```sh
nykteris thresholds > my-rig.toml
nykteris report ~/Pictures/M_33/Light --config my-rig.toml
```

## Testing without a sky

`nykteris synth OUTDIR` writes a synthetic Ekos night: 100 FITS lights with the INDI header set, a matching
`.analyze` file and an internal-guider guide log on one timeline, with faults injected on purpose (cloud passage,
guiding spike, seeing jump, focus drift with falling temperature, meridian flip, satellite trail, target sinking low).
`truth.json` labels every sub, and the tests check that the report finds each fault.

```sh
nykteris synth /tmp/fake-night
nykteris report /tmp/fake-night/Light/L
pytest
```

The synthetic data validates logic and file handling, not thresholds. Gaussian stars and simple noise are much
tidier than a real sky.

## File formats

Pinned from the KStars and INDI sources:

- `.analyze` (KStars `ekos/analyze/analyze.cpp`): `AnalyzeStartTime,<local time>,<tz>`, then one event per line,
  `<Event>,<seconds since start>,...`. Used: `CaptureComplete`, `GuideStats`, `MountCoords`, `Temperature`,
  `AutofocusStarting/Complete/Aborted`, `MeridianFlipState`, `GuideState`. Unknown events are ignored.
- Guide log (KStars `ekos/guide/internalguide/guidelog.cpp`, PHD2 log format 2.5): sections from `Guiding Begins at`
  to `Guiding Ends at`, local time, distances in guide camera pixels converted with the section's pixel scale.
- FITS (INDI `libs/indibase/indiccd.cpp`): `DATE-OBS` (UTC), `EXPTIME`, `FILTER`, `IMAGETYP`/`FRAME`, `SCALE` or
  `PIXSIZE1`+`FOCALLEN`, `RA`/`DEC` or `OBJCTRA`/`OBJCTDEC`, `SITELAT`/`SITELONG`, `PIERSIDE`, `OBJCTALT`,
  `FOCUSPOS`/`FOCUSTEM` (only when the focuser is snooped by the camera driver).

The logs are in local time without a reliable zone. Nykteris derives the offset by matching `CaptureComplete`
file names to FITS `DATE-OBS`; `--utc-offset` overrides it.

## Limits

- Thresholds are provisional until tuned on real data.
- Clouds at low altitude are reported as low altitude.
- One-shot-colour subs are measured on a 2x2 super-pixel luminance image.
- Satellite trails that cross dense star fields may be split by the detector and missed.

## License

MIT
