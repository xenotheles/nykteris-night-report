# Changelog

## 0.1.0 (unreleased)

First release.

- Night report for Ekos/KStars and N.I.N.A. sessions: one HTML page that flags bad subs with a likely cause
  (clouds, guiding, elongation, seeing, focus drift, low altitude, trails, meridian flip), plus CSV and a reject list.
- Inputs: FITS and XISF lights; Ekos `.analyze` files; N.I.N.A. Session Metadata (CSV/JSON) and autofocus reports;
  PHD2 and Ekos internal guider logs. Falls back to image headers alone.
- Thresholds in a TOML file. They are provisional until tuned on real nights.
- `nykteris synth` writes synthetic Ekos or N.I.N.A. nights with labelled faults.
