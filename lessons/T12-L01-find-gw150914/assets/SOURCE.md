# Asset provenance

## `gw150914_h1_l1_32s.npz`

| | |
|---|---|
| Source | Gravitational Wave Open Science Center (GWOSC), event GW150914, release `GWTC-1-confident`, event UID `GW150914-v3` |
| H1 file | <https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt.gz> |
| L1 file | <https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_4KHZ_R1-1126259447-32.txt.gz> |
| Event page | <https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/> |
| DOI | <https://doi.org/10.7935/82H3-HH23> |
| Format downloaded | GWOSC's plain-text (`.txt.gz`) release, NOT HDF5 — this repository ships no `h5py`. One double-precision strain value per line, gzip-compressed, after a 3-line `#` comment header naming the detector, the sample rate and the GPS start/duration. |
| GPS start (both files) | 1126259447 |
| Duration | 32 s |
| Sample rate | 4096 Hz (131,072 samples per detector) |
| Event GPS time | 1126259462.4, as published on the event page (`GPS: 1126259462.4`) — 15.4 s into this window |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/> — stated on <https://gwosc.org/acknowledgement/> (`claims.yaml`: `gwosc-licence`) |
| Required citation | R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo", SoftwareX 13 (2021) 100658, <https://doi.org/10.1016/j.softx.2021.100658> — the reference GWOSC names for the observing run (O1) this event belongs to |
| Changes made | Re-encoded: the two text files were parsed and stored together as float32 arrays in one compressed `.npz`, with the GPS start, the sample rate and GWOSC's published event GPS time. No sample was altered beyond the float32 rounding described below. |
| Retrieved | 2026-09-24 (re-downloaded and checked sample for sample by the independent reviewer the same day) |
| Size (this asset) | 929,060 bytes (< 1 MiB) |
| Gated? | No. Public, ungated, no registration, no API key. |

### Required acknowledgement

GWOSC asks that any work using its data include this acknowledgment, reproduced verbatim from
<https://gwosc.org/acknowledgement/> (read 2026-09-24):

> This research has made use of data or software obtained from the Gravitational Wave Open
> Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo
> Collaboration, and KAGRA. This material is based upon work supported by NSF's LIGO
> Laboratory which is a major facility fully funded by the National Science Foundation, as
> well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the
> Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the
> construction of Advanced LIGO and construction and operation of the GEO600 detector.
> Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo
> is funded, through the European Gravitational Observatory (EGO), by the French Centre
> National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica
> Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium,
> Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. KAGRA is
> supported by Ministry of Education, Culture, Sports, Science and Technology (MEXT), Japan
> Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF)
> and Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science
> and Technology Council (NSTC) in Taiwan.

### How the committed file was built

Both `.txt.gz` files above were downloaded directly from `gwosc.org` and read with
`numpy.loadtxt` — one float64 value per line, no HDF5 library involved anywhere. The two
131,072-sample arrays, the shared GPS start time, the sample rate and GWOSC's published event
GPS time were written into one compact `.npz` with `numpy.savez_compressed`, storing the
strain arrays as **float32** rather than float64.

**Checked, 2026-09-24:** both arrays equal the float32 cast of a fresh download of the two
files above, sample for sample. The float32 rounding moves no sample by more than about
5.5e-8 of that detector's peak absolute strain, and the finished lesson prints exactly the
same output from a float64 copy of the same data as from this float32 asset.

### Why the direct GWOSC URLs, not a mirror

An earlier authoring pass could not reach `gwosc.org` and used a third-party CC-BY-4.0 Zenodo
mirror (`10.5281/zenodo.16284550`) of the same H1/L1 32 s release. The committed asset was
rebuilt from the primary GWOSC files above; the Zenodo copy is not used anywhere.

### The event time, and a window that is not centred

The first version of this asset stored the event time rounded down to GPS 1126259462, which
put the "true event" 0.4 s early in every printout and in the rubric's real-data tolerance.
The independent review replaced that one field with GWOSC's published 1126259462.4; the strain
arrays are byte-for-byte unchanged. GWOSC's 32 s window therefore starts 15.4 s before the
event, not 16 s. `lesson.py` computes `T_EVENT` from the two GPS numbers in the file
(`event_gps - gps_start`) and never types it.
