# Asset provenance

## `camels_aus_v2_two_basins_1985_2014.npz`

| | |
|---|---|
| Source dataset | CAMELS-AUS v2 (Catchment Attributes and Meteorology for Large-sample Studies, Australia, version 2), Zenodo record 14289037, version DOI <https://doi.org/10.5281/zenodo.14289037> (published 2024-12-07); concept DOI for all versions <https://doi.org/10.5281/zenodo.12575680> |
| Record page | <https://zenodo.org/records/14289037> |
| Files read | `03_streamflow.zip` member `streamflow_mmd.csv`; `05_hydrometeorology.zip` members `precipitation_SILO.csv` and `et_morton_wet_SILO.csv`; `01_id_name_metadata.zip` members `id_name_metadata.csv` and `dataset_sources.csv`; `CAMELS_AUS_Attributes&Indices_MasterTable.csv` |
| Variables stored | `flow_mm_d` (observed daily streamflow, mm/day of depth), `precip_mm_d` (catchment-average daily precipitation from SILO, mm/day), `pet_mm_d` (catchment-average daily potential evapotranspiration, SILO Morton wet-environment areal PET, mm/day); each of shape (2 catchments, 10957 days). Also `station_id`, `station_name`, `area_km2` and `start_date` |
| Period | 1985-01-01 to 2014-12-31 (10,957 days, identical for both catchments) |
| Catchments | `607013` Lefroy Brook River at Rainbow Trail (Western Australia, 248.0 km2) and `116015A` Blunder Creek at Wooroora (Queensland, 125.9 km2) |
| Licence | Every array in this file: Creative Commons Attribution 4.0 International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>. The dataset as a whole is CC BY 4.0 by its Zenodo record (`claims.yaml`: `camels-aus-zenodo-licence`) and its paper (`camels-aus-licence`), and each upstream provider states CC BY 4.0 on its own page for the series used here (next row) |
| Upstream sources, each checked on its provider's own page (2026-09-29) | `flow_mm_d`: Bureau of Meteorology Hydrologic Reference Stations (HRS 2022 list); the HRS page states CC BY 4.0 (`bom-hrs-licence`). `precip_mm_d` and `pet_mm_d`: SILO (Queensland Government), catchment averages computed by the dataset authors from SILO grids (`camels-aus-silo-precipitation`); SILO states CC BY 4.0 and asks users to acknowledge SILO as the source and to say whether the data were changed (`silo-licence`) |
| Changes made | Extracted 2 of 561 catchment columns and the 10,957 days of the window, and stored them as float64 in one compressed `.npz`. The decimal values are parsed exactly as they appear in the CSV files; nothing is filtered, gap-filled, rescaled or altered. The chosen catchments have no missing value in flow, precipitation or PET for these days (the dataset marks a gap as -99.99; none falls in the window). The only attribute copied from the master table is the catchment area |
| Retrieved | 2026-09-29 |
| Size (this asset) | about 0.3 MB |
| Gated? | No. Public, ungated, no registration, no API key |

### Required attribution (CC BY 4.0)

Data: Fowler, K., Zhang, Z. and Hou, X., *Dataset for CAMELS-AUS v2: updated
hydrometeorological timeseries and landscape attributes for an enlarged set of catchments in
Australia*, Zenodo, <https://doi.org/10.5281/zenodo.14289037> (concept DOI
<https://doi.org/10.5281/zenodo.12575680>), licensed CC BY 4.0. Paper: Fowler, K. J. A.,
Zhang, Z. and Hou, X., *CAMELS-AUS v2: updated hydrometeorological time series and landscape
attributes for an enlarged set of catchments in Australia*, Earth System Science Data 17,
4079-4095, 2025, <https://doi.org/10.5194/essd-17-4079-2025>.

Streamflow: Bureau of Meteorology, © Commonwealth of Australia. Licensed from the Commonwealth of
Australia under a Creative Commons Attribution 4.0 International licence (the Bureau's preferred
wording, `bom-attribution`).

Precipitation and PET: SILO, Queensland Government, <https://www.longpaddock.qld.gov.au/silo/>,
licensed CC BY 4.0 (`silo-licence`). The catchment averages were computed from SILO's grids by
the CAMELS-AUS v2 authors; this lesson changed nothing in their values, only selected two
catchments and thirty years (see "Changes made").

### How the two catchments were chosen (a rule written from the dataset's own catalogue)

The rule uses only catalogue attributes and record completeness, never a model result. From the
561 catchments, keep those with all of: flagged `InHRS(2022) = 1` in `dataset_sources.csv`; a
catchment area from 100 to 500 km2; aridity (mean PET over mean precipitation, master table
`aridity`) below 1.5; no missing value in flow, SILO precipitation or SILO Morton wet PET on
any day 1985-01-01 to 2014-12-31; and at most 1 percent of those days with zero flow (so the
river is perennial, and a runoff model is not asked to reproduce a dry bed). 31 catchments
qualify. The lesson takes the one with the lowest and the one with the highest master-table
`p_seasonality` (negative is a Jun/Jul rainfall peak, positive a Dec/Jan peak;
`claims.yaml`: `camels-aus-attribute-definitions`): the most winter-dominant and the most
summer-dominant rainfall regime that qualify. That gave `607013` and `116015A`. The catalogue's
`aridity` and `p_seasonality` were computed by the dataset authors from another rainfall product;
they were used only to choose, and are not stored in the asset. The lesson describes each river's
seasonality with its own statistic, computed from the committed rain.

The PET series is the one the dataset authors recommend for hydrological modelling (Morton
wet-environment PET; `claims.yaml`: `camels-aus-forcing-recommendation`). For rain they recommend
a different gridded product; this lesson uses the dataset's SILO rainfall instead, for the reason
given in the provenance note below. The window 1985-2014 is the years common to every gauge source
in the dataset (some gauges end in 2014).

### Choices made after looking at the data

The lesson's section 1 tells the student about these; here are the details.

* The selection rule's thresholds (size band, aridity, zero-flow share) were set by the lesson's
  author after looking at the catalogue attributes of the eligible catchments, though (by the
  author's account) before any model was run on the chosen two. The rule was re-run
  independently on 2026-09-29 from the Zenodo files and gives the same 31 catchments and the same
  two picks, but nearby thresholds pick other rivers: an area band up to 1000 km2 gives 611004 as
  the winter extreme, aridity below 1.0 gives 403214 and below 3.0 gives 616013, and allowing no
  zero-flow day at all gives 207015 as the summer extreme.
* The forcing was first prototyped with the SILO Morton point PET; the lesson switched to the
  recommended Morton wet-environment PET after those runs. Re-run with the point PET (and the
  rain used now), the calibrated model scores lower on 607013's unseen years and about the same
  on 116015A's, and the three forecasters keep their order on both rivers.
* The rain series was swapped from the dataset's recommended product to its SILO rainfall after
  the whole lesson had been run and reviewed on the former, for the licence alone. What changed:
  on the unseen years the calibrated model scores higher on 607013 and lower on 116015A, the ridge
  scores a little lower on both, and the three forecasters keep their order on both rivers. For
  116015A the random-split score now sits further above the chronological hold-out, while the
  random-fitted parameters' lead on the later years is smaller. Every check the lesson's grader
  makes on the real records passes with either series.
* The ridge baseline's 30 days of lagged rain were picked from a comparison of 7, 30 and 90 days on
  these records. The notebook's baseline demo prints that comparison.

### Provenance note: why not the recommended rainfall

The dataset authors recommend their `precipitation_AGCD.csv` series, catchment averages of the
Bureau of Meteorology's Australian Gridded Climate Data (AGCD) v1.0.1. The dataset paper states
that the dataset and the datasets it is based on are CC BY. But the publisher's own record for
AGCD v1.0.1 (NCI, <https://doi.org/10.25914/hjqj-0x55>, metadata read on 2026-09-29 at
<https://geonetwork.nci.org.au/geonetwork/srv/api/records/f1801_8183_3094_1341/formatters/xml>)
gives its licence as Creative Commons Attribution-NonCommercial 4.0 International. With two
statements that disagree, this lesson does not commit AGCD-derived values and uses SILO rainfall,
whose provider states CC BY 4.0. This note records a conflict found while reviewing the lesson;
the lesson makes no claim that depends on it.

### How the committed file was built

The three series were read from the zip archives on Zenodo with HTTP range requests (only the
named members were downloaded), the columns for the two station ids and the rows for the window
were selected by date, and the seven arrays (`station_id`, `station_name`, `area_km2`,
`start_date`, `precip_mm_d`, `pet_mm_d`, `flow_mm_d`) were written with
`numpy.savez_compressed`. The lesson computes its own calendar from `start_date`.

Checked independently on 2026-09-29: the three arrays equal, value for value, the two station
columns of `streamflow_mmd.csv`, `precipitation_SILO.csv` and `et_morton_wet_SILO.csv` for
1985-01-01 to 2014-12-31, read again from the Zenodo archives.
