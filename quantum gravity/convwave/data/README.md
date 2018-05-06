# data

This folder is the standard location (i.e. scripts can expect data to be in this folder) for input data (e.g. generated waveforms), and this README file is necessary so that git keeps the folder in the repository :)

*Note:* Data files (especially large ones) should not be checked into GitHub! This folder will remain empty in the repo; only put stuff in here locally!

**The intended structure of this folder is the following:**
* `predictions`: The subfolders will contain the predictions that are automatically generated at the end of a training run, and can be used to quickly (visually) evaluate the model's performance at the end of this particular training run.
  These data are not meant for properly evaluating the method as such, as this should be done on longer, sparser stretches of data. Data for this purpose should be stored in `testing`.
* `strain`: Raw strain data, taken from the [LIGO Open Science Center](https://losc.ligo.org/events/). The files in here need to obey the following naming conventions to automatically be detected by the sample generation scripts: `<EVENTNAME>_<DETECTOR>_STRAIN_<SAMPLING-RATE>.h5`, i.e. for example: `GW150914_H1_STRAIN_4096.h5`
* `testing`: The subfolders here will contain data intended for testing and evaluating the method. These stretches will generally be longer and more sparse than the data used for training.
* `training`: The subfolders here will contain data used for training (short stretches with potentially a rather high amount of injections).
* `waveforms`: Pre-computed waveforms (using the `waveform_generator.py` script) that are used for making injections into noise taken from raw strain data.

