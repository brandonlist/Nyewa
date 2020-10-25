import mne

raw = mne.io.read_raw_edf('2020_10_20_0.cdt.edf')
events,events_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw,events)


k = 10
d = 19