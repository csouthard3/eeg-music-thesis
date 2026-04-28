import os
import numpy as np
from tensorflow.keras.models import load_model
import pretty_midi


onset_model = load_model('onset.keras')
onset_norm = np.load("global_norm_binary_onset.npz")
onset_params = {"original_samples": int(200/1000 * 128), "step_samples": int(100/1000 * 128), "max_timesteps": 150}

chroma_model = load_model("chroma.keras")
chroma_norm = np.load("global_norm_chroma.npz")
chroma_params = {"original_samples": int(400/1000 * 128), "step_samples": int(200/1000 * 128), "max_timesteps": 100}

p13_data = np.load(os.path.join("/content", "P13-raw_data_v2.npz"))
eeg, labels = p13_data["data"], p13_data["labels"]
results = {}

lengths_v2 = {
    1: 13.301,
    2: 7.7,
    3: 9.7,
    4: 11.6,
    11: 13.5,
    12: 7.7,
    13: 9,
    14:  12.2,
    21: 8.275,
    22: 16,
    23: 9.2,
    24: 6.956
}


def build_windows(trial, params, mean, std): #creates padded windows, this mirorrs how constructed in model code

    length = trial.shape[1]
    starts = list(range(0, length - params["original_samples"] + 1, params["step_samples"]))
    raw_windows = []
    for start in starts:
        raw_windows.append(trial[:, start:start + params['original_samples']].astype(np.float32))
    raw_windows = np.stack(raw_windows)

    pad_width = 128 - params['original_samples'] #eegnet is validated on 1 second windows, so we determine how much padding we need to get windows to 128 samples
    windows = []
    for raw in raw_windows:
        padded = np.pad(raw, ((0,0), (0, pad_width)), constant_values = 0.0)
        padded = padded[np.newaxis,...,np.newaxis]
        w_norm = (padded - mean)/(std + 1e-12)
        windows.append(w_norm[0])
    windows = np.stack(windows).astype(np.float32)

    return windows[:params['max_timesteps']]


# def refine_peak_time(onset_curve, idx, centers): #tried to interpolate onset times, but this didn't seem to noticeably help recognizability

#     if idx <= 0 or idx >= len(onset_curve) - 1:
#         return centers[idx]

#     #used formula from this link to interpolate peaks: https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html 
#     y1 = onset_curve[idx-1]
#     y2 = onset_curve[idx]
#     y3 = onset_curve[idx+1]
#     denom = (y1 - 2*y2 + y3)
#     offset = 0.5 * (y1 - y3) / (denom + 1e-12)

#     step = centers[1] - centers[0]
#     refined_time = centers[idx] + offset * step

#     return refined_time


def midi_from_predictions(preds, stim_id):
    onset_pred = preds["onset_pred"]
    chroma_pred = preds["chroma_pred"]
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

    #find windows where onsets are detected. a threshold of 0.6 seems to be best based on some quick tests of 0.5, 0.6, 0.7
    binary = (onset_pred.squeeze() > 0.6).astype(int)
    onset_indices = np.where((binary[1:] == 1))[0] + 1
    if binary[0] == 1:
        onset_indices = np.insert(onset_indices, 0, 0)

    #build notes only where onset is detected, since chroma model was only trained  on onset windows
    duration = preds['trial_length'] / 128
    onset_center_times = (np.arange(len(onset_pred)) * onset_params["step_samples"] + onset_params["original_samples"] // 2) / 128
    chroma_center_times = (np.arange(len(chroma_pred)) * chroma_params["step_samples"] + chroma_params["original_samples"] // 2)/128
    
    i = 0
    while i < len(onset_indices):
        start_i = onset_indices[i]
        start = onset_center_times[start_i]
        if start >= duration:
            i += 1
            continue
        chroma_index = int(np.argmin(np.abs(chroma_center_times - start)))
        pitch = int(np.argmax(chroma_pred[chroma_index]))

        #extend note duration over consecutive windows w predicted onset and same chroma, but in future there might be a more sophisticated way to address overlapping windows
        end_idx = start_i
        while end_idx + 1 < len(binary) and binary[end_idx + 1] == 1:
            next_chroma = np.argmin(np.abs(chroma_center_times - (onset_center_times[end_idx+1])))
            next_pitch = int(np.argmax(chroma_pred[int(next_chroma)]))
            if next_pitch != pitch:
                break
            end_idx += 1
        
        #also extend note duration through silence, this just sounds better than having long periods of silence since we haven't trained any models to handle/specifically detect silence
        while end_idx + 1 < len(binary) and binary[end_idx + 1] == 0:
            end_idx += 1
        end = duration
        if end_idx + 1 < len(onset_center_times):
            end = onset_center_times[end_idx+1]

        note = pretty_midi.Note(velocity = 60, pitch= 60 + int(pitch), start=start, end=end)
        piano.notes.append(note)

        i = np.searchsorted(onset_indices, end_idx+1)

    pm.instruments.append(piano)
    return pm


for trial_idx in range(labels.shape[0]):
    event_id = int(labels[trial_idx])
    stim_id = event_id // 10
    condition_num = event_id % 10

    if condition_num != 2 or stim_id in results:  #only use imagination trials with cue clicks right before, and also just the first time an iteration of each trial is seen
        continue

    print('predicting ' + str(event_id))

    trial = eeg[trial_idx]
    onset_windows = build_windows(trial, onset_params, onset_norm["mean"], onset_norm["std"])
    onset_pred = onset_model.predict(onset_windows[np.newaxis, ...], batch_size=32)
    onset_pred = onset_pred.squeeze(0)
   

    chroma_windows = build_windows(trial, chroma_params, chroma_norm['mean'], chroma_norm["std"])
    chroma_pred = chroma_model.predict(chroma_windows[np.newaxis, ...], batch_size=32)
    chroma_pred = chroma_pred.squeeze(0)

    results[stim_id] = {"trial_length": trial.shape[1], "onset_pred": onset_pred, "chroma_pred": chroma_pred}
    


for stim_id, preds in results.items():
    pm = midi_from_predictions(preds, stim_id)

    for inst in pm.instruments: #make sure length of generated MIDI matches original MIDI lengths (sometimes it's longer, maybe it's because the model always predicts max_timesteps?)
        new_notes = []
        for note in inst.notes:
            if note.start >= lengths_v2[stim_id]:
                continue
            note.end = min(note.end, lengths_v2[stim_id])
            new_notes.append(note)
        inst.notes = new_notes

    pm.write(os.path.join('/content/reconstructed', f'{stim_id}.mid'))