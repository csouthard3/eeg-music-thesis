import mne
import numpy as np
from mne.preprocessing import ICA
import os
import pandas as pd



meta1 = pd.read_excel(r"C:\Users\skyha\OneDrive\Documents\Thesis\Stimuli_Meta_v1.xlsx")
meta2 = pd.read_excel(r"C:\Users\skyha\OneDrive\Documents\Thesis\Stimuli_Meta_v2.xlsx")
stim_durations1 = dict(zip(meta1['id'], meta1['length of song+cue (sec)']))
stim_durations2 = dict(zip(meta2['id'], meta2['length of song+cue (sec)']))
cue_durations1 = dict(zip(meta1['id'], meta1['length of cue (sec)']))
cue_durations2 = dict(zip(meta2['id'], meta2['length of cue (sec)']))


condition_dict = {
        1: 'listen_with_cue',
        2: 'imagine_with_cue',
        3: 'imagine_without_cue_same_block',
        4: 'imagine_without_cue_diff_block',
    }

eog_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
mastoid_channel=['EXG5','EXG6']


def decode_event_id(event_id): #from paper--trial labels (called event_id here) are a concatenation of stimulus id and condition number

    stim_id = event_id // 10
    condition_num = event_id % 10
    condition = condition_dict[int(condition_num)]
    return stim_id, condition



def preprocess_openmiir(raw_file, filename, bad_channels):
 
    print()
    print("file name: ", filename)

    raw = mne.io.read_raw(raw_file, preload=True)
    raw.set_montage('biosemi64', on_missing='ignore')

    mastoids_present = False
    for channel in raw.ch_names:
        if channel in mastoid_channel:
            mastoids_present = True
    if mastoids_present:
        raw.set_channel_types({m: 'eeg' for m in mastoid_channel})
        raw.set_eeg_reference(mastoid_channel)
        raw.drop_channels(mastoid_channel) #must drop these before interpolating, otherwise got an error when running code
    else:
        raw.set_eeg_reference('average')
    
    ch_type_map = {}
    for channel in eog_channels:
        ch_type_map[channel] = 'eog'
    raw.set_channel_types(ch_type_map)
    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True)
    raw.filter(l_freq=1, h_freq=30) #originally tried a lower bottom frequency value, but captured too much of what was likely drift

    ica = ICA(n_components=20, random_state=97, max_iter='auto') #20 components is a bit of a guesstimate at a happy medium, passing 97 into random_state so same output every time of random number generator
    ica.fit(raw)
    
    eog_indices = []
    for channel in eog_channels:
        idx, scores = ica.find_bads_eog(raw, ch_name=channel)
        eog_indices.extend(idx)
    try: #possibly redundant(?), since mastoids were in theory dropped before interpolation
        mastoid_idx = mne.pick_channels(raw.info["ch_names"], include=mastoid_channel)
    except ValueError:
        mastoid_idx = []
    ica.exclude = list(set(eog_indices)) + list(mastoid_idx)
    raw = ica.apply(raw)


    '''
    Not removing ECG (heartbeats) because they don't have a dedicated channel in data
    Without that dedicated channel, MNE would have to infer the heartbeats based on periodic information
    I'm concerned "periodic information" could pick up things like tempo, which we don't want to remove
    But if the model is struggling, maybe this is worth revisiting to try to increase signal-to-noise ratio
    edit: after talking with people in the field, it seems a lot of people don't worry about filtering out heartbeats
    '''

    #epoching/segmentation
    trial_events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)

    epochs_data = []
    labels = []

    for event in trial_events:
        sample, __, event_id = event
        if event_id < 1000: #captures only listening/imagination events, see code here for more details on event codes: https://github.com/sstober/deepthought/blob/master/deepthought/datasets/openmiir/events.py
            stim_id, condition = decode_event_id(event_id)
           
            if filename in ['P01-raw.fif', 'P04-raw.fif', 'P06-raw.fif', 'P07-raw.fif']:
                print("one of first four participants")
                duration = stim_durations1[stim_id]
                cue_len = cue_durations1[stim_id]
            else: #same pieces but slightly different (e.g. small difference in tempo for a few) were given for other participants
                duration = stim_durations2[stim_id]
                cue_len = cue_durations2[stim_id]

            epoch = mne.Epochs(
                raw,
                events = [event],
                event_id = {condition: event_id},
                tmin = -.5,
                tmax = duration,
                baseline = (-.5, 0), #performing baseline correction during epoching, based on activity at the -.5 to 0 second range before the start of the trial
                preload = True,
            )

            epoch.crop(tmin=cue_len + .5, tmax=None) #need to crop after creating epoch to remove cue clicks for model
            epoch.pick("eeg")
            epoch.resample(128)
            
            data = epoch.get_data()
            #we actually don't want line below yet, since it could cause data leakage since using train and test data
            # data = (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True) 
            epochs_data.append(data)
            labels.append(event_id)


    max_len = max(d.shape[2] for d in epochs_data)
    padded_data = []
    
    for d in epochs_data:
        pad_width = max_len - d.shape[2]
        if pad_width > 0:
            d = np.pad(d, ((0, 0), (0, 0), (0,pad_width)))
        padded_data.append(d)
    
    all_data = np.concatenate(padded_data)
    labels = np.array(labels)
    subj_name, _ = os.path.splitext(raw_file)
    np.savez_compressed(f"{subj_name}_data_v2.npz", data=all_data, labels=labels)
    
    return all_data, epochs_data





bad_channels = {
    'P01-raw.fif': ['P8', 'P10', 'T8'],
    'P04-raw.fif': ['T8'],
    'P06-raw.fif': ["Fp1", 'Iz', 'FT7'], #Fp1 wan't identified by researchers, we found this one
    'P07-raw.fif': [],
    'P09-raw.fif': [],
    'P11-raw.fif': ['T7', 'T8'],
    'P12-raw.fif': ['C3', 'PO3'],
    'P13-raw.fif': ['Iz'],
    'P14-raw.fif': ['T7', 'F7'],
}



for file in os.listdir('C:/Users/skyha/Downloads/OpenMIIR-RawEEG_v1'):
    if file.endswith(".fif") or file.endswith('.bdf'):
        raw_path = os.path.join('C:/Users/skyha/Downloads/OpenMIIR-RawEEG_v1', file)
        subject_data, subject_epochs = preprocess_openmiir(raw_path, file, bad_channels[file])

