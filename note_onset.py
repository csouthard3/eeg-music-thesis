import os
import pretty_midi
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, TimeDistributed, Lambda, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
sys.path.append('./eeg_models')
from eeg_models.EEGModels import EEGNet
from collections import Counter

     
subjects = ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']
original_window_samples = int(200 / 1000 * 128)
step_samples = original_window_samples // 2
output_dim = 1 #only one output per window representing whether an onset is present or not

np.random.seed(42)
tf.random.set_seed(42)


class EEGWindowGenerator(Sequence): #initially the model.fit call was overloading colab gpu vram, but calling this class on the data before allows it to be chunked into more doable sizes

    def __init__(self, X, y, indices, batch_size, shuffle):
        super().__init__()
        self.X = X
        self.y = y
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_inds = self.indices[idx*self.batch_size : (idx+1) * self.batch_size]
        X_batch = self.X[batch_inds]
        y_batch = self.y[batch_inds]
        return X_batch,  y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def imagination_dataset_generator(sequences, targets, sample_weights):
    for i, (seq, tgt) in enumerate(zip(sequences, targets)):
        sw = sample_weights[i]
        yield seq, tgt, sw

def yield_data(sequences, targets, sample_weights, batch_size, shuffle):
    output_signature = (
        tf.TensorSpec(shape=(None, sequences[0].shape[1], sequences[0].shape[2], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(lambda: imagination_dataset_generator(sequences, targets, sample_weights), output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(16 , len(sequences)))
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [None, sequences[0].shape[1], sequences[0].shape[2], 1],
            [None, 1],
            [None]),
        padding_values=(0.0, 0.0, 0.0)
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_subject(subj):
    openmiir_path = os.path.join('/content' , f"{subj}-raw_data_v2.npz")
    print(openmiir_path)
    eeg = np.load(openmiir_path)
    return eeg["data"], eeg["labels"]

def extract_onsets(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    onsets = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            onsets.append(note.start)
    onsets = np.unique(np.array(onsets))
    return np.sort(onsets)


def eeg_windows(trial_data):

    length = trial_data.shape[1]
    starts = list(range(0, length - original_window_samples + 1, step_samples))
    windows = []
    for start in starts:
        windows.append(trial_data[:, start:start+original_window_samples].astype(np.float32))
    return np.stack(windows)


def create_onset_labels(length, onsets_seconds):

    num_windows = 1 + (length - original_window_samples) // step_samples

    starts = np.arange(0, num_windows * step_samples, step_samples)
    targets = np.zeros((num_windows,), dtype=np.float32)

    for i in range(len(starts)):
        start = starts[i]
        window_start = start
        window_end = start + original_window_samples

        onset_samples = (np.array(onsets_seconds) * 128).astype(int)
        if np.any((onset_samples >= window_start) & (onset_samples < window_end)):
            targets[i] = 1.0

    return targets.reshape(-1, 1)


def pretrain_dataset(subjects, onset_dict):

    windows = []
    targets = []
    subjects_list = []

    for subj in subjects:
        eeg, labels = load_subject(subj)
        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx]) #event ids are stim_id concatenated with condition_num (see openmiir paper for details)
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 1:  #only listening trials for pretraining, which have condition num of 1
                continue

            trial = eeg[trial_idx]
            raw_windows = eeg_windows(trial)

            if subj in ['P01','P04','P06','P07']:
                onsets = onset_dict[f"{stim_id}_v1"]
            else:
                onsets = onset_dict[f"{stim_id}_v2"]

            length = trial.shape[1]
            y = create_onset_labels(length, onsets)

            print(y.shape[0])
            if y.shape[0] != raw_windows.shape[0]:
                nmin = min(y.shape[0], raw_windows.shape[0])
                raw_windows = raw_windows[:nmin]
                y = y[:nmin]
            print(y.shape[0])

            pad_width = 128 - original_window_samples  #eegnet is trained specifically for one second (128 sample) windows, so we pad our shorter windows to that length
            for i, w in enumerate(raw_windows):
                w_padded = np.pad(w, ((0,0),(0, pad_width)), constant_values=0.0)
                windows.append(w_padded)
                targets.append(y[i])
                subjects_list.append(subj) 

    X_raw = np.stack(windows).astype(np.float32)
    X_raw = X_raw[..., np.newaxis]

    mean = X_raw.mean(axis=(0,1,2), keepdims = True)
    std  = X_raw.std(axis=(0,1,2), keepdims = True)
    np.savez("/content/drive/MyDrive/global_norm_binary_onset.npz", mean=mean, std=std)
    Xs = (X_raw - mean) / (std + 1e-12)
    y = np.stack(targets).astype(np.float32)

    return Xs, y, subjects_list


def training_dataset(subjects, onset_dict, norm):

    mean = norm["mean"]
    std = norm["std"]
    valid_subjects = []

    sequences = []
    targets = []
    masks = []
    meta = []

    for subj in subjects:
        eeg, labels = load_subject(subj)
        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx])   #event ids are stim_id concatenated with condition_num (see openmiir paper for details)
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 2:
                continue

            print(condition_num)

            trial = eeg[trial_idx]
            length = trial.shape[1]

            raw_windows = eeg_windows(trial)

            pad_width = 128 - original_window_samples #eegnet is trained specifically for one second (128 sample) windows, so we pad our shorter windows to that length
            win_list = []
            for w in raw_windows:
                w_padded = np.pad(w, ((0,0),(0,pad_width)), constant_values=0.0)
                w_padded = w_padded[np.newaxis, ... , np.newaxis]
                w_norm = (w_padded - mean) / (std+1e-12)
                win_list.append(w_norm[0])
            windows = np.stack(win_list).astype(np.float32)

            if windows.shape[0] > 160: #code was overloading colab vram without a sequence cap, but 160 timesteps should be enough to capture an entire trial
                windows = windows[:160]

            if subj in ['P01','P04','P06','P07']:
                version = 'v1'
            else:
                version = 'v2'

            onsets = onset_dict[f"{stim_id}_{version}"]

            y = create_onset_labels(length, onsets)

            if y.shape[0] != windows.shape[0]:
                nmin = min(y.shape[0], windows.shape[0])
                windows = windows[:nmin]
                y = y[:nmin]

            #randomly masking 50% of non-onset windows, because without this the model was heavily biased toward predicting non-onsets
            keep_mask = np.ones(len(y), dtype=bool)
            for i in range(len(y)):
                if y[i] == 0:
                    if np.random.rand() > 0.5:
                        keep_mask[i] = False

            windows = windows[keep_mask]
            y = y[keep_mask]

            if windows.shape[0] < 10:
                continue

            sequences.append(windows.astype(np.float32))
            targets.append(y.astype(np.float32))
            meta.append((subj,  trial_idx, int(stim_id)))
            masks.append(np.ones(len(y), dtype=np.float32))

    return sequences, targets, masks, meta



def pretrain_eegnet(X_listen, y_listen, subjects_list):

    train_subjects, val_subjects = train_test_split(subjects_list, test_size=0.1, random_state=42) #could test different test sizes for pretrain

    train_inds = [i for i, s in enumerate(subjects_list) if s in train_subjects]
    val_inds = [i for i, s in enumerate(subjects_list) if s in val_subjects]

    chans = X_listen.shape[1]
    samples = X_listen.shape[2]
    base = EEGNet(nb_classes=1, Chans=chans, Samples=samples)
    flatten_layer = base.get_layer('flatten').output
    out = Dense(1, activation='sigmoid', name='regression_output')(base.get_layer('flatten').output)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss = 'binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

    checkpoint = ModelCheckpoint("/content/drive/MyDrive/pretrain_binary_onset.keras", save_best_only=True, mode='min')
    early = EarlyStopping(patience=10, mode='min', restore_best_weights=True)

    train_gen = EEGWindowGenerator(X_listen, y_listen, train_inds, 64, True)
    val_gen = EEGWindowGenerator(X_listen, y_listen, val_inds, 64, False)
    model.fit(train_gen, validation_data=val_gen, epochs=80, callbacks=[checkpoint, early]) #80 epochs was chosen somewhat arbitrarily



def build_cnn_rnn(channels, samples, subject_list):

    base = EEGNet(nb_classes=output_dim, Chans=channels, Samples=samples)
    base.load_weights("/content/drive/MyDrive/pretrain_binary_onset.keras")
    
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # flatten_layer = base.get_layer(-3)
    flatten_layer = base.get_layer('flatten') #layer right before final classification layer
    eegnet = Model(inputs=base.input, outputs=flatten_layer.output)

    seq_input = Input(shape=(None, channels, samples, 1), name='seq_input')

    feats = TimeDistributed(eegnet, name='eegnet_time_dist')(seq_input)
    masked_feats = Masking(mask_value=0.0)(feats)
    x = Bidirectional(LSTM(64, return_sequences=True))(masked_feats) #we chose 64 units for lstm because it seems pretty standard
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    out = TimeDistributed(Dense(output_dim, activation='sigmoid'))(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss = 'binary_crossentropy') #Adam optimizer chosen in the hope of performing well on sparse data, but it might be valuable to test other optimizers too
    return model



onset_dict = {}
for stim_id in [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]:
    #in openmiir data collection, tempo was shifted between first and second half of participants so the midi is slightly different
    found_v1 = os.path.join('/content', f"stim_{stim_id}_v1.mid") 
    found_v2 = os.path.join('/content', f"stim_{stim_id}_v2.mid")
    onset_times_v1 = extract_onsets(found_v1)
    onset_times_v2 = extract_onsets(found_v2)
    onset_dict[str(stim_id) + "_v1"] = onset_times_v1
    onset_dict[str(stim_id) + "_v2"] = onset_times_v2

X_listen, y_listen, subjects_list = pretrain_dataset(subjects, onset_dict)
pretrain_eegnet(X_listen, y_listen, subjects_list)

norm = np.load("/content/drive/MyDrive/global_norm_binary_onset.npz")

sequences, targets, masks, meta = training_dataset(subjects, onset_dict, norm)

flat_targets = []
flat_keep = []

for target, mask in zip(targets, masks):
    target = target.squeeze()
    flat_targets.append(target[mask == 1])
    flat_keep.append(mask == 1)

flat_targets = np.concatenate(flat_targets)
flat_keep = np.concatenate(flat_keep)

counts = Counter(flat_targets.tolist())
total = len(flat_targets)

majority_class = counts.most_common(1)[0][0]
majority_acc = counts[majority_class] / total

masked_targets = flat_targets.astype(float)

zero_baseline_mse = np.mean(masked_targets ** 2)
mean_value = np.mean(masked_targets)
mean_baseline_mse = np.mean((masked_targets - mean_value)**2)

print("classes:", dict(counts))
print("majority class baseline accuracy:", majority_acc)
print("total windows:", total)
print("zero baseline mse:", zero_baseline_mse)
print("mean baseline mse:", mean_baseline_mse)

subj_to_indices = {}
for i, (subj, trial_idx, stim_id) in enumerate(meta):
    subj_to_indices.setdefault(subj,  []).append(i)

subject_list = np.array(subjects)
kf = KFold(n_splits=len(subjects))
fold_scores = []

for fold, (train_subj_idx, test_subj_idx) in enumerate(kf.split(subject_list)):
    train_subjs = subject_list[train_subj_idx]
    test_subjs = subject_list[test_subj_idx]

    train_data, train_label = [], []
    test_data, test_labels = [], []
    for subj in train_subjs:
        for i in subj_to_indices.get(subj, []):
            train_data.append(sequences[i])
            train_label.append(targets[i])
    for subj in test_subjs:
        for i in subj_to_indices.get(subj, []):
            test_data.append(sequences[i])
            test_labels.append(targets[i])

    chans = sequences[0].shape[1]
    samples = sequences[0].shape[2]

    model = build_cnn_rnn(chans, samples, subject_list)
    # model.summary()

    train_mask_list = [masks[i] for s in train_subjs for i in subj_to_indices.get(s, [])]
    val_mask_list   = [masks[i] for s in test_subjs for i in subj_to_indices.get(s, [])]

    train_ds = yield_data(train_data, train_label, train_mask_list, 2, True)
    val_ds = yield_data(test_data, test_labels, val_mask_list, 2, False)

    ckpt = ModelCheckpoint(f"/content/drive/MyDrive/fintune_fold{fold+1}_onset.keras", save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[ckpt, early])
    eval_ds = yield_data(test_data, test_labels, val_mask_list, batch_size=1, shuffle=False)
    # model.load_weights("/content/drive/MyDrive/finetune_onset.keras")
    eval_loss = model.evaluate(eval_ds)
    preds = model.predict(eval_ds)

    y_true = []
    y_pred = []

    for (batch_x, batch_y, batch_mask), batch_preds in zip(eval_ds, preds):

        batch_preds = batch_preds.numpy().reshape(-1)
        batch_y = batch_y.numpy().reshape(-1)
        batch_mask = batch_mask.numpy().reshape(-1)

        valid = batch_mask == 1

        y_true.extend(batch_y[valid])
        y_pred.extend((batch_preds[valid] > 0.6).astype(int))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = np.mean(y_true == y_pred)

    print(f"Fold {fold+1} test accuracy:", acc)
    fold_scores.append(acc)


print("All accuracies: ", fold_scores)
print("Mean accuracy:", np.mean(fold_scores))
