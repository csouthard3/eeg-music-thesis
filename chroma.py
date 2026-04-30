import os
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
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
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'cm'
     
subjects = ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']
original_window_samples = int(400 / 1000 * 128)
step_samples = int(200/1000 * 128)
output_dim = 12 #12 outputs per window representing the likelihood of each chroma class

np.random.seed(42)
tf.random.set_seed(42)


class EEGWindowGenerator(Sequence): #initially the model.fit call was overloading colab gpu vram, but calling this class on the data before allows it to be chunked into more doable sizes

    def __init__(self, X, y, mask, indices, batch_window, shuffle):
        super().__init__()
        self.X = X
        self.y = y
        self.mask = mask
        self.indices = indices
        self.batch_size = batch_window
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_inds = self.indices[idx*self.batch_size : (idx+1) * self.batch_size]
        X_batch = self.X[batch_inds]
        y_batch = self.y[batch_inds]
        mask_batch = self.mask[batch_inds]
        # return X_batch,  y_batch
        return X_batch, y_batch, mask_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def imagination_dataset_generator(sequences, targets, masks):

    for seq, tgt, mask in zip(sequences, targets, masks):
        tgt = np.asarray(tgt)
        if tgt.ndim == 1:
            tgt = tgt[:, np.newaxis]
        yield seq, tgt.astype(np.int64), mask.astype(np.float32)


def yield_data(sequences, targets, masks, batch_size, shuffle):
    chans = sequences[0].shape[1]
    samples = sequences[0].shape[2]

    output_signature = (
        tf.TensorSpec(shape=(None, chans, samples, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(lambda: imagination_dataset_generator(sequences, targets, masks), output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(16 , len(sequences)))

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            [None, chans, samples, 1],
            [None, 1],
            [None]),
        padding_values=(
            0.0,
            np.int64(0),
            0.0
        )
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

def load_subject(subj):
    openmiir_path = os.path.join('/content', f"{subj}-raw_data_v2.npz")
    print(openmiir_path)
    eeg = np.load(openmiir_path)
    return eeg["data"], eeg["labels"]

def extract_midi_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.pitch))
    notes.sort(key=lambda x: x[0])
    return notes


def eeg_windows(trial_data):

    length = trial_data.shape[1]
    starts = list(range(0, length - original_window_samples + 1, step_samples))
    windows = []
    for start in starts:
        windows.append(trial_data[:, start:start+original_window_samples].astype(np.float32))
    return np.stack(windows)


def create_chroma_labels(length, midi_notes):

    note_starts = np.array([int(t * 128) for t, _ in midi_notes], dtype=np.int64)

    num_windows = 1 + (length - original_window_samples) // step_samples
    starts = np.arange(0, num_windows * step_samples, step_samples)
    centers = starts + original_window_samples // 2
    y = np.zeros((num_windows,), dtype=np.int64)
    mask = np.zeros((num_windows,), dtype=np.float32)

    for i in range(len(centers)):

        center = centers[i]
        start = starts[i]
        end = start + original_window_samples

        in_window = np.where((note_starts >= start) & (note_starts <= end))[0]

        if len(in_window) > 0:
            chromas = np.array([p % 12 for _, p in midi_notes], dtype=np.int64)
            closest = in_window[np.argmin(np.abs(note_starts[in_window] - center))]  #pick note closest to center of window, we found this resulted in most recognizable midi generation
            y[i] = chromas[closest]
            mask[i] = 1.0 #only label/train on windows that have a note onset, otherwise it seems the model "cheated" with a sort of offset-sustain behavior that's described in the thesis

    return y.reshape(-1, 1), mask.reshape(-1, 1)




def pretrain_dataset(subjects, note_dict):

    windows = []
    targets = []
    subjects_list = []
    masks = []

    for subj in subjects:
        eeg, labels = load_subject(subj)
        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx])  #event ids are stim_id concatenated with condition_num (see openmiir paper for details)
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 1: #only listening trials for pretraining, which have condition num of 1
                continue

            trial = eeg[trial_idx] 
            raw_windows = eeg_windows(trial)

            notes = note_dict[f"{stim_id}_v1"] if subj in ['P01','P04','P06','P07'] else note_dict[f"{stim_id}_v2"]
            length = trial.shape[1]
            y, mask = create_chroma_labels(length, notes)

            if y.shape[0] != raw_windows.shape[0]:
                nmin = min(y.shape[0], raw_windows.shape[0])
                raw_windows = raw_windows[:nmin]
                y = y[:nmin]
                mask = mask[:nmin]

            pad_width = 128 - original_window_samples  #eegnet is trained specifically for one second (128 sample) windows, so we pad our shorter windows to that length
            for i, w in enumerate(raw_windows):
                if mask[i] == 0:
                    continue

                w_padded = np.pad(w, ((0,0),(0, pad_width)), constant_values=0.0)

                windows.append(w_padded)
                targets.append(y[i])
                masks.append(mask[i])
                subjects_list.append(subj)


    X_raw = np.stack(windows).astype(np.float32)
    X_raw = X_raw[..., np.newaxis]

    mean = X_raw.mean(axis=(0,1,2), keepdims = True)
    std  = X_raw.std(axis=(0,1,2), keepdims = True)
    np.savez("/content/drive/MyDrive/global_norm_chroma.npz", mean=mean, std=std)

    Xs = (X_raw - mean) / (std + 1e-12)
    y = np.stack(targets).astype(np.int64)
    mask = np.stack(masks).astype(np.float32)
    return Xs, y, mask, subjects_list


def training_dataset(subjects, note_dict, mean, std):

    valid_subjects = []
    sequences = []
    targets = []
    meta = []
    masks = []

    for subj in subjects:
        eeg, labels = load_subject(subj)
        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx])  #event ids are stim_id concatenated with condition_num (see openmiir paper for details)
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 2:
                continue

            print(condition_num)

            trial = eeg[trial_idx]
            length = trial.shape[1]

            raw_windows = eeg_windows(trial)
 
            pad_width = 128 - original_window_samples  #eegnet is trained specifically for one second (128 sample) windows, so we pad our shorter windows to that length
            win_list = []
            for w in raw_windows:
                w_padded = np.pad(w, ((0,0),(0,pad_width)), constant_values=0.0)
                w_padded = w_padded[np.newaxis, ... , np.newaxis]
                w_norm = (w_padded - mean) / (std+1e-12)
                win_list.append(w_norm[0])
            windows = np.stack(win_list).astype(np.float32)

            if windows.shape[0] > 150: #code was overloading colab vram without a sequence cap, but 150 timesteps should be enough to capture an entire trial
                windows = windows[:150]

            notes = note_dict[f"{stim_id}_v1"] if subj in ['P01','P04','P06','P07'] else note_dict[f"{stim_id}_v2"]

            y, mask = create_chroma_labels(length, notes)

            if y.shape[0] != windows.shape[0]:
                nmin = min(y.shape[0], windows.shape[0])
                windows = windows[:nmin]
                y = y[:nmin]
                mask = mask[:nmin]

            if windows.shape[0] < 10:
                continue

            sequences.append(windows.astype(np.float32))
            targets.append(y.squeeze().astype(np.int64)) 
            meta.append((subj,  trial_idx, int(stim_id)))
            masks.append(mask.squeeze(-1).astype(np.float32))
    
    return sequences, targets, masks, meta



def pretrain_eegnet(X_listen, y_listen, mask_listen, subjects_list):

    train_subjects, val_subjects = train_test_split(subjects_list, test_size=0.1, random_state=42) #could test different test sizes for pretrain

    train_inds = [i for i, s in enumerate(subjects_list) if s in train_subjects]
    val_inds = [i for i, s in enumerate(subjects_list) if s in val_subjects]

    chans = X_listen.shape[1]
    samples = X_listen.shape[2]
    base = EEGNet(nb_classes=12, Chans=chans, Samples=samples)
    flatten_layer = base.get_layer('flatten').output
    out = Dense(12, activation='softmax')(base.get_layer('flatten').output)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss = 'sparse_categorical_crossentropy',)

    checkpoint = ModelCheckpoint("/content/drive/MyDrive/pretrain_chroma.keras", save_best_only=True,mode='min')
    early = EarlyStopping(patience=10, mode='min', restore_best_weights=True)

    train_gen = EEGWindowGenerator(X_listen, y_listen, mask_listen, train_inds, 64, True)
    val_gen = EEGWindowGenerator(X_listen, y_listen, mask_listen, val_inds, 64, False)
    model.fit(train_gen, validation_data=val_gen, epochs=80, callbacks=[checkpoint, early]) #80 epochs was chosen somewhat arbitrarily



def build_cnn_rnn(chans, samples, subject_list):
    
    base = EEGNet(nb_classes=output_dim, Chans=chans, Samples=samples)
    base.load_weights("/content/drive/MyDrive/pretrain_chroma.keras")

    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # flatten_layer = base.get_layer(-3)
    flatten_layer = base.get_layer('flatten') #layer right before final classification layer
    eegnet = Model(inputs=base.input, outputs=flatten_layer.output)
    seq_input = Input(shape=(None, chans, samples, 1), name='seq_input')

    feats = TimeDistributed(eegnet, name='eegnet_time_dist')(seq_input)
    masked_feats = Masking(mask_value=0.0)(feats)
    x = Bidirectional(LSTM(64, return_sequences=True))(masked_feats) #we chose 64 units for lstm because it seems pretty standard
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    out = TimeDistributed(Dense(output_dim, activation='softmax'))(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss = 'sparse_categorical_crossentropy') #Adam optimizer chosen in the hope of performing well on sparse data, but it might be valuable to test other optimizers too
    return model


def circular_distance_mod12(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    return np.minimum(diff, 12 - diff)




note_dict = {}
stim_ids = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]

for stim_id in stim_ids:
    #in openmiir data collection, tempo was shifted between first and second half of participants so the midi is slightly different
    v1 = os.path.join('/content', f"stim_{stim_id}_v1.mid")
    v2 = os.path.join('/content', f"stim_{stim_id}_v2.mid")
    note_dict[f"{stim_id}_v1"] = extract_midi_notes(v1)
    note_dict[f"{stim_id}_v2"] = extract_midi_notes(v2)

X_listen, y_listen, mask_listen, subjects_list = pretrain_dataset(subjects, note_dict, step_samples)
pretrain_eegnet(X_listen, y_listen, mask_listen, subjects_list)

norm = np.load("/content/drive/MyDrive/global_norm_chroma.npz")
mean_pre = norm["mean"]
std_pre  = norm["std"]

sequences, targets, masks, meta = training_dataset(subjects, note_dict, mean=mean_pre, std=std_pre)

all_targets = np.concatenate(targets)

zero_baseline_mse = np.mean(all_targets ** 2)
mean_value = np.mean(all_targets)
mean_baseline_mse = np.mean((all_targets - mean_value) ** 2)

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

chance_acc = 1.0 / 12.0

masked_targets = flat_targets.astype(float)

zero_baseline_mse = np.mean(masked_targets ** 2)
mean_value = np.mean(masked_targets)
mean_baseline_mse = np.mean((masked_targets - mean_value)**2)

print("classes:", dict(counts))
print("majority class baseline accuracy:", majority_acc)
print("total evaluated windows:", total)

subj_to_indices = {}
for i, (subj, trial_idx, stim_id) in enumerate(meta):
    subj_to_indices.setdefault(subj,  []).append(i)

subject_list = np.array(subjects)
kf = KFold(n_splits=len(subjects))
fold_scores = []
fold_stim_scores = []

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
    val_mask_list = [masks[i] for s in test_subjs for i in subj_to_indices.get(s, [])]

    train_ds = yield_data(train_data, train_label, train_mask_list, 2, True)
    val_ds = yield_data(test_data, test_labels, val_mask_list, 2, False)

    ckpt = ModelCheckpoint(f"/content/drive/MyDrive/finetune_fold{fold+1}_chroma.keras", save_best_only=True)
    early = EarlyStopping(patience=6, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[ckpt, early])

    eval_ds = yield_data(test_data, test_labels, val_mask_list, 1, False)
    # model.load_weights("/content/drive/MyDrive/finetune_chroma.keras")
    eval_loss = model.evaluate(eval_ds)

    preds = model.predict(eval_ds)

    y_true_all = []
    y_pred_all = []

    for (x, y, w), p in zip(eval_ds, preds):
        y = y.numpy().flatten()
        w = w.numpy().flatten()
        p = np.argmax(p, axis=-1).flatten()

        keep = w == 1
        y_true_all.append(y[keep])
        y_pred_all.append(p[keep])

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = np.mean(y_true == y_pred)

    circ_dist = circular_distance_mod12(y_true, y_pred)
    mean_circ = np.mean(circ_dist)

    print(f"Accuracy: {acc:.3f}")
    print(f"Mean circular distance: {mean_circ:.3f} (chance = 3.0)")

    print(f"Fold {fold+1} test accuracy:", acc)
    fold_scores.append(acc)

    plt.figure(figsize=(12,4))
    plt.plot(y_true_all[0], label='True chroma')
    plt.plot(y_pred_all[0], label='Predicted chroma')
    plt.xlabel("Unmasked windows")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Chroma class")
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [r'C', r'C\#', r'D', r'D\#', r'E', r'F', r'F\#', r'G', r'G\#', r'A', r'A\#', r'B'])
    plt.legend()
    plt.title("Example True vs. Predicted Chroma (Current)")
    plt.savefig('/content/drive/MyDrive/current_exampe_prediction_' + str(fold+1) + '_chroma.pdf')
    plt.savefig('/content/drive/MyDrive/current_exampe_prediction_' + str(fold+1) + '_chroma.svg')
    # plt.show()    
    plt.close()


print("All accuracies:", fold_scores)
print("Mean accuracy: ", np.mean(fold_scores))

