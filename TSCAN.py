import os
import random
import math
import zipfile
import struct
import ast
from pathlib import Path

import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import (
    Input, Lambda, Dot, Dense, Conv1D, GlobalAveragePooling1D,
    MaxPooling1D, Dropout, Flatten, Activation, Multiply, Add,
    Reshape, ELU, SeparableConv1D, Layer
)
from keras import optimizers
from keras.callbacks import CSVLogger


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def _parse_npy_header(buf: bytes):
    assert buf[:6] == b"\x93NUMPY", "Not a .npy file"
    major = buf[6]

    if major == 1:
        hlen = struct.unpack_from("<H", buf, 8)[0]
        offset = 10 + hlen
        header_start = 10
    else:
        hlen = struct.unpack_from("<I", buf, 8)[0]
        offset = 12 + hlen
        header_start = 12

    hdict = ast.literal_eval(buf[header_start:offset].decode("latin1").strip())
    return np.dtype(hdict["descr"]), tuple(hdict["shape"]), offset


def _array_info(zip_path: Path, entry: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(entry)
        lh_offset = info.header_offset

    with open(zip_path, "rb") as f:
        f.seek(lh_offset)
        lh = f.read(30)
        fname_len = struct.unpack_from("<H", lh, 26)[0]
        extra_len = struct.unpack_from("<H", lh, 28)[0]
        data_start = lh_offset + 30 + fname_len + extra_len
        f.seek(data_start)
        npy_buf = f.read(512)

    dtype, shape, npy_hdr_size = _parse_npy_header(npy_buf)
    array_data_offset = data_start + npy_hdr_size
    return array_data_offset, dtype, shape


def load_array_rows(zip_path: Path, entry: str, target_rows: np.ndarray) -> np.ndarray:
    data_off, dtype, shape = _array_info(zip_path, entry)
    target_rows = np.asarray(target_rows, dtype=np.int64)

    mm = np.memmap(str(zip_path), dtype=dtype, mode="r", offset=data_off, shape=shape)
    result = np.array(mm[target_rows])
    del mm
    return result


def load_full_array(zip_path: Path, entry: str) -> np.ndarray:
    data_off, dtype, shape = _array_info(zip_path, entry)
    mm = np.memmap(str(zip_path), dtype=dtype, mode="r", offset=data_off, shape=shape)
    result = np.array(mm)
    del mm
    return result


NPZ_PATH = Path("dataset/WTF-PAD.npz")

NUM_CLASSES = 0
TRACES_PER_CLASS = 0
SEQ_LEN = 0

alpha = 0
batch_size_value = 0
emb_size = 0
number_epoch = 0
learning_rate = 0


def get_label_based_indices(
    zip_path: Path,
    num_classes: int = 70,
    traces_per_class: int = 40,
    label_entry: str = "y.npy"
):
    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)

    selected_indices = []
    selected_labels = []

    monitored_labels = sorted([lab for lab in np.unique(labels) if lab >= 0])[:num_classes]

    if len(monitored_labels) < num_classes:
        raise ValueError(f"可用类别数不足：需要 {num_classes} 类，但只找到 {len(monitored_labels)} 类")

    for new_label, old_label in enumerate(monitored_labels):
        class_indices = np.where(labels == old_label)[0]

        if len(class_indices) < traces_per_class:
            raise ValueError(
                f"类别 {old_label} 样本不足：需要 {traces_per_class} 条，实际只有 {len(class_indices)} 条"
            )

        chosen = np.random.choice(class_indices, size=traces_per_class, replace=False)
        selected_indices.extend(chosen.tolist())
        selected_labels.extend([new_label] * traces_per_class)

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selected_labels = np.asarray(selected_labels, dtype=np.int64)

    perm = np.random.permutation(len(selected_indices))
    return selected_indices[perm], selected_labels[perm]


def build_dual_channel_signed_temporal(traces: np.ndarray, L: int = 5000) -> np.ndarray:

    traces = np.asarray(traces, dtype=np.float32)

    if traces.ndim != 2:
        raise ValueError(f"期望输入为二维数组 (N, raw_len)，但得到 shape={traces.shape}")

    if traces.shape[1] >= L:
        x = traces[:, :L]
    else:
        pad_width = L - traces.shape[1]
        x = np.pad(traces, ((0, 0), (0, pad_width)), mode="constant")

    direction = np.sign(x).astype(np.float32)
    temporal = np.abs(x).astype(np.float32)

    X = np.stack([direction, temporal], axis=-1).astype(np.float32)
    return X


print("=== 正在按 label 加载 WF1 预训练数据 ===")
indices, all_class = get_label_based_indices(
    NPZ_PATH,
    num_classes=NUM_CLASSES,
    traces_per_class=TRACES_PER_CLASS,
    label_entry="y.npy"
)

raw_traces = load_array_rows(NPZ_PATH, "X.npy", indices)
print(f"原始 X shape: {raw_traces.shape}")
print(f"原始 label shape: {all_class.shape}")

all_traces = build_dual_channel_signed_temporal(raw_traces, L=SEQ_LEN)

print(f"双通道输入 shape: {all_traces.shape}")
print("应为: (样本数, 5000, 2)")


total_trace = NUM_CLASSES * TRACES_PER_CLASS
id_list = list(range(total_trace))


def build_pos_pairs_for_class(class_id):
    class_indices = np.where(all_class == class_id)[0].tolist()
    pos_pairs = [
        (class_indices[i], class_indices[j])
        for i in range(len(class_indices))
        for j in range(i + 1, len(class_indices))
    ]
    random.shuffle(pos_pairs)
    return pos_pairs


def build_positive_pairs(class_id_range):
    listX1, listX2 = [], []

    for class_id in class_id_range:
        pos = build_pos_pairs_for_class(class_id)
        for pair in pos:
            listX1.append(pair[0])
            listX2.append(pair[1])

    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]


Xa_train, Xp_train = build_positive_pairs(range(NUM_CLASSES))
all_traces_train_idx = np.arange(len(all_traces))

print(f"Anchor 数量: {Xa_train.shape}")
print(f"Positive 数量: {Xp_train.shape}")


def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def cosine_triplet_loss(X):
    positive_sim, negative_sim = X
    losses = tf.maximum(0.0, negative_sim - positive_sim + float(alpha))
    return tf.reduce_mean(losses)


def build_similarities(conv, all_imgs):
    embs = conv.predict(all_imgs, batch_size=32, verbose=0)
    norms = np.linalg.norm(embs, axis=-1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-12)
    return np.dot(embs, embs.T)


def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    if similarities is None:
        final_neg = []
        for anc_idx in anc_idxs:
            anchor_class = all_class[anc_idx]
            candidates = [idx for idx in neg_imgs_idx if all_class[idx] != anchor_class]
            final_neg.append(random.choice(candidates))
        return final_neg

    final_neg = []

    for anc_idx, pos_idx in zip(anc_idxs, pos_idxs):
        anchor_class = all_class[anc_idx]
        sim_ap = similarities[anc_idx, pos_idx]

        possible_ids = np.where(similarities[anc_idx] > sim_ap - float(alpha))[0]
        possible_ids = [
            idx for idx in possible_ids
            if idx in neg_imgs_idx and all_class[idx] != anchor_class
        ]

        if possible_ids:
            final_neg.append(random.choice(possible_ids))
        else:
            candidates = [idx for idx in neg_imgs_idx if all_class[idx] != anchor_class]
            final_neg.append(random.choice(candidates))

    return final_neg


class SemiHardTripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.neg_traces_idx = list(neg_traces_idx)
        self.num_samples = len(Xa_train)

        self.similarities = None
        if conv is not None:
            self.similarities = build_similarities(conv, self.traces)

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.num_samples)

        traces_a = self.Xa[start:end]
        traces_p = self.Xp[start:end]
        traces_n = build_negatives(
            traces_a,
            traces_p,
            self.similarities,
            self.neg_traces_idx
        )

        anchor = tf.convert_to_tensor(self.traces[traces_a], dtype=tf.float32)
        positive = tf.convert_to_tensor(self.traces[traces_p], dtype=tf.float32)
        negative = tf.convert_to_tensor(self.traces[traces_n], dtype=tf.float32)

        anchor = tf.ensure_shape(anchor, [None, SEQ_LEN, 2])
        positive = tf.ensure_shape(positive, [None, SEQ_LEN, 2])
        negative = tf.ensure_shape(negative, [None, SEQ_LEN, 2])

        y = tf.zeros((len(traces_a),), dtype=tf.float32)

        return {"anchor": anchor, "positive": positive, "negative": negative}, y


class GlobalTemporalGating(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1d = Conv1D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid"
        )

    def call(self, inputs):
        avg_pool = GlobalAveragePooling1D(keepdims=True)(inputs)
        max_pool = tf.reduce_max(inputs, axis=1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attn = self.conv1d(concat)
        return Multiply()([inputs, attn])


def eca_block(inputs, b=1, gamma=2):
    in_channel = int(inputs.shape[-1])

    kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    x = GlobalAveragePooling1D()(inputs)
    x = Reshape((in_channel, 1))(x)
    x = Conv1D(1, kernel_size, padding="same", use_bias=False)(x)
    x = Activation("sigmoid")(x)
    x = Reshape((1, in_channel))(x)

    return Multiply()([inputs, x])


def residual_block(inputs, filters, kernel_size=8, strides=1):
    shortcut = inputs

    x = Conv1D(filters, kernel_size, strides=strides, padding="same")(inputs)
    x = Activation("relu")(x)
    x = Conv1D(filters, kernel_size, strides=strides, padding="same")(x)
    x = Activation("relu")(x)

    if strides != 1 or int(inputs.shape[-1]) != filters:
        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding="same")(shortcut)

    return Add()([x, shortcut])


def TSCAN_Encoder(input_shape=(5000, 2), emb_size=64):
    input_data = Input(shape=input_shape)

    x = SeparableConv1D(32, 8, strides=1, padding="same")(input_data)
    x = ELU(alpha=1.0)(x)
    x = SeparableConv1D(32, 8, strides=1, padding="same")(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling1D(pool_size=8, strides=4, padding="same")(x)
    x = Dropout(0.1)(x)
    x = Add()([x, eca_block(x)])

    x = residual_block(x, 64)
    x = MaxPooling1D(8, strides=4, padding="same")(x)
    x = Dropout(0.1)(x)
    x = Add()([x, eca_block(x)])

    x = residual_block(x, 128)
    x = GlobalTemporalGating()(x)
    x = MaxPooling1D(8, strides=4, padding="same")(x)
    x = Dropout(0.1)(x)
    x = Add()([x, eca_block(x)])

    x = residual_block(x, 256)
    x = MaxPooling1D(8, strides=4, padding="same")(x)

    x = Flatten()(x)
    output = Dense(emb_size, name="FeaturesVec")(x)

    return Model(inputs=input_data, outputs=output, name="TSCAN_Encoder")


shared_encoder = TSCAN_Encoder(input_shape=(SEQ_LEN, 2), emb_size=emb_size)

anchor = Input((SEQ_LEN, 2), name="anchor")
positive = Input((SEQ_LEN, 2), name="positive")
negative = Input((SEQ_LEN, 2), name="negative")

a = shared_encoder(anchor)
p = shared_encoder(positive)
n = shared_encoder(negative)

pos_sim = Dot(axes=-1, normalize=True)([a, p])
neg_sim = Dot(axes=-1, normalize=True)([a, n])

loss = Lambda(cosine_triplet_loss, name="triplet_loss")([pos_sim, neg_sim])

model_triplet = Model(
    inputs=[anchor, positive, negative],
    outputs=loss,
    name="TSCA_Model"
)

model_triplet.compile(
    loss=identity_loss,
    optimizer=optimizers.Adam(learning_rate=learning_rate)
)

print(model_triplet.summary())


os.makedirs("log", exist_ok=True)
csv_logger = CSVLogger(
    "log/Training_Log_TSCAN.csv",
    append=True,
    separator=";"
)

steps_per_epoch = len(Xa_train) // batch_size_value

for epoch in range(number_epoch):
    print(f"\nEpoch {epoch + 1}/{number_epoch}")

    gen_hard = SemiHardTripletGenerator(
        Xa_train=Xa_train,
        Xp_train=Xp_train,
        batch_size=batch_size_value,
        all_traces=all_traces,
        neg_traces_idx=all_traces_train_idx,
        conv=shared_encoder if epoch > 0 else None
    )

    model_triplet.fit(
        gen_hard,
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        verbose=1,
        callbacks=[csv_logger]
    )


save_dir = "trained_model"
os.makedirs(save_dir, exist_ok=True)

shared_encoder.save(os.path.join(save_dir, "TSCAN_WTF-PAD.h5"))

print("=== 预训练完成！模型已保存 ===")
