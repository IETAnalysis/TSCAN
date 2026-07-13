import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
import zipfile
import struct
import ast
import traceback
import logging

from keras.models import load_model
from keras.layers import Conv1D, GlobalAveragePooling1D, Multiply, Layer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPUs available: {len(gpus)}")
    except RuntimeError as e:
        logging.error(f"Error configuring GPU: {e}")


NPZ_PATH = Path("dataset/Undefended.npz")

MODEL_PATH = "TSCAN_Undefended.h5"

NUM_CLASSES = 0

TRACES_PER_CLASS_TRAIN = 0

TRACES_PER_CLASS_TEST = 0

SEQ_LEN = 0

TEST_START_OFFSET = 0


def _parse_npy_header(buf: bytes):
    assert buf[:6] == b"\x93NUMPY", "Not a .npy file"

    major = buf[6]

    if major == 1:
        hlen = struct.unpack_from("<H", buf, 8)[0]
        header_start = 10
        offset = 10 + hlen
    elif major in (2, 3):
        hlen = struct.unpack_from("<I", buf, 8)[0]
        header_start = 12
        offset = 12 + hlen
    else:
        raise ValueError(f"Unsupported npy version: {major}")

    hdict = ast.literal_eval(
        buf[header_start:offset].decode("latin1").strip()
    )

    return np.dtype(hdict["descr"]), tuple(hdict["shape"]), offset


def _array_info(zip_path: Path, entry: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(entry)
        lh_offset = info.header_offset

    with open(zip_path, "rb") as f:
        f.seek(lh_offset)
        local_header = f.read(30)

        fname_len = struct.unpack_from("<H", local_header, 26)[0]
        extra_len = struct.unpack_from("<H", local_header, 28)[0]

        data_start = lh_offset + 30 + fname_len + extra_len

        f.seek(data_start)
        npy_buf = f.read(4096)

    dtype, shape, npy_header_size = _parse_npy_header(npy_buf)
    array_data_offset = data_start + npy_header_size

    return array_data_offset, dtype, shape


def load_full_array(zip_path: Path, entry: str) -> np.ndarray:
    data_off, dtype, shape = _array_info(zip_path, entry)

    mm = np.memmap(
        str(zip_path),
        dtype=dtype,
        mode="r",
        offset=data_off,
        shape=shape
    )

    result = np.array(mm)
    del mm

    return result


def load_array_rows(zip_path: Path, entry: str, target_rows: np.ndarray) -> np.ndarray:
    data_off, dtype, shape = _array_info(zip_path, entry)

    target_rows = np.asarray(target_rows, dtype=np.int64)

    mm = np.memmap(
        str(zip_path),
        dtype=dtype,
        mode="r",
        offset=data_off,
        shape=shape
    )

    result = np.array(mm[target_rows])
    del mm

    return result


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


def reconstruct_train_indices_like_training(
    zip_path: Path,
    num_classes: int = 70,
    traces_per_class_train: int = 40,
    label_entry: str = "y.npy",
    seed: int = 42
):
    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)

    rng = np.random.RandomState(seed)

    monitored_labels = sorted([lab for lab in np.unique(labels) if lab >= 0])[:num_classes]

    if len(monitored_labels) < num_classes:
        raise ValueError(
            f"可用类别数不足：需要 {num_classes} 类，但只找到 {len(monitored_labels)} 类"
        )

    train_indices = []
    train_labels = []

    for new_label, old_label in enumerate(monitored_labels):
        class_indices = np.where(labels == old_label)[0]

        if len(class_indices) < traces_per_class_train:
            raise ValueError(
                f"类别 {old_label} 样本不足："
                f"需要 {traces_per_class_train} 条，实际只有 {len(class_indices)} 条"
            )

        chosen = rng.choice(
            class_indices,
            size=traces_per_class_train,
            replace=False
        )

        train_indices.extend(chosen.tolist())
        train_labels.extend([new_label] * traces_per_class_train)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)

    return train_indices, train_labels


def get_nonoverlap_test_indices(
    zip_path: Path,
    num_classes: int = 70,
    traces_per_class_test: int = 70,
    traces_per_class_train: int = 40,
    start_offset: int = 130,
    label_entry: str = "y.npy",
    seed: int = 42
):

    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)

    train_indices, _ = reconstruct_train_indices_like_training(
        zip_path=zip_path,
        num_classes=num_classes,
        traces_per_class_train=traces_per_class_train,
        label_entry=label_entry,
        seed=seed
    )

    train_index_set = set(train_indices.tolist())

    monitored_labels = sorted([lab for lab in np.unique(labels) if lab >= 0])[:num_classes]

    if len(monitored_labels) < num_classes:
        raise ValueError(
            f"可用类别数不足：需要 {num_classes} 类，但只找到 {len(monitored_labels)} 类"
        )

    selected_support_indices = []
    selected_support_labels = []
    selected_test_indices = []
    selected_test_labels = []

    for new_label, old_label in enumerate(monitored_labels):
        class_indices = np.where(labels == old_label)[0]

        if len(class_indices) <= start_offset:
            raise ValueError(
                f"类别 {old_label} 样本数量不足，无法从 offset={start_offset} 开始取测试样本。"
            )

        support_candidate_indices = [
            idx for idx in class_indices[:start_offset]
            if idx not in train_index_set
        ]

        if len(support_candidate_indices) < 25:
            raise ValueError(
                f"类别 {old_label} 可用支持样本不足："
                f"最大 N-shot 需要 25 条，排除训练集后只有 "
                f"{len(support_candidate_indices)} 条。"
            )

        test_candidate_indices = [
            idx for idx in class_indices[start_offset:]
            if idx not in train_index_set
        ]

        if len(test_candidate_indices) < traces_per_class_test:
            raise ValueError(
                f"类别 {old_label} 可用测试样本不足："
                f"需要 {traces_per_class_test} 条，"
                f"排除训练集后只剩 {len(test_candidate_indices)} 条。"
            )

        chosen_test = test_candidate_indices[:traces_per_class_test]

        selected_support_indices.extend(support_candidate_indices)
        selected_support_labels.extend([new_label] * len(support_candidate_indices))
        selected_test_indices.extend(chosen_test)
        selected_test_labels.extend([new_label] * traces_per_class_test)

    selected_support_indices = np.asarray(selected_support_indices, dtype=np.int64)
    selected_support_labels = np.asarray(selected_support_labels, dtype=np.int64)
    selected_test_indices = np.asarray(selected_test_indices, dtype=np.int64)
    selected_test_labels = np.asarray(selected_test_labels, dtype=np.int64)

    overlap_train_support = np.intersect1d(train_indices, selected_support_indices)
    overlap_train_test = np.intersect1d(train_indices, selected_test_indices)
    overlap_support_test = np.intersect1d(selected_support_indices, selected_test_indices)

    print("\n========== 训练/支持/测试重叠检查 ==========")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"支持候选样本数: {len(selected_support_indices)}")
    print(f"固定测试集样本数: {len(selected_test_indices)}")
    print(f"训练/支持重叠样本数: {len(overlap_train_support)}")
    print(f"训练/测试重叠样本数: {len(overlap_train_test)}")
    print(f"支持/测试重叠样本数: {len(overlap_support_test)}")

    if len(overlap_train_support) > 0:
        raise RuntimeError("训练集和支持候选集存在重叠，请检查划分逻辑。")

    if len(overlap_train_test) > 0:
        raise RuntimeError("训练集和测试集存在重叠，请检查划分逻辑。")

    if len(overlap_support_test) > 0:
        raise RuntimeError("支持候选集和测试集存在重叠，请检查划分逻辑。")

    expected_test_samples = num_classes * traces_per_class_test
    if len(selected_test_indices) != expected_test_samples:
        raise RuntimeError(
            f"固定测试集数量错误：期望 {expected_test_samples} 条，"
            f"实际 {len(selected_test_indices)} 条。"
        )

    print("训练集、支持候选集和固定测试集互不重叠。")
    print(f"固定测试集总数: {len(selected_test_indices)}")

    return (
        selected_support_indices,
        selected_support_labels,
        selected_test_indices,
        selected_test_labels
    )


def load_fixed_test_pool():
    logging.info("正在加载与训练集不重叠的支持候选集和固定测试集：70 类 × 70 条测试样本...")

    (
        support_indices,
        y_support_pool,
        test_indices,
        y_test_pool
    ) = get_nonoverlap_test_indices(
        zip_path=NPZ_PATH,
        num_classes=NUM_CLASSES,
        traces_per_class_test=TRACES_PER_CLASS_TEST,
        traces_per_class_train=TRACES_PER_CLASS_TRAIN,
        start_offset=TEST_START_OFFSET,
        label_entry="y.npy",
        seed=SEED
    )

    raw_support_traces = load_array_rows(NPZ_PATH, "X.npy", support_indices)
    X_support_pool = build_dual_channel_signed_temporal(raw_support_traces, L=SEQ_LEN)

    raw_test_traces = load_array_rows(NPZ_PATH, "X.npy", test_indices)
    X_test_pool = build_dual_channel_signed_temporal(raw_test_traces, L=SEQ_LEN)

    logging.info(f"支持候选集 X.shape = {X_support_pool.shape}")
    logging.info(f"支持候选集 y.shape = {y_support_pool.shape}")
    logging.info(f"固定测试集 X.shape = {X_test_pool.shape}")
    logging.info(f"固定测试集 y.shape = {y_test_pool.shape}")
    logging.info("输入应为: (样本数, 5000, 2)")

    return X_support_pool, y_support_pool, X_test_pool, y_test_pool


def create_test_set_traffic(
    features_model,
    X_support_pool,
    y_support_pool,
    X_test_pool,
    y_test_pool,
    n_shot,
    type_exp,
    n_instance=70
):
    support_site_dict = {}
    test_site_dict = {}

    for s in range(NUM_CLASSES):
        support_class_indices = np.where(y_support_pool == s)[0]
        test_class_indices = np.where(y_test_pool == s)[0]

        support_site_dict[str(s)] = [
            X_support_pool[idx] for idx in support_class_indices
        ]
        test_site_dict[str(s)] = [
            X_test_pool[idx] for idx in test_class_indices
        ]

    signature_dict = {}
    test_dict = {}

    for s in range(NUM_CLASSES):
        s_str = str(s)
        support_data = support_site_dict[s_str][:]
        test_data = test_site_dict[s_str][:]

        if len(support_data) < n_shot:
            logging.warning(
                f"Class {s_str} has only {len(support_data)} support traces, "
                f"less than n_shot = {n_shot}"
            )
            continue

        if len(test_data) < TRACES_PER_CLASS_TEST:
            logging.warning(
                f"Class {s_str} has only {len(test_data)} test traces, "
                f"less than {TRACES_PER_CLASS_TEST}"
            )
            continue

        random.shuffle(support_data)
        signature_dict[s_str] = support_data[:n_shot]

        test_dict[s_str] = test_data[:min(n_instance, len(test_data))]

        if len(test_dict[s_str]) != TRACES_PER_CLASS_TEST:
            raise RuntimeError(
                f"类别 {s_str} 测试样本数错误："
                f"期望 {TRACES_PER_CLASS_TEST}，"
                f"实际 {len(test_dict[s_str])}。"
            )

    total_test_samples = sum(len(v) for v in test_dict.values())
    expected_test_samples = NUM_CLASSES * TRACES_PER_CLASS_TEST

    if total_test_samples != expected_test_samples:
        raise RuntimeError(
            f"封闭世界测试集总数错误：期望 {expected_test_samples} 条，"
            f"实际 {total_test_samples} 条。"
        )

    signature_vector_dict = {}

    for i in range(NUM_CLASSES):
        i_str = str(i)

        if i_str not in signature_dict or len(signature_dict[i_str]) == 0:
            logging.warning(f"No signature data for class {i_str}")
            continue

        signature_instance = np.array(signature_dict[i_str], dtype=np.float32)

        signature_vector = features_model.predict(
            signature_instance,
            batch_size=32,
            verbose=0
        )

        signature_vector = signature_vector / np.maximum(
            np.linalg.norm(signature_vector, axis=-1, keepdims=True),
            1e-12
        )

        if type_exp == "N-MEV":
            signature_vector = np.array([signature_vector.mean(axis=0)])
            signature_vector = signature_vector / np.maximum(
                np.linalg.norm(signature_vector, axis=-1, keepdims=True),
                1e-12
            )

        signature_vector_dict[i_str] = signature_vector

    test_vector_dict = {}

    for i in range(NUM_CLASSES):
        i_str = str(i)

        if i_str not in test_dict or len(test_dict[i_str]) == 0:
            logging.warning(f"No test data for class {i_str}")
            continue

        test_instance = np.array(test_dict[i_str], dtype=np.float32)

        test_vector = features_model.predict(
            test_instance,
            batch_size=32,
            verbose=0
        )

        test_vector = test_vector / np.maximum(
            np.linalg.norm(test_vector, axis=-1, keepdims=True),
            1e-12
        )

        test_vector_dict[i_str] = test_vector

    return signature_vector_dict, test_vector_dict


def kNN_metrics(signature_vector_dict, test_vector_dict, size_of_problem, n_shot):
    if not signature_vector_dict or not test_vector_dict:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    X_train = []
    y_train = []

    site_labels = list(signature_vector_dict.keys())
    random.shuffle(site_labels)

    tested_sites = site_labels[:min(size_of_problem, len(site_labels))]

    for s in tested_sites:
        for each_idx in range(len(signature_vector_dict[s])):
            X_train.append(signature_vector_dict[s][each_idx])
            y_train.append(s)

    X_test = []
    y_test = []

    for s in tested_sites:
        if s in test_vector_dict:
            for i in range(len(test_vector_dict[s])):
                X_test.append(test_vector_dict[s][i])
                y_test.append(s)

    if len(X_train) == 0 or len(X_test) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    k_for_predict = min(max(1, n_shot), len(X_train))

    knn = KNeighborsClassifier(
        n_neighbors=k_for_predict,
        weights="distance",
        p=2,
        metric="cosine",
        algorithm="brute"
    )

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc_top1 = accuracy_score(y_test, y_pred)
    recall_top1 = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_top1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    k_for_top = min(5, len(X_train))
    _, neighbor_indices = knn.kneighbors(X_test, n_neighbors=k_for_top)

    count_top2 = 0
    count_top5 = 0

    for idx in range(len(X_test)):
        true_label = y_test[idx]
        top_labels = y_train[neighbor_indices[idx]]

        top2_labels = top_labels[:min(2, len(top_labels))]
        top5_labels = top_labels[:min(5, len(top_labels))]

        if true_label in top2_labels:
            count_top2 += 1

        if true_label in top5_labels:
            count_top5 += 1

    acc_top2 = count_top2 / len(X_test)
    acc_top5 = count_top5 / len(X_test)

    return acc_top1, recall_top1, f1_top1, acc_top2, acc_top5


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


def Traffic_Experiment():
    try:
        features_model = load_model(
            MODEL_PATH,
            custom_objects={"SpatialAttentionLayer": GlobalTemporalGating},
            compile=False
        )
        logging.info(f"模型加载成功: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        traceback.print_exc()
        return

    logging.info(f"模型输入 shape: {features_model.input_shape}")
    logging.info(f"模型输出 shape: {features_model.output_shape}")

    expected_input_shape = (None, SEQ_LEN, 2)
    if features_model.input_shape != expected_input_shape:
        logging.warning(
            f"当前模型输入为 {features_model.input_shape}，"
            f"但测试数据输入为 {expected_input_shape}。"
            f"请确认加载的是双通道模型。"
        )

    X_support_pool, y_support_pool, X_test_pool, y_test_pool = load_fixed_test_pool()

    type_exp = "N-MEV"
    SOP_list = [70]
    n_shot_list = [1,5,10,15,20,25]

    results = {}

    for size_of_problem in SOP_list:
        results[size_of_problem] = {}
        logging.info(f"SOP: {size_of_problem}")

        for n_shot in n_shot_list:
            acc_list_top1 = []
            recall_list_top1 = []
            f1_list_top1 = []
            acc_list_top2 = []
            acc_list_top5 = []

            for i in range(10):
                try:
                    signature_vector_dict, test_vector_dict = create_test_set_traffic(
                        features_model=features_model,
                        X_support_pool=X_support_pool,
                        y_support_pool=y_support_pool,
                        X_test_pool=X_test_pool,
                        y_test_pool=y_test_pool,
                        n_shot=n_shot,
                        type_exp=type_exp,
                        n_instance=90
                    )

                    acc1, rec1, f11, acc2, acc5 = kNN_metrics(
                        signature_vector_dict=signature_vector_dict,
                        test_vector_dict=test_vector_dict,
                        size_of_problem=size_of_problem,
                        n_shot=n_shot
                    )

                    acc_list_top1.append(float(f"{acc1:.5f}"))
                    recall_list_top1.append(float(f"{rec1:.5f}"))
                    f1_list_top1.append(float(f"{f11:.5f}"))
                    acc_list_top2.append(float(f"{acc2:.5f}"))
                    acc_list_top5.append(float(f"{acc5:.5f}"))

                    logging.info(
                        f"Iter {i + 1}/10 | "
                        f"N={n_shot} | "
                        f"Top1={acc1:.5f} | "
                        f"Top2={acc2:.5f} | "
                        f"Top5={acc5:.5f}"
                    )

                except Exception as e:
                    logging.warning(f"Error in iteration {i + 1} for n_shot={n_shot}: {e}")
                    traceback.print_exc()
                    continue

            results[size_of_problem][n_shot] = {
                "Top-1 Accuracy": {
                    "mean": np.mean(acc_list_top1),
                    "std": np.std(acc_list_top1)
                },
                "Top-2 Accuracy": {
                    "mean": np.mean(acc_list_top2),
                    "std": np.std(acc_list_top2)
                },
                "Top-5 Accuracy": {
                    "mean": np.mean(acc_list_top5),
                    "std": np.std(acc_list_top5)
                },
                "Recall": {
                    "mean": np.mean(recall_list_top1),
                    "std": np.std(recall_list_top1)
                },
                "F1 Score": {
                    "mean": np.mean(f1_list_top1),
                    "std": np.std(f1_list_top1)
                }
            }

            print(f"\n================ N_shot: {n_shot} ================")
            print(f"Top-1 Accuracy: {str(acc_list_top1).strip('[]')}")
            print(f"Top-2 Accuracy: {str(acc_list_top2).strip('[]')}")
            print(f"Top-5 Accuracy: {str(acc_list_top5).strip('[]')}")
            print(f"Recall: {str(recall_list_top1).strip('[]')}")
            print(f"F1 Score: {str(f1_list_top1).strip('[]')}")

            print(
                f"Top-1 Accuracy Mean: "
                f"{np.mean(acc_list_top1):.5f}, "
                f"Std: {np.std(acc_list_top1):.5f}"
            )
            print(
                f"Top-2 Accuracy Mean: "
                f"{np.mean(acc_list_top2):.5f}, "
                f"Std: {np.std(acc_list_top2):.5f}"
            )
            print(
                f"Top-5 Accuracy Mean: "
                f"{np.mean(acc_list_top5):.5f}, "
                f"Std: {np.std(acc_list_top5):.5f}"
            )
            print(
                f"Recall Mean: "
                f"{np.mean(recall_list_top1):.5f}, "
                f"Std: {np.std(recall_list_top1):.5f}"
            )
            print(
                f"F1 Score Mean: "
                f"{np.mean(f1_list_top1):.5f}, "
                f"Std: {np.std(f1_list_top1):.5f}"
            )

    return results


if __name__ == "__main__":
    Traffic_Experiment()
