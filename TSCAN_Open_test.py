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
import csv

from keras.models import load_model
from keras.layers import Conv1D, GlobalAveragePooling1D, Multiply, Layer
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

DATA_NPZ = Path("dataset/WTF-PAD.npz")

OW_NPZ = DATA_NPZ

MODEL_PATH = Path("TSCAN_WTF-PAD.h5")
CSV_PATH = Path("TSCAN_WTF-PAD_OpenWorld_results_.csv")

WF1_SOURCE_LABELS = tuple(range(0, 0))
WF2_SOURCE_LABELS = tuple(range(0, 0))
OW_UNKNOWN_LABELS = tuple(range(0, 0))

WF1_NUM_CLASSES = len(WF1_SOURCE_LABELS)
WF2_NUM_CLASSES = len(WF2_SOURCE_LABELS)

TRACES_PER_CLASS_TRAIN = 0

MAX_N_SHOT = 0
FIXED_KNOWN_TEST_PER_CLASS = 0
MONITORED_POOL_PER_CLASS = MAX_N_SHOT + FIXED_KNOWN_TEST_PER_CLASS

WF2_TEST_START_OFFSET = 0

OW_TRACES_PER_CLASS = 0
NUM_OW_TRACES = len(OW_UNKNOWN_LABELS) * OW_TRACES_PER_CLASS
BALANCE_OW_ACROSS_LABELS = 0
UNMONITORED_LABEL = 0

SEQ_LEN = 0

N_SHOT_LIST = [1, 5, 10, 15, 20, 25]
REPEAT_TIMES = 0

TYPE_EXP = "N-MEV"
SIMILARITY_THRESHOLD = 0

OW_SELECTION_MODE = "label_is_unknown"
OW_LABEL_NAMESPACE_MATCHES_DATA = 0

MONITORED_DEFENSE_NAME = "WTF-PAD"
UNMONITORED_DEFENSE_NAME = "WTF-PAD"
REQUIRE_SAME_DEFENSE = 0

if max(N_SHOT_LIST) > MAX_N_SHOT:
    raise ValueError(
        f"N_SHOT_LIST 中存在大于 MAX_N_SHOT={MAX_N_SHOT} 的取值。"
    )

if MONITORED_POOL_PER_CLASS < MAX_N_SHOT + FIXED_KNOWN_TEST_PER_CLASS:
    raise ValueError(
        "监控类样本池不足：至少需要 "
        f"{MAX_N_SHOT + FIXED_KNOWN_TEST_PER_CLASS} 条/类，"
        f"当前只有 {MONITORED_POOL_PER_CLASS} 条/类。"
    )

if NUM_OW_TRACES != len(OW_UNKNOWN_LABELS) * OW_TRACES_PER_CLASS:
    raise ValueError("未监控样本总数与每类抽样数不一致。")

def _npz_key(entry: str) -> str:
    return entry[:-4] if entry.endswith(".npy") else entry

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

def _array_info(zip_path: Path, entry: str = "X.npy"):
    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(entry)

        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{zip_path} 中的 {entry} 是压缩存储，不能直接 memmap。"
            )

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
    try:
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

    except Exception as e:
        logging.warning(
            f"memmap 加载 {zip_path}:{entry} 失败，改用 np.load。原因: {e}"
        )
        key = _npz_key(entry)
        with np.load(zip_path, allow_pickle=False) as data:
            return np.asarray(data[key])

def load_array_rows(zip_path: Path, entry: str, target_rows: np.ndarray) -> np.ndarray:
    target_rows = np.asarray(target_rows, dtype=np.int64)

    try:
        data_off, dtype, shape = _array_info(zip_path, entry)

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

    except Exception as e:
        logging.warning(
            f"memmap 按行加载 {zip_path}:{entry} 失败，改用 np.load。原因: {e}"
        )
        key = _npz_key(entry)
        with np.load(zip_path, allow_pickle=False) as data:
            return np.asarray(data[key][target_rows])

def get_array_shape(zip_path: Path, entry: str):
    try:
        _, _, shape = _array_info(zip_path, entry)
        return shape

    except Exception:
        key = _npz_key(entry)
        with np.load(zip_path, allow_pickle=False) as data:
            return data[key].shape

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

def get_known_labels(labels: np.ndarray):
    return sorted([lab for lab in np.unique(labels) if lab >= 0])

def get_wf_source_labels(
    labels: np.ndarray,
    wf1_num_classes: int,
    wf2_num_classes: int
):
    """按明确的原始标签范围返回 WF1 与 WF2 标签。"""
    available = set(get_known_labels(labels))

    wf1_labels = np.asarray(WF1_SOURCE_LABELS, dtype=np.int64)
    wf2_labels = np.asarray(WF2_SOURCE_LABELS, dtype=np.int64)

    if len(wf1_labels) != wf1_num_classes:
        raise ValueError(
            f"WF1 配置不一致：期望 {wf1_num_classes} 类，"
            f"实际配置 {len(wf1_labels)} 类。"
        )

    if len(wf2_labels) != wf2_num_classes:
        raise ValueError(
            f"WF2 配置不一致：期望 {wf2_num_classes} 类，"
            f"实际配置 {len(wf2_labels)} 类。"
        )

    required_labels = (
        set(wf1_labels.tolist())
        | set(wf2_labels.tolist())
        | set(OW_UNKNOWN_LABELS)
    )
    missing = sorted(required_labels - available)
    if missing:
        raise ValueError(
            f"数据文件中缺少配置所需的原始标签：{missing}"
        )

    return wf1_labels, wf2_labels

def _same_path(path_a: Path, path_b: Path) -> bool:
    return path_a.expanduser().resolve(strict=False) == path_b.expanduser().resolve(strict=False)

def validate_open_world_configuration():
    if REQUIRE_SAME_DEFENSE and MONITORED_DEFENSE_NAME != UNMONITORED_DEFENSE_NAME:
        raise ValueError(
            "开放世界评测被拒绝：监控类与未监控类的防御域不同："
            f"{MONITORED_DEFENSE_NAME!r} != {UNMONITORED_DEFENSE_NAME!r}。"
        )

    if OW_SELECTION_MODE not in {"outside_wf1_wf2", "label_is_unknown"}:
        raise ValueError(
            "OW_SELECTION_MODE 只能是 'outside_wf1_wf2' 或 'label_is_unknown'。"
        )

    if (
        OW_SELECTION_MODE == "outside_wf1_wf2"
        and not _same_path(DATA_NPZ, OW_NPZ)
        and not OW_LABEL_NAMESPACE_MATCHES_DATA
    ):
        raise ValueError(
            "OW_NPZ 与 DATA_NPZ 不同，且标签空间不一致，无法按 WF1/WF2 标签排除污染。"
            "请把 OW_SELECTION_MODE 改为 'label_is_unknown'，或统一两个文件的网站标签编号。"
        )

def sample_strict_unmonitored_indices(
    zip_path: Path,
    num_traces: int,
    monitored_source_labels: np.ndarray,
    selection_mode: str = "outside_wf1_wf2",
    unknown_labels=(-1,),
    label_entry: str = "y.npy",
    seed: int = 42,
    balance_across_unknown_labels: bool = False
):
    """
    只从标签确认的未监控候选中抽样，并对 WF1/WF2 污染执行 fail-closed 检查。

    outside_wf1_wf2：候选标签不属于 WF1/WF2。
    label_is_unknown：候选标签必须属于 unknown_labels（通常为 -1）。
    """
    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)
    x_rows = get_array_shape(zip_path, "X.npy")[0]

    if len(labels) != x_rows:
        raise ValueError(
            f"{zip_path} 的 X/y 行数不一致：X={x_rows}, y={len(labels)}。"
        )

    monitored_source_labels = np.asarray(monitored_source_labels, dtype=np.int64)

    if selection_mode == "outside_wf1_wf2":
        candidate_mask = ~np.isin(labels, monitored_source_labels)
    elif selection_mode == "label_is_unknown":
        unknown_labels = np.asarray(tuple(unknown_labels), dtype=np.int64)
        candidate_mask = np.isin(labels, unknown_labels)
    else:
        raise ValueError(f"未知 selection_mode: {selection_mode}")

    candidate_indices = np.flatnonzero(candidate_mask)
    excluded_monitored_rows = int(np.sum(np.isin(labels, monitored_source_labels)))

    logging.info("========== OW label audit ==========")
    logging.info(f"OW source: {zip_path}")
    logging.info(f"OW selection mode: {selection_mode}")
    logging.info(f"Total rows in OW source: {len(labels)}")
    logging.info(f"Rows carrying WF1/monitored labels: {excluded_monitored_rows}")
    logging.info(f"Strict unmonitored candidates: {len(candidate_indices)}")

    if len(candidate_indices) < num_traces:
        raise ValueError(
            f"严格过滤后未监控样本不足：需要 {num_traces} 条，"
            f"但 {zip_path} 中只有 {len(candidate_indices)} 条候选。"
            "不要回退到未防御 OW.npz；应先用相同防御生成更多未监控流量。"
        )

    rng = np.random.RandomState(seed)

    if balance_across_unknown_labels:
        if selection_mode != "label_is_unknown":
            raise ValueError(
                "balance_across_unknown_labels=True 仅适用于 label_is_unknown 模式。"
            )

        unknown_labels_arr = np.asarray(tuple(unknown_labels), dtype=np.int64)
        if len(unknown_labels_arr) == 0:
            raise ValueError("unknown_labels 不能为空。")
        if num_traces % len(unknown_labels_arr) != 0:
            raise ValueError(
                f"num_traces={num_traces} 不能被未知类别数 "
                f"{len(unknown_labels_arr)} 整除，无法均衡抽样。"
            )

        per_unknown_class = num_traces // len(unknown_labels_arr)
        selected_parts = []

        for source_label in unknown_labels_arr:
            class_candidates = np.flatnonzero(labels == source_label)
            if len(class_candidates) < per_unknown_class:
                raise ValueError(
                    f"未知标签 {source_label} 只有 {len(class_candidates)} 条，"
                    f"均衡抽样需要 {per_unknown_class} 条。"
                )
            selected_parts.append(
                rng.choice(
                    class_candidates,
                    size=per_unknown_class,
                    replace=False
                )
            )

        selected_indices = np.concatenate(selected_parts).astype(np.int64)
        rng.shuffle(selected_indices)
    else:
        selected_indices = rng.choice(
            candidate_indices,
            size=num_traces,
            replace=False
        )

    selected_labels = labels[selected_indices]

    contamination_mask = np.isin(selected_labels, monitored_source_labels)
    contamination_count = int(np.sum(contamination_mask))

    if contamination_count != 0:
        bad_labels, bad_counts = np.unique(
            selected_labels[contamination_mask], return_counts=True
        )
        detail = dict(zip(bad_labels.tolist(), bad_counts.tolist()))
        raise RuntimeError(
            f"未监控抽样仍含 {contamination_count} 条 WF1/监控类流量：{detail}"
        )

    unique_selected_labels, unique_selected_counts = np.unique(
        selected_labels, return_counts=True
    )

    audit = {
        "source_rows": int(len(labels)),
        "candidate_count": int(len(candidate_indices)),
        "excluded_monitored_rows": excluded_monitored_rows,
        "sampled_count": int(len(selected_indices)),
        "contamination_count": contamination_count,
        "sampled_label_histogram": dict(
            zip(unique_selected_labels.tolist(), unique_selected_counts.tolist())
        )
    }

    logging.info(f"Sampled OW rows: {audit['sampled_count']}")
    logging.info(f"Sampled WF1/monitored contamination: {contamination_count}")

    return selected_indices.astype(np.int64), selected_labels, audit

def reconstruct_wf1_train_indices_like_training(
    zip_path: Path,
    wf1_num_classes: int = 70,
    traces_per_class_train: int = 40,
    label_entry: str = "y.npy",
    seed: int = 42
):
    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)
    known_labels = get_known_labels(labels)

    if len(known_labels) < wf1_num_classes:
        raise ValueError(
            f"可用类别不足：需要至少 {wf1_num_classes} 类，"
            f"但只找到 {len(known_labels)} 类"
        )

    wf1_labels = list(WF1_SOURCE_LABELS)

    if len(wf1_labels) != wf1_num_classes:
        raise ValueError(
            f"WF1 配置不一致：期望 {wf1_num_classes} 类，"
            f"实际配置 {len(wf1_labels)} 类。"
        )

    missing = sorted(set(wf1_labels) - set(known_labels))
    if missing:
        raise ValueError(f"数据文件中缺少 WF1 标签：{missing}")

    rng = np.random.RandomState(seed)

    train_indices = []
    train_labels = []

    for new_label, old_label in enumerate(wf1_labels):
        class_indices = np.where(labels == old_label)[0]

        if len(class_indices) < traces_per_class_train:
            raise ValueError(
                f"WF1 类别 {old_label} 样本不足："
                f"需要 {traces_per_class_train} 条，实际只有 {len(class_indices)} 条"
            )

        chosen = rng.choice(
            class_indices,
            size=traces_per_class_train,
            replace=False
        )

        train_indices.extend(chosen.tolist())
        train_labels.extend([new_label] * traces_per_class_train)

    return np.asarray(train_indices, dtype=np.int64), np.asarray(train_labels, dtype=np.int64)

def get_wf1_all_indices(
    zip_path: Path,
    wf1_num_classes: int = 70,
    label_entry: str = "y.npy"
):
    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)
    known_labels = get_known_labels(labels)

    if len(known_labels) < wf1_num_classes:
        raise ValueError(
            f"可用类别不足：需要至少 {wf1_num_classes} 类，"
            f"但只找到 {len(known_labels)} 类"
        )

    wf1_labels = list(WF1_SOURCE_LABELS)

    if len(wf1_labels) != wf1_num_classes:
        raise ValueError(
            f"WF1 配置不一致：期望 {wf1_num_classes} 类，"
            f"实际配置 {len(wf1_labels)} 类。"
        )

    missing = sorted(set(wf1_labels) - set(known_labels))
    if missing:
        raise ValueError(f"数据文件中缺少 WF1 标签：{missing}")

    wf1_indices = []

    for old_label in wf1_labels:
        class_indices = np.where(labels == old_label)[0]
        wf1_indices.extend(class_indices.tolist())

    return np.asarray(wf1_indices, dtype=np.int64)

def get_nonoverlap_wf2_indices(
    zip_path: Path,
    wf1_num_classes: int = 70,
    wf2_num_classes: int = 20,
    traces_per_class_test: int = MONITORED_POOL_PER_CLASS,
    traces_per_class_train: int = 40,
    start_offset: int = 30,
    label_entry: str = "y.npy",
    seed: int = 42
):
    labels = load_full_array(zip_path, label_entry).astype(np.int64).reshape(-1)
    wf1_labels, wf2_labels = get_wf_source_labels(
        labels=labels,
        wf1_num_classes=wf1_num_classes,
        wf2_num_classes=wf2_num_classes
    )

    wf1_train_indices, _ = reconstruct_wf1_train_indices_like_training(
        zip_path=zip_path,
        wf1_num_classes=wf1_num_classes,
        traces_per_class_train=traces_per_class_train,
        label_entry=label_entry,
        seed=seed
    )

    wf1_all_indices = get_wf1_all_indices(
        zip_path=zip_path,
        wf1_num_classes=wf1_num_classes,
        label_entry=label_entry
    )

    wf1_train_set = set(wf1_train_indices.tolist())
    wf1_all_set = set(wf1_all_indices.tolist())

    selected_wf2_indices = []
    selected_wf2_labels = []

    logging.info(f"WF1 label range: {wf1_labels[0]} to {wf1_labels[-1]}")
    logging.info(f"WF2 label range: {wf2_labels[0]} to {wf2_labels[-1]}")

    for local_label, old_label in enumerate(wf2_labels):
        class_indices = np.where(labels == old_label)[0]

        if len(class_indices) <= start_offset:
            raise ValueError(
                f"WF2 类别 {old_label} 样本数量不足，无法从 offset={start_offset} 开始取。"
            )

        candidate_indices = class_indices[start_offset:]

        candidate_indices = [
            idx for idx in candidate_indices
            if idx not in wf1_train_set and idx not in wf1_all_set
        ]

        if len(candidate_indices) < traces_per_class_test:
            raise ValueError(
                f"WF2 类别 {old_label} 可用测试样本不足："
                f"需要 {traces_per_class_test} 条，"
                f"排除 WF1/训练集后只剩 {len(candidate_indices)} 条。"
            )

        chosen = candidate_indices[:traces_per_class_test]

        selected_wf2_indices.extend(chosen)
        selected_wf2_labels.extend([local_label] * traces_per_class_test)

    selected_wf2_indices = np.asarray(selected_wf2_indices, dtype=np.int64)
    selected_wf2_labels = np.asarray(selected_wf2_labels, dtype=np.int64)

    overlap_with_train = np.intersect1d(selected_wf2_indices, wf1_train_indices)
    overlap_with_wf1 = np.intersect1d(selected_wf2_indices, wf1_all_indices)

    print("\n========== WF2 overlap check ==========")
    print(f"WF1 training samples: {len(wf1_train_indices)}")
    print(f"WF1 all samples: {len(wf1_all_indices)}")
    print(f"WF2 test pool samples: {len(selected_wf2_indices)}")
    print(f"Overlap with WF1 training set: {len(overlap_with_train)}")
    print(f"Overlap with WF1 full set: {len(overlap_with_wf1)}")

    if len(overlap_with_train) > 0:
        print("WF2 overlaps with WF1 training set. First 20 indices:")
        print(overlap_with_train[:20])
        raise RuntimeError("WF2 测试集与之前训练集存在重叠。")

    if len(overlap_with_wf1) > 0:
        print("WF2 overlaps with WF1 full set. First 20 indices:")
        print(overlap_with_wf1[:20])
        raise RuntimeError("WF2 测试集与 WF1 数据集存在重叠。")

    print("WF2 test set has no overlap with WF1 training/full set.")

    return (
        selected_wf2_indices,
        selected_wf2_labels,
        np.asarray(wf1_labels, dtype=np.int64),
        np.asarray(wf2_labels, dtype=np.int64)
    )

def load_open_world_test_pool(num_ow_traces: int = 1000):
    validate_open_world_configuration()
    logging.info("Loading WF2 known-class test pool...")

    (
        wf2_indices,
        wf2_labels,
        wf1_source_labels,
        wf2_source_labels
    ) = get_nonoverlap_wf2_indices(
        zip_path=DATA_NPZ,
        wf1_num_classes=WF1_NUM_CLASSES,
        wf2_num_classes=WF2_NUM_CLASSES,
        traces_per_class_test=MONITORED_POOL_PER_CLASS,
        traces_per_class_train=TRACES_PER_CLASS_TRAIN,
        start_offset=WF2_TEST_START_OFFSET,
        label_entry="y.npy",
        seed=SEED
    )

    raw_wf2 = load_array_rows(DATA_NPZ, "X.npy", wf2_indices)
    logging.info(f"Raw WF2 data shape: {raw_wf2.shape}")

    monitored_source_labels = np.concatenate([
        wf1_source_labels,
        wf2_source_labels
    ])

    logging.info(f"Strictly sampling {num_ow_traces} unmonitored traces...")
    ow_indices, ow_source_labels, ow_audit = sample_strict_unmonitored_indices(
        zip_path=OW_NPZ,
        num_traces=num_ow_traces,
        monitored_source_labels=monitored_source_labels,
        selection_mode=OW_SELECTION_MODE,
        unknown_labels=OW_UNKNOWN_LABELS,
        label_entry="y.npy",
        seed=SEED,
        balance_across_unknown_labels=BALANCE_OW_ACROSS_LABELS
    )

    if _same_path(DATA_NPZ, OW_NPZ):
        row_overlap = np.intersect1d(wf2_indices, ow_indices)
        if len(row_overlap) > 0:
            raise RuntimeError(
                f"WF2 与未监控集合发生行重叠，共 {len(row_overlap)} 条。"
            )

    raw_ow = load_array_rows(OW_NPZ, "X.npy", ow_indices)
    logging.info(f"Raw OW data shape: {raw_ow.shape}")

    raw_combined = np.concatenate([raw_wf2, raw_ow], axis=0)
    X_pool = build_dual_channel_signed_temporal(raw_combined, L=SEQ_LEN)

    y_pool = np.concatenate([
        wf2_labels,
        np.full(num_ow_traces, UNMONITORED_LABEL, dtype=np.int64)
    ])

    logging.info("Open-world test pool loaded.")
    logging.info(f"Monitored defense: {MONITORED_DEFENSE_NAME}")
    logging.info(f"Unmonitored defense: {UNMONITORED_DEFENSE_NAME}")
    logging.info(f"X_pool.shape = {X_pool.shape}")
    logging.info(f"y_pool.shape = {y_pool.shape}")
    logging.info(f"Total pool samples: {len(y_pool)}")
    logging.info(f"Strict OW samples: {np.sum(y_pool == UNMONITORED_LABEL)}")
    logging.info(f"OW source labels in sample: {np.unique(ow_source_labels).tolist()}")
    logging.info(f"OW sampled-label histogram: {ow_audit['sampled_label_histogram']}")
    logging.info(
        f"Known test set: {WF2_NUM_CLASSES} classes x "
        f"{FIXED_KNOWN_TEST_PER_CLASS} traces = "
        f"{WF2_NUM_CLASSES * FIXED_KNOWN_TEST_PER_CLASS}"
    )
    logging.info(
        f"Unknown test set: {len(OW_UNKNOWN_LABELS)} classes x "
        f"{OW_TRACES_PER_CLASS} traces = {NUM_OW_TRACES}"
    )
    logging.info(
        f"Expected total test size per repeat: "
        f"{WF2_NUM_CLASSES * FIXED_KNOWN_TEST_PER_CLASS + NUM_OW_TRACES}"
    )
    logging.info("Model input should be: (samples, 5000, 2)")

    return X_pool, y_pool, ow_audit

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

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)

def extract_all_embeddings(features_model, X_pool: np.ndarray, batch_size: int = 32) -> np.ndarray:
    logging.info("Extracting embeddings for the entire open-world pool...")

    embeddings = features_model.predict(
        X_pool,
        batch_size=batch_size,
        verbose=1
    )

    embeddings = l2_normalize(embeddings)

    logging.info(f"Embedding extraction finished, embeddings.shape = {embeddings.shape}")

    return embeddings

def create_open_world_split_from_embeddings(
    embeddings: np.ndarray,
    y_pool: np.ndarray,
    n_shot: int,
    rng: np.random.RandomState,
    size_of_problem: int,
    max_n_shot: int = MAX_N_SHOT,
    test_instances_per_class: int = FIXED_KNOWN_TEST_PER_CLASS,
    type_exp: str = "N-MEV"
):
    """
    对同一 repeat，所有 N 使用同一个类内排列：
    - 前 max_n_shot 条是候选支持池，N-shot 使用其中前 N 条；
    - 后 test_instances_per_class 条是所有 N 共用的固定测试集。
    因而 N 的变化不会改变测试样本和类别比例。
    """
    if not 1 <= n_shot <= max_n_shot:
        raise ValueError(f"n_shot={n_shot} 必须位于 [1, {max_n_shot}]。")

    num_tested_classes = min(size_of_problem, WF2_NUM_CLASSES)
    prototype_dict = {}
    test_embeddings = []
    test_labels = []

    for s in range(num_tested_classes):
        class_indices = np.where(y_pool == s)[0]
        required = max_n_shot + test_instances_per_class

        if len(class_indices) < required:
            raise ValueError(
                f"Class {s} 只有 {len(class_indices)} 条，"
                f"固定 support/test 划分需要至少 {required} 条。"
            )

        shuffled_indices = class_indices.copy()
        rng.shuffle(shuffled_indices)

        support_indices = shuffled_indices[:max_n_shot]
        signature_indices = support_indices[:n_shot]
        test_indices = shuffled_indices[
            max_n_shot:max_n_shot + test_instances_per_class
        ]

        if len(test_indices) != test_instances_per_class:
            raise RuntimeError(
                f"Class {s} 固定测试样本数异常："
                f"expected={test_instances_per_class}, actual={len(test_indices)}"
            )

        if np.intersect1d(signature_indices, test_indices).size != 0:
            raise RuntimeError(f"Class {s} 的支持集与测试集发生重叠。")

        sig_vec = embeddings[signature_indices]

        if type_exp == "N-MEV":
            proto = np.mean(sig_vec, axis=0, keepdims=True)
            proto = l2_normalize(proto)
        else:
            proto = l2_normalize(sig_vec)

        prototype_dict[s] = proto
        test_embeddings.append(embeddings[test_indices])
        test_labels.extend([s] * len(test_indices))

    ow_indices = np.where(y_pool == UNMONITORED_LABEL)[0]

    if len(ow_indices) > 0:
        test_embeddings.append(embeddings[ow_indices])
        test_labels.extend([UNMONITORED_LABEL] * len(ow_indices))

    X_test = np.concatenate(test_embeddings, axis=0).astype(np.float32)
    y_test = np.asarray(test_labels, dtype=np.int64)

    expected_known = num_tested_classes * test_instances_per_class
    actual_known = int(np.sum(y_test != UNMONITORED_LABEL))
    if actual_known != expected_known:
        raise RuntimeError(
            f"固定 known 测试集数量异常：expected={expected_known}, actual={actual_known}"
        )

    return prototype_dict, X_test, y_test

def compute_class_rankings(
    prototype_dict,
    X_test: np.ndarray,
    size_of_problem: int
):
    class_labels = sorted([lab for lab in prototype_dict.keys() if lab != UNMONITORED_LABEL])
    tested_sites = class_labels[:min(size_of_problem, len(class_labels))]

    if len(tested_sites) == 0:
        raise ValueError("没有可用类别原型。")

    class_score_matrix = np.zeros((len(X_test), len(tested_sites)), dtype=np.float32)

    for j, lab in enumerate(tested_sites):
        proto = prototype_dict[lab]
        proto = l2_normalize(proto)

        sims = np.matmul(X_test, proto.T)

        class_score_matrix[:, j] = np.max(sims, axis=1)

    sorted_col_indices = np.argsort(-class_score_matrix, axis=1)
    sorted_scores = np.take_along_axis(class_score_matrix, sorted_col_indices, axis=1)

    tested_sites_arr = np.asarray(tested_sites, dtype=np.int64)
    ranked_labels = tested_sites_arr[sorted_col_indices]

    best_scores = sorted_scores[:, 0]
    best_labels = ranked_labels[:, 0]

    return ranked_labels, sorted_scores, best_labels, best_scores, tested_sites

def metrics_open_world_only(
    prototype_dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    size_of_problem: int,
    similarity_threshold: float = 0.5
):
    ranked_labels, sorted_scores, best_labels, best_scores, tested_sites = compute_class_rankings(
        prototype_dict=prototype_dict,
        X_test=X_test,
        size_of_problem=size_of_problem
    )

    y_pred_top1 = np.where(
        best_scores < similarity_threshold,
        UNMONITORED_LABEL,
        best_labels
    ).astype(np.int64)

    all_eval_labels = [UNMONITORED_LABEL] + tested_sites

    metrics = {}

    for k in [1, 2, 5]:
        k_eff = min(k, len(tested_sites))

        correct_all = 0
        total_all = len(y_test)

        for i in range(len(y_test)):
            true_label = y_test[i]
            accepted = best_scores[i] >= similarity_threshold
            topk_labels = ranked_labels[i, :k_eff]

            if true_label == UNMONITORED_LABEL:
                if not accepted:
                    correct_all += 1
            else:
                if accepted and true_label in topk_labels:
                    correct_all += 1

        metrics[f"OW_Top{k}_Accuracy"] = correct_all / total_all if total_all > 0 else 0.0

    metrics["Macro_Recall"] = recall_score(
        y_test,
        y_pred_top1,
        labels=all_eval_labels,
        average="macro",
        zero_division=0
    )

    metrics["Macro_F1"] = f1_score(
        y_test,
        y_pred_top1,
        labels=all_eval_labels,
        average="macro",
        zero_division=0
    )

    metrics["Known_Macro_Recall"] = recall_score(
        y_test,
        y_pred_top1,
        labels=tested_sites,
        average="macro",
        zero_division=0
    )

    metrics["Known_Macro_F1"] = f1_score(
        y_test,
        y_pred_top1,
        labels=tested_sites,
        average="macro",
        zero_division=0
    )

    metrics["OW_Overall_Accuracy"] = accuracy_score(y_test, y_pred_top1)

    ow_mask = y_test == UNMONITORED_LABEL
    total_ow = int(np.sum(ow_mask))

    ow_reject = int(np.sum((y_pred_top1 == UNMONITORED_LABEL) & ow_mask))
    ow_false_accept = int(np.sum((y_pred_top1 != UNMONITORED_LABEL) & ow_mask))

    metrics["Unmonitored_TNR"] = ow_reject / total_ow if total_ow > 0 else 0.0
    metrics["Unmonitored_FAR"] = ow_false_accept / total_ow if total_ow > 0 else 0.0

    known_mask = y_test != UNMONITORED_LABEL
    total_known = int(np.sum(known_mask))
    known_false_reject = int(
        np.sum((y_pred_top1 == UNMONITORED_LABEL) & known_mask)
    )
    metrics["Known_FRR"] = (
        known_false_reject / total_known if total_known > 0 else 0.0
    )

    metrics["Total_Test_Samples"] = int(len(y_test))
    metrics["Total_OW_Samples"] = total_ow

    return metrics

def different_Experiment():
    logging.info(
        "Protocol: same-defense class holdout; "
        "WF1 labels=0--69 (websites 1--70), "
        "monitored labels=70--89 (websites 71--90), "
        "unknown labels=90--94 (websites 91--95)."
    )
    logging.info(
        "This protocol evaluates 20 monitored target websites, not 25. "
        "It reuses the encoder trained only on WF1 labels 0--69."
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 未找到")

    try:
        features_model = load_model(
            MODEL_PATH,
            custom_objects={
                "SpatialAttentionLayer": GlobalTemporalGating,
                "GlobalTemporalGating": GlobalTemporalGating
            },
            compile=False
        )

        logging.info(f"模型加载成功: {MODEL_PATH}")
        logging.info(f"模型输入形状: {features_model.input_shape}")
        logging.info(f"模型输出形状: {features_model.output_shape}")

    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        traceback.print_exc()
        return

    expected_input_shape = (None, SEQ_LEN, 2)

    if features_model.input_shape != expected_input_shape:
        logging.warning(
            f"当前模型输入为 {features_model.input_shape}，"
            f"但当前测试数据输入为 {expected_input_shape}。"
            f"请确认模型是否为双通道模型。"
        )

    X_pool, y_pool, ow_audit = load_open_world_test_pool(num_ow_traces=NUM_OW_TRACES)

    embeddings = extract_all_embeddings(
        features_model=features_model,
        X_pool=X_pool,
        batch_size=32
    )

    SOP_list = [WF2_NUM_CLASSES]

    metric_names = [
        "OW_Top1_Accuracy",
        "OW_Top2_Accuracy",
        "OW_Top5_Accuracy",
        "OW_Overall_Accuracy",
        "Macro_Recall",
        "Macro_F1",
        "Known_Macro_Recall",
        "Known_Macro_F1",
        "Unmonitored_TNR",
        "Unmonitored_FAR",
        "Known_FRR"
    ]

    results_summary = {}
    csv_rows = []

    for size_of_problem in SOP_list:
        results_summary[size_of_problem] = {}

        logging.info(f"开始开放世界测试 SOP = {size_of_problem}")

        for n_shot in N_SHOT_LIST:
            metric_values = {name: [] for name in metric_names}

            total_test_samples_last = 0
            total_ow_samples_last = 0

            for repeat_id in range(REPEAT_TIMES):
                logging.info(
                    f"运行 {repeat_id + 1}/{REPEAT_TIMES}，"
                    f"SOP={size_of_problem}, n_shot={n_shot}"
                )

                try:

                    rng = np.random.RandomState(SEED + repeat_id)

                    prototype_dict, X_test, y_test = create_open_world_split_from_embeddings(
                        embeddings=embeddings,
                        y_pool=y_pool,
                        n_shot=n_shot,
                        rng=rng,
                        size_of_problem=size_of_problem,
                        max_n_shot=MAX_N_SHOT,
                        test_instances_per_class=FIXED_KNOWN_TEST_PER_CLASS,
                        type_exp=TYPE_EXP
                    )

                    metrics = metrics_open_world_only(
                        prototype_dict=prototype_dict,
                        X_test=X_test,
                        y_test=y_test,
                        size_of_problem=size_of_problem,
                        similarity_threshold=SIMILARITY_THRESHOLD
                    )

                    for name in metric_names:
                        metric_values[name].append(float(metrics[name]))

                    total_test_samples_last = metrics["Total_Test_Samples"]
                    total_ow_samples_last = metrics["Total_OW_Samples"]

                    logging.info(
                        f"Iter {repeat_id + 1}/{REPEAT_TIMES} | "
                        f"N={n_shot} | "
                        f"OW-Top1={metrics['OW_Top1_Accuracy']:.5f} | "
                        f"OW-Top2={metrics['OW_Top2_Accuracy']:.5f} | "
                        f"OW-Top5={metrics['OW_Top5_Accuracy']:.5f} | "
                        f"Recall={metrics['Macro_Recall']:.5f} | "
                        f"F1={metrics['Macro_F1']:.5f} | "
                        f"OW-TNR={metrics['Unmonitored_TNR']:.5f} | "
                        f"OW-FAR={metrics['Unmonitored_FAR']:.5f}"
                    )

                except Exception as e:
                    logging.error(f"运行 {repeat_id + 1} 失败: {e}")
                    traceback.print_exc()
                    continue

            results_summary[size_of_problem][n_shot] = {}

            print(f"\n================ Open-World | SOP: {size_of_problem}, N_shot: {n_shot} ================")
            print(f"Total test samples: {total_test_samples_last}")
            print(f"Total OW samples: {total_ow_samples_last}")
            print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

            row = {
                "SOP": size_of_problem,
                "N_shot": n_shot,
                "Threshold": SIMILARITY_THRESHOLD,
                "Type": TYPE_EXP,
                "Known_Test_Per_Class": FIXED_KNOWN_TEST_PER_CLASS,
                "Monitored_Defense": MONITORED_DEFENSE_NAME,
                "Unmonitored_Defense": UNMONITORED_DEFENSE_NAME,
                "OW_Selection_Mode": OW_SELECTION_MODE,
                "WF1_Source_Labels": "0-69 (websites 1-70)",
                "Known_Source_Labels": "70-89 (websites 71-90)",
                "Unknown_Source_Labels": "90-94 (websites 91-95)",
                "OW_Source_Rows": ow_audit["source_rows"],
                "OW_Candidate_Count": ow_audit["candidate_count"],
                "OW_Excluded_Monitored_Rows": ow_audit["excluded_monitored_rows"],
                "OW_Contamination_Count": ow_audit["contamination_count"],
                "Total_Test_Samples": total_test_samples_last,
                "Total_OW_Samples": total_ow_samples_last
            }

            for name in metric_names:
                values = metric_values[name]
                mean_value = float(np.mean(values)) if values else 0.0
                std_value = float(np.std(values)) if values else 0.0

                results_summary[size_of_problem][n_shot][name] = {
                    "values": values,
                    "mean": mean_value,
                    "std": std_value
                }

                print(f"{name}: {str(values).strip('[]')}")
                print(f"{name} Mean: {mean_value:.5f} ± {std_value:.5f}")

                row[f"{name}_values"] = str(values)
                row[f"{name}_mean"] = mean_value
                row[f"{name}_std"] = std_value

            csv_rows.append(row)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "SOP",
        "N_shot",
        "Threshold",
        "Type",
        "Known_Test_Per_Class",
        "Monitored_Defense",
        "Unmonitored_Defense",
        "OW_Selection_Mode",
        "WF1_Source_Labels",
        "Known_Source_Labels",
        "Unknown_Source_Labels",
        "OW_Source_Rows",
        "OW_Candidate_Count",
        "OW_Excluded_Monitored_Rows",
        "OW_Contamination_Count",
        "Total_Test_Samples",
        "Total_OW_Samples"
    ]

    for name in metric_names:
        fieldnames.extend([
            f"{name}_values",
            f"{name}_mean",
            f"{name}_std"
        ])

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logging.info(f"开放世界实验结果已保存到 CSV 文件: {CSV_PATH}")

    return results_summary

if __name__ == "__main__":
    different_Experiment()
