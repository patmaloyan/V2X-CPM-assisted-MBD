import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from catch_checks import CatchChecks
from data_processing import CatchMLPDecision, prepare_messages_dataframe, perform_catch_checks
from data_structures import Parameters

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / 'model' / 'sybil_model.json'

FEATURE_NAMES = CatchMLPDecision.DEFAULT_FEATURES


def build_parameters(args) -> Parameters:
    if args.parameter is not None:
        with open(args.parameter, 'r', encoding='utf-8') as f:
            data = json.load(f)

        p = data['parameters']
        return Parameters(
            MAX_PLAUSIBLE_RANGE=p['mpr'],
            MAX_SA_RANGE=args.msar if args.msar is not None else Parameters.MAX_SA_RANGE,
            MAX_PLAUSIBLE_DIST_NEGATIVE=p['mpdn'],
            MAX_PLAUSIBLE_SPEED=p['mps'],
            MAX_PLAUSIBLE_ACCEL=p['mpa'],
            MAX_PLAUSIBLE_DECEL=p['mpd'],
            MAX_HEADING_CHANGE=p['mhc'],
            MAX_DELTA_INTERSECTION=p['mdi'],
            MAX_TIME_DELTA=p['mtd'],
            POS_HEADING_TIME=p['pht'],
            MAX_MGT_RNG_UP=p['mmru'],
            MAX_MGT_RNG_DOWN=p['mmrd'],
            MAX_SA_TIME=args.msat if args.msat is not None else Parameters.MAX_SA_TIME,
            MAX_NON_ROUTE_SPEED=p['mnrs']
        )

    if args.train == 1:
        return Parameters(
            MAX_PLAUSIBLE_RANGE=args.mpr,
            MAX_SA_RANGE=args.msar,
            MAX_PLAUSIBLE_DIST_NEGATIVE=args.mpdn,
            MAX_PLAUSIBLE_SPEED=args.mps,
            MAX_PLAUSIBLE_ACCEL=args.mpa,
            MAX_PLAUSIBLE_DECEL=args.mpd,
            MAX_HEADING_CHANGE=args.mhc,
            MAX_DELTA_INTERSECTION=args.mdi,
            MAX_TIME_DELTA=args.mtd,
            POS_HEADING_TIME=args.pht,
            MAX_MGT_RNG_UP=args.mmru,
            MAX_MGT_RNG_DOWN=args.mmrd,
            MAX_SA_TIME=args.msat,
            MAX_NON_ROUTE_SPEED=args.mnrs
        )

    return Parameters()


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
        if 'source_file' not in df.columns:
            df['source_file'] = path.stem
        return df

    if path.is_file() and path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        frame = pd.json_normalize(data, sep='_')
        frame['source_file'] = path.stem
        return frame

    if not path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {path}')

    parquet_files = list(path.glob('*.parquet'))
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        if 'source_file' not in df.columns:
            df['source_file'] = parquet_files[0].stem
        return df

    frames = []
    for json_file in sorted(path.glob('*.json')):
        if 'ground_truth' in json_file.name.lower():
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        if not data:
            continue

        frame = pd.json_normalize(data, sep='_')
        frame['source_file'] = json_file.stem
        frames.append(frame)

    if not frames:
        raise RuntimeError(f'No usable JSON or Parquet data found in {path}')

    return pd.concat(frames, ignore_index=True)


def load_split_data(base_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = base_path / 'Train'
    validation_path = base_path / 'Validation'
    test_path = base_path / 'Test'

    if not (train_path.exists() and validation_path.exists() and test_path.exists()):
        train_path = base_path / 'train'
        validation_path = base_path / 'validation'
        test_path = base_path / 'test'

    if train_path.exists() and validation_path.exists() and test_path.exists():
        return load_raw_dataframe(train_path), load_raw_dataframe(validation_path), load_raw_dataframe(test_path)

    raise FileNotFoundError(
        'Expected train/validation/test folders under input_folder, or pass explicit split folders.'
    )


def detect_features(raw_df: pd.DataFrame, params: Parameters) -> pd.DataFrame:
    prepared = prepare_messages_dataframe(raw_df)
    checks = CatchChecks(params)
    return perform_catch_checks(prepared, checks, mlp_model=None, decision_type='threshold')


def prepare_split_features(split_name: str, raw_df: pd.DataFrame, params: Parameters):
    """Worker-friendly split preprocessing: normalize rows and run CaTCH checks."""
    prepared = prepare_messages_dataframe(raw_df)
    checks = CatchChecks(params)
    return split_name, perform_catch_checks(prepared, checks, mlp_model=None, decision_type='threshold')


def extract_xy(results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    feature_columns = [f'check_{name}' for name in FEATURE_NAMES]
    for column in feature_columns:
        if column not in results.columns:
            results[column] = 0.0

    x = results[feature_columns].fillna(0.0).to_numpy(dtype=np.float64)
    x = np.clip(x, 0.0, 1.0)
    y = results['attacker'].astype(np.float64).to_numpy()
    return x, y


class ScipyMLP:
    def __init__(self, layer_sizes: Sequence[int], l2: float = 0.0):
        self.layer_sizes = list(layer_sizes)
        self.l2 = float(l2)
        self.shapes = [(out_size, in_size) for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    @staticmethod
    def _unpack(theta: np.ndarray, shapes: Sequence[Tuple[int, int]]):
        weights = []
        biases = []
        offset = 0
        for out_size, in_size in shapes:
            weight_size = out_size * in_size
            weights.append(theta[offset:offset + weight_size].reshape(out_size, in_size))
            offset += weight_size
            biases.append(theta[offset:offset + out_size])
            offset += out_size
        return weights, biases

    @staticmethod
    def _pack(weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]) -> np.ndarray:
        parts = []
        for weight, bias in zip(weights, biases):
            parts.append(weight.ravel())
            parts.append(bias.ravel())
        return np.concatenate(parts)

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(np.float64)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x: np.ndarray, theta: np.ndarray):
        weights, biases = self._unpack(theta, self.shapes)
        activations = [x]
        pre_activations = []
        current = x

        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            z = current @ weight.T + bias
            pre_activations.append(z)
            current = self._sigmoid(z) if layer_index == len(weights) - 1 else self._relu(z)
            activations.append(current)

        return activations, pre_activations, weights, biases

    def loss_and_grad(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray):
        activations, pre_activations, weights, biases = self.forward(x, theta)
        y_hat = activations[-1].reshape(-1, 1)
        y = y.reshape(-1, 1)

        eps = 1e-9
        y_hat_clip = np.clip(y_hat, eps, 1.0 - eps)
        loss = -np.mean(y * np.log(y_hat_clip) + (1.0 - y) * np.log(1.0 - y_hat_clip))
        loss += 0.5 * self.l2 * sum(np.sum(weight ** 2) for weight in weights)

        delta = (y_hat - y) / x.shape[0]
        grad_weights = [None] * len(weights)
        grad_biases = [None] * len(biases)

        for layer_index in reversed(range(len(weights))):
            grad_weights[layer_index] = delta.T @ activations[layer_index] + self.l2 * weights[layer_index]
            grad_biases[layer_index] = delta.sum(axis=0)
            if layer_index > 0:
                delta = (delta @ weights[layer_index]) * self._relu_grad(pre_activations[layer_index - 1])

        grad = self._pack(grad_weights, grad_biases)
        return float(loss), grad

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray,
            maxiter: int, threshold: float):
        rng = np.random.default_rng(42)
        total_params = 0
        for out_size, in_size in self.shapes:
            total_params += out_size * in_size + out_size

        weights = []
        biases = []
        for layer_index, (out_size, in_size) in enumerate(self.shapes):
            scale = np.sqrt(2.0 / max(1, in_size))
            weights.append(rng.normal(0.0, scale, size=(out_size, in_size)).astype(np.float64))
            biases.append(np.zeros(out_size, dtype=np.float64))

        theta0 = self._pack(weights, biases)

        def objective(theta):
            loss, grad = self.loss_and_grad(theta, x_train, y_train)
            return loss, grad

        result = minimize(
            fun=lambda theta: objective(theta)[0],
            x0=theta0,
            jac=lambda theta: objective(theta)[1],
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )

        self.theta = result.x if result.success else theta0
        return result

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        activations, _, _, _ = self.forward(x, self.theta)
        return activations[-1].reshape(-1)

    def predict(self, x: np.ndarray, threshold: float) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def to_json(self, threshold: float, feature_names: List[str], training_config: Dict[str, object], metrics: Dict[str, object]):
        weights, biases = self._unpack(self.theta, self.shapes)
        layers = []
        for weight, bias in zip(weights, biases):
            layers.append({'weights': weight.tolist(), 'bias': bias.tolist()})

        return {
            'threshold': threshold,
            'feature_names': feature_names,
            'layers': layers,
            'training_config': training_config,
            'metrics': metrics,
        }


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    parser = argparse.ArgumentParser(description='Train a binary MLP on CATCH scores using SciPy')
    parser.add_argument('--input_folder', required=True, help='Folder with train/validation/test folders')
    parser.add_argument('--output_model', required=False, default=str(DEFAULT_MODEL_PATH),
                        help='Where to write the trained JSON model (default: model/sybil_model.json next to this script)')
    parser.add_argument('--output_report', required=False, default=None, help='Optional JSON report output')
    parser.add_argument('--parameter', required=False, default=None)
    parser.add_argument('--train', type=float)
    parser.add_argument('--mpr', required=False, type=float)
    parser.add_argument('--msar', required=False, type=float)
    parser.add_argument('--mpdn', required=False, type=float)
    parser.add_argument('--mps', required=False, type=float)
    parser.add_argument('--mpa', required=False, type=float)
    parser.add_argument('--mpd', required=False, type=float)
    parser.add_argument('--mhc', required=False, type=float)
    parser.add_argument('--mdi', required=False, type=float)
    parser.add_argument('--mtd', required=False, type=float)
    parser.add_argument('--pht', required=False, type=float)
    parser.add_argument('--mmru', required=False, type=float)
    parser.add_argument('--mmrd', required=False, type=float)
    parser.add_argument('--msat', required=False, type=float)
    parser.add_argument('--mnrs', required=False, type=float)
    parser.add_argument('--hidden_layers', required=False, default='16,8', help='Comma-separated hidden layer sizes')
    parser.add_argument('--maxiter', required=False, type=int, default=200, help='SciPy optimizer iterations')
    parser.add_argument('--threshold', required=False, type=float, default=0.5, help='Binary decision threshold')
    parser.add_argument('--workers', required=False, type=int, default=os.cpu_count() or 4,
                        help='Number of worker processes for split preprocessing')
    args = parser.parse_args()

    output_model = Path(args.output_model)
    if not output_model.is_absolute():
        output_model = (SCRIPT_DIR / output_model).resolve()
    args.output_model = str(output_model)

    params = build_parameters(args)
    print('Loading train/validation/test splits...', file=sys.stderr)
    train_raw, validation_raw, test_raw = load_split_data(Path(args.input_folder))

    print(
        f"Loaded splits: train={len(train_raw)}, validation={len(validation_raw)}, test={len(test_raw)}",
        file=sys.stderr,
    )

    split_frames = {
        'train': train_raw,
        'validation': validation_raw,
        'test': test_raw,
    }

    if args.workers > 1:
        max_workers = min(args.workers, len(split_frames))
        print(f'Running CATCH feature extraction with {max_workers} workers...', file=sys.stderr)
        prepared_splits = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(prepare_split_features, split_name, split_frame, params): split_name
                for split_name, split_frame in split_frames.items()
            }
            for future in as_completed(futures):
                split_name = futures[future]
                returned_split_name, split_results = future.result()
                prepared_splits[returned_split_name] = split_results
                print(f'Finished CATCH feature extraction for {split_name} split', file=sys.stderr)
    else:
        print('Running CATCH feature extraction sequentially...', file=sys.stderr)
        prepared_splits = {}
        for split_name, split_frame in split_frames.items():
            prepared_splits[split_name] = detect_features(split_frame, params)
            print(f'Finished CATCH feature extraction for {split_name} split', file=sys.stderr)

    train_results = prepared_splits['train']
    validation_results = prepared_splits['validation']
    test_results = prepared_splits['test']

    x_train, y_train = extract_xy(train_results)
    x_validation, y_validation = extract_xy(validation_results)
    x_test, y_test = extract_xy(test_results)

    hidden_layers = [int(value.strip()) for value in args.hidden_layers.split(',') if value.strip()]
    layer_sizes = [len(FEATURE_NAMES), *hidden_layers, 1]
    mlp = ScipyMLP(layer_sizes, l2=0.0)

    print('Starting SciPy optimization...', file=sys.stderr)
    optimization_result = mlp.fit(x_train, y_train, x_validation, y_validation, args.maxiter, args.threshold)
    print(
        f"Optimization finished: success={optimization_result.success}, iterations={optimization_result.nit}",
        file=sys.stderr,
    )

    train_threshold_metrics = metrics_dict(y_train, (train_results['prediction_threshold'].to_numpy() == 1).astype(int))
    validation_threshold_metrics = metrics_dict(y_validation, (validation_results['prediction_threshold'].to_numpy() == 1).astype(int))
    test_threshold_metrics = metrics_dict(y_test, (test_results['prediction_threshold'].to_numpy() == 1).astype(int))

    train_mlp_metrics = metrics_dict(y_train, mlp.predict(x_train, args.threshold))
    validation_mlp_metrics = metrics_dict(y_validation, mlp.predict(x_validation, args.threshold))
    test_mlp_metrics = metrics_dict(y_test, mlp.predict(x_test, args.threshold))

    report = {
        'threshold': args.threshold,
        'feature_names': FEATURE_NAMES,
        'hidden_layers': hidden_layers,
        'optimizer_success': bool(optimization_result.success),
        'optimizer_message': optimization_result.message,
        'train': {'threshold': train_threshold_metrics, 'mlp': train_mlp_metrics},
        'validation': {'threshold': validation_threshold_metrics, 'mlp': validation_mlp_metrics},
        'test': {'threshold': test_threshold_metrics, 'mlp': test_mlp_metrics},
    }

    model_payload = mlp.to_json(
        threshold=args.threshold,
        feature_names=FEATURE_NAMES,
        training_config={
            'optimizer': 'scipy.optimize.minimize',
            'method': 'L-BFGS-B',
            'maxiter': args.maxiter,
        },
        metrics=report,
    )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, 'w', encoding='utf-8') as f:
        json.dump(model_payload, f, indent=2)

    output_report = Path(args.output_report) if args.output_report else output_model.with_suffix('.report.json')
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('Training complete.', file=sys.stderr)

    print(test_mlp_metrics['f1'])


if __name__ == '__main__':
    main()
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from catch_checks import CatchChecks
from data_processing import CatchMLPDecision, perform_catch_checks, prepare_messages_dataframe
from data_structures import Parameters


FEATURE_NAMES = list(CatchMLPDecision.DEFAULT_FEATURES)


def load_messages_frame(path: Path) -> pd.DataFrame:
    """Load one JSON/Parquet source into a flat DataFrame."""
    if path.is_file():
        if path.suffix.lower() == '.parquet':
            frame = pd.read_parquet(path)
            if 'source_file' not in frame.columns:
                frame['source_file'] = path.stem
            return frame

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        frame = pd.json_normalize(data, sep='_')
        frame['source_file'] = path.stem
        return frame

    if not path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {path}')

    parquet_files = sorted(path.glob('*.parquet'))
    if parquet_files:
        frame = pd.read_parquet(parquet_files[0])
        if 'source_file' not in frame.columns:
            frame['source_file'] = parquet_files[0].stem
        return frame

    json_frames = []
    for json_file in sorted(path.glob('*.json')):
        if 'ground_truth' in json_file.name.lower():
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not data:
            continue

        frame = pd.json_normalize(data, sep='_')
        frame['source_file'] = json_file.stem
        json_frames.append(frame)

    if not json_frames:
        raise RuntimeError(f'No usable JSON or Parquet data found in {path}')

    return pd.concat(json_frames, ignore_index=True)


def build_parameters(args) -> Parameters:
    if args.parameter is not None:
        with open(args.parameter, 'r', encoding='utf-8') as f:
            data = json.load(f)
        p = data['parameters']
        return Parameters(
            MAX_PLAUSIBLE_RANGE=p['mpr'],
            MAX_SA_RANGE=args.msar if args.msar is not None else Parameters.MAX_SA_RANGE,
            MAX_PLAUSIBLE_DIST_NEGATIVE=p['mpdn'],
            MAX_PLAUSIBLE_SPEED=p['mps'],
            MAX_PLAUSIBLE_ACCEL=p['mpa'],
            MAX_PLAUSIBLE_DECEL=p['mpd'],
            MAX_HEADING_CHANGE=p['mhc'],
            MAX_DELTA_INTERSECTION=p['mdi'],
            MAX_TIME_DELTA=p['mtd'],
            POS_HEADING_TIME=p['pht'],
            MAX_MGT_RNG_UP=p['mmru'],
            MAX_MGT_RNG_DOWN=p['mmrd'],
            MAX_SA_TIME=args.msat if args.msat is not None else Parameters.MAX_SA_TIME,
            MAX_NON_ROUTE_SPEED=p['mnrs'],
        )

    if args.train == 1:
        return Parameters(
            MAX_PLAUSIBLE_RANGE=args.mpr,
            MAX_SA_RANGE=args.msar,
            MAX_PLAUSIBLE_DIST_NEGATIVE=args.mpdn,
            MAX_PLAUSIBLE_SPEED=args.mps,
            MAX_PLAUSIBLE_ACCEL=args.mpa,
            MAX_PLAUSIBLE_DECEL=args.mpd,
            MAX_HEADING_CHANGE=args.mhc,
            MAX_DELTA_INTERSECTION=args.mdi,
            MAX_TIME_DELTA=args.mtd,
            POS_HEADING_TIME=args.pht,
            MAX_MGT_RNG_UP=args.mmru,
            MAX_MGT_RNG_DOWN=args.mmrd,
            MAX_SA_TIME=args.msat,
            MAX_NON_ROUTE_SPEED=args.mnrs,
        )

    return Parameters()


def split_groups(groups: Sequence[str], train_ratio: float, validation_ratio: float,
                 test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    """Split source files so one vehicle/file never leaks across train/val/test."""
    shuffled = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    total_ratio = train_ratio + validation_ratio + test_ratio
    train_ratio /= total_ratio
    validation_ratio /= total_ratio

    total = len(shuffled)
    if total < 3:
        raise RuntimeError('Need at least 3 source groups for automatic train/validation/test splitting')

    train_end = max(1, int(round(total * train_ratio)))
    validation_end = max(train_end + 1, int(round(total * (train_ratio + validation_ratio))))
    validation_end = min(validation_end, total - 1)

    train_groups = shuffled[:train_end]
    validation_groups = shuffled[train_end:validation_end]
    test_groups = shuffled[validation_end:]

    if not validation_groups or not test_groups:
        raise RuntimeError('Automatic group split produced an empty validation or test split')

    return train_groups, validation_groups, test_groups


def subset_by_groups(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    return frame[frame['source_file'].isin(groups)].copy()


def prepare_catch_frame(raw_frame: pd.DataFrame, params: Parameters) -> pd.DataFrame:
    prepared = prepare_messages_dataframe(raw_frame)
    checks = CatchChecks(params)
    return perform_catch_checks(prepared, checks)


def extract_features(results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = [f'check_{name}' for name in FEATURE_NAMES]
    for column in feature_columns:
        if column not in results.columns:
            results[column] = 0.0

    x = results[feature_columns].fillna(0.0).to_numpy(dtype=np.float64)
    x = np.clip(x, 0.0, 1.0)
    y = results['attacker'].astype(np.float64).to_numpy()
    threshold_predictions = results['prediction_threshold'].astype(int).to_numpy()
    return x, y, threshold_predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


class SciPyMLP:
    """One-hidden-layer MLP trained with SciPy's L-BFGS-B optimizer."""

    def __init__(self, input_dim: int, hidden_units: int, l2: float, seed: int):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.theta = self._init_theta()
        self.threshold = 0.5

    def _init_theta(self) -> np.ndarray:
        w1 = self.rng.normal(0.0, 0.1, size=(self.hidden_units, self.input_dim))
        b1 = np.zeros(self.hidden_units, dtype=np.float64)
        w2 = self.rng.normal(0.0, 0.1, size=(1, self.hidden_units))
        b2 = np.zeros(1, dtype=np.float64)
        return self.pack(w1, b1, w2, b2)

    def pack(self, w1, b1, w2, b2) -> np.ndarray:
        return np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()])

    def unpack(self, theta: np.ndarray):
        index = 0
        w1_size = self.hidden_units * self.input_dim
        w1 = theta[index:index + w1_size].reshape(self.hidden_units, self.input_dim)
        index += w1_size
        b1 = theta[index:index + self.hidden_units]
        index += self.hidden_units
        w2_size = self.hidden_units
        w2 = theta[index:index + w2_size].reshape(1, self.hidden_units)
        index += w2_size
        b2 = theta[index:index + 1]
        return w1, b1, w2, b2

    def forward(self, theta: np.ndarray, x: np.ndarray):
        w1, b1, w2, b2 = self.unpack(theta)
        z1 = x @ w1.T + b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ w2.T + b2
        y_hat = expit(z2)
        return w1, b1, w2, b2, z1, h1, z2, y_hat

    def loss_and_grad(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray):
        w1, b1, w2, b2, z1, h1, z2, y_hat = self.forward(theta, x)
        y = y.reshape(-1, 1)
        eps = 1e-9
        y_hat = np.clip(y_hat, eps, 1.0 - eps)

        loss = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
        loss += 0.5 * self.l2 * (np.sum(w1 ** 2) + np.sum(w2 ** 2))

        n = x.shape[0]
        dz2 = (y_hat - y) / n
        grad_w2 = dz2.T @ h1 + self.l2 * w2
        grad_b2 = dz2.sum(axis=0)

        dh1 = dz2 @ w2
        dz1 = dh1 * (z1 > 0).astype(np.float64)
        grad_w1 = dz1.T @ x + self.l2 * w1
        grad_b1 = dz1.sum(axis=0)

        grad = self.pack(grad_w1, grad_b1, grad_w2, grad_b2)
        return float(loss), grad

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray,
            max_iter: int, threshold: float):
        self.threshold = threshold

        def objective(theta):
            return self.loss_and_grad(theta, x_train, y_train)

        result = minimize(
            objective,
            self.theta,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )
        self.theta = result.x

        train_prob = self.predict_proba(x_train)
        validation_prob = self.predict_proba(x_validation)
        train_metrics = compute_metrics(y_train, (train_prob >= threshold).astype(int))
        validation_metrics = compute_metrics(y_validation, (validation_prob >= threshold).astype(int))

        history = {
            'optimizer_success': bool(result.success),
            'optimizer_message': str(result.message),
            'optimizer_iterations': int(result.nit),
            'final_loss': float(result.fun),
        }
        return train_metrics, validation_metrics, history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, _, _, _, _, _, _, y_hat = self.forward(self.theta, x)
        return y_hat.reshape(-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold).astype(int)

    def to_json(self, feature_names: Sequence[str], training_config: Dict[str, object], metrics: Dict[str, object]):
        w1, b1, w2, b2 = self.unpack(self.theta)
        return {
            'threshold': self.threshold,
            'feature_names': list(feature_names),
            'layers': [
                {'weights': w1.tolist(), 'bias': b1.tolist()},
                {'weights': w2.tolist(), 'bias': b2.tolist()},
            ],
            'training_config': training_config,
            'metrics': metrics,
        }


def load_or_split_data(input_path: Path, train_folder: Optional[str], validation_folder: Optional[str],
                       test_folder: Optional[str], train_ratio: float, validation_ratio: float,
                       test_ratio: float, seed: int):
    if train_folder and validation_folder and test_folder:
        return {
            'train': load_messages_frame(Path(train_folder)),
            'validation': load_messages_frame(Path(validation_folder)),
            'test': load_messages_frame(Path(test_folder)),
        }

    auto_train = input_path / 'train'
    auto_validation = input_path / 'validation'
    auto_test = input_path / 'test'
    if auto_train.exists() and auto_validation.exists() and auto_test.exists():
        return {
            'train': load_messages_frame(auto_train),
            'validation': load_messages_frame(auto_validation),
            'test': load_messages_frame(auto_test),
        }

    raw_frame = load_messages_frame(input_path)
    groups = sorted(raw_frame['source_file'].dropna().astype(str).unique().tolist())
    train_groups, validation_groups, test_groups = split_groups(
        groups,
        train_ratio,
        validation_ratio,
        test_ratio,
        seed,
    )
    return {
        'train': subset_by_groups(raw_frame, train_groups),
        'validation': subset_by_groups(raw_frame, validation_groups),
        'test': subset_by_groups(raw_frame, test_groups),
    }


def main():
    parser = argparse.ArgumentParser(description='Train a SciPy-backed MLP on CATCH check scores')
    parser.add_argument('--input_folder', required=True, help='Folder, parquet file, or train/validation/test base folder')
    parser.add_argument('--output_model', required=True, help='Path for the exported MLP JSON model')
    parser.add_argument('--output_report', required=False, default=None, help='Optional JSON report path')
    parser.add_argument('--train_folder', required=False, default=None, help='Explicit train split folder')
    parser.add_argument('--validation_folder', required=False, default=None, help='Explicit validation split folder')
    parser.add_argument('--test_folder', required=False, default=None, help='Explicit test split folder')
    parser.add_argument('--split_seed', required=False, type=int, default=42, help='Seed for automatic group splitting')
    parser.add_argument('--train_ratio', required=False, type=float, default=0.5, help='Automatic train split ratio')
    parser.add_argument('--validation_ratio', required=False, type=float, default=0.1, help='Automatic validation split ratio')
    parser.add_argument('--test_ratio', required=False, type=float, default=0.4, help='Automatic test split ratio')
    parser.add_argument('--hidden_units', required=False, type=int, default=8, help='Hidden layer size for the MLP')
    parser.add_argument('--max_iter', required=False, type=int, default=200, help='Maximum L-BFGS-B iterations')
    parser.add_argument('--threshold', required=False, type=float, default=0.5, help='Binary cutoff for the MLP output')
    parser.add_argument('--l2', required=False, type=float, default=0.0, help='L2 regularization strength')
    parser.add_argument('--parameter', required=False, default=None)
    parser.add_argument('--train', type=float)
    parser.add_argument('--mpr', required=False, type=float)
    parser.add_argument('--msar', required=False, type=float)
    parser.add_argument('--mpdn', required=False, type=float)
    parser.add_argument('--mps', required=False, type=float)
    parser.add_argument('--mpa', required=False, type=float)
    parser.add_argument('--mpd', required=False, type=float)
    parser.add_argument('--mhc', required=False, type=float)
    parser.add_argument('--mdi', required=False, type=float)
    parser.add_argument('--mtd', required=False, type=float)
    parser.add_argument('--pht', required=False, type=float)
    parser.add_argument('--mmru', required=False, type=float)
    parser.add_argument('--mmrd', required=False, type=float)
    parser.add_argument('--msat', required=False, type=float)
    parser.add_argument('--mnrs', required=False, type=float)
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    params = build_parameters(args)
    split_frames = load_or_split_data(
        input_path,
        args.train_folder,
        args.validation_folder,
        args.test_folder,
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio,
        args.split_seed,
    )

    prepared_splits = {}
    for split_name, raw_frame in split_frames.items():
        if raw_frame.empty:
            raise RuntimeError(f'{split_name} split is empty')
        prepared_splits[split_name] = prepare_catch_frame(raw_frame, params)

    train_x, train_y, train_threshold_predictions = extract_features(prepared_splits['train'])
    validation_x, validation_y, validation_threshold_predictions = extract_features(prepared_splits['validation'])
    test_x, test_y, test_threshold_predictions = extract_features(prepared_splits['test'])

    threshold_report = {
        'train': compute_metrics(train_y, train_threshold_predictions),
        'validation': compute_metrics(validation_y, validation_threshold_predictions),
        'test': compute_metrics(test_y, test_threshold_predictions),
    }

    mlp = SciPyMLP(input_dim=train_x.shape[1], hidden_units=args.hidden_units, l2=args.l2, seed=args.split_seed)
    train_metrics, validation_metrics, optimizer_info = mlp.fit(
        train_x,
        train_y,
        validation_x,
        validation_y,
        args.max_iter,
        args.threshold,
    )

    test_mlp_predictions = mlp.predict(test_x)
    mlp_report = {
        'train': train_metrics,
        'validation': validation_metrics,
        'test': compute_metrics(test_y, test_mlp_predictions),
    }

    report = {
        'feature_names': FEATURE_NAMES,
        'threshold': args.threshold,
        'hidden_units': args.hidden_units,
        'threshold_report': threshold_report,
        'mlp_report': mlp_report,
        'optimizer': optimizer_info,
        'training_config': {
            'max_iter': args.max_iter,
            'l2': args.l2,
            'split_seed': args.split_seed,
            'train_ratio': args.train_ratio,
            'validation_ratio': args.validation_ratio,
            'test_ratio': args.test_ratio,
        },
    }

    model_payload = mlp.to_json(
        feature_names=FEATURE_NAMES,
        training_config=report['training_config'],
        metrics=report,
    )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, 'w', encoding='utf-8') as f:
        json.dump(model_payload, f, indent=2)

    output_report = Path(args.output_report) if args.output_report else output_model.with_suffix('.report.json')
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Trained model saved to {output_model}')
    print(f'Report saved to {output_report}')
    print(mlp_report['test']['f1'])


if __name__ == '__main__':
    main()
