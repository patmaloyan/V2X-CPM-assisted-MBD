import json
import sys
from pathlib import Path

import numpy as np

np.seterr(all='warn', over='raise')

from data_structures import Mapper, Parameters, Coord
from catch_checks import CatchChecks
from legacy_checks import LegacyChecks
from mdm_lib import MDMLib

import pandas as pd


class CatchMLPDecision:
    """Minimal feed-forward scorer for CATCH check vectors."""

    DEFAULT_FEATURES = (
        'range_plausibility',
        'position_plausibility_check',
        'speed_plausibility_check',
        'position_consistency_check',
        'speed_consistency_check',
        'position_speed_consistency_check',
        'position_heading_consistency_check',
        'intersection_check',
    )

    def __init__(self, layers, threshold: float = 0.5, feature_names=None):
        self.layers = layers
        self.threshold = float(threshold)
        self.feature_names = tuple(feature_names) if feature_names else self.DEFAULT_FEATURES

    @staticmethod
    def from_file(model_path: str):
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        layers = data.get('layers')
        if not layers:
            raise ValueError('MLP model file must contain a non-empty "layers" list')

        return CatchMLPDecision(
            layers=layers,
            threshold=data.get('threshold', 0.5),
            feature_names=data.get('feature_names')
        )

    def _prepare_input(self, features):
        if isinstance(features, dict):
            return np.array([
                float(np.clip(features.get(name, 0.0), 0.0, 1.0))
                for name in self.feature_names
            ], dtype=np.float64)

        vector = np.asarray(features, dtype=np.float64)
        return np.clip(vector, 0.0, 1.0)

    def predict_score(self, features) -> float:
        activation = self._prepare_input(features)

        for layer_index, layer in enumerate(self.layers):
            weights = np.asarray(layer['weights'], dtype=np.float64)
            bias = np.asarray(layer['bias'], dtype=np.float64)
            activation = activation @ weights.T + bias

            if layer_index < len(self.layers) - 1:
                activation = np.maximum(activation, 0.0)
            else:
                activation = 1.0 / (1.0 + np.exp(-activation))

        return float(np.squeeze(activation))

    def predict(self, features) -> int:
        return 1 if self.predict_score(features) >= self.threshold else 0


def save_messages(results: pd.DataFrame, input_file: Path, source_file):
    output_dir = input_file.parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{source_file}_predicted.json"

    nested_data = results.apply(Mapper.row_to_json, axis=1).tolist()
    with open(output_file, 'w') as f:
        json.dump(nested_data, f, indent=4)


def calculate_metrics(results: pd.DataFrame, prediction_column: str = 'prediction', suffix: str = '') -> dict:
    tp = ((results['attacker'] == 1) & (results[prediction_column] == 1)).sum()
    tn = ((results['attacker'] == 0) & (results[prediction_column] == 0)).sum()
    fp = ((results['attacker'] == 0) & (results[prediction_column] == 1)).sum()
    fn = ((results['attacker'] == 1) & (results[prediction_column] == 0)).sum()

    return {
        f'tp{suffix}': int(tp),
        f'tn{suffix}': int(tn),
        f'fp{suffix}': int(fp),
        f'fn{suffix}': int(fn)
    }


def prepare_messages_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw rows into the flat layout expected by the CaTCH checks."""
    df = df.copy()

    df['rcvTime'] = df['rcvTime'].astype(int)
    df['sendTime'] = df['sendTime'].astype(int)
    df['sender_id'] = df['sender_id'].astype(str)
    df['sender_alias'] = df['sender_alias'].astype(int)
    df['messageID'] = df['messageID'].astype(int)
    df['attacker'] = df['attacker'].astype(int)

    df['receiver_spd'] = df['receiver_spd'].astype(np.float64)
    df['receiver_spd_noise'] = df['receiver_spd_noise'].astype(np.float64)
    df['receiver_acl'] = df['receiver_acl'].astype(float)
    df['receiver_acl_noise'] = df['receiver_acl_noise'].astype(float)
    df['receiver_hed'] = df['receiver_hed'].astype(float)
    df['receiver_hed_noise'] = df['receiver_hed_noise'].astype(float)
    df['receiver_driversProfile'] = df['receiver_driversProfile'].astype(str)

    df['sender_spd'] = df['sender_spd'].astype(np.float64)
    df['sender_spd_noise'] = df['sender_spd_noise'].astype(np.float64)
    df['sender_acl'] = df['sender_acl'].astype(float)
    df['sender_acl_noise'] = df['sender_acl_noise'].astype(float)
    df['sender_hed'] = df['sender_hed'].astype(float)
    df['sender_hed_noise'] = df['sender_hed_noise'].astype(float)
    df['sender_driversProfile'] = df['sender_driversProfile'].astype(str)
    df['sender_distance_to_road_edge'] = df['sender_distance_to_road_edge'].astype(float)

    # The dataset may store coordinates as comma-separated strings or as lists.
    if isinstance(df.iloc[0].get('sender_pos', '')[0], str):
        pos_data = df['receiver_pos'].str.split(',', expand=True)
        df['receiver_pos_lat'] = pos_data[0].astype(np.float64)
        df['receiver_pos_lon'] = pos_data[1].astype(np.float64)
        df['receiver_pos_alt'] = pos_data[2].astype(np.float64)

        noise_data = df['receiver_pos_noise'].str.split(',', expand=True)
        df['receiver_pos_lat_noise'] = noise_data[0].astype(np.float64)
        df['receiver_pos_lon_noise'] = noise_data[1].astype(np.float64)
        df['receiver_pos_alt_noise'] = noise_data[2].astype(np.float64)

        sender_pos_data = df['sender_pos'].str.split(',', expand=True)
        df['sender_pos_lat'] = sender_pos_data[0].astype(np.float64)
        df['sender_pos_lon'] = sender_pos_data[1].astype(np.float64)
        df['sender_pos_alt'] = sender_pos_data[2].astype(np.float64)

        sender_noise_data = df['sender_pos_noise'].str.split(',', expand=True)
        df['sender_pos_lat_noise'] = sender_noise_data[0].astype(np.float64)
        df['sender_pos_lon_noise'] = sender_noise_data[1].astype(np.float64)
        df['sender_pos_alt_noise'] = sender_noise_data[2].astype(np.float64)
    else:
        df[['receiver_pos_lat', 'receiver_pos_lon', 'receiver_pos_alt']] = pd.DataFrame(
            df['receiver_pos'].tolist(), index=df.index, columns=['lat', 'lon', 'alt']
        )

        df[['receiver_pos_lat_noise', 'receiver_pos_lon_noise', 'receiver_pos_alt_noise']] = pd.DataFrame(
            df['receiver_pos_noise'].tolist(), index=df.index, columns=['lat_noise', 'lon_noise', 'alt_noise']
        )

        df[['sender_pos_lat', 'sender_pos_lon', 'sender_pos_alt']] = pd.DataFrame(
            df['sender_pos'].tolist(), index=df.index, columns=['lat', 'lon', 'alt']
        )

        df[['sender_pos_lat_noise', 'sender_pos_lon_noise', 'sender_pos_alt_noise']] = pd.DataFrame(
            df['sender_pos_noise'].tolist(), index=df.index, columns=['lat_noise', 'lon_noise', 'alt_noise']
        )

    return df


def perform_catch_checks(messages: pd.DataFrame, checks: CatchChecks,
                         mlp_model=None, decision_type: str = 'threshold') -> pd.DataFrame:
    history = dict()
    history_data = dict()

    def process_row(df_msg: pd.Series):
        msg = Mapper.row_to_message(df_msg)
        prediction_results = dict()
        activations = dict()
        result = dict()

        prediction_results['range_plausibility'] = checks.range_plausibility_check(
            msg.receiver.pos, msg.receiver.pos_noise, msg.sender.pos, msg.sender.pos_noise)

        if prediction_results['range_plausibility'] < 0.5:
            activations['range_plausibility'] = True

        prediction_results['position_plausibility_check'] = checks.position_plausibility_check(
            msg.sender.pos_noise, msg.sender.spd, msg.sender.spd_noise, msg.sender.distance_to_road_edge)

        if prediction_results['position_plausibility_check'] < 0.5:
            activations['position_plausibility_check'] = True

        prediction_results['speed_plausibility_check'] = checks.speed_plausibility_check(
            msg.sender.spd, msg.sender.spd_noise)

        if prediction_results['speed_plausibility_check'] < 0.5:
            activations['speed_plausibility_check'] = True

        #if (msg.sender_id not in history.keys() or MDMLib.ns_to_seconds(
         #       msg.rcvTime - history[msg.sender_id]) > Parameters.MAX_SA_TIME):
          #  prediction_results['sudden_appearance_check'] = checks.sudden_appearance_check(
           #     msg.receiver.pos, msg.receiver.pos_noise, msg.sender.pos, msg.sender.pos_noise)

            #if prediction_results['sudden_appearance_check'] < 0.5:
             #   activations['sudden_appearance_check'] = True

        #history[msg.sender_id] = msg.rcvTime

        sender_history = messages[
            (messages['sender_id'] == msg.sender_id) &
            (messages['sendTime'] < msg.sendTime)
            ].sort_values('sendTime', ascending=False)

        prev_msg = Mapper.row_to_message(sender_history.iloc[0]) if not sender_history.empty else None

        if prev_msg is not None:
            delta_time = MDMLib.ns_to_seconds(msg.sendTime - prev_msg.sendTime)

            prediction_results['position_consistency_check'] = checks.position_consistency_check(
                msg.sender.pos, msg.sender.pos_noise, prev_msg.sender.pos, prev_msg.sender.pos_noise, delta_time)

            if prediction_results['position_consistency_check'] < 0.5:
                activations['position_consistency_check'] = True

            prediction_results['speed_consistency_check'] = checks.speed_consistency_check(
                msg.sender.spd, msg.sender.spd_noise, prev_msg.sender.spd, prev_msg.sender.spd_noise, delta_time)

            if prediction_results['speed_consistency_check'] < 0.5:
                activations['speed_consistency_check'] = True

            prediction_results['position_speed_consistency_check'] = checks.position_speed_consistency_check(
                msg.sender.pos, msg.sender.pos_noise, prev_msg.sender.pos, prev_msg.sender.pos_noise, msg.sender.spd,
                msg.sender.spd_noise, prev_msg.sender.spd, prev_msg.sender.spd_noise, delta_time)

            if prediction_results['position_speed_consistency_check'] < 0.5:
                activations['position_speed_consistency_check'] = True

            prediction_results['position_heading_consistency_check'] = checks.position_heading_consistency_check(
                msg.sender.hed, msg.sender.hed_noise, prev_msg.sender.pos, prev_msg.sender.pos_noise, msg.sender.pos,
                msg.sender.pos_noise, delta_time, msg.sender.spd, msg.sender.spd_noise)

            if prediction_results['position_heading_consistency_check'] < 0.5:
                activations['position_heading_consistency_check'] = True

        prediction_results['intersection_check'] = 0
        for sender_id, hist_data in history_data.items():
            if len(hist_data) > 0:
                data = hist_data[-1]

                if (MDMLib.ns_to_seconds(msg.rcvTime - data.rcvTime) <= checks.params.MAX_DELTA_INTERSECTION and
                        data.sender_id != msg.sender_id):

                    delta_time = MDMLib.ns_to_seconds(msg.sendTime - data.sendTime)

                    res = checks.intersection_check(
                        msg.sender.pos, msg.sender.pos_noise,
                        data.sender.pos, data.sender.pos_noise,
                        msg.sender.hed, data.sender.hed,
                        Coord(5, 1.8, 1.5), delta_time)

                    prediction_results['intersection_check'] += res

                    if res < 0.5:
                        activations['intersection_check'] = True
                        break

        if msg.sender_id not in history_data:
            history_data[msg.sender_id] = []

        history_data[msg.sender_id].append(msg)
        history_data[msg.sender_id] = history_data[msg.sender_id][-10:]

        threshold_prediction = 1 if any(activations.values()) else 0
        result['prediction_threshold'] = threshold_prediction
        result['prediction'] = threshold_prediction

        if mlp_model is not None:
            # Keep the MLP input order identical to training.
            feature_vector = np.array([
                float(np.clip(prediction_results.get(name, 0.0), 0.0, 1.0))
                for name in mlp_model.feature_names
            ], dtype=np.float64)
            result['prediction_mlp_score'] = mlp_model.predict_score(feature_vector)
            result['prediction_mlp'] = mlp_model.predict(feature_vector)

            if decision_type == 'mlp':
                result['prediction'] = result['prediction_mlp']

        for check_name, check_value in prediction_results.items():
            result[f'check_{check_name}'] = check_value

        return pd.Series(result)

    check_results = messages.apply(process_row, axis=1)
    for col in check_results.columns:
        messages[col] = check_results[col].values

    return messages


def perform_legacy_checks(messages: pd.DataFrame, checks: LegacyChecks) -> pd.DataFrame:
    history = dict()
    history_data = dict()

    def process_row(df_msg: pd.Series):
        msg = Mapper.row_to_message(df_msg)
        prediction_results = dict()
        activations = dict()
        result = dict()

        prediction_results['range_plausibility'] = checks.range_plausibility_check(msg.sender.pos, msg.receiver.pos)

        if prediction_results['range_plausibility'] == 0:
            activations['range_plausibility'] = True

        prediction_results['position_plausibility_check'] = checks.position_plausibility_check(
            msg.sender.spd, msg.sender.distance_to_road_edge)

        if prediction_results['position_plausibility_check'] == 0:
            activations['position_plausibility_check'] = True

        prediction_results['speed_plausibility_check'] = checks.speed_plausibility_check(msg.sender.spd)

        if prediction_results['speed_plausibility_check'] == 0:
            activations['speed_plausibility_check'] = True

        if (msg.sender_id not in history.keys() or MDMLib.ns_to_seconds(
                msg.rcvTime - history[msg.sender_id]) > Parameters.MAX_SA_TIME):
            prediction_results['sudden_appearance_check'] = checks.sudden_appearance_check(msg.sender.pos,
                                                                                           msg.receiver.pos)

            if prediction_results['sudden_appearance_check'] == 0:
                activations['sudden_appearance_check'] = True

        history[msg.sender_id] = msg.rcvTime

        sender_history = messages[
            (messages['sender_id'] == msg.sender_id) &
            (messages['sendTime'] < msg.sendTime)
            ].sort_values('sendTime', ascending=False)

        prev_msg = Mapper.row_to_message(sender_history.iloc[0]) if not sender_history.empty else None

        if prev_msg is not None:
            delta_time = MDMLib.ns_to_seconds(msg.sendTime - prev_msg.sendTime)

            prediction_results['position_consistency_check'] = checks.position_consistency_check(
                msg.sender.pos, prev_msg.sender.pos, delta_time)

            if prediction_results['position_consistency_check'] == 0:
                activations['position_consistency_check'] = True

            prediction_results['speed_consistency_check'] = checks.speed_consistency_check(
                msg.sender.spd, prev_msg.sender.spd, delta_time)

            if prediction_results['speed_consistency_check'] == 0:
                activations['speed_consistency_check'] = True

            prediction_results['position_speed_consistency_check'] = checks.position_speed_consistency_check(
                msg.sender.pos, prev_msg.sender.pos, msg.sender.spd, prev_msg.sender.spd, delta_time)

            if prediction_results['position_speed_consistency_check'] == 0:
                activations['position_speed_consistency_check'] = True

            prediction_results['position_heading_consistency_check'] = checks.position_heading_consistency_check(
                msg.sender.hed, msg.sender.pos, prev_msg.sender.pos, delta_time, msg.sender.spd)

            if prediction_results['position_heading_consistency_check'] == 0:
                activations['position_heading_consistency_check'] = True

        prediction_results['intersection_check'] = 0

        for sender_id, hist_data in history_data.items():
            if len(hist_data) > 0:
                data = hist_data[-1]

                if (MDMLib.ns_to_seconds(msg.rcvTime - data.rcvTime) <= checks.params.MAX_DELTA_INTERSECTION and
                        data.sender_id != msg.sender_id):

                    result = checks.intersection_check(msg.sender.pos, data.sender.pos, Coord(4, 2, 2),
                                                       Coord(4, 2, 2),
                                                       msg.sender.hed, data.sender.hed)

                    prediction_results['intersection_check'] += result
                    if result < 0.5:
                        activations['intersection_check'] = True
                        break

        if msg.sender_id not in history_data:
            history_data[msg.sender_id] = []

        history_data[msg.sender_id].append(msg)
        history_data[msg.sender_id] = history_data[msg.sender_id][-10:]

        result['prediction'] = 1 if any(activations.values()) else 0
        for check_name, check_value in prediction_results.items():
            result[f'check_{check_name}'] = check_value

        return pd.Series(result)

    check_results = messages.apply(process_row, axis=1)
    for col in check_results.columns:
        messages[col] = check_results[col].values

    return messages


def process(input_file: Path, option: int, params: Parameters, df: pd.DataFrame, source_file,
            mlp_model_path: str = None, decision_type: str = 'threshold'):
    if df.empty:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if len(data) < 1:
            print(f"Datei {input_file} ist leer, überspringe...", file=sys.stderr)
            return
        df = pd.json_normalize(data, sep='_')

    df = prepare_messages_dataframe(df)

    if option == 0:
        checks = CatchChecks(params)
        mlp_model = None
        if decision_type in ('mlp', 'both'):
            if mlp_model_path is None:
                raise ValueError(f"decision_type='{decision_type}' requires --mlp_model")
            mlp_model = CatchMLPDecision.from_file(mlp_model_path)

        results = perform_catch_checks(df, checks, mlp_model=mlp_model, decision_type=decision_type)
    elif option == 1:
        checks = LegacyChecks(params)
        results = perform_legacy_checks(df, checks)
    else:
        raise RuntimeError("Incorrect type chosen: ", option)

    #save_messages(results, input_file, source_file)

    metrics = calculate_metrics(results, 'prediction_threshold', '') if 'prediction_threshold' in results.columns else calculate_metrics(results)

    if 'prediction_mlp' in results.columns:
        metrics.update(calculate_metrics(results, 'prediction_mlp', '_mlp'))

    return metrics
