import json
from pathlib import Path

import numpy as np

np.seterr(all='warn', over='raise')

from data_structures import Mapper, Parameters, Coord
from catch_checks import CatchChecks
from mdm_lib import MDMLib

import pandas as pd


def save_messages(results: pd.DataFrame, input_file: Path, source_file):
    output_dir = input_file.parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{source_file}_predicted.json"

    nested_data = results.apply(Mapper.row_to_json, axis=1).tolist()
    with open(output_file, 'w') as f:
        json.dump(nested_data, f, indent=4)


def calculate_metrics(results: pd.DataFrame) -> dict:
    tp = ((results['attacker'] == 1) & (results['prediction'] == 1)).sum()

    tn = ((results['attacker'] == 0) & (results['prediction'] == 0)).sum()

    fp = ((results['attacker'] == 0) & (results['prediction'] == 1)).sum()

    fn = ((results['attacker'] == 1) & (results['prediction'] == 0)).sum()

    metrics = {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_messages': int(len(results)),
    }
    metrics['alias_grace_messages'] = int(
        results.get('catch_alias_grace', pd.Series(dtype=bool)).sum()
    )
    message_types = results.get('message_type', pd.Series(dtype=str))
    metrics['cam_messages'] = int((message_types == 'CAM').sum())
    metrics['cpm_messages'] = int((message_types == 'CPM').sum())
    metrics['check_activations'] = {
        column.removeprefix('check_'): int((results[column] < 0.5).sum())
        for column in results.columns if column.startswith('check_')
    }
    return metrics


def load_catch_messages(cam_path=None, cpm_path=None):
    records = []
    for message_type, path in (("CAM", cam_path), ("CPM", cpm_path)):
        if path is None:
            continue
        with Path(path).open("r", encoding="utf-8") as file:
            for message in json.load(file):
                record = dict(message)
                record["message_type"] = message_type
                records.append(record)
    frame = pd.json_normalize(records, sep='_')
    if frame.empty:
        return frame
    return frame.sort_values(
        ['rcvTime', 'sendTime', 'messageID', 'message_type'],
        kind='mergesort', ignore_index=True,
    )


def perform_catch_checks(messages: pd.DataFrame, checks: CatchChecks, use_alias=False) -> pd.DataFrame:
    histories = {}
    latest_messages = {}
    first_seen = {}
    results = []

    for _, row in messages.iterrows():
        msg = Mapper.row_to_message(row)
        message_type = row.get('message_type', 'CAM')
        identity = msg.sender_alias if use_alias else msg.sender_id
        history_key = (message_type, identity)
        first_seen.setdefault(history_key, msg.rcvTime)
        alias_age = MDMLib.ns_to_seconds(msg.rcvTime - first_seen[history_key])
        grace = use_alias and alias_age < checks.params.MAX_DELTA_INTERSECTION
        skipped = []

        factors = {
            'range_plausibility': checks.range_plausibility_check(
                msg.receiver.pos, msg.receiver.pos_noise,
                msg.sender.pos, msg.sender.pos_noise,
            ),
            'position_plausibility_check': (
                checks.position_plausibility_check(
                    msg.sender.pos_noise, msg.sender.spd, msg.sender.spd_noise,
                    msg.sender.distance_to_road_edge,
                ) if checks.params.POSITION_PLAUSIBILITY_ENABLED else 1.0
            ),
            'speed_plausibility_check': checks.speed_plausibility_check(
                msg.sender.spd, msg.sender.spd_noise,
            ),
        }

        prior_messages = histories.get(history_key, [])
        earlier_messages = [item for item in prior_messages if item.sendTime < msg.sendTime]
        previous = max(earlier_messages, key=lambda item: item.sendTime, default=None)
        if previous is None:
            skipped.extend([
                'position_consistency_check',
                'speed_consistency_check',
                'position_speed_consistency_check',
                'position_heading_consistency_check',
            ])
        else:
            delta_time = MDMLib.ns_to_seconds(msg.sendTime - previous.sendTime)
            factors['position_consistency_check'] = checks.position_consistency_check(
                msg.sender.pos, msg.sender.pos_noise,
                previous.sender.pos, previous.sender.pos_noise, delta_time,
            )
            factors['speed_consistency_check'] = checks.speed_consistency_check(
                msg.sender.spd, msg.sender.spd_noise,
                previous.sender.spd, previous.sender.spd_noise, delta_time,
            )
            factors['position_speed_consistency_check'] = checks.position_speed_consistency_check(
                msg.sender.pos, msg.sender.pos_noise,
                previous.sender.pos, previous.sender.pos_noise,
                msg.sender.spd, msg.sender.spd_noise,
                previous.sender.spd, previous.sender.spd_noise, delta_time,
            )
            factors['position_heading_consistency_check'] = checks.position_heading_consistency_check(
                msg.sender.hed, msg.sender.hed_noise,
                previous.sender.pos, previous.sender.pos_noise,
                msg.sender.pos, msg.sender.pos_noise, delta_time,
                msg.sender.spd, msg.sender.spd_noise,
            )

        if grace:
            skipped.append('intersection_check')
        else:
            intersection_factors = []
            for (other_type, other_identity), other in latest_messages.items():
                time_delta = MDMLib.ns_to_seconds(msg.rcvTime - other.rcvTime)
                if (other_type != message_type or other_identity == identity
                        or not 0 <= time_delta <= checks.params.MAX_DELTA_INTERSECTION):
                    continue
                intersection_factors.append(checks.intersection_check(
                    msg.sender.pos, msg.sender.pos_noise,
                    other.sender.pos, other.sender.pos_noise,
                    msg.sender.hed, other.sender.hed,
                    Coord(5, 1.8, 1.5),
                    MDMLib.ns_to_seconds(msg.sendTime - other.sendTime),
                ))
            factors['intersection_check'] = min(intersection_factors, default=1.0)

        result = {
            'prediction': int(any(value < 0.5 for value in factors.values())),
            'catch_identity': str(identity),
            'catch_alias_grace': grace,
            'catch_skipped_checks': skipped,
        }
        result.update({f'check_{name}': value for name, value in factors.items()})
        results.append(result)

        histories.setdefault(history_key, []).append(msg)
        latest_messages[history_key] = msg

    check_results = pd.DataFrame(results, index=messages.index)
    for column in check_results.columns:
        messages[column] = check_results[column]
    return messages


def apply_catch_gate(messages: pd.DataFrame, params: Parameters) -> pd.DataFrame:
    if messages.empty:
        return messages.copy()

    normalized = pd.json_normalize(messages.to_dict(orient='records'), sep='_')
    results = apply_alias_catch_checks(normalized, params)
    check_columns = [
        column for column in results.columns if column.startswith('check_')
    ]

    gated = messages.reset_index(drop=True).copy()
    gated['catch_prediction'] = results['prediction'].astype(int)
    gated['catch_failed_checks'] = results.apply(
        lambda row: [
            column.removeprefix('check_')
            for column in check_columns
            if pd.notna(row[column]) and row[column] < 0.5
        ],
        axis=1,
    )
    return gated


def apply_alias_catch_checks(df: pd.DataFrame, params: Parameters):
    # Metadaten Felder
    df['rcvTime'] = df['rcvTime'].astype(int)
    df['sendTime'] = df['sendTime'].astype(int)
    df['sender_id'] = df['sender_id'].astype(str)
    df['sender_alias'] = df['sender_alias'].astype(int)
    df['messageID'] = df['messageID'].astype(str)
    df['attacker'] = df['attacker'].astype(int)
    df['prediction'] = 0

    # Receiver Felder
    df['receiver_spd'] = df['receiver_spd'].astype(np.float64)
    df['receiver_spd_noise'] = df['receiver_spd_noise'].astype(np.float64)
    df['receiver_acl'] = df['receiver_acl'].astype(float)
    df['receiver_acl_noise'] = df['receiver_acl_noise'].astype(float)
    df['receiver_hed'] = df['receiver_hed'].astype(float)
    df['receiver_hed_noise'] = df['receiver_hed_noise'].astype(float)
    df['receiver_driversProfile'] = df['receiver_driversProfile'].astype(str)

    # Sender Felder
    df['sender_spd'] = df['sender_spd'].astype(np.float64)
    df['sender_spd_noise'] = df['sender_spd_noise'].astype(np.float64)
    df['sender_acl'] = df['sender_acl'].astype(float)
    df['sender_acl_noise'] = df['sender_acl_noise'].astype(float)
    df['sender_hed'] = df['sender_hed'].astype(float)
    df['sender_hed_noise'] = df['sender_hed_noise'].astype(float)
    df['sender_driversProfile'] = df['sender_driversProfile'].astype(str)
    if 'sender_distance_to_road_edge' not in df:
        df['sender_distance_to_road_edge'] = 0.0
    df['sender_distance_to_road_edge'] = df['sender_distance_to_road_edge'].astype(float)

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

    df = df.sort_values(['rcvTime', 'sendTime', 'messageID'], kind='mergesort').reset_index(drop=True)

    checks = CatchChecks(params)
    return perform_catch_checks(df, checks, use_alias=True)
