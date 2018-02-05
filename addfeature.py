import argparse

import pandas as pd
import numpy as np


NEW_FETURES = ['enter_times',
               'win_rate',
               'mean_prise',
               'jocky_enter_times',
               'jocky_win_rate',
               'jocky_mean_prise',
               'trainer_times',
               'trainer_win_rate',
               'trainer_mean_prise',
               'owner_times',
               'owner_win_rate',
               'owner_mean_prise']


def calculate_metrics(data, date):
    past_data = data[data['date'] < date]

    times = past_data['race_id'].count()

    if times == 0:
        return 0.0, 0.0, 0.0

    times = float(times)
    win_times = float(past_data.query('order==1.0')['race_id'].count())
    win_rate = win_times / times
    prise = past_data['prise'].sum()
    mean_prise = prise / times

    return times, win_rate, mean_prise


def search_data(df, key, name, memo):
    if name in memo:
        return memo[name]

    past_data = df.query(key + '=="' + name + '"')
    memo[name] = past_data

    return past_data


def add_new_features(df):
    df = df.fillna({'prise': 0.0})
    df['date'] = pd.to_datetime(df['date'])

    length = len(df)
    new_array = np.zeros((length, len(NEW_FETURES)), dtype=np.float32)
    memo = {'name': {}, 'jocky': {}, 'trainer': {}, 'owner': {}}

    for i, (_, row) in enumerate(df.iterrows()):
        j = 0

        for key in ['name', 'jocky', 'trainer', 'owner']:
            data = search_data(df, key, row[key], memo[key])
            times, win_rate, mean_prise = calculate_metrics(data, row['date'])
            new_array[i][j] = times
            j += 1
            new_array[i][j] = win_rate
            j += 1
            new_array[i][j] = mean_prise
            j += 1

    return pd.concat([df, pd.DataFrame(new_array, columns=NEW_FETURES)], axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--infile',
                        help='入力となる CSV ファイル',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--outfile',
                        help='出力となる CSV ファイル',
                        type=str,
                        required=True)
    args = parser.parse_args()
    df = add_new_features(pd.read_csv(args.infile))
    df.to_csv(args.outfile)

