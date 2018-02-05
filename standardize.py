import argparse

import pandas as pd
import numpy as np


DB_NUMERICAL= ['distance',
               'age',
               'weight',
               'horse_weight',
               'horse_weight_difference',
               'enter_times',
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

RACE_NUMERICAL= ['age',
                 'weight',
                 'horse_weight',
                 'horse_weight_difference',
                 'enter_times',
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


def db_standardize(df):
    df.dropna(subset=DB_NUMERICAL, inplace=True)
    df.loc[:,'horse_weight_difference'] /= df['horse_weight']
    df.update(df[DB_NUMERICAL].apply(lambda x: (x - x.mean()) / x.std()))

    return df


def race_standardize(df):
    df.dropna(subset=RACE_NUMERICAL, inplace=True)
    df.update(
        df.groupby('race_id')[RACE_NUMERICAL].transform(
            lambda x: (x - x.mean()) / x.std()))

    return df


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

    df = db_standardize(pd.read_csv(args.infile))
    race_standardize(df).to_csv(args.outfile, index=False)

