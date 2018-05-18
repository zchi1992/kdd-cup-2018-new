from utils.data_utils import *
import numpy as np
from datetime import datetime
import pandas as pd
from utils.STMVL import STMVL
import os

post_paths = {
    'bj_post': './data/Beijing/post/',
    'ld_post': './data/London/post/'
}


def preprocess_main(city):
    if city == 'bj':
        aq_city_preprocess(city, bj_particles)
    else:
        aq_city_preprocess(city, ld_particles)
    meter_preprocess(city)

    # validate results
    print('Validate results:\n')
    _validate_results(city)


def aq_city_preprocess(city, particles):
    print('start processing {} air quality'.format(city))
    stations = get_stations(city, grid=False, which='all')

    if city == 'bj':
        df_temp = bj_aq_merge()
    else:
        df_temp = ld_aq_merge()

    # fill particles_aq with all stations
    for particle in particles:
        data = df_temp.pivot_table(index='utc_time',
                                   columns='stationId',
                                   values=particle)
        # rename columns
        data.columns = [name + '_{}'.format(particle) for name in data.columns.tolist()]

        # special process for london air quality
        if city == 'ld':
            # find missing stations
            stations_all = stations['stationId'].tolist()
            stations_exist = [name.rsplit('_', maxsplit=1)[0] for name in data.columns.tolist()]
            stations_missing = list(np.setdiff1d(stations_all, stations_exist))

            # find nearest stations
            nearest_stations = find_nearest(stations, stations)

            if len(stations_missing) != 0:
                # find the nearest station to fill the air quality data
                missing_df = pd.DataFrame(index=data.index, columns=stations_missing)
                for station in stations_missing:
                    col = nearest_stations[station][0] + '_{}'.format(particle) # the nearest of 'BX9' is 'BX1'
                    missing_df[station] = data[col]
                missing_df.columns = [col + '_{}'.format(particle) for col in missing_df.columns.tolist()]

                data = pd.concat([data, missing_df], axis=1)
                data.sort_index(axis=1, inplace=True)

        print('Filling missing data for {}\n'.format(particle))

        data_filled_hour = fill_missing_time_with_na(data)
        missing_len = data_filled_hour.loc[data_filled_hour.isnull().any(axis=1), :].shape[0]
        print('There are {} missing in the data\n'.format(missing_len))

        # fill missing values
        data_filled_na = fill_missing_single_data(data_filled_hour, stations)
        missing_still_len = data_filled_na.loc[data_filled_na.isnull().any(axis=1), :].shape[0]
        print('There are {} still missing in the data \n'.format(missing_still_len))

        # for rest of missing data, we use forward fill
        data_filled_na.fillna(method='ffill', inplace=True)

        # for negative values, we replace them with 0
        data_filled_na[data_filled_na < 0] = 0

        # write out to disk
        if city == 'bj':
            data_filled_na.to_csv(post_paths['bj_post'] + '{}_{}_filled.csv'.format(city, particle))
        else:
            data_filled_na.to_csv(post_paths['ld_post'] + '{}_{}_filled.csv'.format(city, particle))


def meter_preprocess(city):
    print('start working on {} meterology '.format(city))
    stations = get_stations(city, grid=False, which='all')
    grids = get_stations(city, grid=True)

    stations_grids = find_nearest(stations, grids)

    metero_hist, metero_new = load_city_meter_grid(city)

    # process historical data
    metero_hist.drop(['longitude', 'latitude'], axis=1, inplace=True)
    metero_hist.rename(columns={'stationName': 'stationId', 'wind_speed/kph': 'wind_speed'}, inplace=True)

    # process new data
    metero_new.drop('id', axis=1, inplace=True)
    metero_new.rename(columns={'time': 'utc_time', 'station_id': 'stationId'}, inplace=True)
    metero_new.drop('weather', axis=1, inplace=True)  # TODO: Add weather later

    df_temp = pd.concat([metero_hist, metero_new], ignore_index=True)

    for metero in meteros:
        data = df_temp.pivot_table(index='utc_time',
                                    columns='stationId',
                                    values=metero)
        stations_metero = pd.DataFrame() # build meterological features
        col_names = []
        for station, nearest_grids in stations_grids.items():
            stations_metero = pd.concat([stations_metero, data[nearest_grids[0]]], axis=1)
            col_names.append(station + '_{}'.format(metero))
        stations_metero.columns = col_names

        print('Fill missing data for {}'.format(metero))

        # fill missing dates
        data_filled_time = fill_missing_time_with_na(stations_metero)
        missing_len = data_filled_time.loc[data_filled_time.isnull().any(axis=1), :].shape[0]
        print('There are {} missing in the data'.format(missing_len))

        # if there's no missing value, skip filling process
        if missing_len == 0:
            print('There"s no missing value in the data')
            continue

        # fill missing values
        data_filled_na = fill_missing_single_data(data_filled_time, stations)

        missing_still_len = data_filled_na.loc[data_filled_na.isnull().any(axis=1), :].shape[0]
        print('There are {} still missing in the data \n'.format(missing_still_len))

        # for rest of missing data, we use forward fill
        data_filled_na.fillna(method='ffill', inplace=True)

        if city == 'bj':
            data_filled_na.to_csv(post_paths['bj_post'] + '{}_{}_filled.csv'.format(city, metero))
        else:
            data_filled_na.to_csv(post_paths['ld_post'] + '{}_{}_filled.csv'.format(city, metero))

    # # deduplication
    # # there's no duplication in the data
    #
    # for metero, data in metero_stations.items():
    #     print('Fill missing data for {}'.format(metero))
    #
    #     # fill missing dates
    #     data_filled_time = fill_missing_time_with_na(data)
    #     missing_len = data_filled_time.loc[data_filled_time.isnull().any(axis=1), :].shape[0]
    #     print('There are {} missing in the data'.format(missing_len))
    #
    #     # if there's no missing value, skip filling process
    #     if missing_len == 0:
    #         print('There"s no missing value in the data')
    #         continue
    #
    #     # fill missing values
    #     data_filled_na = fill_missing_single_data(data_filled_time, stations)
    #
    #     missing_still_len = data_filled_na.loc[data_filled_na.isnull().any(axis=1), :].shape[0]
    #     print('There are {} still missing in the data \n'.format(missing_still_len))
    #     # deal with missing left
    #     # don't do anything
    #     metero_stations[metero] = data_filled_na
    #
    #     if city == 'bj':
    #         data_filled_na.to_csv(r'./Data/Beijing/{}_{}_filled.csv'.format(city, metero), index=True)
    #     else:
    #         data_filled_na.to_csv(r'./Data/London/{}_{}_filled.csv'.format(city, metero), index=True)
    # return metero_stations

# ==============================================================
# Private functions
# ==============================================================


def bj_aq_merge():
    aq, aq_historical, aq_new = load_city_aq('bj')

    # process aq_new data
    aq_new.drop('id', axis=1, inplace=True)
    aq_new.rename(columns={'station_id': 'stationId', 'time': 'utc_time', 'PM25_Concentration': 'PM2.5',
                           'PM10_Concentration': 'PM10', 'NO2_Concentration': 'NO2', 'CO_Concentration': 'CO',
                           'O3_Concentration': 'O3', 'SO2_Concentration': 'SO2'},
                  inplace=True)

    # merge
    df_temp = pd.concat([aq, aq_historical, aq_new], ignore_index=True)
    df_temp.sort_index(inplace=True)
    return df_temp


def ld_aq_merge():
    aq, aq_other, aq_new = load_city_aq('ld')

    # drop, rename ld_aq data to have same format as beijing data
    aq.drop('Unnamed: 0', axis=1, inplace=True)
    aq.rename(columns={'station_id': 'stationId', 'MeasurementDateGMT': 'utc_time', 'PM2.5 (ug/m3)': 'PM2.5',
                       'PM10 (ug/m3)': 'PM10', 'NO2 (ug/m3)': 'NO2'}, inplace=True)

    # reformat time
    times = aq['utc_time'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M'))
    times = times.apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    aq.loc[:, 'utc_time'] = times

    # other data
    aq_other.drop(['Unnamed: 5', 'Unnamed: 6'], axis=1, inplace=True)
    aq_other.rename(
        columns={'Station_ID': 'stationId', 'MeasurementDateGMT': 'utc_time', 'PM2.5 (ug/m3)': 'PM2.5',
                 'PM10 (ug/m3)': 'PM10', 'NO2 (ug/m3)': 'NO2'}, inplace=True) \

    # remove stations of NA
    aq_other = aq_other.loc[aq_other['stationId'].notnull(), :]

    times = aq_other['utc_time'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M'))
    times = times.apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    aq_other.loc[:, 'utc_time'] = times

    # new data
    aq_new.drop(['id', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration'], axis=1, inplace=True)
    aq_new.rename(columns={'station_id': 'stationId', 'time': 'utc_time', 'PM25_Concentration': 'PM2.5',
                           'PM10_Concentration': 'PM10', 'NO2_Concentration': 'NO2'}, inplace=True)

    df_temp = pd.concat([aq, aq_other, aq_new], ignore_index=True)
    df_temp.sort_index(inplace=True)

    # reorder columns to be the same with beijing data
    df_temp = df_temp[['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2']]

    return df_temp


def _get_missing_time(missing_time):
    missing_time_list = []
    for time in missing_time:
        ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        date_time = datetime.utcfromtimestamp(ts)
        missing_time_list.append(datetime.strftime(date_time, '%Y-%m-%d %H:%M:%S'))
    return missing_time_list


def _validate_results(city):
    dir_path = post_paths[city + '_post']
    if os.path.isdir(dir_path):
        files = os.listdir(dir_path)

    for file in files:
        file_path = os.path.join(dir_path, file)
        print('reading file from {}'.format(file_path))
        # if len(file.split('_')) > 3: # features like wind_speed and wind_direction
        #     feature_name = '_'.join(file.split('_')[1:3]) # extract "wind_speed" from "bj_wind_speed_filled"
        # else:
        #     feature_name = file.split('_')[1] # bj_PM2.5_filled will extract PM2.5

        df_filled = pd.read_csv(file_path, index_col=0)
        print('shape of {} is '.format(file), df_filled.shape)
        print('number of missing values of {} is '.format(file), df_filled.loc[df_filled.isnull().any(axis=1), :].shape[0])


def fill_missing_time_with_na(df):
    start_time = df.index.min()
    end_time = df.index.max()
    all_times = pd.date_range(start_time, end_time, freq='H')
    print('There should be {} date time in total'.format(len(all_times)))
    missing_time = np.setdiff1d(all_times.values, df.index.values.astype(np.datetime64))
    missing_time = _get_missing_time(missing_time)

    # fill missing dates
    missing_df = pd.DataFrame(np.nan, index=missing_time, columns=df.columns)
    missing_df.index = missing_time
    df_fill = pd.concat([df, missing_df], axis=0)
    print('After filling, there are {} date time in total'.format(df_fill.shape[0]))

    return df_fill.sort_index(axis=0)


def fill_missing_single_data(df, stations):
    latitude = np.zeros([len(stations)])
    longitude = np.zeros([len(stations)])
    for i, row in stations.iterrows():
        station = row['stationId']
        longitude[i] = stations.loc[stations['stationId'] == station, 'longitude']
        latitude[i] = stations.loc[stations['stationId'] == station, 'latitude']
    stmvl = STMVL()
    data_filled = stmvl.fit_transform(df.values, latitude, longitude)
    # convert to pandas data frame
    return pd.DataFrame(data_filled, index=df.index, columns=df.columns)


# ==============================================================
# Main
# ==============================================================

if __name__ == '__main__':
    for city in ['bj','ld']:
        preprocess_main(city)