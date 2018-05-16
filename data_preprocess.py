from utils.data_utils import *
import numpy as np
from datetime import datetime
import pandas as pd
from utils.STMVL import STMVL


def aq_city_preprocess(city):
    print('start processing {} air quality'.format(city))
    stations = get_stations(city, grid=False, which='all')
    aq, aq_historical, aq_new = load_city_aq(city)

    # process aq_new data
    aq_new.drop('id', axis=1, inplace=True)
    aq_new.rename(columns={'station_id': 'stationId', 'time': 'utc_time', 'PM25_Concentration': 'PM2.5', 'PM10_Concentration': 'PM10', 'NO2_Concentration': 'NO2', 'CO_Concentration': 'CO', 'O3_Concentration': 'O3', 'SO2_Concentration': 'SO2'}, 
                            inplace=True)

    # merge
    df_temp = pd.concat([aq, aq_historical, aq_new], ignore_index=True)
    df_temp.sort_index(inplace=True)

    # preprocess for every particle
    if city == 'bj':
        for particle in bj_particles:
            data = df_temp.pivot_table(index='utc_time', 
                               columns='stationId', 
                               values=particle)
            # rename columns
            data.columns = [name+'_{}'.format(particle) for name in data.columns.tolist()]
            # filling missing data
            data_filled_hour = fill_missing_time_with_na(data)
            missing_len = data_filled_hour.loc[data_filled_hour.isnull().any(axis=1), :].shape[0]
            print('There are {} missing in the data'.format(missing_len))

            # fill missing values
            data_filled_na = fill_missing_single_data(data_filled_hour, stations)
            missing_still_len = data_filled_na.loc[data_filled_na.isnull().any(axis=1), :].shape[0]
            print('There are {} still missing in the data \n'.format(missing_still_len))
            data_filled_na.to_csv(r'./data/Beijing/post/{}_{}_filled.csv'.format(city, particle))
    # deal with missing left
    #     # don't do anything
    #     # use forward fill to

    #     # finish filling missing data

    # # de-duplication
    # # already handled by pivot table
    # # london air quality data needs to be filled with all staions first
    # if city == 'ld':
    #     # fill particles_aq with all stations
    #     for particle, data in particles_aq.items():
    #         # find missing stations
    #         stations_all = stations['stationId'].tolist()
    #         stations_exist = [name.rsplit('_', maxsplit=1)[0] for name in data.columns.tolist()]
    #         stations_missing = list(np.setdiff1d(stations_all, stations_exist))

    #         # find nearest stations
    #         nearest_stations = find_nearest(stations, stations)

    #         if len(stations_missing) == 0:
    #             continue

    #         # find the nearest station to fill the air quality data
    #         missing_df = pd.DataFrame(index=data.index, columns=stations_missing)
    #         for station in stations_missing:
    #             col = nearest_stations[station] + '_{}'.format(particle)
    #             missing_df[station] = data[col]
    #         data = pd.concat([data, missing_df], axis=1)
    #         data.sort_index(axis=1, inplace=True)
    #         particles_aq[particle] = data

    # # missing dates
    # for particle, data in particles_aq.items():
    #     print('Fill missing data for {}'.format(particle))

    #     # fill missing dates
    #     data_filled_time = fill_missing_time_with_na(data)
    #     missing_len = data_filled_time.loc[data_filled_time.isnull().any(axis=1), :].shape[0]
    #     print('There are {} missing in the data'.format(missing_len))

    #     # fill missing values
    #     data_filled_na = fill_missing_single_data(data_filled_time, stations)
    #     missing_still_len = data_filled_na.loc[data_filled_na.isnull().any(axis=1), :].shape[0]
    #     print('There are {} still missing in the data \n'.format(missing_still_len))
    #     # deal with missing left
    #     # don't do anything
    #     # use forward fill to 

    #     particles_aq[particle] = data_filled_na
    #     if city == 'bj':
    #         data_filled_na.to_csv(r'./Data/Beijing/{}_{}_filled.csv'.format(city, particle), index=True)
    #     else:
    #         data_filled_na.to_csv(r'./Data/London/{}_{}_filled.csv'.format(city, particle), index=True)
    
    # return particles_aq

    
def meter_preprocess(city):
    print('start working on {} meterology '.format(city))
    metero_stations = load_city_meter(city)
    stations = get_stations(city, grid=False, which='all')
    # deduplication
    # there's no duplication in the data

    for metero, data in metero_stations.items():
        print('Fill missing data for {}'.format(metero))

        # fill missing dates
        data_filled_time = fill_missing_time_with_na(data)
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
        # deal with missing left
        # don't do anything
        metero_stations[metero] = data_filled_na

        if city == 'bj':
            data_filled_na.to_csv(r'./Data/Beijing/{}_{}_filled.csv'.format(city, metero), index=True)
        else:
            data_filled_na.to_csv(r'./Data/London/{}_{}_filled.csv'.format(city, metero), index=True)
    return metero_stations


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

# def fill_missing_grid_metero(df):


# def normalization(df):
#     return df.describe()

#==============================================================
# Private functions
#==============================================================
def _get_missing_time(missing_time):
    missing_time_list = []
    for time in missing_time:
        ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        date_time = datetime.utcfromtimestamp(ts)
        missing_time_list.append(datetime.strftime(date_time, '%Y-%m-%d %H:%M:%S'))
    return missing_time_list


#==============================================================
# Main
#==============================================================
if __name__ == '__main__':
    for city in ['bj','ld']:
        city_aq = aq_city_preprocess(city)
        print('Verify {} air quality data ...'.format(city))
        for particle, data in city_aq.items():
            print('shape of {}_{} is '.format(city, particle), data.shape)
            print('number of missing values of {}_{} is '.format(city, particle), data.loc[data.isnull().any(axis=1), :].shape)
        
        print('Verify {} meterology data ...'.format(city))
        city_meo = meter_preprocess(city)
        for meo, data in city_meo.items():
            print('shape of {}_{} is '.format(city, meo), data.shape)
            print('number of missing values of {}_{} is '.format(city, meo), data.loc[data.isnull().any(axis=1), :].shape)