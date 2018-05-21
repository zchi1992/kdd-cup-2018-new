import pandas as pd
from utils.STMVL import STMVL
import numpy as np


bj_particles = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
ld_particles = ['PM2.5', 'PM10', 'NO2']
meteros = ["temperature","pressure","humidity","wind_direction","wind_speed"]

file_paths = {
    'bj_aq': './data/Beijing/aq/beijing_17_18_aq.csv',
    'bj_aq_201802_201803': './data/Beijing/aq/beijing_201802_201803_aq.csv', 
    'bj_aq_new': './data/Beijing/aq/bj_aq_new.csv',
    'bj_meter_grid_hist': './data/Beijing/meo/Beijing_historical_meo_grid.csv',
    'bj_meter_grid_new': './data/Beijing/meo/bj_meo_grid_new.csv',
    'ld_aq': r'./data/London/aq/London_historical_aqi_forecast_stations_20180331.csv',
    'ld_aq_new': r'./data/London/aq/ld_aq_new.csv',
    'ld_aq_other': r'./data/London/aq/London_historical_aqi_other_stations_20180331.csv',
    'ld_meter_grid_hist': r'./data/London/meo/London_historical_meo_grid.csv',
    'ld_meter_grid_new': r'./data/London/meo/ld_meo_grid_new.csv'
}


def datetime_formatter(date_time):
    year, month, day, hour = date_time.year, date_time.month, date_time.day, date_time.hour
    return '{}-{:02}-{:02}-{}'.format(year, month, day, hour)


def load_city_aq(city):
    if city == 'bj':
        bj_aq = pd.read_csv(file_paths['bj_aq'])
        bj_aq_201802_201803 = pd.read_csv(file_paths['bj_aq_201802_201803'])
        bj_aq_new = pd.read_csv(file_paths['bj_aq_new'])
        return bj_aq, bj_aq_201802_201803, bj_aq_new
    else:
        ld_aq = pd.read_csv(file_paths['ld_aq'])
        ld_aq_other = pd.read_csv(file_paths['ld_aq_other'], dtype={'Station_ID': str, 'MeasurementDateGMT': str})
        ld_aq_new = pd.read_csv(file_paths['ld_aq_new'])
        return ld_aq, ld_aq_other, ld_aq_new


def get_stations(city, grid=False, which='predict'):
    if city == 'bj':
        if grid:
            stations = pd.read_csv(r'./data/Beijing/Beijing_grid_weather_station.csv')
        else:
            stations = pd.read_excel(r'./data/Beijing/Beijing_AirQuality_Stations_locations.xlsx')
    else:
        if grid:
            stations = pd.read_csv(r'./data/London/London_grid_weather_station.csv')
        else:
            stations = pd.read_csv(r'./data/London/London_AirQuality_Stations.csv')
            if which == 'predict':
                stations = stations.loc[stations['need_prediction'], :][['Unnamed: 0', 'Longitude', 'Latitude']]
            elif which == 'need':
                stations = stations.loc[stations['need_prediction'].isnull(), :][['Unnamed: 0', 'Longitude', 'Latitude']]
            else:
                stations = stations[['Unnamed: 0', 'Longitude', 'Latitude']]
            # rename
            stations.rename(columns={'Unnamed: 0': 'stationId', 'Longitude': 'longitude', 'Latitude': 'latitude'}, inplace=True)
    return stations.reset_index(drop=True)


def load_city_meter_grid(city):
    if city == 'bj':
        bj_meter_grid_hist = pd.read_csv(file_paths['bj_meter_grid_hist'])
        bj_meter_grid_new = pd.read_csv(file_paths['bj_meter_grid_new'])
        return bj_meter_grid_hist, bj_meter_grid_new
    else:
        ld_meter_grid_hist = pd.read_csv(file_paths['ld_meter_grid_hist'])
        ld_meter_grid_new = pd.read_csv(file_paths['ld_meter_grid_new'])
        return ld_meter_grid_hist, ld_meter_grid_new


# def load_city_meter(city):
#     grid_meteros_stations = load_city_meter_grid(city)
#
#     stations = get_stations(city, grid=False, which='all')
#     grids = get_stations(city, grid=True)
#
#     # find nearest grids
#     stations_grids = find_nearest(stations, grids)
#
#     # use nearest grid meterology data as station meterological condition
#     metero_stations = {}
#     for metero, data in grid_meteros_stations.items():
#         temp_df = pd.DataFrame()
#         col_names = []
#         for station, nearest_grid in stations_grids.items():
#             temp_df = pd.concat([temp_df, data[nearest_grid]], axis=1)
#             col_names.append(station + '_{}'.format(metero))
#         temp_df.columns = col_names
#         metero_stations[metero] = temp_df
#
#     return metero_stations


def find_nearest_single(station, all_stations):
    lat1 = station['latitude']
    lng1 = station['longitude']
    grids_list = []
    dist_list = []
    for _, series in all_stations.iterrows():
        if series['stationId'] == station['stationId']:
            continue
        lat2 = series['latitude']
        lng2 = series['longitude']
        dist = STMVL.geo_distance(lat1, lng1, lat2, lng2)
        grids_list.append(series['stationId'])
        dist_list.append(dist)

    return [grids_list[i] for i in np.argsort(dist_list)]


def find_nearest(stations, all_stations):
    stations_grids = {}
    for _, station in stations.iterrows():
        stations_grids[station['stationId']] = find_nearest_single(station, all_stations)
    return stations_grids


def flat_data(data_dict):
    df = pd.DataFrame()
    for feature, data in data_dict.items():
        df = pd.concat([df, data], axis=1)
    return df