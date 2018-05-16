import pandas as pd
from datetime import datetime
import requests
import io
import os
from utils.STMVL import STMVL
import numpy as np


bj_particles = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
ld_particles = ['PM2.5', 'PM10', 'NO2']
meteros = ["temperature","pressure","humidity","wind_direction","wind_speed"]

file_paths = {
    'bj_aq': './data/Beijing/aq/beijing_17_18_aq.csv',
    'bj_aq_201802_201803': './data/Beijing/aq/beijing_201802_201803_aq.csv', 
    'bj_aq_new': './data/Beijing/aq/bj_aq_new.csv',
    'bj_meter_grid_hist': r'./Data/Beijing/Beijing_historical_meo_grid.csv',
    'bj_meter_grid_new': r'./Data/Beijing/bj_grid_meo_new.csv',
    'ld_aq': r'./Data/London/London_historical_aqi_forecast_stations_20180331.csv',
    'ld_aq_new': r'./Data/London/ld_aq_new.csv',
    'ld_aq_other': r'./Data/London/London_historical_aqi_other_stations_20180331.csv',
    'ld_meter_grid_hist': r'./Data/London/London_historical_meo_grid.csv',
    'ld_meter_grid_new': r'./Data/London/ld_meo_new.csv'
}

#================================================================
# main functions
#=================================================================

def datetime_formatter(date_time):
	year, month, day, hour = date_time.year, date_time.month, date_time.day, date_time.hour
	return '{}-{:02}-{:02}-{}'.format(year, month, day, hour)


def load_city_aq(city):
    # particles_aq = {}
    if city == 'bj':
        bj_aq = pd.read_csv(file_paths['bj_aq'])
        bj_aq_201802_201803 = pd.read_csv(file_paths['bj_aq_201802_201803'])
        bj_aq_new = pd.read_csv(file_paths['bj_aq_new'])
    return bj_aq, bj_aq_201802_201803, bj_aq_new
    #     bj_aq_new = pd.read_csv(file_paths['bj_aq_new'])
    #     df_temp = pd.concat([bj_aq, bj_aq_201802_201803, bj_aq_new], ignore_index=True)
    #     df_temp.sort_index(inplace=True)
    #     # return df_temp
    #     # deduplication
        # get all particles
        # particles = df_temp.columns.tolist()[2:]
        # for particle in particles:
        #     data = df_temp.pivot_table(index='utc_time', 
        #                        columns='stationId', 
        #                        values=particle)
        #     # rename columns
        #     data.columns = [name+'_{}'.format(particle) for name in data.columns.tolist()]
        #     particles_aq[particle] = data

    # if city == 'ld':
    #     ld_aq = pd.read_csv(file_paths['ld_aq'])
    #     ld_aq_other = pd.read_csv(file_paths['ld_aq_other'])
    #     ld_aq_new = pd.read_csv(file_paths['ld_aq_new'])

    #     # drop, rename ld_aq data to have same format as beijing data
    #     ld_aq.drop('Unnamed: 0', axis=1, inplace=True)
    #     ld_aq.rename(columns = {'station_id': 'stationId', 'MeasurementDateGMT': 'utc_time', 'PM2.5 (ug/m3)': 'PM2.5', 'PM10 (ug/m3)': 'PM10', 'NO2 (ug/m3)': 'NO2'}, inplace=True)
        
    #     # reformat time
    #     times = ld_aq['utc_time'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M'))
    #     ld_aq['utc_time'] = times.apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
        
    #     # other data
    #     ld_aq_other.drop(['Unnamed: 5', 'Unnamed: 6'], axis=1, inplace=True)
    #     ld_aq_other.rename(columns={'Station_ID': 'stationId', 'MeasurementDateGMT': 'utc_time', 'PM2.5 (ug/m3)': 'PM2.5', 'PM10 (ug/m3)': 'PM10', 'NO2 (ug/m3)': 'NO2'}, inplace=True)\
    #     # remove stations of NA
    #     ld_aq_other = ld_aq_other.loc[ld_aq_other['stationId'].notnull(), :]

    #     times = ld_aq_other['utc_time'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M'))
    #     ld_aq_other['utc_time'] = times.apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))

    #     df_temp = pd.concat([ld_aq, ld_aq_other, ld_aq_new], ignore_index=True)
    #     df_temp.sort_index(inplace=True)
    #     # reorder columns to be the same with beijing data
    #     df_temp = df_temp[['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2']]

    #     particles = df_temp.columns.tolist()[2:]
    #     for particle in particles:
    #         data = df_temp.pivot_table(index='utc_time', 
    #                            columns='stationId', 
    #                            values=particle)
    #         # rename columns
    #         data.columns = [name+'_{}'.format(particle) for name in data.columns.tolist()]
    #         particles_aq[particle] = data

    # return particles_aq # a dictionary of air quality data


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
            # reformat
            stations.drop(['SiteType', 'SiteName'], axis=1, inplace=True)
            if which == 'predict':
                stations = stations.loc[stations['need_prediction'] == True, :][['stationId', 'longitude', 'latitude']]
            elif which == 'need':
                stations = stations.loc[stations['need_prediction'].isnull(), :][['stationId', 'longitude', 'latitude']]
            else:
                stations = stations[['stationId', 'longitude', 'latitude']]
    return stations.reset_index(drop=True)


def load_city_meter_grid(city):
    grid_meteros_stations = {}
    if city == 'bj':
        bj_meter_grid_hist = pd.read_csv(file_paths['bj_meter_grid_hist'])
        bj_meter_grid_new = pd.read_csv(file_paths['bj_meter_grid_new'])

        bj_meter_grid_hist.drop(['longitude', 'latitude'], axis=1, inplace=True)
        bj_meter_grid_hist.rename(columns={'stationName': 'stationId', 'wind_speed/kph': 'wind_speed'}, inplace=True)
        
        # drop weather column from bj_meter_grid_new first, will add that feature later
        bj_meter_grid_new.drop('weather', axis=1, inplace=True)

        df_temp = pd.concat([bj_meter_grid_hist, bj_meter_grid_new])
        meteros = df_temp.columns.tolist()[2:]

        grid_meteros_stations = {}
        for metero in meteros:
            print(metero)
            data = df_temp.pivot_table(index='utc_time', 
                                        columns='stationId',
                                        values=metero)
            # meteros_stations = pd.concat([meteros_stations, data], axis=1)
            grid_meteros_stations[metero] = data

    if city == 'ld':
        ld_meter_grid_hist = pd.read_csv(file_paths['ld_meter_grid_hist'])
        ld_meter_grid_new = pd.read_csv(file_paths['ld_meter_grid_new'])

        ld_meter_grid_hist.drop(['longitude', 'latitude'], axis=1, inplace=True)
        ld_meter_grid_hist.rename(columns={'stationName': 'stationId', 'wind_speed/kph': 'wind_speed'}, inplace=True)
        
        # drop weather column
        ld_meter_grid_new.drop('weather', axis=1, inplace=True)

        df_temp = pd.concat([ld_meter_grid_hist, ld_meter_grid_new])
        meteros = df_temp.columns.tolist()[2:]

        grid_meteros_stations = {}
        for metero in meteros:
            print(metero)
            data = df_temp.pivot_table(index='utc_time', 
                                        columns='stationId',
                                        values=metero)
            # meteros_stations = pd.concat([meteros_stations, data], axis=1)
            grid_meteros_stations[metero] = data

    return grid_meteros_stations

def load_city_meter(city):
    grid_meteros_stations = load_city_meter_grid(city)
    stations = get_stations(city, grid=False, which='all')
    grids = get_stations(city, grid=True)
    # find nearest grids
    stations_grids = find_nearest(stations, grids)

    # use nearest grid meterology data as station weather condition
    metero_stations = {}
    for metero, data in grid_meteros_stations.items():
        temp_df = pd.DataFrame()
        col_names = []
        for station, nearest_grid in stations_grids.items():
            temp_df = pd.concat([temp_df, data[nearest_grid]], axis=1)
            col_names.append(station + '_{}'.format(metero))
        temp_df.columns = col_names
        metero_stations[metero] = temp_df

    return metero_stations


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

    # return grids_list[np.argsort(dist_list)[0]]
    print(np.argsort(dist_list))
    # return grids_list[np.argsort(dist_list)]
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