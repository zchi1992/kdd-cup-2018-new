from utils.data_utils import datetime_formatter
from datetime import datetime, timedelta
import requests
import pandas as pd
import io

url_templates = {
    'bj_aq': 'https://biendata.com/competition/airquality/bj/{start_time}/{end_time}/2k0d1d8', 
    'bj_meteorology': 'https://biendata.com/competition/meteorology/bj/{start_time}/{end_time}/2k0d1d8',
    'bj_meteorology_grid': 'https://biendata.com/competition/meteorology/bj_grid/{start_time}/{end_time}/2k0d1d8',
    'ld_aq': 'https://biendata.com/competition/airquality/ld/{start_time}/{end_time}/2k0d1d8',
    'ld_meteorology_grid': 'https://biendata.com/competition/meteorology/ld_grid/{start_time}/{end_time}/2k0d1d8'
}

save_paths = {
    'bj_aq': r'./data//Beijing/aq/bj_aq_new.csv',
    'bj_meteorology_grid': r'./data/Beijing/meo/bj_meo_grid_new.csv',
    'ld_aq': r'./data/London/aq/ld_aq_new.csv',
    'ld_meteorology_grid': r'./data/London/meo/ld_meo_grid_new.csv'
}

def request_data(url):
    respones = requests.get(url, timeout=100)
    # write to files
    if respones.status_code == requests.codes.ok:
        df_url = pd.read_csv(io.StringIO(respones.content.decode('utf-8')))
        return df_url
    return


def request_main(start_time, end_time):
    
    # request beijing air quality data
    print('requesting beijing air quality data to {}'.format(end_time))
    bj_aq_new = request_data(url_templates['bj_aq'].format(start_time=start_time, end_time=end_time))
    bj_aq_new.to_csv(save_paths['bj_aq'], index=False)
    # bj_aq_new.drop('id', axis=1, inplace=True)
    # bj_aq_new.rename(columns={'station_id': 'stationId', 'time': 'utc_time', 'PM25_Concentration': 'PM2.5', 'PM10_Concentration': 'PM10', 'NO2_Concentration': 'NO2', 'CO_Concentration': 'CO', 'O3_Concentration': 'O3', 'SO2_Concentration': 'SO2'}, 
    #                 inplace=True)

    # # request beijing grid meo data
    # bj_aq_new.to_csv(r'./Data/Beijing/bj_aq_new.csv', index=False)
    # print('data is saved to ./Data/Beijing/bj_aq_new.csv \n')

    print('requesting beijing grid meo data to {}'.format(end_time))
    bj_grid_meo_new = request_data(url_templates['bj_meteorology_grid'].format(start_time=start_time, end_time=end_time))
    bj_grid_meo_new.to_csv(save_paths['bj_meteorology_grid'], index=False)
    # bj_grid_meo_new.drop('id', axis=1, inplace=True)
    # bj_grid_meo_new.rename(columns={'time': 'utc_time', 'station_id': 'stationId'}, inplace=True)
    # bj_grid_meo_new.to_csv(r'./Data/Beijing/bj_grid_meo_new.csv', index=False)
    # print('data is saved to ./Data/Beijing/bj_meteo_grid_new.csv \n')

    # request london air quality data\
    print('requesting london air quality data to {}'.format(end_time))
    ld_aq_new = request_data(url_templates['ld_aq'].format(start_time=start_time, end_time=end_time))
    ld_aq_new.to_csv(save_paths['ld_aq'], index=False)
    # ld_aq_new.drop(['id', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration'], axis=1, inplace=True)
    # ld_aq_new.rename(columns={'station_id': 'stationId', 'time': 'utc_time', 'PM25_Concentration': 'PM2.5', 'PM10_Concentration': 'PM10', 'NO2_Concentration': 'NO2'}, inplace=True)
    # print('data is saved to ./Data/London/ld_aq_new.csv \n')

    # # request london grid meo data
    print('requesting london meo data to {}'.format(end_time))
    ld_grid_meo_new = request_data(url_templates['ld_meteorology_grid'].format(start_time=start_time, end_time=end_time))
    ld_grid_meo_new.to_csv(save_paths['ld_meteorology_grid'], index=False)
    # ld_grid_meo_new.drop('id', axis=1, inplace=True)
    # ld_grid_meo_new.rename(columns={'station_id': 'stationId', 'time': 'utc_time'}, inplace=True)
    # ld_grid_meo_new.to_csv(r'./Data/London/ld_meo_new.csv', index=False)
    # print('data is saved to ./Data/London/ld_meo_new.csv \n')


if __name__ == '__main__':
    start_time = '2018-03-31-0'
    end_time = datetime_formatter(datetime.utcnow())

    request_main(start_time, end_time)





