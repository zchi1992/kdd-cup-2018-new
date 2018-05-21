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

    print('requesting beijing air quality data to {}'.format(end_time))
    bj_aq_new = request_data(url_templates['bj_aq'].format(start_time=start_time, end_time=end_time))
    bj_aq_new.to_csv(save_paths['bj_aq'], index=False)

    print('requesting beijing grid meo data to {}'.format(end_time))
    bj_grid_meo_new = request_data(url_templates['bj_meteorology_grid'].format(start_time=start_time, end_time=end_time))
    bj_grid_meo_new.to_csv(save_paths['bj_meteorology_grid'], index=False)

    print('requesting london air quality data to {}'.format(end_time))
    ld_aq_new = request_data(url_templates['ld_aq'].format(start_time=start_time, end_time=end_time))
    ld_aq_new.to_csv(save_paths['ld_aq'], index=False)

    print('requesting london meo data to {}'.format(end_time))
    ld_grid_meo_new = request_data(url_templates['ld_meteorology_grid'].format(start_time=start_time, end_time=end_time))
    ld_grid_meo_new.to_csv(save_paths['ld_meteorology_grid'], index=False)


if __name__ == '__main__':
    start_time = '2018-03-31-0'
    end_time = datetime_formatter(datetime.utcnow())

    request_main(start_time, end_time)





