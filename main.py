from utils.request_data import *
from preprocess.data_preprocess import *
from datetime import datetime
from utils.train_test_split import *
from model.train import train_and_dev
import argparse

parser = argparse.ArgumentParser()
parser.parse_args()
# set positional arguments for training
parser.add_argument('--gap', type=int, default=0, help='gap coulbe 0, 12, 24, depending on when the program is running')
args = parser.parse_args(args=[])
gap = args.gap

if __name__ == '__main__':
    cities = ['bj', 'ld']
    particles = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    meteros = ["temperature", "pressure", "humidity", "wind_direction", "wind_speed"]

    # request latest dataset
    start_time = '2018-03-31-0'
    end_time = datetime_formatter(datetime.utcnow())
    request_main(start_time, end_time)

    # data preprocessing
    for city in cities:
        preprocess_main(city)

    # load filled data
    data_aq_all = {}
    data_metero_all = {}
    for city in cities:
        city_aq_all = pd.DataFrame()
        city_meter_all = pd.DataFrame()
        if city == 'bj':
            files = os.listdir(r'./data/Beijing/post')
            for file in files:
                tmp = pd.read_csv(os.path.join(r'./data/Beijing/post', file), index_col=0)
                if any([p in file for p in particles]):
                    city_aq_all = pd.concat([city_aq_all, tmp], ignore_index=True)
                else:
                    city_meter_all = pd.concat([city_meter_all, tmp], ignore_index=True)
            assert any(city_aq_all.isnull().any(axis=1)) is False, 'There"s still missing data in {}_aq dataset'.format(city)
            assert any(city_meter_all.isnull().any(axis=1)) is False, 'There"s still missing data in {}_meter dataset'.format(city)
        else:
            files = os.listdir(r'./data/London/post')
            for file in files:
                tmp = pd.read_csv(os.path.join(r'./data/Beijing/post', file), index_col=0)
                if any([p in file for p in particles]):
                    city_aq_all = pd.concat([city_aq_all, tmp], ignore_index=True)
                else:
                    city_meter_all = pd.concat([city_meter_all, tmp], ignore_index=True)
            assert any(city_aq_all.isnull().any(axis=1)) is False, 'There"s still missing data in {}_aq dataset'.format(city)
            assert any(city_meter_all.isnull().any(axis=1)) is False, 'There"s still missing data in {}_meter dataset'.format(city)

    # split data into training and testing
    train_start_time = '2017-01-02 00:00:00'
    train_end_time = '2018-03-31 23:00:00'
    test_start_time = '2018-04-01 00:00:00'

    bj_split = train_test_split(train_start_time, train_end_time, test_start_time)
    bj_train_aq, bj_test_aq, bj_train_meo, bj_test_meo = bj_split(data_aq_all['bj'], data_metero_all['bj'])

    ld_split = train_test_split(train_start_time, train_end_time, test_start_time)
    ld_train_aq, ld_test_aq, ld_train_meo, ld_test_meo = ld_split(data_aq_all['ld'], data_metero_all['ld'])

    bj_train_aq.to_csv(r'./data/Beijing/bj_train_aq.csv', index=True)
    bj_train_meo.to_csv(r'./data/Beijing/bj_train_meo.csv', index=True)
    bj_test_aq.to_csv(r'./data/Beijing/bj_test_aq.csv', index=True)
    bj_test_meo.to_csv(r'./data/Beijing/bj_test_meo.csv', index=True)
    ld_train_aq.to_csv(r'./data/London/ld_train_aq.csv', index=True)
    ld_train_meo.to_csv(r'./data/London/ld_train_meo.csv', index=True)
    ld_test_aq.to_csv(r'./data/London/ld_test_aq.csv', index=True)
    ld_test_meo.to_csv(r'./data/London/ld_test_meo.csv', index=True)

    # train models and output prediction with the best model trained
    print('Start training model...')
    results = {}
    pre_days = 5
    loss_function = 'L2'
    iterations = 500
    print("Starting training, validating and make prediction with the best models")

    for city in cities:
        results[city] = {}
        # make directories to store all trained models
        datetime = datetime_formatter(datetime.utcnow())
        path = './results/{datetime}/{city}/'.format(datetime=datetime, city=city)
        if not os.path.isdir(path):
            os.makedirs(path) # recursively create a directory to store all models
        print("city: {city}".format(city=city))
        aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features, forecast_df = train_and_dev(
            path, city, pre_days, gap, loss_function, iterations)
        print("best_SAMPE: {:.5f}".format(aver_smapes_best))
        # write out model_preds_on_test and forecast_df
        model_preds_on_test.save(path + 'model_preds_on_test')
        forecast_df.save(path + 'forecast_df')
        results[city] = forecast_df

    bj_forecast = results['bj']
    ld_forecast = results['ld']

    # print("Getting the best results")
    # submit = {}
    # for city, results_of_city in results.items():
    #     _, _, model_preds_on_test, output_features, forecast_df = results_of_city
    #     submit[city] = [np.squeeze(model_preds_on_test), output_features]  # len of 2 for each city

    # post process on beijing forecast
    # rename station names
    station_need_rename = {
        'aotizhongxin_aq': 'aotizhongx_aq',
        'fengtaihuayuan_aq': 'fengtaihua_aq',
        'miyunshuiku_aq': 'miyunshuik_aq',
        'nongzhanguan_aq': 'nongzhangu_aq',
        'wanshouxigong_aq': 'wanshouxig_aq',
        'xizhimenbei_aq': 'xizhimenbe_aq',
        'yongdingmennei_aq': 'yongdingme_aq'
    }

    # rename stations
    for index in bj_forecast.index:
        station, time = bj_forecast.at[index, 'station'].split('#') # aotizhongxin_aq#0 to aotizhongxin_aq, 0
        if station in station_need_rename.keys():
            bj_forecast.at[index, 'station'] = station_need_rename[station] + '#' + time
    bj_forecast.rename(columns={'station': 'test_id'}, inplace=True)
    bj_forecast = bj_forecast[['test_id', 'PM2.5', 'PM10', 'O3']]
    print('Finish post processing beijing forecast')

    ld_forecast['O3'] = 0
    ld_forecast.rename(columns={'station': 'test_id'}, inplace=True)
    ld_forecast = ld_forecast[['test_id', 'PM2.5', 'PM10', 'O3']]
    print('Finish post processing london forecast')

    submission_new = pd.concat([bj_forecast, ld_forecast], axis=0)
    datetime = datetime_formatter(datetime.utcnow())
    submission_new.to_csv(r'./submit/submission_{datetime}'.format(datetime=datetime), index=False)
    print('Done!')