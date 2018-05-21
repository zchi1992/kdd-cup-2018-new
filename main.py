from utils.request_data import *
from preprocess.data_preprocess import *
from datetime import datetime
from utils.train_test_split import *
from utils.utils import *
from model.train import train_and_dev

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

    # prepare test dataset
    print('Start training model...')
    results = {}
    # gap = 24 # should be set as argument on command line
    # total_iterations = 10 # should be set as argument on command line
    pre_days_list = [5]
    # pre_days_list = [5,6,7]
    loss_functions = ["L2"]
    # loss_functions = ["L2", "L1", "huber_loss"]
    print("Starting training, validating and make prediction with the best models")

    # remember to change it back to cities
    for city in cities:
        results[city] = {}
        for pre_days in pre_days_list:
            for loss_function in loss_functions:
                # make directories to store all trained models
                datetime = datetime_formatter(datetime.utcnow())
                path = './results/{datetime}/{city}/{pre_days}_{loss_function}/'.format(datetime=datetime, city=city,
                                                                                        pre_days=pre_days, loss_function=loss_function)
                if not os.path.isdir(path):
                    os.makedirs(path) # recursively create a directory to store all models
                print("city: {city}, pre_days: {pre_days}, loss_function: {loss_function}".format(city=city,
                                                                                                  pre_days=pre_days,
                                                                                                  loss_function=loss_function))
                aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features, forecast_df = train_and_dev(
                    path, city, pre_days, gap, loss_function, iterations)
                print("best_SAMPE: {:.5f}".format(aver_smapes_best))
                results[city][aver_smapes_best] = [model_preds_on_dev, dev_y_original, model_preds_on_test,
                                                   output_features, forecast_df]
                print(model_preds_on_test)

    print("Getting the best results")
    submit = {}
    for city, results_of_city in results.items():
        min_smape = min(results_of_city.keys())
        _, _, model_preds_on_test, output_features, forecast_df = results_of_city[min_smape]
        submit[city] = [np.squeeze(model_preds_on_test), output_features]  # len of 2 for each city

    # post processing if necessary

    
