import numpy as np
import pandas as pd


def symmetric_mean_absolute_percentage_error(actual, forecast, y_mean, y_std, norm=False):
    '''
    Compute the Symmetric mean absolute percentage error (SMAPE or sMAPE) on a single data of the dev set or test set.
    Details of SMAPE here : https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Args:
        actual : an actual values in the dev/test dataset.
        forecast : an model forecast values.
        y_mean : mean value used when doing preprocess.
        y_std : std value used when doing preprocess.
        
    '''

    actual = np.squeeze(actual)
    forecast = np.squeeze(forecast)

    assert len(actual) == len(forecast), "The shape of actual value and forecast value are not the same."

    np.seterr(over='raise')
    length = len(actual)
    if norm:
        # print('Not correct')
        actual = actual * y_std + y_mean
        forecast = forecast * y_std + y_mean

    r = 0

    for i in range(length):
        f = forecast[i]
        a = actual[i]

        r += abs(f-a) / ((abs(a)+abs(f))/2.0)

    # if np.isnan(r):
    #     print((abs(a)+abs(f))/2.0)
        # print(((abs(a)+abs(f))/2.0) == 0)
    # print(r)

    return r/length, forecast


def SMAPE_on_dataset(actual_data, forecast_data, feature_list, y_mean, y_std, forecast_duration=24):
    '''
    Compute SMAPE value on the dataset of actual and forecast.
	
    Args:
        actual_data : actual data in the dev set or test set, shape [number_of_examples_in_the_dev_set, output_seq_length, num_of_output_features]
        forecast_data : forecast data which are predicted by the seq2seq model using x data in the dev set or test set, same shape as actual_data.
        forecast_duration : predict every 24 hours, so forecast_duration is set default to 1, because the dev set is sampled every 24 hours.
        feature_list : a list of features that is caculated in the forecast.
    Return:
        aver_smapes : average smape for all features on the test data.
        smapes_of_features : smapes of different features on the test data.
    '''
    assert actual_data.shape == forecast_data.shape, "The shape of actual data and perdiction data must match."

    number_of_features = actual_data.shape[2]
    smapes_list_of_features = {feature:[] for feature in feature_list}

    for i in range(0, actual_data.shape[0], forecast_duration):
        actual_data_item = actual_data[i]
        forecast_data_item = forecast_data[i]
        for j in range(number_of_features):
            feature = feature_list[j]
            a = actual_data_item[:,j]
            f = forecast_data_item[:,j]
            smape_a_feature_a_day = symmetric_mean_absolute_percentage_error(a, f, y_mean, y_std)
            smapes_list_of_features[feature].append(smape_a_feature_a_day)

    smapes_of_features = {feature:np.mean(value) for feature, value in smapes_list_of_features.items()}
    aver_smapes = np.mean(list(smapes_of_features.values()))

    return aver_smapes, smapes_of_features
    
        
# For new seq2seq model
def SMAPE_on_dataset_v1(actual_data, forecast_data, feature_list, statistics, forecast_duration=1, norm=False):
    '''
    Compute SMAPE value on the dataset of actual and forecast.
    
    Args:
        actual_data : actual data in the dev set or test set, shape [number_of_examples_in_the_dev_set, output_seq_length, num_of_output_features]
        forecast_data : forecast data which are predicted by the seq2seq model using x data in the dev set or test set, same shape as actual_data.
        forecast_duration : predict every 24 hours, so forecast_duration is set default to 1, because the dev set is sampled every 24 hours.
        feature_list : a list of features that is caculated in the forecast. Need to be in the right order!!
        statistics : a pandas dataframe of statistics.
    Return:
        aver_smapes : average smape for all features on the test data.
        smapes_of_features : smapes of different features on the test data.
    '''
    assert actual_data.shape == forecast_data.shape, "The shape of actual data and perdiction data must match."

    forecast_original = np.zeros(forecast_data.shape)

    number_of_features = actual_data.shape[2]
    smapes_list_of_features = {feature:[] for feature in feature_list}

    for i in range(0, actual_data.shape[0], forecast_duration):
        actual_data_item = actual_data[i]
        forecast_data_item = forecast_data[i]
        for j in range(number_of_features):
            feature = feature_list[j]
            a = actual_data_item[:,j]
            f = forecast_data_item[:,j]
            # use y_mean, y_std when norm=True
            y_mean = statistics.loc['mean'][feature]
            y_std = statistics.loc['std'][feature]

            smape_a_feature_a_day, f_original_a_feature_a_day = symmetric_mean_absolute_percentage_error(a, f, y_mean, y_std, norm=norm)

            # if i%100 == 0:
            #     print(smape_a_feature_a_day)

            smapes_list_of_features[feature].append(smape_a_feature_a_day)
            forecast_original[i,:,j] = f_original_a_feature_a_day

    smapes_of_features = {feature:np.mean(value) for feature, value in smapes_list_of_features.items()}
    aver_smapes = np.mean(list(smapes_of_features.values()))

    # transform forecast values to conform to submission requirement
    # convert
    forecast_df = pd.DataFrame()
    for index in range(number_of_features):
        temp = pd.DataFrame()
        feature = feature_list[index]
        station, particle = feature.rsplit('_', maxsplit=1)

        stations = [station + '#' + str(i) for i in range(0, 48)]
        particles = [particle for i in range(0, 48)]
        values = forecast_original[0, :, index]
        temp_df = pd.DataFrame({'stations': stations, 'particle': particles, 'value': values})
        forecast_df = pd.concat([forecast_df, temp_df])

    forecast_df = forecast_df.pivot_table(index='stations', columns='particle', values='value').reset_index()
    # change order and rename columns

    return aver_smapes, smapes_of_features, forecast_original, forecast_df


