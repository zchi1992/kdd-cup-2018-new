import utils
from request_data import *
from data_preprocess import *
from datetime import datetime

if __name__ == '__main__':
    cities = ['bj', 'ld']

    # request latest dataset
    start_time = '2018-03-31-0'
    end_time = datetime_formatter(datetime.utcnow())
    request_main(start_time, end_time)

    # data preprocessing
    for city in cities:
        preprocess_main(city)


    # split data into training and testing

    # train models and output prediction with the best model trained

    # post processing if necessary

    
