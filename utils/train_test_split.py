from utils.data_utils import flat_data

def normalize(df, mean, std):
	df_norm = (df - mean) / std
	return df_norm

if __name__ == '__main__':
	
	# train data starts from 2017-01-02 since starts time are different
	train_start_time = '2017-01-02 00:00:00'
	train_end_time = '2018-03-31 23:00:00'

	# use April as testing data
	test_start_time = '2018-04-01 00:00:00'
	test_end_time = '2018-04-30 23:00:00'

	# load data
	# normalization