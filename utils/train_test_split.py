
def normalize(df, mean, std):
    df_norm = (df - mean) / std
    return df_norm


def normalize_train_test(df_train, df_test, city='bj', particle='aq'):
    desc = df_train.describe()

    # write out to disk. Logic needs to be improved
    desc.to_csv('./Data/data/{city}_{particle}_desc.csv'.format(city=city, particle=particle))

    mean, std = desc.loc['mean'], desc.loc['std']
    df_train_norm = (df_train - mean) / std
    df_test_norm = (df_test - mean) / std
    return df_train_norm, df_test_norm


def get_end_time(df_aq, df_meo):
    aq_end_time = df_aq.index.max()
    meo_end_time = df_meo.index.max()
    return min(aq_end_time, meo_end_time)


def train_test_split(train_start_time, train_end_time, test_start_time):
    def split(df_aq, df_meo):
        test_end_time = get_end_time(df_aq, df_meo)
        train_aq = df_aq[train_start_time:train_end_time]
        test_aq = df_aq[test_start_time:test_end_time]
        train_meo = df_meo[train_start_time:train_end_time]
        test_meo = df_meo[test_start_time:test_end_time]
        return train_aq, test_aq, train_meo, test_meo

    return split


if __name__ == '__main__':

    # train data starts from 2017-01-02 since starts time are different
    train_start_time = '2017-01-02 00:00:00'
    train_end_time = '2018-03-31 23:00:00'

    # use April as testing data
    test_start_time = '2018-04-01 00:00:00'
    test_end_time = '2018-04-30 23:00:00'

    # load data
    # normalization