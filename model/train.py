import os
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from metrics.metrics import SMAPE_on_dataset_v1
from .seq2seq_data_util import get_training_statistics, generate_training_set, generate_dev_set, generate_X_test_set
from .multi_variable_seq2seq_model_parameters import build_graph


# configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_config = tf.ConfigProto(log_device_placement=False)
gpu_config.gpu_options.allow_growth = True
session = tf.Session(config=gpu_config)
KTF.set_session(session)

# args
bj_station_list = ['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq',
            'nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq',
            'yungang_aq','gucheng_aq','fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq',
            'shunyi_aq','pingchang_aq','mentougou_aq','pinggu_aq','huairou_aq','miyun_aq',
            'yanqin_aq','dingling_aq','badaling_aq','miyunshuiku_aq','donggaocun_aq',
            'yongledian_aq','yufa_aq','liulihe_aq','qianmen_aq','yongdingmennei_aq',
            'xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq']            
bj_X_aq_list = ["PM2.5","PM10","O3","CO","SO2","NO2"]  
bj_y_aq_list = ["PM2.5","PM10","O3"]
bj_X_meo_list = ["temperature","pressure","humidity","wind_direction","wind_speed"]

ld_station_list = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']            
ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']  
ld_y_aq_list = ['PM10', 'PM2.5'] 
ld_X_meo_list = ["temperature","pressure","humidity","wind_direction","wind_speed"] 


def train_and_dev(path, city='bj', pre_days=5, gap=0, loss_function="L2", total_iterations=10, norm=False):
    '''
    city='bj' or 'ld' : 针对某个城市的数据进行训练
    pre_days : 使用 pre_days 天数的数据进行预测
    gap : 0,12,24
        0 : 当天 23点以后进行的模型训练
        12 : 当天中午进行的模型训练
        24 : 不使用当天数据进行的训练
    loss_function : 使用不同的损失函数
    '''
    if city=="bj":
        station_list = bj_station_list
        X_aq_list = bj_X_aq_list
        y_aq_list = bj_y_aq_list
        X_meo_list = bj_X_meo_list
    elif city=="ld":
        station_list = ld_station_list
        X_aq_list = ld_X_aq_list
        y_aq_list = ld_y_aq_list
        X_meo_list = ld_X_meo_list

    use_day=True
    learning_rate=1e-3
    batch_size=128
    input_seq_len = pre_days * 24 - gap
    output_seq_len = 48
    hidden_dim = 256
    input_dim = len(station_list) * (len(X_aq_list) + len(X_meo_list))
    output_dim = len(station_list) * len(y_aq_list)
    num_stacked_layers = 3

    lambda_l2_reg=0.003
    GRADIENT_CLIPPING=2.5
    total_iterations = total_iterations
    KEEP_RATE = 0.5

    output_features = []
    for station in station_list:
        for aq_feature in y_aq_list:
            output_features.append(station + "_" + aq_feature)
    output_features.sort()

    # use this when norm=True
    statistics = get_training_statistics(city)

    # Define training model
    rnn_model = build_graph(feed_previous=False, 
                            input_seq_len=input_seq_len, 
                            output_seq_len=output_seq_len, 
                            hidden_dim=hidden_dim, 
                            input_dim=input_dim, 
                            output_dim=output_dim, 
                            num_stacked_layers=num_stacked_layers, 
                            learning_rate=learning_rate,
                            lambda_l2_reg=lambda_l2_reg,
                            GRADIENT_CLIPPING=GRADIENT_CLIPPING,
                            loss_function=loss_function)
    # training process
    train_losses = []
    val_losses = []
    saved_iteractions = []

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        losses = []
        print("Training losses: ")
        for i in range(total_iterations):
            batch_input, batch_output = generate_training_set(city=city,
                                                              station_list=station_list,
                                                              X_aq_list=X_aq_list,
                                                              y_aq_list=y_aq_list,
                                                              X_meo_list=X_meo_list,
                                                              use_day=use_day,
                                                              pre_days=pre_days,
                                                              batch_size=batch_size,
                                                              gap=gap)
            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t,:] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t,:] for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict) 
            
            if i%5 == 0:
                print("loss after %d/%d iteractions : %.3f" %(i, total_iterations, loss_t))

                temp_saver = rnn_model['saver']()
                name = '%d pre_days, %d gap, %s loss_function, multivariate_%d_iteractions' %(pre_days, gap, loss_function, i)
                saved_iteractions.append(name)
                save_path = temp_saver.save(sess, os.path.join(path, name))
                print("Checkpoint saved at: ", save_path)

            losses.append(loss_t)

    print('Finishing training model, models are saved in {}'.format(path))

    print('Generating test data for the model to validate all models...')
    # Generate test data for the model
    dev_x, dev_y = generate_dev_set(city=city,
                                  station_list=station_list,
                                  X_aq_list=X_aq_list,
                                  y_aq_list=y_aq_list,
                                  X_meo_list=X_meo_list,
                                  pre_days=pre_days,
                                  gap=gap)
    # ??? What does this step do
    _, _, dev_y_original, _ = SMAPE_on_dataset_v1(dev_y, dev_y, output_features, statistics, 1, norm=norm)


    # predicting using different model on dev set
    rnn_model = build_graph(feed_previous=True, 
                            input_seq_len=input_seq_len, 
                            output_seq_len=output_seq_len, 
                            hidden_dim=hidden_dim, 
                            input_dim=input_dim, 
                            output_dim=output_dim, 
                            num_stacked_layers=num_stacked_layers, 
                            learning_rate=learning_rate,
                            lambda_l2_reg=lambda_l2_reg,
                            GRADIENT_CLIPPING=GRADIENT_CLIPPING,
                            loss_function="L1")


    # aver_smapes_on_iteractions = {}
    aver_smapes_best = 10
    model_preds_on_dev = None

    for name in saved_iteractions:

        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)
            
            # print("Using checkpoint: ", name)
            saver = rnn_model['saver']().restore(sess,  os.path.join(path, name))
            
            feed_dict = {rnn_model['enc_inp'][t]: dev_x[:, t, :] for t in range(input_seq_len)} # batch prediction
            feed_dict.update({rnn_model['target_seq'][t]: np.zeros([dev_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
            final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
            
            final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
            final_preds = np.concatenate(final_preds, axis = 1)

        aver_smapes, smapes_of_features, forecast_original, _ = SMAPE_on_dataset_v1(dev_y, final_preds, output_features, statistics, 1, norm=norm)

        # aver_smapes_on_iteractions[name] = aver_smapes
        if aver_smapes < aver_smapes_best :
            aver_smapes_best = aver_smapes
            model_preds_on_dev = forecast_original  
            model_name = name

        print(aver_smapes_best)

    print('Finishing validating models and the best model is {}'.format(model_name))

    print('Starting prediction...')

    # Use best model to make prediction
    X_predict = generate_X_test_set(city=city,
                                 station_list=station_list,
                                 X_aq_list=X_aq_list,
                                 X_meo_list=X_meo_list,
                                 pre_days=pre_days,
                                 gap=gap)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        saver = rnn_model['saver']().restore(sess,  os.path.join(path, model_name))

        feed_dict = {rnn_model['enc_inp'][t]: X_predict[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([X_predict.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_test_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    
        final_test_preds = [np.expand_dims(pred, 1) for pred in final_test_preds]
        final_test_preds = np.concatenate(final_test_preds, axis = 1)

    _, _, model_preds_on_test, forecast_df = SMAPE_on_dataset_v1(final_test_preds, final_test_preds, output_features, statistics, 1, norm=norm)

    return aver_smapes_best, model_preds_on_dev, dev_y_original, model_preds_on_test, output_features, forecast_df  # 将在这种情况下表现最好的模型 的预测结果 和 模型的位置信息返回
