import tensorflow as tf   ### version 1.01
import numpy as np
import pandas as pd
import tflearn
from tflearn.layers.estimator import regression

import data_input as da
import cnn_model




def train_model(df, prop_df, sample_id, tb_dir, learning_rate, training_steps, linear=False, seed=123):
    tf.reset_default_graph()

    ### get convolutional neural network
    if linear is False:
        cnn = regression(cnn_model.get_cnn(df), optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
    else:
        cnn = regression(cnn_model.get_cnn_linear(df,seed), optimizer='adam', learning_rate=learning_rate, loss='mean_square')

    cnn = tflearn.DNN(cnn, tensorboard_verbose=3, tensorboard_dir=tb_dir) ### for tensorboard analysis

    ### get data
    train_x, train_y, test_x, test_y = da.split_train_test(df, prop_df, sample_id, linear)
    train_x = train_x.reshape([-1, len(df), 1])
    test_x = test_x.reshape([-1, len(df), 1])



    ### fit model
    cnn.fit(train_x, train_y, n_epoch=training_steps, show_metric=True, run_id='run_'+sample_id, shuffle=False, batch_size=len(train_x))
    pred_y = cnn.predict(test_x)

    ### get output info
    out_df = pd.DataFrame(columns=['sample_id','day','loss','acc','pred_raw','pred_bool'])
    for day in range(test_y.shape[0]):
        day_df = pd.DataFrame( [[sample_id, day, cnn.trainer.training_state.loss_value, cnn.trainer.training_state.acc_value,
                                 pred_y[day], get_pred_bool(test_y[day], pred_y[day])]],
                                columns=['sample_id', 'day', 'loss', 'acc', 'pred_raw', 'pred_bool'])
        out_df = out_df.append(day_df)

    return {'model':cnn, 'test_y':test_y, 'pred_y':pred_y, 'out_df':out_df}




### helper method for classifier probability per day
def get_pred_bool(test_y, pred_y):
    if (test_y[0] == 0 and pred_y[0] < 0.5) | (test_y[0] == 1 and pred_y[0] > 0.5):
        return True
    return False












