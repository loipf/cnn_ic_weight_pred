import tflearn
import pandas as pd
import numpy as np


import data_input as da
import cnn_model
import pred_methods as pm
import utilis
import random



def run_prediction():
    learning_rate = 0.01
    training_steps = 100
    linear = True

    ## get data
    data_file_path = 'male_chow_rer.csv'
    prop_file_path = 'property_table.csv'
    df_pred = da.read_in_data(data_file_path)
    df_pred = cnn_model.normalize_input(df_pred)
    df_prop = da.read_in_data(prop_file_path)

    ### reset stuff
    tb_dir = '../tensorboard_dir'
    utilis.remove_dir(tb_dir)

    ### leave one out approach
    rand_seed = random.randrange(1,500)
    # rand_seed = 128
    print('random seed: {}'.format(rand_seed))

    out_df = pd.DataFrame()
    for sample_id in np.unique(df_prop['id']):
        pred_out = pm.train_model(df_pred, df_prop, sample_id, tb_dir=tb_dir, learning_rate = learning_rate,
                                  training_steps = training_steps, linear=linear, seed=rand_seed)
        out_df = out_df.append(pred_out['out_df'], ignore_index=True)
    merge_df = df_prop.drop(['id_day','sex','TSE'], axis=1).drop_duplicates()
    out_df = out_df.merge(merge_df, left_on='sample_id', right_on='id')

    # print(out_df)
    # print("mean acc of network: {}".format(np.mean(out_df['acc'])) )
    # print("mean loss of network: {}".format(np.mean(out_df['loss'])) )
    # print("prediction acc true: {}".format(np.mean(out_df['pred_bool'])) )


    utilis.plot_regression(out_df)










if __name__ == '__main__':
    for i in range(40):
        run_prediction()







