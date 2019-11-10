import numpy as np
import pandas as pd
import random



def read_in_data(df_path):
    df = pd.read_csv(df_path)
    return df


### leave one out split
def split_train_test(df, prop_df, sample_id, linear = False):
    sample_train = prop_df[prop_df['id'] == sample_id]
    test_x = df.loc[:, sample_train['id_day']]
    train_x = df.loc[:, ~df.columns.isin(sample_train['id_day'])]

    random_cols = list(train_x.columns)
    random.shuffle(random_cols)
    train_x = train_x[random_cols]



    # test_y = prop_df.loc[prop_df['id_day'].isin(list(train_x.columns) ),: ] # ['pred_class']
    # print(test_y)

    cp_prop = prop_df.copy()
    cp_prop.set_index('id_day', inplace=True)

    if linear is False:
        test_y = prop_df.loc[prop_df['id'] == sample_id,: ]['pred_class']
        test_y = make_pred_one_hot(test_y)
        train_y = cp_prop.loc[train_x.columns,: ]['pred_class']
        train_y = make_pred_one_hot(train_y)
    else:
        ### regression task
        test_y = prop_df.loc[prop_df['id'] == sample_id,: ]['weight_gain_per_z']
        test_y = test_y.reshape([len(test_y), 1])
        train_y = cp_prop.loc[train_x.columns, :]['weight_gain_per_z']
        train_y = train_y.reshape([len(train_y), 1])

    return np.array(train_x), train_y, np.array(test_x), test_y




def make_pred_one_hot(pred):
    out_df = pd.DataFrame(columns=['fat','skinny'])
    out_df['fat'] = np.array(pred)
    out_df['skinny'] = 1-np.array(pred)
    return np.array(out_df)









