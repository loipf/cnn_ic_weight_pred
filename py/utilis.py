import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats



def remove_dir(dir_path):
    deleteFiles = []
    deleteDirs = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            deleteFiles.append(os.path.join(root, f))
        for d in dirs:
            deleteDirs.append(os.path.join(root, d))
    for f in deleteFiles:
        os.remove(f)
    for d in deleteDirs:
        os.rmdir(d)
    # os.rmdir(dir_path)



def plot_regression(out_df):
    pred_y = out_df['pred_raw']
    pred_y = [ x[0] for x in pred_y]
    true_y = list(out_df['weight_gain_per_z'])

    # plt.plot(true_y, pred_y, '.')
    plt.clf()
    plt.scatter(true_y, pred_y, marker='.')
    plt.plot([-10.0, 10.0], [-10.0, 10.0], color='orange')

    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # plt.plot(x, intercept + slope * x, 'r-')
    # statsText = ' simple lin reg\n p-value: ' + hm.roundPvalue(p_value) + ' ' + hm.getAsterix(
    #         p_value) + '\n r: ' + hm.roundPvalue(r_value)


    plt.xlabel('true')
    plt.ylabel('prediction')

    corrCoeff, corr_pVal = stats.pearsonr(true_y, pred_y)
    rmse = round(mean_squared_error(true_y, pred_y),3)
    print('\n> rmse: ' + str(rmse))

    plt.title('weight_gain_per_z   [rmse: {}, pear_coef: {}]'.format(rmse, corrCoeff))
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.tight_layout()
    plt.savefig('plot_'+str(rmse)+'.png')


    # plt.show()




