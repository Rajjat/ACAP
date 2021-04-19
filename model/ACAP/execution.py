import dinet_base as dinet
import tensorflow as tf
import os
import logging
import argparse
import configparser
import numpy as np
import sys
import pandas as pd
import random as rn
import tensorflow as tf

import talos
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

rn.seed(0)
np.random.seed(0)
tf.compat.v1.set_random_seed(0)



from ACAP.ACAP import ACAP

def Train_Model(city,method):
    def initialte_class():
        mypred = ACAP(city=city)
        return mypred

    def do_rest(pred,i):
        pred.load_data(city,method)
        pred.create_model()
        pred.compile_model()
        pred.train(i)
        return pred

    def process_frame(df, i):
        new_df = df[['0', '1', 'weighted avg', 'micro avg', 'macro avg']].drop('support', axis=0)
        new_df = new_df.stack().swaplevel()
        new_df.index = new_df.index.map('{0[0]}_{0[1]}'.format)
        new_df = new_df.to_frame().T
        new_df['run'] = i
        new_df = new_df.set_index('run')
        return new_df

    def rerun(classname):
        df_list = []
        for i in range(int(config['global']['no_of_loops_model'])):

            print("*" * 20, classname, "*" * 20)
            print('*' * 10, ' round ', i)
            mypred = initialte_class()
            mypred = do_rest(mypred,i)
            res = mypred.evaluate(i,city,method)
            df_list.append(process_frame(res, i))
            #K.clear_session()
        df = pd.concat(df_list)
        return pd.DataFrame(df.mean(), columns=[classname])

    return rerun('ACAP')


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise Exception("You must set PYTHONHASHSEED=0 when starting the Jupyter server to get reproducible results.")

    # get TF logger
    tensorflow_logger = tf.get_logger()
    tensorflow_logger.setLevel(logging.DEBUG)
    tensorflow_logger.addHandler(file_handler)

    # create dinet handler
    logger = logging.getLogger("dinet")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # add stdout
    root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # first restrict used gpus and afterwards set memory growth
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # session set-up
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        device_name = '/device:GPU:0'
        feed_device = '/cpu:0'
    else:
        logger.warning("No GPU available")
        device_name = '/device:CPU:0'
        feed_device = '/cpu:0'    
    cities=['LS/hannover']
    methods = ['GG','dbscan','kmeans']

    for city in cities:
        for method in methods:
            result=Train_Model(city,method)
            print(result)
            result.to_csv("../../../data_preprocessing/data/regions/"+city+"/"+method+'/result_ACAP.csv')

