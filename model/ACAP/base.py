import pandas as pd
import logging
import argparse
import configparser
from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import optimizers
import keras_metrics as km

from sklearn.metrics import classification_report

from keras.wrappers.scikit_learn import KerasClassifier

from  dinet_base.utils_cities import *

SEQ = 8  # sequence for LSTM

verbose = 2
dropout = 0.2
VAL_SPLIT = 0.2
patience = 15
lr = 0.01
weight_decay = 0.0000
lr_decay = 1e-7
ADD_ON_LAYERS = True
ACT_PRIOR = 'sigmoid'
ACT_POSTERIOR = 'relu'
LSTM_UNIT = 128
GEOHASH_UNIT = 128
EMBEDDING_UNIT = 128
Embedding_outdim = 128
NLP_UNIT = 128
SEQ_UNIT = 256
DENSE_CONCAT = 512
CONV_UNIT = 32
weights = np.array([1, 1])



class base_model(object):
    def __init__(self, n_jobs=10, act=ACT_POSTERIOR, city='hann'):
        self.n_jobs = n_jobs
        self.CITY = city
        self.act = act
        args = parser.parse_args()

    def load_data(self, city,method,category=None, with_geocode=False):        
        self.X_train = np.load("../"+city+"/"+method+"/X_train.npy",allow_pickle=True)
        self.y_train = np.load("../"+city+"/"+method+"/y_train.npy",allow_pickle=True)
        self.X_test = np.load("../"+city+"/"+method+"/X_test.npy",allow_pickle=True)
        self.y_test = np.load("../"+city+"/"+method+"/y_test.npy",allow_pickle=True)
        self.X_val = np.load("../"+city+"/"+method+"/X_val.npy",allow_pickle=True)
        self.y_val = np.load("../"+city+"/"+method+"/y_val.npy", allow_pickle=True) 

        if not with_geocode:
            self.X_train = self.X_train[:, 0:-1]
            self.X_test = self.X_test[:, 0:-1]
            self.X_val = self.X_val[:, 0:-1]            
        self.update_y()

        if category != None:
            l_train = []
            l_test = []
            l_val=[]
            for cat in category:
                l_train.append(reshape_cat(self.X_train, cat))
                l_test.append(reshape_cat(self.X_test, cat))
                l_val.append(reshape_cat(self.X_val, cat))

            self.X_train = np.concatenate(l_train, axis=1)
            self.X_test = np.concatenate(l_test, axis=1)
            self.X_val = np.concatenate(l_val, axis=1)


        print('load and test: shapes for train and test, X/Y')
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        
        print(self.X_val.shape)
        print(self.y_val.shape)
        print(self.y_test)

    def update_y(self):
        self.y_train = to_categorical(self.y_train, 2)
        self.y_test = to_categorical(self.y_test, 2)
        self.y_val = to_categorical(self.y_val, 2)

    def last_layers(self, model_in):
        model_in = Dense(DENSE_CONCAT,
                         kernel_regularizer=regularizers.l2(self.weight_decay),
                         kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                         activation=self.act)(model_in)

        model_in = Dense(units=int(DENSE_CONCAT / 2),
                         kernel_regularizer=regularizers.l2(self.weight_decay),
                         kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                         activation=None)(model_in)
        if ADD_ON_LAYERS:
            model_in = BatchNormalization()(model_in)
        model_in = Activation(self.act)(model_in)
        model_in = Dense(units=int(DENSE_CONCAT / 8),
                         kernel_regularizer=regularizers.l2(self.weight_decay),
                         kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                         activation=None)(model_in)
        if ADD_ON_LAYERS:
            model_in = BatchNormalization()(model_in)
        model_in = Activation(self.act)(model_in)
        main_output = Dense(self.output_dim, kernel_initializer=keras.initializers.glorot_uniform(seed=0),activation=self.activation)(model_in)
        return main_output


class keras_model(base_model):
    def __init__(self, city='LS', activation='softmax', batch_size=256, epoch=60, n_jobs=1, act=ACT_POSTERIOR):
        super(keras_model, self).__init__(act=act, city=city)
        self.output_dim = 2
        self.activation = activation
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_jobs = n_jobs
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_decay = lr_decay

    def reshape(self, x):
        print('x before=', (x.shape))
        x=x[:,0:296] # past 8
        m=reshape_cat(x,'time')
        t = m.reshape((m.shape[0],SEQ,int(m.shape[1]/SEQ)))
        return t

    def reshapeSimple(self, x):
        print('x before=', (x.shape))
        x=x[:,0:296] # past 8
        return x


    def compile_model(self, model=None):
        f1_score = km.categorical_f1_score(label=1)
        self.earlyStopping = EarlyStopping(monitor='val_f1_score',
                                           restore_best_weights=True,
                                           patience=patience, verbose=0, mode='max'  ,min_delta=0.01
                                           )
        adam = optimizers.Adam(lr=self.lr, decay=self.lr_decay)
        loss = weighted_categorical_crossentropy(weights)
        self.model.compile(optimizer=adam, loss=loss 

    def create_model(self):
        self.model = KerasClassifier(build_fn=self.build_model, epochs=self.epoch, batch_size=self.batch_size,
                                     verbose=1)

    def make_report(self,i, y_true, y_pred,city,method):
        data_frame = classification_report(y_true.argmax(axis=-1), y_pred.argmax(axis=-1), output_dict=True)
        df = pd.DataFrame(data_frame)
        df = df.reset_index()
        roc_dict = self.roc_auc(i,y_true, y_pred,city,method)
        df = df.append({'index': 'auc', '0': roc_dict[0], '1': roc_dict[1],
                        'micro avg': roc_dict['micro'],
                        'macro avg': roc_dict['macro']}, ignore_index=True)
        df = df.set_index('index')
        return df

    def roc_auc(self,i, y_test, y_score,city,method):
        fpr, tpr, roc_auc = roc_auc_compute(y_test, y_score)
        return roc_auc