

from keras.models import Model
from sklearn.utils import class_weight
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt1
from ACAP.base_cities import *


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
GRU_UNIT = 128
GEOHASH_UNIT = 128
EMBEDDING_UNIT = 128
Embedding_outdim = 128
NLP_UNIT = 128
SEQ_UNIT = 256
DENSE_CONCAT = 512
CONV_UNIT = 32
weights = np.array([1, 1]) 

class ACAP(keras_model):
    def load_data(self,city,method,):
        super(ACAP, self).load_data(city,method,with_geocode=False)

        self.X_train1 = self.reshape(self.X_train)
        self.X_test1 = self.reshape(self.X_test)
        self.X_val1= self.reshape(self.X_val)
        
        self.X_train5 = self.reshapeSimple(self.X_train)
        self.X_test5 = self.reshapeSimple(self.X_test)
        self.X_val5= self.reshapeSimple(self.X_val)


        self.X_train2 = reshape_cat(self.X_train, 'geohash') 
        self.X_train3 = reshape_cat(self.X_train, 'NLP')  

        self.X_test2 = reshape_cat(self.X_test, 'geohash')
        self.X_test3 = reshape_cat(self.X_test, 'NLP')

        
        self.X_val2 = reshape_cat(self.X_val, 'geohash')
        self.X_val3 = reshape_cat(self.X_val, 'NLP')
        self.X_train4 = self.X_train[:,-1]
        self.X_val4 = self.X_val[:,-1]
        self.X_test4 = self.X_test[:,-1]

        print(self.X_train1.shape)
        print(self.X_train2.shape)
        print(self.X_train3.shape)
        print (self.X_train4.shape)
    def create_model(self):
        input1 = Input(shape=(self.X_train1.shape[1], self.X_train1.shape[2]), dtype='float32',
                       name='main_input')
        lstm = GRU(units=GRU_UNIT, return_sequences=True,
                    kernel_regularizer=regularizers.l2(self.weight_decay),
                    recurrent_regularizer=regularizers.l2(self.weight_decay),
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    unroll=True)(input1)

        lstm = GRU(units=GRU_UNIT, return_sequences=False,
            kernel_regularizer=regularizers.l2(self.weight_decay),
            recurrent_regularizer=regularizers.l2(self.weight_decay),
            dropout=dropout,
            recurrent_dropout=dropout,
            unroll=True)(lstm)

        #######################################
        input2 = Input(shape=(self.X_train2.shape[1],), dtype='float32', name='geohash_input')
        geohash_vec = Dense(GEOHASH_UNIT, activation=ACT_PRIOR)(input2)
        ######################################
        input3 = Input(shape=(self.X_train3.shape[1],), dtype='float32', name='nlp_input')
        nlp_vec = Dense(NLP_UNIT, activation=ACT_PRIOR)(input3)
        #####################################
        level_2 = concatenate([lstm,geohash_vec, nlp_vec])
        main_output = self.last_layers(level_2)

        self.model = Model(inputs=[input1,input2, input3], outputs=main_output) 
        print(self.model.summary())

    def train(self,i):
        history = self.model.fit([self.X_train1,self.X_train2, self.X_train3], self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epoch, verbose=verbose,
                                 validation_data=([self.X_val1,self.X_val2,self.X_val3], 
                                                                                     self.y_val),
                                 callbacks=[self.earlyStopping])


    def evaluate(self,i,city,method):
        y_true, y_pred = self.y_test, self.model.predict([self.X_test1, self.X_test2, self.X_test3], verbose=verbose)
        return self.make_report(i,y_true, y_pred,city,method)
