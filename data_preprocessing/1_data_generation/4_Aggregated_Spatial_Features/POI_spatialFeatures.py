import tensorflow as tf
import logging
import argparse
import configparser

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn import cluster
from sklearn.cluster import KMeans
import hdbscan




def poi_pkl(city,method):
    city='new_method/gridgrowing'
    
    ob = pd.read_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/'+city+'/geohash_cluster7.csv',
                      header=0)
    df = ob[['geohash', 'cluster_id']]
    poi = pd.read_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/munichgeohash7/allPoi_geohash7.csv',
                     header=0)
    print(poi)
    poi_hash = pd.merge(poi, ob, on='geohash',how='right')
    poi_hash = poi_hash[
        ['cluster_id', 'aamenity_count', 'count_cross', 'count_junc',
       'railway_count', 'station_count', 'stopsign_count', 'trafsignal_count',
       'turning_loop', 'giveway_count', 'count']]
    poi_hash.columns=['geohash', 'amenity_count', 'count_junc', 'length',
       'railway_count', 'station_count', 'stopsign_count', 'trafsignal_count',
       'turning_loop', 'giveway_count', 'htype_count']
    poi_hash=poi_hash.drop_duplicates()
    a = poi_hash.groupby('geohash').sum().reset_index()
    a.to_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata/clustered_data_poi_acc_merged.csv',
             index=False)
    geohash_vec = a[[u'amenity_count', u'count_junc', u'length', u'railway_count',
                     u'station_count', u'stopsign_count', u'trafsignal_count',
                     u'turning_loop', u'giveway_count',u'htype_count']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(geohash_vec.loc[:, 'amenity_count':])
    scaled_values = scaler.transform(geohash_vec.loc[:, 'amenity_count':])
    geohash_vec.loc[:, 'amenity_count':] = scaled_values
    geohash_dict = {}
    for index, row in a.iterrows():
        geohash_dict[row.geohash] = np.array(geohash_vec.iloc[index])

    f = open(
        '/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata/clustered_data_poi_acc_merged.pkl',
        "wb")
    pickle.dump(geohash_dict, f)
    f.close()

def to_scalar(df, name):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df.iloc[:, 1:])
    scaled_values = scaler.transform(df.iloc[:, 1:])
    df.loc[:, 1:] = scaled_values
    #print('-----------------')

    # display(geohash_vec)
    geohash_dict = {}
    #print(geohash_dict)
    for index, row in df.iterrows():
        geohash_dict[row.geohash] = np.array(df.iloc[index])
    geohash_dict1 = geohash_dict.copy()
    for key, values in geohash_dict1.items():
        geohash_dict1[key]=np.array([values[1]])
    print('name=',name)
    print(geohash_dict1)
    print('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata/' + name + '.pkl')
    f = open('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata/' + name + '.pkl', "wb")
    pickle.dump(geohash_dict1, f)
    f.close()


def acc_count(joined_data):
    df = pd.read_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/Alldata_baveria/geodata/acc_count.csv',
                     header=0)
    df_accCount = df.groupby('geohash').mean().reset_index()
    joined_data=joined_data[['geohash', 'cluster_id']]
    joined_data=joined_data.drop_duplicates()
    poi_hash = pd.merge(joined_data, df_accCount, on='geohash',how='left')
    poi_hash=poi_hash[['cluster_id','acc_count_roun1kmfrom1km']]
    poi_hash=poi_hash.drop_duplicates()
    poi_hash.columns=['geohash','acc_count']
    count_acc = poi_hash.groupby('geohash').sum().reset_index()
    name = 'acc_count'
    count_acc.to_csv(
        '/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/som_clustering_30x30/geodata//acc_count.csv', index=False)
    to_scalar(count_acc, name)


# for uart, utype,zustand
def to_savepkl(df, name):
    col = df.iloc[:, 1:].values
    NLP_dict_uart = {}
    for index, row in df.iterrows():
        NLP_dict_uart[row.geohash] = np.array(col[index])
    f = open('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata/' + name + '.pkl', "wb")
    pickle.dump(NLP_dict_uart, f)





def to_htype(df1):
    df = pd.read_csv("/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/munichgeohash7/htype_onehotgeohash5x5grid1.csv")
    df_htype = df.groupby('geohash').mean().reset_index()
    joined_data=df1[['geohash', 'cluster_id']]
    joined_data=joined_data.drop_duplicates()
    poi_hash = pd.merge(joined_data, df_htype, on='geohash',how='left')

    poi_hash=poi_hash[['cluster_id','secondary','motorway','primary_link','motorway_link','construction','service',
    'footway','track','cycleway','trunk_link','tertiary_link','trunk','pedestrian','living_street','tertiary','secondary_link',
                          'residential','primary','unclassified']]
    poi_hash.columns=['geohash','secondary','motorway','primary_link','motorway_link','construction','service',
    'footway','track','cycleway','trunk_link','tertiary_link','trunk','pedestrian','living_street','tertiary','secondary_link',
                          'residential','primary','unclassified']
    poi_hash = poi_hash.groupby('geohash').mean().reset_index()
    poi_hash=poi_hash.fillna(0)
    poi_hash.to_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata//df_htype.csv', index=False)
    print(poi_hash.count)

    name = 'htype'
    to_savepkl(poi_hash, name)


def to_maxspeed(df1):
    df=pd.read_csv("/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/munichgeohash7/speed.csv")
   df_maxspeed = df.groupby('geohash').mean().reset_index()
    joined_data=df1[['geohash', 'cluster_id']]
    joined_data=joined_data.drop_duplicates()
    poi_hash = pd.merge(joined_data, df_maxspeed, on='geohash',how='left')

    poi_hash=poi_hash[['cluster_id','speed']]
    poi_hash.columns=['geohash','speed']
    poi_hash = poi_hash.groupby('geohash').mean().reset_index()
    poi_hash=poi_hash.fillna(0)
    poi_hash.to_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/new_method/gridgrowing/geodata/df_maxspeed.csv', index=False)
    print(poi_hash.count)

    
    name = 'maxspeed'
    to_scalar(poi_hash, name)


def regionattributes(city,method):
    city='new_method/gridgrowing'
    joined_data = pd.read_csv('/data/dadwal/data/DAP_data/dataPrepTrainTestCluster/Baveria/'+city+'/geohash_cluster7.csv',
    to_htype(joined_data)
    to_maxspeed(joined_data)



if __name__ == "__main__":
    cities = ['hannover']#,'osna','gott','olden','osna','LS']
    methods = ['dbscan']
    poi_pkl(city, method)
    regionattributes(city, method)