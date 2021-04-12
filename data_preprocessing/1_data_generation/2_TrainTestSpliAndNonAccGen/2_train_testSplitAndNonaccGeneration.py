from shapely import geometry
from shapely.geometry import shape, Point
import geohash as gh
import numpy as np
import pandas as pd
import numpy as np
import sys
import pandas as pd
import datetime
import random
from random import choices



# get train acc and test acc
# Train acc from 2017 to 2019 May
#Test acc from 2019 May to Dec 2019

def traintestdataAcc(data, city):
    train_data_2017_acc = data.loc[(data['UJAHR'] == 2017)]
    train_data_2018_acc = data.loc[
         (data['UJAHR'] == 2018)]
    train_data_2019_acc = data.loc[
    (data['UMONAT'] <= 5) & (data['UJAHR'] == 2019)]

    frames = [train_data_2017_acc, train_data_2018_acc,train_data_2019_acc]
    train_data_acc = pd.concat(frames)
    
    train_data_acc.to_csv('../data/regions/'+city+'/train_acc.csv',index=False)
    test_data_acc = data.loc[
        (data['UMONAT'] > 5) & (data['UJAHR'] == 2019)]
    test_data_acc.to_csv('../data/regions/'+city+'/test_acc.csv',index=False)
    return train_data_acc,test_data_acc


def random_latlong(geohash):
    dic = gh.bbox(geohash)
    # getting min, max lat/lng
    min_lng = dic.get('w')
    min_lat = dic.get('s')
    max_lng = dic.get('e')
    max_lat = dic.get('n')
    # generate random float between [min_lng, max_lng)
    long = np.random.uniform(min_lng, max_lng)
    # generate random float between [min_lat, max_lat)
    lat = np.random.uniform(min_lat, max_lat)
    return lat, long


def dow(date):
    dayNumber = date.weekday()
    day = -1
    if dayNumber == 6:
        day = 1
    else:
        day = dayNumber + 2
    return day



def find_t_nonACC(t):
    tm = str(t)
    dateTimesplit = tm.split(' ')
    dateFind = dateTimesplit[0]
    timeFind = dateTimesplit[1]
    datesplit = dateFind.split('-')
    timesplit = timeFind.split(':')
    frmt = '%Y-%m-%d'
    datsend = datetime.datetime.strptime(dateFind, frmt)
    dayofweek = dow(datsend)
    year, month, day = datesplit[0], datesplit[1], datesplit[2]
    month = int(month)
    hour = int(timesplit[0])
    return year, month, dayofweek, hour


def randomtimes(geohash, stime, etime, n):
    frmt = '%d-%m-%Y %H:%M:%S'
    stime = datetime.datetime.strptime(stime, frmt)
    etime = datetime.datetime.strptime(etime, frmt)
    td = etime - stime
    k = []
    t = random.random() * td + stime
    year, month, dayofweek, hour = find_t_nonACC(t)
    year = int(year)
    lat, long = random_latlong(geohash)
    return True, lat, long, year, month, dayofweek,hour

def trainNonacc(hann_grid_zeroacc,train,city):
    t = []
    a=[]
    no_of_acc=len(train.index)
    print('no of acc=',no_of_acc)
    no_of_nonacc=no_of_acc*3
    print('no of non acc in train=',no_of_nonacc)

    for i in range(0,no_of_nonacc):
        geohashVal=hann_grid_zeroacc['geohash'].values # 153m x153 m all geohashes
        geoSelect=choices(geohashVal) # select one geohash with replacement
        bol, lat, long, year, month, dayofweek,hour = randomtimes(geoSelect[0], '01-01-2017 00:00:00',
                                                                     '31-05-2019 23:00:00', i)
        if bol and [year, month, dayofweek,hour] not in t:
            p = (year, month, dayofweek,hour)
            k = (geoSelect[0], lat, long, year, month, dayofweek,hour)
            a.append(k)
            t.append(p)
            i = i + 1
        else:
            continue

    dt = pd.DataFrame(a)
    dt.columns = ['geohash', 'non_acclat', 'non_acclong', 'UJAHR', 'UMONAT', 'UWOCHENTAG','hour']
    dt['UMONAT'] = dt["UMONAT"].astype(str).astype(int)
    dt['UJAHR'] = dt["UJAHR"].astype(str).astype(int)
    train_non_acc_data=dt.loc[((dt['UJAHR']==2017) & (dt['UMONAT']<=12)|(dt['UJAHR']==2018) & (dt['UMONAT']<=12) | ((dt['UJAHR']==2019) & (dt['UMONAT']<=5)))]
    train_non_acc_data.to_csv('../data/regions/'+city+'/train_nonaccdata.csv', index=False)

    
def testNonacc(hann_grid_zeroacc,test,city):
    a=[]
    t=[]
    no_of_acc=len(test.index)
    print('no of acc=',no_of_acc)
    no_of_nonacc=no_of_acc*3
    print('no of non acc in test=',no_of_nonacc)

    for i in range(0,no_of_nonacc):
        geohashVal=hann_grid_zeroacc['geohash'].values # 153m x153 m all geohashes
        geoSelect=choices(geohashVal) # select one geohash with replacement
        bol, lat, long, year, month, dayofweek,hour = randomtimes(geoSelect[0], '01-06-2019 00:00:00',
                                                                     '31-12-2019 23:00:00', i)
        if bol and [year, month, dayofweek,hour] not in t:
            p = (year, month, dayofweek,hour)
            k = (geoSelect[0], lat, long, year, month, dayofweek,hour)
            a.append(k)
            t.append(p)
            i = i + 1
        else:
            continue

    dt = pd.DataFrame(a)
    dt.columns = ['geohash', 'non_acclat', 'non_acclong', 'UJAHR', 'UMONAT', 'UWOCHENTAG','hour']
    dt['UMONAT'] = dt["UMONAT"].astype(str).astype(int)
    dt['UJAHR'] = dt["UJAHR"].astype(str).astype(int)
    test_data = dt.loc[(dt['UMONAT'] > 5) & (dt['UJAHR'] == 2019)]
    test_data.to_csv('../data/regions/'+city+'/test_nonaccdata.csv', index=False)


if __name__ == "__main__":
    cities = ['LS/hannover']#,'Bayern/munich','Bayern/nurenberg']
    for city in cities:                  
        region_grid=pd.read_csv('../data/regions/'+city+'/numberofGridRegionGeo7.csv',header=0)

        region_selectedWithacc=pd.read_csv('../data/regions/'+city+'/acc_threeyear.csv',header=0)
        

        train,test=traintestdataAcc(region_selectedWithacc, city)

        # non acc cases generation
        trainNonacc(region_grid,train,city)
        testNonacc(region_grid,test,city)
        
        print('finished for city=',city)
