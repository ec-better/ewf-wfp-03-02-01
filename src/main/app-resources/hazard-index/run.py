#!/opt/anaconda/envs/env_hazard_index/bin/python

import atexit
import cioppy
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from urllib.parse import urlparse


SUCCESS = 0
ERR_RESOLUTION = 10
ERR_STAGEIN = 20
ERR_NO_OUTPUT = 30

# add a trap to exit gracefully
def clean_exit(exit_code):
    log_level = 'INFO'
    if exit_code != SUCCESS:
        log_level = 'ERROR'  
   
    msg = {SUCCESS: 'Processing successfully concluded',
           ERR_RESOLUTION: 'Could not resolve Sentinel-1 product enclosure',
           ERR_STAGEIN: 'Could not stage-in Sentinel-1 product', 
           ERR_NO_OUTPUT: "Missing output"
    }
 
    ciop.log(log_level, msg[exit_code])  



def get_vsi_url(enclosure, username=None, api_key=None):

    parsed_url = urlparse(enclosure)

    if username is not None:
        url = '/vsigzip//vsicurl/%s://%s:%s@%s%s' % (list(parsed_url)[0],
                                                       username,
                                                       api_key,
                                                       list(parsed_url)[1],
                                                       list(parsed_url)[2])

    else:

        url = '/vsigzip//vsicurl/%s://%s%s' % (list(parsed_url)[0],
                                            list(parsed_url)[1],
                                            list(parsed_url)[2])


    return url

def to_ds(search, username, api_key):
    
    chirps_ds = []
    dates = []
    datasets = []
    
    for index, row in search.iterrows():

            # read the vsicurl geotiff
            da = xr.open_rasterio(get_vsi_url(row['enclosure'], username, api_key))

            # remove the band dimension
            da = da.squeeze().drop(labels='band')

            # add the variable
            da = da.to_dataset(name='rainfall')

            dates.append(row['startdate'])
            datasets.append(da)
   
    ds = xr.concat(datasets, dim=pd.DatetimeIndex(dates, name='date'))
    
    return ds

def get_weights(ds):
    
    zigma=ds['rainfall'].sum(dim=['date'])
    
    for index, d in enumerate(ds['rainfall'].date.values):
       
        w.append((ds['rainfall'].sel(date=d)/zigma)) 

    ds_w = xr.concat(w, 
                     dim=pd.DatetimeIndex(ds['rainfall'].date.values, 
                                             name='date'))

    # Throw away nan entries to make it arithmetically multipicable
    remove_nan = lambda x: 0 if np.isnan(x) else x
    
    vfunc_remove_nan = np.vectorize(remove_nan,otypes=[np.float64])
    
    return vfunc_remove_nan(ds_w)

def percentile(x):
    ''' this function calculates impirical percentile of input array'''
    nan_indeces=[]
    if not np.isnan(x).all():
        
        rank = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            
            if np.isnan(x[i]):
                
                nan_indeces.append(i)
                rank[i]=0
                
            else:
                for j in range(x.shape[0]):
                    
                    if not np.isnan(x[j]) and not i==j  and x[j]<=x[i]:
                        
                        rank[i]+=1
                    
                    
        percentil = ((rank + 0.5) / (x.shape[0] - len(nan_indeces)))
    

        for i in nan_indeces:
            
            percentil[i] = np.nan
            
    else:
        
        percentil = np.empty(x.shape[0])
        percentil[:] = np.nan

    return percentil

def teta_sp(x,y):
    '''this function computes sum of inner product of two 1d input array'''
    if x.shape==y.shape:
        x[np.isnan(x)]=0
        y[np.isnan(y)]=0
        return np.sum(x*y)
    else:
        return np.nan


def inv_logit(p):
    '''maps from the linear predictor to the probabilities'''
    return np.exp(p) / float(1 + np.exp(p))

def main():
    
    ciop = cioppy.Cioppy()
    
    
    parameters = dict()
    
    parameters['username'] = ciop.getparam('_T2Username')
    parameters['api_key'] = ciop.getparam('_T2ApiKey')

    
    enclosures = []
    
    creds = '{}:{}'.format(parameters['username'],
                           parameters['api_key'])

    
    search_params = dict()
    
    temp_results = []
    
    for line in sys.stdin:
        
        entry = cioppy.search(end_point=line.rstrip(),
                               params=search_params,
                               output_fields='self,startdate,enddate,enclosure,title',
                               model='GeoTime',
                               timeout=1200000,
                               creds=creds)[0]
    
        enclosures.append(search['enclosure'])
        
        temp_results.append(entry)  

    search = gp.GeoDataFrame(temp_results)
    
    # Convert startdate to pd.datetime and sort by date
    search['startdate_dt'] = pd.to_datetime(search.startdate)
    search['enddate_dt'] = pd.to_datetime(search.enddate)
    
    search = search.sort_values(by='startdate_dt')
    
    
    ciop.log('Create xarray dataset')
    ds = to_ds(search,
               username=parameters['username'], 
               api_key=parameters['api_key'])
    
    
    # compute impirical percentile for each pixel over the vector 'date'
    result = np.apply_along_axis(percentile, 
                                 0, 
                                 ds['rainfall'])
    
    
    ciop.log('Get weights')
    w = get_weights(ds)
    
    ciop.log('pixel-wise weighted percentile')
    # pixel-wise weighted percentile 
    for i in range(result.shape[1]):
        for j in range(result.shape[2]):
            
            teta[i,j]=teta_sp(result[:,i,j],
                              w[:,i,j])
    
    vfunc_inv_logit = np.vectorize(inv_logit,
                                   otypes=[np.float64])
    
    ciop.log('Precipitation hazard index')
    q = 100 * vfunc_inv_logit(teta)
    
    ciop.log('Save as geotiff')
    
    output_name = 'rainfall_hazard_index_{}_{}.nc'.format(search['startdate_dt'].min().strftime('%Y_%m_%d'), 
                                                       search['enddate_dt'].max().strftime('%Y_%m_%d'))
    
    q.to_netcdf(output_name)
    
    ciop.log('Publish geotiff')
    
    ciop.publish(output_name)