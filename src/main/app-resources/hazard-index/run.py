#!/opt/anaconda/envs/env_hazard_index/bin/python

import atexit
import cioppy
ciop = cioppy.Cioppy()
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
import sys
import gdal
from urllib.parse import urlparse


SUCCESS = 0
ERR_RESOLUTION = 10
ERR_STAGEIN = 20
ERR_NO_OUTPUT = 30

ciop = cioppy.Cioppy()

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
    w=[]
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
    
    parameters = dict()
    
    parameters['username'] = None if ciop.getparam('_T2Username') == '' else ciop.getparam('_T2Username')
    parameters['api_key'] = None if ciop.getparam('_T2ApiKey') == '' else ciop.getparam('_T2ApiKey')

    ciop.log('INFO', 'username: "{}"'.format(parameters['username']))
   
    search_params = dict()
    
    temp_results = []
    
    for line in sys.stdin:
        
        ciop.log('INFO', 'Line: {}'.format(line.rstrip()))
        
        
        if parameters['username'] is not None:
            
            creds = '{}:{}'.format(parameters['username'],
                                   parameters['api_key'])
        
            entry = ciop.search(end_point=line.rstrip(),
                                   params=search_params,
                                   output_fields='self,startdate,enddate,enclosure,title',
                                   model='GeoTime',
                                   timeout=1200000,
                                   creds=creds)[0]
           
        else:
    
            entry = ciop.search(end_point=line.rstrip(),
                                   params=search_params,
                                   output_fields='self,startdate,enddate,enclosure,title',
                                   model='GeoTime',
                                   timeout=1200000)[0]
        
        temp_results.append(entry)  

    search = gpd.GeoDataFrame(temp_results)
    
    # Convert startdate to pd.datetime and sort by date
    search['startdate_dt'] = pd.to_datetime(search.startdate)
    search['enddate_dt'] = pd.to_datetime(search.enddate)
    
    search = search.sort_values(by='startdate_dt')
    
    
    ciop.log('DEBUG', 'Create xarray dataset')
    ds = to_ds(search,
               username=parameters['username'], 
               api_key=parameters['api_key'])
    
    
    
    # Geo-Info reterived from input
    temp = get_vsi_url(search.iloc[0]['enclosure'], parameters['username'], parameters['api_key'])
    temp_ds = gdal.Open(temp)
    geo_transform = temp_ds.GetGeoTransform()
    projection = temp_ds.GetProjection()
    temp_ds = None
    
    # compute impirical percentile for each pixel over the vector 'date'
    ds_date_index,ds_x_index, ds_y_index= ds['rainfall'].shape
    result=np.zeros((ds_date_index,ds_x_index,ds_y_index),dtype=float)
    x_block=int(np.ceil(ds_x_index/1000))
    y_block=int(np.ceil(ds_y_index/1000))
    
    for i in range(x_block):
        for j in range(y_block):
            
            x_low=1000*(i)
            x_high=1000*(i+1)
            y_low=1000*(j)
            y_high=1000*(j+1)
            if x_high>ds_x_index:
                x_high=ds_x_index
            if y_high>ds_y_index:
                y_high=ds_y_index     

            result[:,x_low:x_high,y_low:y_high] = np.apply_along_axis(percentile, 
                                 0, 
                                 ds['rainfall'][:,x_low:x_high,y_low:y_high])
    

    
    
    
    
    ciop.log('DEBUG', 'Get weights')
    w = get_weights(ds)
    
    ciop.log('DEBUG','pixel-wise weighted percentile')
    # pixel-wise weighted percentile 
    teta=np.zeros((result.shape[1],result.shape[2]),dtype=float)
    
    for i in range(result.shape[1]):
        for j in range(result.shape[2]):
            
            teta[i,j]=teta_sp(result[:,i,j],
                              w[:,i,j])
    
    vfunc_inv_logit = np.vectorize(inv_logit,
                                   otypes=[np.float64])
    
    ciop.log('DEBUG', 'Precipitation hazard index')
    q = 100 * vfunc_inv_logit(teta)
    
    ciop.log('DEBUG', 'Save as geotiff')
    
    output_name = 'rainfall_hazard_index_{}_{}.nc'.format(search['startdate_dt'].min().strftime('%Y_%m_%d'), 
                                                       search['enddate_dt'].max().strftime('%Y_%m_%d'))
    
    
    
    cols=q.shape[1]
    rows=q.shape[0]
    drv = gdal.GetDriverByName('GTiff')

    ds_tif = drv.Create(output_name, 
                        cols, rows, 
                        1, 
                        gdal.GDT_Float32)

        
    ds_tif.SetGeoTransform(geo_transform)
    ds_tif.SetProjection(projection)
    ds_tif.GetRasterBand(1).WriteArray(q)
    ds_tif.GetRasterBand(1).SetDescription('Q')
    ds_tif.FlushCache()
    
    
    
    ciop.log('DEBUG', 'Publish geotiff')
    
    ciop.publish(output_name)
    
try:
    main()
except SystemExit as e:
    if e.args[0]:
        clean_exit(e.args[0])
    raise
else:
    atexit.register(clean_exit, 0)
