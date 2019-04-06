# -*- coding: utf-8 -*-
"""
Spyder Editor.
"""
####################################    Spatial Analyses: SYRACUSE   #######################################
#######################################  Analyse data from Census #######################################
#This script performs basic analyses for the Exercise 1 of the workshop using Census data.
# The overall goal is to explore spatial autocorrelation and aggregation of units of analyses.     
#
#AUTHORS: Benoit Parmentier                                             
#DATE CREATED: 12/29/2018 
#DATE MODIFIED: 04/04/2019
#Version: 1
#PROJECT: AAG 2019 workshop preparation
#TO DO:
#
#COMMIT: added Moran'I and spatial regression, AAG workshop
#Useful links:
#sudo mount -t vboxsf C_DRIVE ~/c_drive

##################################################################################################

###### Library used in this script

import gdal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import subprocess
import pandas as pd
import os, glob
from rasterio import plot
import geopandas as gpd
import descartes
import libpysal as lp #new pysal interface
#from cartopy import crs as ccrs
from pyproj import Proj
from osgeo import osr
from shapely.geometry import Point
import pysal as ps
import os
import splot #for hotspot analysis
from esda.moran import Moran
from libpysal.weights.contiguity import Queen
from sklearn.preprocessing import StandardScaler

################ NOW FUNCTIONS  ###################

##------------------
# Functions used in the script 
##------------------

def create_dir_and_check_existence(path):
    #Create a new directory
    try:
        os.makedirs(path)
    except:
        print ("directory already exists")

############################################################################
#####  Parameters and argument set up ########### 

#ARG 1
in_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/Exercise_1/data"
#ARG 2
out_dir = "/home/bparmentier/c_drive/Users/bparmentier/Data/python/Exercise_1/outputs"
#ARG 3:
create_out_dir=True #create a new ouput dir if TRUE
#ARG 4
out_suffix = "exercise1_03182019" #output suffix for the files and ouptut folder
#ARGS 5
NA_value = -9999 # number of cores
file_format = ".tif"

ct_2000_fname = "ct_00.shp" # CT_00: Cencus Tracts 2000
bg_2000_fname = "bg_00.shp" # BG_00: Census Blockgroups 2000
bk_2000_fname = "bk_00.shp" # BK_00: Census Blocks 2000

census_table_fname = "census.csv" #contains data from census to be linked
soil_PB_table_fname = "Soil_PB.csv" #same as census table
tgr_shp_fname = "tgr36067lkA.shp" #contains data from census to be linked

metals_table_fname = "SYR_metals.xlsx" #contains metals data to be linked

################# START SCRIPT ###############################

######### PART 0: Set up the output dir ################

#set up the working directory
#Create output directory

if create_out_dir==True:
    #out_path<-"/data/project/layers/commons/data_workflow/output_data"
    out_dir = "output_data_"+out_suffix
    out_dir = os.path.join(in_dir,out_dir)
    create_dir_and_check_existence(out_dir)
    os.chdir(out_dir)        #set working directory
else:
    os.chdir(create_out_dir) #use working dir defined earlier
    
    
#######################################
### PART 1: Read in datasets #######

## Census tracks for Syracuse in 2000
ct_2000_filename = os.path.join(in_dir,ct_2000_fname)
## block groups for Syracuse in 2000
bg_2000_filename = os.path.join(in_dir,bg_2000_fname)
## block for Syracuse in 200
bk_2000_filename = os.path.join(in_dir,bk_2000_fname)

#Read spatial data 
ct_2000_gpd = gpd.read_file(ct_2000_filename)
bg_2000_gpd = gpd.read_file(bg_2000_filename)
bk_2000_gpd = gpd.read_file(bk_2000_filename)

#Explore datasets:
ct_2000_gpd.describe()
ct_2000_gpd.plot(column="CNTY_FIPS")
ct_2000_gpd.head()

#Read tabular data
metals_df = pd.read_excel(os.path.join(in_dir,metals_table_fname))
census_syr_df = pd.read_csv(os.path.join(in_dir,census_table_fname),sep=",",header=0) #census information
#This soil lead in UTM 18 coordinate system
soil_PB_df = pd.read_csv(os.path.join(in_dir,soil_PB_table_fname),sep=",",header=None) #point locations

#Check size
ct_2000_gpd.shape #57 spatial entities (census)
bg_2000_gpd.shape #147 spatial entities (block groups)
bk_2000_gpd.shape #2025 spatial entities (blocks)
census_syr_df.shape #147 spatial entities
metals_df.shape #57 entities

#########################################################
####### PART 2: Visualizing population in 2000 at Census track level with geopandas layers 
#### We explore  also two ways of joining and aggregating data at census track level #########
#### Step 1: First join census information data to blockgroups
#### Step 2: Summarize/aggregate poppulation at census track level ###
#### Step 3: Plot population 2000 by tracks

### Step 1: First join census data to blockgroups

bg_2000_gpd.columns # missing census information:check columns' name for the data frame
census_syr_df.columns #contains census variables to join
#Key is "TRACT" but with a different format/data type
#First fix the format
bg_2000_gpd.head()
bg_2000_gpd.shape
census_syr_df.BKG_KEY.head()
#ct_2000_gpd.TRACT.dtype
census_syr_df.dtypes #check all the data types for all the columns
bg_2000_gpd.BKG_KEY.dtypes #check data type for the "BKG_KEY"" note dtype is 'O"
census_syr_df.BKG_KEY.dtypes

#Change data type for BKG_KEY column from object 'O" to int64
bg_2000_gpd['BKG_KEY'] = bg_2000_gpd['BKG_KEY'].astype('int64')

# Join data based on common ID after matching data types
bg_2000_gpd = bg_2000_gpd.merge(census_syr_df, on='BKG_KEY')
# Check if data has been joined 
bg_2000_gpd.head()

#Quick visualization of population 
bg_2000_gpd.plot(column='POP2000',cmap="OrRd")
plt.title('POPULATION 2000')

#############
#### Step 2: Summarize/aggregate poppulation at census track level

### Method 1: Summarize by census track using DISSOLVE geospatial operation

#To keep geometry, we must use dissolve method from geopanda
census_2000_gpd = bg_2000_gpd.dissolve(by='TRACT',
                                       aggfunc='sum')
type(census_2000_gpd)
census_2000_gpd.index
#Note that the TRACT field has become the index
census_2000_gpd=census_2000_gpd.reset_index() # reset before comparing data
census_2000_gpd.shape #Dissolved results shows aggregation from 147 to 57.

### Method 2: Summarize using groupby aggregation and joining

##Note losing TRACT field
census_2000_df = bg_2000_gpd.groupby('TRACT',as_index=False).sum()
type(census_2000_df) #This is a panda object, we lost the geometry after the groupby operation.
census_2000_df.shape #Groupby results shows aggregation from 147 to 57.

### Let's join the dataFrame to the geopanda object to the census track layer 
census_2000_df['TRACT'].dtype == ct_2000_gpd['TRACT'].dtype #Note that the data type for the common Key does not mach.  
census_2000_df['TRACT'].dtype # check data type field from table
ct_2000_gpd['TRACT'].dtype # check data type field from census geopanda layer
ct_2000_gpd['TRACT'] = ct_2000_gpd.TRACT.astype('int64') #Change data type to int64
ct_2000_gpd.shape #57 rows and 8 columns

ct_2000_gpd = ct_2000_gpd.merge(census_2000_df, on='TRACT')
ct_2000_gpd.shape #57 rows and 50 columns

#### Step 3: Plot population 2000 by tracks in Syracuse

### Check if the new geometry of entities is the same as census
fig, ax = plt.subplots(figsize=(12,8))
ax.set_aspect('equal') # set aspect to equal, done automatically in *geopandas* plot but not in pyplot
census_2000_gpd.plot(ax=ax,column='POP2000',cmap='OrRd')
ct_2000_gpd.plot(ax=ax,color='white',edgecolor="red",alpha=0.7) # Check if outputs from two methods match
ax.set_title("Population", fontsize= 20)

#### Generate population maps with two different class intervals

title_str = "Population by census tract in 2000"
census_2000_gpd.plot(column='POP2000',cmap="OrRd",
                 scheme='quantiles')
plt.title(title_str)

### Let's use more option with matplotlib

fig, ax = plt.subplots(figsize=(14,6))
census_2000_gpd.plot(column='POP2000',cmap="OrRd",
                 scheme='equal_interval',k=7,
                 ax=ax,
                 legend=False)

fig, ax = plt.subplots(figsize=(14,6))
census_2000_gpd.plot(column='POP2000',cmap="OrRd",
                 scheme='quantiles',k=7,
                 ax=ax,
                 legend=True)
ax.set_title('POP2000')

##############################################
##### PART 3: SPATIAL QUERY #############
### We generate a dataset with metals and lead information by census tracks.
### To do so we use the following steps:
##Step 1: Join metals to census tracks 
##Step 2: Generate geopanda from PB sample measurements 
##Step 3: Join lead (pb) measurements to census tracks
##Step 4: Find average lead by census track

##### Step 1: Join metals to census tracks ###### 

metals_df.head()
metals_df.describe # 57 rows  
##Number of rows suggests matching to the following spatial entities
metals_df.shape[0]== ct_2000_gpd.shape[0]
#Check data types before joining tables with "merge"
metals_df.dtypes
ct_2000_gpd.dtypes
ct_2000_gpd.shape
census_metals_gpd = ct_2000_gpd.merge(metals_df,left_on='TRACT',right_on='ID')
census_metals_gpd.shape #census information has been joined

##### Step 2: Generate geopanda from PB sample measurements ##### 
# Processing lead data to generate a geopanda object using shapely points

soil_PB_df.columns #Missing names for columns
soil_PB_df.columns = ["x","y","ID","ppm"]
soil_PB_df.head()

soil_PB_gpd = soil_PB_df.copy()
type(soil_PB_df)
soil_PB_gpd['Coordinates']=list(zip(soil_PB_gpd.x,soil_PB_gpd.y)) #create a new column with tuples of coordinates
type(soil_PB_gpd)
soil_PB_gpd['Coordinates']= soil_PB_gpd.Coordinates.apply(Point) #create a point for each tupple row
type(soil_PB_gpd.Coordinates[0]) #This shows that we created a shapely geometry point
type(soil_PB_gpd) #This is still an panda DataFrame
soil_PB_gpd = gpd.GeoDataFrame(soil_PB_gpd,geometry='Coordinates') #Create a gpd by setting the geometry column
type(soil_PB_gpd) # This is now a GeoDataFrame

## Checking and setting the coordinates reference system
soil_PB_gpd.crs #No coordinate reference system (CRS) is set
census_metals_gpd.crs # Let's use the metal geopanda object to set the CRS

## Find out more about the CRS using the epsg code
epsg_code = census_metals_gpd.crs.get('init').split(':')[1]
inproj = osr.SpatialReference()
inproj.ImportFromEPSG(int(epsg_code))
inproj.ExportToProj4() # UTM 18: this is the coordinate system in Proj4 format
## Assign projection system
soil_PB_gpd.crs= census_metals_gpd.crs #No coordinate system is set
soil_PB_gpd.head()

## Now plot the points
fig, ax = plt.subplots()
census_metals_gpd.plot(ax=ax,color='white',edgecolor='red')
soil_PB_gpd.plot(ax=ax,marker='*',
                 color='black',
                 markersize=0.8)

##### Step 3: Join lead (pb) measurements to census tracks #####
# Spatial query: associate points of pb measurements to each census tract

soil_PB_joined_gpd =gpd.tools.sjoin(soil_PB_gpd,census_2000_gpd,
                     how="left")
soil_PB_joined_gpd.columns
soil_PB_joined_gpd.shape #every point is associated with information from the census track it is contained in

len(soil_PB_joined_gpd.BKG_KEY.value_counts()) #associated BKG Key to points: 57 unique identifiers
len(soil_PB_joined_gpd.index_right.value_counts()) #associated BKG Key to points: 57 unique identifiers

#### Step 4: Find average lead by census track #####

grouped_PB_ct_df = soil_PB_joined_gpd[['ppm','TRACT','index_right']].groupby(['index_right']).mean() #compute average by census track
grouped_PB_ct_df = grouped_PB_ct_df.reset_index()
grouped_PB_ct_df.shape
grouped_PB_ct_df.head()

grouped_PB_ct_df = grouped_PB_ct_df.rename(columns={'ppm': 'pb_ppm' })
type(grouped_PB_ct_df)

census_metals_gpd = census_metals_gpd.merge(grouped_PB_ct_df,on="TRACT")
census_metals_gpd.shape
census_metals_gpd.columns #check for duplicate columns

outfile_metals_shp = "census_metals_pb_"+'_'+out_suffix+'.shp'
census_metals_gpd.to_file(os.path.join(outfile_metals_shp))

census_metals_df = pd.DataFrame(census_metals_gpd.drop(columns='geometry'))
outfile = "census_metals_pb_"+'_'+out_suffix+'.csv'

census_metals_df.to_csv(os.path.join(outfile))

census_metals_gpd.head()

#################################################
##### PART IV: Spatial regression: Vulnerability to metals #############
#Examine the relationship between  Pb and vulnerable populations in Syracuse

######## Step 1: Explore neighbors with pysal

w = Queen.from_dataframe(census_metals_gpd)
type(w)
w.transform = 'r'
w.n # number of observations (spatial features)
census_metals_gpd.index #this is the index used for neighbors
w.neighbors # list of neighbours per census track
w.mean_neighbors #average number of neighbours

### Visualizaing neighbors:
ax = census_metals_gpd.plot(edgecolor='grey', 
                            facecolor='w')
f,ax = w.plot(census_metals_gpd, ax=ax, 
        edge_kws=dict(color='r', linestyle=':', 
                      linewidth=1),
        node_kws=dict(marker=''))
ax.set_axis_off()
ax.set_title("Queen Neighbors links")

########## Step 2: Explore Moran's I ##########

#http://pysal.org/notebooks/viz/splot/esda_morans_viz
y = census_metals_gpd['pb_ppm'] 
w_queen = ps.lib.weights.Queen.from_shapefile(outfile_metals_shp)
y_lag = ps.model.spreg.lag_spatial(w_queen,y)
census_metals_gpd['y'] = census_metals_gpd['pb_ppm']
census_metals_gpd['y_lag'] = y_lag

ax= sns.regplot(x='y',y='y_lag',data=census_metals_gpd)
ax.set_title("Moran's scatter plot")

### Visualize the Moran's I with standardized values
scaler = StandardScaler()
census_metals_gpd['y_std'] = scaler.fit_transform(census_metals_gpd['pb_ppm'].values.reshape(-1,1))
census_metals_gpd['y_lag_std'] = ps.model.spreg.lag_spatial(w_queen,
                                                census_metals_gpd['y_std']) #this is a numpy array

ax= sns.regplot(x='y_std',y='y_lag_std',data=census_metals_gpd)
ax.set_title("Moran's scatter plot")
ax.axhline(0, color='black')
ax.axvline(0, color='black')
#for more in depth understanding take a look at: http://darribas.org/gds15/content/labs/lab_06.html

#This suggests autocorrelation.
#Let's us Moran's I.

moran = Moran(y, w)
moran.I # Moran's I value
moran.p_sim # Permutation test suggests that autocorrelation is significant

########## Step 3: Spatial Regression ##########

y.values.shape #not the right dimension
y = y.values.reshape(len(y),1)

y_lag = y_lag.reshape(len(y_lag),1)
x = census_metals_gpd['perc_hispa']
x = x.values.reshape(len(x),1)

mod_ols = ps.model.spreg.OLS(y,x)
mod_ols.u 
m_I_residuals = ps.explore.esda.Moran(mod_ols.u,w_queen)
m_I_residuals.I
m_I_residuals.p_sim #significant autocorrelation

#take into account autocorr in spreg
mod_ols.summary
mod_ols_test = ps.model.spreg.OLS(y,x,w_queen)
mod_ols_test.summary

mod_ml_lag = ps.model.spreg.ML_Lag(y,x,w_queen)
mod_ml_lag.summary
mod_ml_lag.u
m_ml_I_residuals = ps.explore.esda.Moran(mod_ml_lag.u,w_queen)
m_ml_I_residuals.I # Moran's I is lower now!!!

m_ml_I_residuals.p_sim #not significant autocorrelation

################################## END OF SCRIPT ########################################

















