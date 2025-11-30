import IGCAnalysis
#import igcanalysis
from IGCAnalysis import AsymPlotLib as apl


import numpy as np
# import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# import time

from io import StringIO
# import math
#cimport openpyxl
from scipy import signal

#import datetime  # https://stackoverflow.com/questions/29385868/plotting-datetime-objects-with-pyqtgraph
#from datetime import datetime

def test_AsymPlot():
    # Initialisierung

    h        = []
    p        = []
    line     = []
    my_region= []
    l_plt    = 1
    l_suchkr = 1
    l_wechsel = 1
    
    
    
    ####   Daten einlesen

    # Use POSIX path (forward slash) so backslash escapes aren't interpreted
    filename = '/home/helmut/Python/IGCAnalysis/.venv_kubuntu/src/37KVQ2I1.igc'
    findstrings, kfindstrings, hardware, software, FlapSensor = apl.readfile(filename)

    findstr    =  findstrings['findstr']
    findstr_u  =  findstrings['findstr_u']
    kfindstr   =  kfindstrings['kfindstr']
    kfindstr_u =  kfindstrings['kfindstr_u']


    #### Panda Data Frame erzeugen
    
    df, dfk = apl.mpdf(findstrings, kfindstrings)

    # Latitude und Longitude berechnen 

    BDG, LDG = apl.CalcTrack (df)

    # Asymplot

    lat1     = BDG[0:-1].reset_index(drop=True)
    lat2     = BDG[1:].reset_index(drop=True)
    lon1     = LDG[0:-1].reset_index(drop=True)
    lon2     = LDG[1:].reset_index(drop=True)

    df2                                                            = pd.DataFrame(data={'lon1':lon1,'lon2':lon2,'lat1':lat1,'lat2':lat2})
    df2['angle'], df2['distance'], df2['alpha']                    = apl.haversine_np(df2['lon1'],df2['lat1'],df2['lon2'],df2['lat2'])        
    #df2['alpha'].loc [df2['alpha'][df2['alpha']<0].dropna().index] = 360+df2['alpha']
    df2.loc[df2['alpha'][df2['alpha']<0].dropna().index,'alpha'] = 360+df2['alpha']

    # Height = df['Press Alt.'][0:-1].reset_index(drop=True)
    Height   = df['GNS Alt.'][0:-1].reset_index(drop=True)
    Sekunden = df['Seconds'][0:-1].reset_index(drop=True)
    AOR      = df['AOR'][0:-1].reset_index(drop=True)
    TRT      = df['TRT'][0:-1].reset_index(drop=True)

    Kreisengamma,df_Kr,df_Gl,AnzRK, AnzLK                          =  apl.EstimateTrajectoryAngle (BDG,LDG,Height, Sekunden, df2, AOR, l_suchkr, l_wechsel, TRT)

    apl.AsymPlot(AnzRK,AnzLK,df,df_Kr,filename,hardware, software, FlapSensor, l_plt)
    #apl.AsymPlot(AnzRK,AnzLK,df,df_Kr,filename)
