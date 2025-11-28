import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import seaborn as sns
sns.set_theme()

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


from io import StringIO
import math
#cimport openpyxl
from scipy import signal

import datetime  # https://stackoverflow.com/questions/29385868/plotting-datetime-objects-with-pyqtgraph
from datetime import datetime

# Subroutinen

def asymplot(filename,l_plt=1,l_suchkr=1,l_wechsel=0):
    # Initialisierung

    h        = []
    p        = []
    line     = []
    my_region= []
    aveLK    = np.nan
    aveRK    = np.nan
    np.aveRKw = np.nan
    np.aveLKw = np.nan

    df_Kr = pd.DataFrame()

    findstrings, kfindstrings, hardware, software, FlapSensor = readfile(filename)

    findstr    =  findstrings['findstr']
    findstr_u  =  findstrings['findstr_u']
    kfindstr   =  kfindstrings['kfindstr']
    kfindstr_u =  kfindstrings['kfindstr_u']

    if 'NET' in findstr:
        

        #### Panda Data Frame erzeugen

        df, dfk = mpdf(findstrings, kfindstrings)

        # Latitude und Longitude berechnen 

        if df['NET'].sum()>0:
            BDG, LDG = CalcTrack (df)

            # Asymplot

            lat1     = BDG[0:-1].reset_index(drop=True)
            lat2     = BDG[1:].reset_index(drop=True)
            lon1     = LDG[0:-1].reset_index(drop=True)
            lon2     = LDG[1:].reset_index(drop=True)

            df2                                                            = pd.DataFrame(data={'lon1':lon1,'lon2':lon2,'lat1':lat1,'lat2':lat2})
            df2['angle'], df2['distance'], df2['alpha']                    = haversine_np(df2['lon1'],df2['lat1'],df2['lon2'],df2['lat2'])        
            # df2['alpha'].loc [df2['alpha'][df2['alpha']<0].dropna().index] = 360+df2['alpha']
            df2.loc[df2['alpha'][df2['alpha']<0].dropna().index,'alpha'] = 360+df2['alpha']

            # Height = df['Press Alt.'][0:-1].reset_index(drop=True)
            Height   = df['GNS Alt.'][0:-1].reset_index(drop=True)
            Sekunden = df['Seconds'][0:-1].reset_index(drop=True)
            AOR      = df['AOR'][0:-1].reset_index(drop=True)
            if 'TRT' in df:
                TRT      = df['TRT'][0:-1].reset_index(drop=True)
            else:
                TRT      = pd.DataFrame({'TRT' : []}) # np.zeros(len(AOR))

            Kreisengamma,df_Kr,df_Gl,AnzRK, AnzLK                          =  EstimateTrajectoryAngle (BDG,LDG,Height, Sekunden, df2, AOR, l_suchkr, l_wechsel, TRT)

            aveRK, aveLK, np.aveRKw,np.aveLKw, df_Kr = AsymPlot(AnzRK,AnzLK,df,df_Kr,filename,hardware, software, FlapSensor, l_plt)

            l_hawk = True
        else:
            l_hawk = False
        
    else:
        l_hawk = False
        
    return  l_hawk, aveRK, aveLK,  np.aveRKw, np.aveLKw, df_Kr, hardware, software, FlapSensor
        
        
      



def readfile(filename):
   
    searchstrg     = ['VAT', 'TAS', 'GSP', 'TRT', 'OAT', 'NET', 'ACX', 'ACY', 'ACZ','AOR', 'AOP', 'HDM',  'IAS', 'SIU', 'MOP', 'FXA', 'ENL']
    unitstrg       = ['m/s', 'km/h','km/h','°',   '°C',  'm/s', 'g',   'g',   'g',  '°',   '°',   'deg',  'km/h','-',   '-',  '-',   '-'  ]
    faktors        = [100,   100,    100,   1,     10,   100,    1,     1,     100,  1,     1,     1,      1,     1,     1,    1,     1 ]   

    ksearchstrg      = ['WDI', 'WVE', 'WSP', 'SIU', 'VAR'];
    kunitstrg        = ['°',   'km/h','km/h','-',   'm/s'];
    kfaktor          = [1,      100,   100  , 1,    '0.1'];

    mylist     = []   # first append to a list and convert is later 

    IRecord    = {}
    findstr    = []
    findstr_u  = []
    findstr_f  = []
    findstr_d  = {}

    findstr_d['Zeit']       = []
    findstr_d['Latitude']   = []
    findstr_d['NorthSouth'] = []
    findstr_d['Longitude']  = []
    findstr_d['EastWest']   = []
    findstr_d['Press Alt.'] = []
    findstr_d['GNS Alt.']   = []

    findstr.append('Zeit')
    findstr_u.append ('hh:mm:ss')
    findstr_f.append(1)

    findstr.append('Latitude')
    findstr_u.append ('Deg')
    findstr_f.append(1)

    findstr.append('NorthSouth')
    findstr_u.append ('°')
    findstr_f.append(1)

    findstr.append('Longitude')
    findstr_u.append ('Deg')
    findstr_f.append(1)

    findstr.append('EastWest')
    findstr_u.append ('°')
    findstr_f.append(1)

    findstr.append('Press Alt.')
    findstr_u.append ('m')
    findstr_f.append(1)

    findstr.append('GNS Alt.')
    findstr_u.append ('m')
    findstr_f.append(1)

    IRecord['Zeit'+'_anf']        = 2
    IRecord['Zeit'+'_end']        = 7
    IRecord['Latitude'+'_anf']    = 8
    IRecord['Latitude'+'_end']    = 14
    IRecord['NorthSouth'+'_anf']  = 15 
    IRecord['NorthSouth'+'_end']  = 15
    IRecord['Longitude'+'_anf']   = 16
    IRecord['Longitude'+'_end']   = 23
    IRecord['EastWest'+'_anf']    = 24
    IRecord['EastWest'+'_end']    = 24
    IRecord['Press Alt.'+'_anf']  = 26
    IRecord['Press Alt.'+'_end']  = 30
    IRecord['GNS Alt.'+'_anf']    = 31
    IRecord['GNS Alt.'+'_end']    = 35

    JRecord     = {}
    kfindstr    = []
    kfindstr_u  = []
    kfindstr_f  = []
    kfindstr_d  = {}

    kfindstr_d['Zeit']      =[]
    kfindstr.append('Zeit')
    kfindstr_u.append ('hh:mm:ss')
    kfindstr_f.append(1)

    JRecord['Zeit'+'_anf']  = 2
    JRecord['Zeit'+'_end']  = 7
    
    FlapSensor = False


    with open( filename ) as fn:
        ln = fn.readline()

        while ln:

            ln = fn.readline()
            if ln:
                if (ln[0]=='I'):
                    #print(f'I-Record',ln)
                    #  I113638FXA3941ENL4246TAS4751GSP5254TRT5559VAT6063OAT6468NET6972ACZ7376AOR7779AOP
                    for sstrg in searchstrg:
                        help   = ln.find(sstrg)
                        if help != -1:
                            pos_anf               = ln.split(sstrg)[0][-4:-2]
                            pos_end               = ln.split(sstrg)[0][-2:]
                            IRecord[sstrg+'_anf'] = pos_anf
                            IRecord[sstrg+'_end'] = pos_end

                            findstr.append(sstrg)
                            findstr_d[sstrg]      = [] 
                            pos                   = searchstrg.index(sstrg)

                            findstr_u.append(unitstrg[pos])
                            findstr_f.append(faktors[pos])

                elif (ln[0]=='B'):

                    mylist.append(ln)

                    #B0909244901702N01101359EA00580006780040080000000001012-000502360000000490005050

                    for variable in findstr_d.keys():
                        a = int(IRecord[variable+'_anf'])-1
                        o = int(IRecord[variable+'_end'])
                        findstr_d[variable].append(ln[a:o])

                elif (ln[0]=='J'):
                    #print(f'J-Record',ln)
                    #  J020810WDI1115WVE
                    for kstrg in ksearchstrg:
                        help   = ln.find(kstrg)
                        if help != -1:
                            pos_anf               = ln.split(kstrg)[0][-4:-2]
                            pos_end               = ln.split(kstrg)[0][-2:]
                            JRecord[kstrg+'_anf'] = pos_anf
                            JRecord[kstrg+'_end'] = pos_end

                            kfindstr.append(kstrg)
                            kfindstr_d[kstrg]     = []
                            pos                   = ksearchstrg.index(kstrg)

                            kfindstr_u.append(kunitstrg[pos])
                            kfindstr_f.append(kfaktor[pos])

                elif (ln[0]=='K'):

                    mylist.append(ln)

                    #K09130834300668

                    for variable in kfindstr_d.keys():
                        a = int(JRecord[variable+'_anf'])-1
                        o = int(JRecord[variable+'_end'])
                        kfindstr_d[variable].append(ln[a:o])
                elif (ln[0]=='H'):
                    if (ln[0:11]=='HFFTYFRTYPE'):
                        hardware = ln[12:-1]
                        try:
                            hardware = hardware.split(',')[1]
                        except:
                            pass
                    elif (ln[0:20]=='HFRFWFIRMWAREVERSION'):
                        software = ln[21:-1]
                elif (ln[0]=='L'):
                    if (ln[0:7]=='LLXVFLP'):
                        FlapSensor = True

    findstrings  = {}
    findstrings['findstr']  = findstr
    findstrings['findstr_u']  = findstr_u
    findstrings['findstr_f']  = findstr_f
    findstrings['findstr_d']  = findstr_d
    
    kfindstrings  = {}
    kfindstrings['kfindstr']  = kfindstr
    kfindstrings['kfindstr_u']  = kfindstr_u
    kfindstrings['kfindstr_f']  = kfindstr_f
    kfindstrings['kfindstr_d']  = kfindstr_d
    
    return findstrings, kfindstrings, hardware, software, FlapSensor

def mpdf (findstrings, kfindstrings):
    
    findstr    = findstrings['findstr']
    findstr_f  = findstrings['findstr_f'] 
    findstr_d  = findstrings['findstr_d'] 
    kfindstr   = kfindstrings['kfindstr'] 
    kfindstr_f = kfindstrings['kfindstr_f'] 
    kfindstr_d = kfindstrings['kfindstr_d']
    
    # Erzeugen eines Pandas DataFrame
    df  = pd.DataFrame.from_dict(findstr_d, orient='columns')
    dfk = pd.DataFrame.from_dict(kfindstr_d, orient='columns')
    
    lst  = ['NorthSouth', 'EastWest']

    # Wandeln der DataFrame Daten-Typen in float
    lst.append('Zeit')
    for var in df.columns:
        if var in lst:
            if var == 'Zeit':
                #Die Spalte Zeit in Datumsformat ändern
                df['Zeit'] = pd.to_datetime(df['Zeit'],format='%H%M%S')
                # df['Zeit'] = df['Zeit'].dt.tz_localize('utc').dt.tz_convert('Europe/Berlin')
                df_time = df['Zeit']
                df['Seconds'] = ((df_time.dt.hour)*60+df_time.dt.minute)*60 + df_time.dt.second
        else:
            #print('VAR =', var,' Type:', type(var))
            df[var]=df[var].astype(float)
            if var in findstr:
                #pos = findstrg.index(var)
                df[var]=df[var]/findstr_f[findstr.index(var)]

    for var in dfk.columns:

        if var == 'Zeit':
                #Die Spalte Zeit in Datumsformat ändern
                dfk['Zeit'] = pd.to_datetime(dfk['Zeit'],format='%H%M%S')
                dfk['Zeit'] = dfk['Zeit'].dt.tz_localize('utc').dt.tz_convert('Europe/Berlin')
                dfk_time = dfk['Zeit']
                dfk['Seconds'] = (dfk_time.dt.hour*60+dfk_time.dt.minute)*60 + dfk_time.dt.second
        else:
            #print('VAR =', var,' Type:', type(var))
            dfk[var]=dfk[var].astype(float)
            if var in kfindstr:
                #pos = findstrg.index(var)
                dfk[var]=dfk[var]/kfindstr_f[kfindstr.index(var)]

    return df, dfk

def CalcTrack (df):
    # Breitengrad
    H1 = (df['Latitude']%1e3)/1000
    H2 = np.floor(df['Latitude']%1e5/1000)
    H2 = (H1+H2)/60;
    H1 = np.floor(df['Latitude']/1e5);
    BDG = H1+H2;
    
    # Methode 1:
    # s = df['NorthSouth']
    # idx = s[s=='N'].index
    
    # for i in idx:
    #    BDG[i]=-BDG[i]
    
    # oder:
    # tmp = map(lambda x: -x, BDG[idx])
    # BDG = list(tmp)
    
    # oder sehr elegant  
    # Another method would be to create a boolean mask, drop the NaN rows, call loc on the index and assign the negative values:
    # https://stackoverflow.com/questions/29299597/python-pandas-replace-values-by-their-opposite-sign
    BDG.loc[BDG[df['NorthSouth']=='S'].dropna().index]=-BDG
    
    H1 = (df['Longitude'][:]%1e3)/1000
    H2 = np.floor(df['Longitude'][:]%1e5/1000)
    H2 = (H1+H2)/60;
    H1 = np.floor(df['Longitude']/1e5);
    LDG = H1+H2;
    
    LDG.loc[LDG[df['EastWest']=='W'].dropna().index]=-LDG
    
    # s = df['EastWest']
    # idx = s[s=='W'].index
    # 
    # for i in idx:
    #     LDG[i]=-LDG[i]
    # 
    return BDG, LDG

def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = np.rad2deg(brng)

    return brng

def bearing(origin, destination):
    # Haversine formula example in Python for Calculating Bearing
    # Inspired from http://www.movable-type.co.uk/scripts/latlong.html

	lat1, lon1 = origin
	lat2, lon2 = destination
	# radius = 6371 # km

	rlat1 = math.radians(lat1)
	rlat2 = math.radians(lat2)
	rlon1 = math.radians(lon1)
	rlon2 = math.radians(lon2)
	dlon = math.radians(lon2-lon1)

	b = math.atan2(math.sin(dlon)*math.cos(rlat2),math.cos(rlat1)*math.sin(rlat2)-math.sin(rlat1)*math.cos(rlat2)*math.cos(dlon)) # bearing calc
	bd = math.degrees(b)
	br,bn = divmod(bd+360,360) # the bearing remainder and final bearing
	
	return bn

def FindShortSequence( Kreisengamma, Ngrenz):
    # df['vim1_L']   = h999['LDG'].diff(periods=1)    
    
    h1 = np.asarray(np.where(np.diff(Kreisengamma)!=0)[0])  # Position, an der ein Wechsel stattfindet, zu Matlab natürlich verschoben
    # h1 = h1[0]  # wandelt von tuple in array 
    # h1b = np.insert(h1,0,0)
    h2 = np.diff(np.insert(h1,0,0))

    h1 = h1 + 1 # addiert jetzt noch eine 1 zu jedem Element hinzu

    h3 = (h2<Ngrenz).astype(int)   # wird nicht benötigt / verwendet

    h4 = np.where(h2<Ngrenz)[0]

    h5 = h1[h4]
    h6 = h2[h4]
    h7 = h5-h6

    return h5, h7

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    h1 = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(h1))
    a = 2 * np.arctan2(np.sqrt(h1), np.sqrt(1-h1))

    # Calculate the angle in degrees
    angle = np.rad2deg(a)
    km = 6378.137 * c

    # Calculate alpha
       
    alpha = np.arctan2(dlon,dlat)  # oder acttan2
        
    alpha = np.rad2deg(alpha)
    
    # dummy = FindShortSequence(Kreisengamma, Ngrenz)

    return angle, km, alpha

def SegmentSearch(SegStart,SegDura, Height, Sekunden, BDGg, LDGg):
    
    h4 = SegDura[:-1]
    h5 = SegStart[:-1]
    Height = np.asarray(Height)
    Sekunden = np.asarray(Sekunden)
    
    Endhoehe          = np.asarray(Height[h5+h4])
    Starthoehe        = np.asarray(Height[h5])
    Endzeit           = np.asarray(Sekunden[h5+h4])
    Startzeit         = np.asarray(Sekunden[h5])
    
    dH                = Height[h5+h4]-Height[h5]
    dT                = Sekunden[h5+h4]-Sekunden[h5]
    aveSteigen        = np.divide(dH,dT)
    Nr                = np.asarray(range(0,len(Startzeit)))
    
    EndPos            = np.asarray([BDGg[h5+h4], LDGg[h5+h4]])
    StartPos          = np.asarray([BDGg[h5], LDGg[h5]])
    # data              = np.array([[Endhoehe], [Starthoehe], [Endzeit], [Startzeit], [dH], [dT], [aveSteigen], [Nr], [EndPos[0]], [EndPos[1]], [StartPos[0]], [StartPos[1]]])
    data              = np.array([h5, h4+h5, Endhoehe, Starthoehe, Endzeit, Startzeit, dH, dT, aveSteigen, Nr, EndPos[0], EndPos[1], StartPos[0], StartPos[1]])
    columns           = ['idx_a','idx_e','Endhoehe', 'Starthoehe', 'Endzeit', 'Startzeit', 'dH', 'dT', 'aveSteigen', 'Nr', 'EndPos_lat', 'EndPos_lon', 'StartPos_lat', 'StartPos_lon']
    df_Seg             = pd.DataFrame(np.transpose(data),columns=columns)
    #return Endhoehe, Starthoehe, Endzeit, Startzeit, dH, dT, aveSteigen, Nr, EndPos, StartPos
    pass
    return df_Seg

def EstimateTrajectoryAngle (BDG,LDG, Height, Sekunden, df2,AOR=[],l_suchkr=1, l_wechsel=0, TRT=[]):
    
    df = pd.DataFrame()
    df['vim1_L']   = LDG.diff(periods=1).fillna(0)    # vim1(:,1)
    df['vim1_B']   = BDG.diff(periods=1).fillna(0)   # vim1(:,2)
    df['vi_L']     = np.roll(df['vim1_L'],1)         # vi(:,1)
    df['vi_B']     = np.roll(df['vim1_B'],1)         # vi(:,2)
    
    g1          = (df['vi_L'] * df['vim1_B'] - df['vi_B'] * df['vim1_L'])
    g2          = (df['vi_L'] * df['vim1_L'] + df['vi_B'] * df['vim1_B'])
    g2.replace(-0,0)
    
    gamma2      = np.arctan2(g1,g2) * 180/math.pi
    gamma2      = gamma2.replace (180,0) 
    # gamma2      = gamma2.mask(gamma2 > 170, 0)

    
    l_debug = False
    if l_debug:
        dfdbg=pd.DataFrame({'g1': g1, 'g2': g2, 'gamma2': gamma2})
        dfdbg.to_excel('debug.xlsx')
        fig, axs = plt.subplots(3)   
        axs[0].plot(g1)
        axs[1].plot(g2)
        axs[2].plot(gamma2)
        plt.show()

    gamma2sg     = signal.savgol_filter(gamma2, window_length=41, polyorder=5, mode="nearest") 
    DeltaNg      = 5;   # 7 ist wie 10 ist zu groß, 5 ist super (28.02.2024)
    Grenzgamma   = 5                              # Winkel ist unabhängig von Zeitschrittweite
    Kreisengamma = abs(gamma2sg)>Grenzgamma       # Kurbeln liegt vor, wenn Kreisengamma==1 
    Kreisengamma = Kreisengamma.astype(int)
    
    # Jetzt sollen "vereinzelte" Nullen zu Einsen werden, also dem Kreisen zugeordnet werden
    
    DeltaT = 1
    Ngrenz = (25-DeltaNg) / DeltaT   # war 25
    
    h5, h7 = FindShortSequence(Kreisengamma, Ngrenz)
    
    for i in range(0,len(h7)):   
        Kreisengamma[h7[i]:h5[i]]=1
    
    Kreisengamma = np.append(Kreisengamma,0)
    
    # Nachgelagert
    # sollen "vereinzelte" Einsen gelöscht werden, dh Steigen wird zu Gleiten
    
    Ngrenz = (20-DeltaNg) / DeltaT  # war 20
    
    [h5,h7] = FindShortSequence(Kreisengamma, Ngrenz)
        
    for i in range (0,len(h7)):  
        Kreisengamma[h7[i]:h5[i]]=0
        
        
    # Suchkreise bestimmen
    Suchkreis=np.zeros(len(Kreisengamma))
    
    [h5,h7] = FindShortSequence(Kreisengamma, 60)   # war 45 geändert wie in Matlab 02.03.24
    
    for i in range (0,len(h7)-1):
        Suchkreis[h7[i]:h5[i]]=1
    
    if l_suchkr:
        # Wenn Suchkreise elemniert werden sollen 
        Kreisengamma = np.logical_and(Kreisengamma, np.logical_not(Suchkreis)).astype(int)
    
    
    Kreisengamma[-2]    = 1                                                        # Hier wird künstlich eine letzter Wechsel hinzugefügt (11.09.2021)
    h1                  = np.asarray(np.where(np.diff(Kreisengamma)!=0)[0])        # Position, an der ein Wechsel stattfindet, zu Matlab natürlich verschoben
    if h1[0]!=0:
        h1                  = np.insert(h1,0,0)
    #h1                  = h1 + 1                                                   # addiert jetzt noch eine 1 zu jedem Element hinzu    
    h2                  = np.asarray(np.diff(h1))                                  # Wie oft der gleiche Status vorliegt
    # h2                  = np.append(h2,0)
    h3                  = (Kreisengamma[h1]==1).astype(int)                        # An diesen Stellen von h1 beginnt Steigen
    h3                  = np.delete(h3, 0)
    h4                  = h2[h3==1]                                               # So lange dauert das Steigen
    h5                  = h1[0:-1][h3==1]                                                # An diesen Stellen von Kreisgamma beginnt Steigen
    
    h5m1                = h5[0:-1]
    h4m1                = h4[0:-1]
    
    h8                  = np.gradient(df2['alpha'])
    
    if np.any(h4==0):
        print('Stop')
    
    AnzLK = []
    AnzRK = []
    
    help1 = len(h5m1)
    x2    = np.zeros((help1,5))
    
    for i in range(0,len(h5m1)):  # war len(h5)-1 geändert 08.04.2024
        AnzLK.append(np.round(np.sum(h8[h5[i]:h5[i]+h4[i]]>80)/2))
        AnzRK.append(np.round(np.sum(h8[h5[i]:h5[i]+h4[i]]<-80)/2))
        
        # Neu 23.03.2024
        if AnzLK[i]*AnzRK[i]!=0:
            x1 = Kurvenwechsel (AOR,BDG,LDG,h4,h5,h8,i,0,TRT)
            if len(x1)>0:
                x2[i][0:len(x1)]=x1[0:5]
    
    Wechsel = x2
    Wechsel.astype(int)
    
    if l_wechsel:
        help1    = help1 + np.count_nonzero(Wechsel)
        i = 0
        while(i<=help1-1):
            if Wechsel[i][0]!=0:
                h5m1 = np.insert(h5m1,i+1,Wechsel[i][0])
                temp = Wechsel[i][0]-h5m1[i]
                temp2 = h4m1[i]-temp
                h4m1 = np.insert(h4m1,i,temp)  # 09.04.2024 16:12 Uhr
                np.put(h4m1,i+1,temp2)
                ins  = np.append(Wechsel[i][1:],0)
                Wechsel = np.insert(Wechsel,i+1,ins,axis=0)
            i += 1 
    
    AnzLK2       = np.zeros(help1) 
    AnzRK2       = np.zeros(help1)
    for i in range(0,help1-1):                     
        AnzLK2[i]      = np.round(np.sum(h8[h5m1[i]:h5m1[i]+h4m1[i]]>80)/2)
        AnzRK2[i]      = np.round(np.sum(h8[h5m1[i]:h5m1[i]+h4m1[i]]<-80)/2)
    
    AnzRK  = AnzRK2
    AnzLK  = AnzLK2
    AnzK=AnzLK+AnzRK
  
    # Auswertung fürs Kreisen
    lat1     = BDG[0:-1].reset_index(drop=True)
    lon1     = LDG[0:-1].reset_index(drop=True)

    df_Kr=SegmentSearch (h5m1, h4m1, Height, Sekunden, lat1, lon1)

    g3 = np.logical_not(h3)
    g3 = g3.astype(int)
    g4 = h2[g3==1]
    g5 = h1[0:-1][g3==1]

    # Auswertung fürs Gleiten
    df_Gl=SegmentSearch (g5, g4, Height, Sekunden, lat1, lon1)
    
    return Kreisengamma,df_Kr,df_Gl,np.array(AnzRK), np.array(AnzLK)

def Kurvenwechsel (AOR,BDG,LDG,h4,h5,h8,i,l_wmplot=0,TRT=[]):
    
    if (AOR.empty):
        AOR = np.zeros(len(BDG))
        
    if (TRT.empty):
        TRT = np.zeros(len(BDG))
    
    
    signal = h8[h5[i]:h5[i]+h4[i]]
    signal = signal/max(signal)
    pos = np.asarray(np.where(np.absolute(signal)<0.025)[0])
    #pos = np.insert(pos,0,0)
    
    if pos.size>1:
        # Berechnen Sie den Abstand zwischen aufeinanderfolgenden Indizes
        differences = np.diff(pos)
        differences = np.insert(differences,0,0)

        # Definieren Sie einen Schwellenwert, der groß genug ist, um Gruppen zu trennen
        # Dieser Schwellenwert hängt von der spezifischen Anwendung und den Daten ab
        threshold = 7  #Beispielwert, der angepasst werden muss  % War bisher immer auf 10!!!! (28.01.2024)
        pos = np.delete(pos,np.where(differences<threshold))

        x1 = h5[i]+pos
        abstand  = 25   # war 15
        u  = x1-abstand
        o  = x1+abstand

        h11 = np.zeros(len(x1))
        h12 = np.zeros(len(x1))
        h13 = np.zeros(len(x1))
        h14 = np.zeros(len(x1))
        h15 = np.zeros(len(x1))
        h16 = np.zeros(len(x1))

        # Hier mal TRT mit einführen und schauen, ob das nicht viel einfacher ist as mit h8!

        diffTRT = np.diff(TRT) # [0; diff(TRT)];
        diffTRT = np.insert(diffTRT,0,0)
        diffTRT[diffTRT > 50] = -10 
        diffTRT[diffTRT < -50] = 10

        for j in range(0,len(x1-1)):
            h11[j] = np.sign(np.mean(h8[u[j]:x1[j]]))
            h12[j] = np.sign(np.mean(h8[x1[j]:o[j]]))
            
            h13[j] = np.sign(np.mean(AOR[u[j]:x1[j]]))
            h14[j] = np.sign(np.mean(AOR[x1[j]:o[j]]))
            
            h15[j] = np.sign(np.mean(diffTRT[u[j]:x1[j]]))
            h16[j] = np.sign(np.mean(diffTRT[x1[j]:o[j]]))
            
            
        x1   = np.delete(x1,np.where(h15==h16))
        pos  = np.delete(pos,np.where(h15==h16))

        # Wenn der Kurvenwechsel sehr nahe am Anfang liegt, wird er verworfen
        if pos.size!=0:
            if (pos[0]<15):
                pos = np.delete(pos,0)
                x1  = np.delete(x1,0)
            
        # Wenn der Kurvenwechsel sehr nahe am Ende liegt, wird er verworfen
        if pos.size!=0:
            if (h4[i]-pos[-1])<15:
                pos = np.delete(pos,-1)
                x1  = np.delete(x1,-1)

        a    = h5[i]
        e    = a + h4[i]

        u    = x1-5
        o    = x1+5

        condition = np.ones(len(x1),dtype=bool)
        # print('size of x1', x1.size)

        if np.any(x1) and x1.size > 0:
            for j in range(0,len(x1)):
                if ((e-x1[j])<20) and ((x1[j]-a)<20):
                    condition[j]=True
                else:
                    condition[j]=False
            x1 = np.delete(x1,np.where(condition==True))
    else:
        x1=[]
        
    return x1            


def AsymPlot (AnzRK,AnzLK,df,df_Kr,filename,hardware, software, FlapSensor, l_plt=1):

    dH_TEK   = np.array([])
    dH_Netto = np.array([])

    for j in range(0,len(df_Kr['Nr'])):
        a = np.int32(df_Kr['Startzeit'][j])
        a = df.loc[(df['Seconds']>=a)].index[0]
        b = np.int32(df_Kr['Endzeit'][j])
        b = df.loc[(df['Seconds']<=b)].index[-1]
        dH_Netto = np.append(dH_Netto, np.trapz(df['NET'][a:b], x = df['Seconds'][a:b]))
        dH_TEK   = np.append(dH_TEK, np.trapz(df['VAT'][a:b], x = df['Seconds'][a:b]))

    dT = df_Kr['dT'].to_numpy()
    deltaNT = np.divide(np.subtract(dH_TEK,dH_Netto),dT)
    
    df_Kr['dHdT(TEK)']   = dH_TEK/dT
    df_Kr['dHfT(Netto)'] = dH_Netto/dT
    df_Kr['deltaNT']     = deltaNT


    AnzLK = np.delete(AnzLK, -1)
    AnzRK = np.delete(AnzRK, -1)
    
    df_Kr['AnzRK']       = AnzRK
    df_Kr['AnzLK']       = AnzLK
    df_Kr['Type']        = np.nan
    df_Kr=df_Kr.astype({'Type': str})
    df_Kr['#Thermals']   = np.maximum(AnzRK,AnzLK)
    j=0
    while j < AnzRK.size:
        if (AnzRK[j]>0 and AnzLK[j] == 0):
            df_Kr.loc[j,'Type'] = 'R'
            j=j+1
        elif (AnzLK[j]>0 and AnzRK[j] == 0):
            df_Kr.loc[j,'Type'] = 'L'
            j=j+1
        else:
            if AnzLK[j]> AnzRK[j]:
                Typ = 'L'
            else:
                Typ = 'R'
            df_Kr.loc[j,'Type'] = Typ
            df_Kr.loc[j,'#Thermals']=df_Kr['AnzRK'][j]+df_Kr['AnzLK'][j]
            j=j+1
            # df_Kr['Type'][j] = 'R'
            # df_Kr['#Thermals'][j]=df_Kr['AnzRK'][j]
            # j=j+1
            # df_Kr['Type'][j] = 'L'
            # df_Kr['#Thermals'][j]=df_Kr['AnzLK'][j]
            # j=j+1
        

    fig1, ax1 = plt.subplots(figsize=(16, 9))

    if FlapSensor:
        FlS = ', w. FS'
    else:
        FlS = ''
        

    ax1.set_title("Asymmetry Plot: " + filename+" (" +hardware+ ", "+ software + FlS + ")", fontsize='large')
    ax1.grid(True)

    ax1.scatter (AnzRK[AnzRK>0],deltaNT[AnzRK>0],color='g', edgecolors='black', label='right circles')
    ax1.scatter (AnzLK[AnzLK>0],deltaNT[AnzLK>0],color='r', edgecolors='black', label='left circles')
    ax1.set_xlabel('No of Circles in Thermals (right and left) / -')
    ax1.set_ylabel('(ave. VAT - ave. NET) / m/s')

    ax1.set_xticks(np.arange(0, np.max([AnzLK,AnzRK])+2, 2))

    plt.xlim([0,np.max([AnzLK,AnzRK])+1])

    aveRK = deltaNT[AnzRK>0].mean()
    aveLK = deltaNT[AnzLK>0].mean()
    ax1.hlines(aveRK, 0, np.max([AnzLK,AnzRK])+1,  linestyle='dashed' ,color='g' , label = 'average right')
    ax1.hlines(aveLK, 0, np.max([AnzLK,AnzRK])+1,  linestyle='dashed' ,color='r' , label = 'average left')
 
    # weighted Average
    np.aveRKw = (df_Kr['deltaNT'].loc[df_Kr['AnzRK']>0].mul(df_Kr['dT'].loc[df_Kr['AnzRK']>0]))
    np.aveRKw = np.aveRKw.sum()/df_Kr['dT'].loc[df_Kr['AnzRK']>0].sum()
    np.aveLKw = (df_Kr['deltaNT'].loc[df_Kr['AnzLK']>0].mul(df_Kr['dT'].loc[df_Kr['AnzLK']>0]))
    np.aveLKw = np.aveLKw.sum()/df_Kr['dT'].loc[df_Kr['AnzLK']>0].sum()

    ax1.hlines(np.aveRKw, 0, np.max([AnzLK,AnzRK])+1,  linestyle='dotted' ,color='g' , label = 'weighted av. right')
    ax1.hlines(np.aveLKw, 0, np.max([AnzLK,AnzRK])+1,  linestyle='dotted' ,color='r' , label = 'weighted av. left')  

    s = 'ave. Climb Rate (right): %.2f' % aveRK
    plt.text(np.max([AnzLK,AnzRK])-5, aveRK+0.02 ,s, color='g')
    s = 'ave. Climb Rate (left): %.2f' % aveLK
    plt.text(np.max([AnzLK,AnzRK])-5, aveLK+0.02 ,s, color='r')

  
    s = 'ave. Climb Rate (weighted): %.2f' % np.aveRKw
    plt.text(np.min([AnzLK,AnzRK])+1, np.aveRKw-0.04 ,s, color='g')
    s = 'ave. Climb Rate (weighted): %.2f' % np.aveLKw
    plt.text(np.min([AnzLK,AnzRK])+1, np.aveLKw-0.04 ,s, color='r')


    ax1.legend()

    if l_plt==1:
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
       
        plt.show()
    else:
        savename = f"{(filename.split('.')[0])}{'.png'}"
        plt.savefig(savename)
        plt.close()
        savename = f"{(filename.split('.')[0])}{'_py.xlsx'}"
        df_Kr.to_excel(savename)
    
    return aveRK, aveLK, np.aveRKw, np.aveLKw,  df_Kr
    pass


