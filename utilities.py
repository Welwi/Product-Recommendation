import pandas as pd
import numpy as np

import gc; gc.enable()


newtype = {
     'fecha_dato': 'str',
     'ncodpers': 'int32',
     'ind_empleado': 'str',
     'pais_residencia': 'str',
     'sexo': 'str',
     'age': 'str',
     'fecha_alta': 'str',
     'ind_nuevo': 'float32',
     'antiguedad': 'str',
     'indrel': 'float32',
     'ult_fec_cli_1t': 'str',
     'indrel_1mes': 'str',
     'tiprel_1mes': 'str',
     'indresi': 'str',
     'indext': 'str',
     'conyuemp': 'str',
     'canal_entrada': 'str',
     'indfall': 'str',
     'tipodom': 'float32',
     'cod_prov': 'float32',
     'nomprov': 'str',
     'ind_actividad_cliente': 'float32',
     'renta': 'float32',
     'segmento': 'str',
     'ind_ahor_fin_ult1': 'uint8',
     'ind_aval_fin_ult1': 'uint8',
     'ind_cco_fin_ult1': 'uint8',
     'ind_cder_fin_ult1': 'uint8',
     'ind_cno_fin_ult1': 'uint8',
     'ind_ctju_fin_ult1': 'uint8',
     'ind_ctma_fin_ult1': 'uint8',
     'ind_ctop_fin_ult1': 'uint8',
     'ind_ctpp_fin_ult1': 'uint8',
     'ind_deco_fin_ult1': 'uint8',
     'ind_deme_fin_ult1': 'uint8',
     'ind_dela_fin_ult1': 'uint8',
     'ind_ecue_fin_ult1': 'uint8',
     'ind_fond_fin_ult1': 'uint8',
     'ind_hip_fin_ult1': 'uint8',
     'ind_plan_fin_ult1': 'uint8',
     'ind_pres_fin_ult1': 'uint8',
     'ind_reca_fin_ult1': 'uint8',
     'ind_tjcr_fin_ult1': 'uint8',
     'ind_valo_fin_ult1': 'uint8',
     'ind_viv_fin_ult1': 'uint8',
     'ind_nomina_ult1': 'float32',
     'ind_nom_pens_ult1': 'float32',
     'ind_recibo_ult1': 'uint8'
}

cols = list(newtype.keys())
chunksize = 1e6

def SampleLowMemory(filename):
    
    i = 0
    samples = []
    for df in pd.read_csv(filename, usecols=cols, dtype=newtype, chunksize=chunksize):
        
        # for each chunk get daily samples
        frames = []
        for dato in df['fecha_dato'].unique().tolist():
            # get 10% sample for each day
            df_dato = df.loc[df['fecha_dato'] == dato].copy().sample(frac=0.10)
            
            # add to frames list
            frames.append(df_dato)
            gc.collect()
            
        # add to samples list
        sample_dato = pd.concat(frames)
        samples.append(sample_dato)
        gc.collect()
              
        # increase the count
        i += 1
        
    trial = pd.concat(samples)
    
    return trial


def DfLowMemory(filename):
    
    samples = []
    for df in pd.read_csv(filename, usecols=cols, dtype=newtype, chunksize=chunksize):
        samples.append(df)
    
    data = pd.concat(samples)
    
    return data


def DataCleaning(df):
    
    