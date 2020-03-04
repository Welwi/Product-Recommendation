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

new_type_test = {
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
     'ind_actividad_cliente': 'str',
     'renta': 'float32',
     'segmento': 'str'}

cols = list(newtype.keys())
test_cols = list(new_type_test.keys())
chunksize = 1e6



def DfLowMemory(filename):
    
    samples = []
    for df in pd.read_csv(filename, usecols=cols, dtype=newtype, chunksize=chunksize):
        samples.append(df)
    
    data = pd.concat(samples)
    
    return data

def DfLowMemoryTest(test_filename):
    
    test_data = pd.read_csv(test_filename, usecols=test_cols, dtype=new_type_test)

    return test_data

    
def CleanData(df_data):
    
    # ind_empleado - 27734. Employee index: A active, B ex employed, F filial, N not employee, P pasive. 
    #I am noticing that there is a value that is not in the description so I believe it is a typo. 
    #I tried looking more into the data but in the end I decided to drop since it is just one person.
    df = df_data[df_data['ind_empleado'] != 'S']
    
    # pais_residencia - 27734. One aspect that I found interesting is that customers with pais_residenciamissing had all other features missing. 
    #For now I am going to drop all of them
    df.dropna(subset=['pais_residencia'], inplace=True)
    
    # Now there are 70 missing values for gender. The value distribution doesn't seem very different. I am going to replace them with the mode.
    df['sexo'].fillna(df['sexo'].mode()[0], inplace=True)
    
    # ult_fec_cli_1t has a lot of missing values. This represents the last day the the customer was the primary costumer.
       # Looking at the indrel_1mes column, it shows that most of the customers at the beginning of the month were the primary costumers. So 
    # I am assuming that the customers with missing last date as the primary costumers are still the primary costumers. So I am going to impute 
    # this with 'primary'
    df.loc[df['ult_fec_cli_1t'].isnull(), 'ult_fec_cli_1t'] = 'PRIMARY'
    
    # Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner).
    # some values that were supposed to be 1 are 1.0 and for the other categories as well. So I am going to use 1,2,3,4 like in the description
    # This has been suggested by @StephenSmith
    map_dict = {'1.0' : '1',
                '1' : '1',
                '2' : '2',
                '2.0' : '2',
                '3' : '3',
                '3.0' : '3', 
                '4' : '4',
                '4.0' : '4'}

    df.indrel_1mes.fillna('P', inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
    df.indrel_1mes = df.indrel_1mes.astype('category')

    # Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
    # There are some described as N that doesn't fit into the column description.
    # Taking a closer look at these 4 rows and comparing them with the ind_actividad: df[df['tiprel_1mes'] == 'N'].iloc[:,8:24]
    # The ones that had ind_actividad = 1 were imputed as active and the ones that had 0 were imputed as I
    # For the nan values, there is another column named 'ind_actividad_cliente' that is 1 for active costumers and 0 for inactive. I am 
    # going to use this to impute the nan values for tiprel_1mes
    df.loc[df['tiprel_1mes'].isnull(), 'tiprel_1mes'] = df['ind_actividad_cliente']

    map_tip = {1 : 'A',
               0 : 'I'}        

    df.tiprel_1mes = df.tiprel_1mes.apply(lambda x: map_tip.get(x,x))
    df.loc[6603017, 'tiprel_1mes'] = 'A'
    df.loc[10123924, 'tiprel_1mes'] = 'A'
    df.loc[10124648, 'tiprel_1mes'] = 'I'
    df.loc[11247349, 'tiprel_1mes'] = 'I'
    df.tiprel_1mes = df.tiprel_1mes.astype('category')
    
    # conyuemp Spouse index. 1 if the customer is spouse of an employee
    # I am assuming that most customers are not spouses and I am going to impute these with the mode
    # Now there are 70 missing values for gender. The value distribution doesn't seem very different. I am going to replace them with the mode.
    df['conyuemp'].fillna(df['conyuemp'].mode()[0], inplace=True)
    df.conyuemp = df.conyuemp.astype('category')
    
    
    # canal_entrada. Channel used by the customer to join
    # I am going to fill the missing values with the mode
    df['canal_entrada'].fillna(df['canal_entrada'].mode()[0], inplace=True)
    
    # tipodom. Addres type. 1, primary address. It is only 1 value missing so I am just going to replace it with the mode
    df['tipodom'].fillna(df['tipodom'].mode()[0], inplace=True)
    
    # cod_prov. Province code (customer's address)
    # nomprov. Province name. 
    # These two have the same number of missing values. I am going to replace these with unknow
    # THESE IS SUGGESTED BY ALAN PRYOR
    for c in ['cod_prov', 'nomprov']:
        df.loc[df[c].isnull(), c ] = 'UNKOWN'
    
    # Renta. Gross income of the household. I am going to replace the missing values with the mean salary per providence. 
    salaries = dict(df.groupby('nomprov')['renta'].mean().round(0))
    df.loc[df['renta'].isnull(), 'renta'] = df['nomprov']
    df.renta = df.renta.apply(lambda x: salaries.get(x,x))
    
    # Segmento. segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
    for c in ['segmento']:
        df.loc[df[c].isnull(), c ] = 'UNKOWN'
        
    df.loc[df['antiguedad'] == '-999999', 'antiguedad'] = 'UNKNOWN'
    df['ind_nomina_ult1'].fillna(df['ind_nomina_ult1'].mode()[0], inplace=True)
    df['ind_nom_pens_ult1'].fillna(df['ind_nom_pens_ult1'].mode()[0], inplace=True)
            
    return df


    
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