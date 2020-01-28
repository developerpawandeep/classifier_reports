import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np


#models=['eucli','cos','manhat']
xyz=[]
models=["manhat"]

for model in models:
    #model="manhat"
    
    if(model=="eucli"):
        threshold=[0.05,0.11,0.16,0.21,0.27,0.33,0.38,0.44,0.49,0.55]
    
    elif(model=="cos"):
        threshold=[0.00,0.01,0.03,0.04,0.05,0.07,0.09,0.13,0.14,0.15]
    
    elif(model=="manhat"):
        threshold=[10,17,24,31,38,45,52,59,66,71]

    if(model=="eucli"):
        data=pd.read_csv("/home/dk-tanmay/Desktop/euclidean.csv")
    
    elif(model=="cos"):
        data=pd.read_csv("/home/dk-tanmay/Desktop/cosine.csv")
    
    elif(model=="manhat"):
        data=pd.read_csv("/home/dk-tanmay/Desktop/manhatten.csv")
    
    
    total_miss_rate=0
    total_fmr=0
    total_intrusion=0
    
    for thre in threshold:
        for index,value in data.iterrows():
            thre_name = round(float(thre),3)
            locals()['total_miss_rate_{}'.format(str(thre_name))] = 0
            locals()['total_fmr_{}'.format(str(thre_name))] = 0
            locals()['total_intrusion_{}'.format(str(thre_name))] = 0
    
    
    for thre in threshold:
        count=0
        for index,value in data.iterrows():
            thre_name = float(thre)
            xyz.append(data['Threshold'][index])
            if(data['Threshold'][index]==thre):
                print(model,thre)
                locals()['total_miss_rate_{}'.format(str(thre_name))] += data['Miss_R'][index]
                locals()['total_fmr_{}'.format(str(thre_name))] += data['FMR'][index]
                locals()['total_intrusion_{}'.format(str(thre_name))] += data['Intrusion'][index]
                count+=1
    
    print(count)
    for thre in threshold:
        thre_name = float(thre)
        locals()['total_miss_rate_{}'.format(str(thre_name))] = locals()['total_miss_rate_{}'.format(str(thre_name))]/count
        locals()['total_fmr_{}'.format(str(thre_name))] = locals()['total_fmr_{}'.format(str(thre_name))]/count
        locals()['total_intrusion_{}'.format(str(thre_name))] = locals()['total_intrusion_{}'.format(str(thre_name))]/count
    
    
    miss_rates=[]
    fmr_rates=[]
    intrusion_rates=[]
    
    for thre in threshold:
        thre_name = float(thre)
        miss_rates.append(locals()['total_miss_rate_{}'.format(str(thre_name))]) 
        fmr_rates.append(locals()['total_fmr_{}'.format(str(thre_name))]) 
        intrusion_rates.append(locals()['total_intrusion_{}'.format(str(thre_name))]) 
    
    df= pd.DataFrame()
    
    df['miss_rates']=miss_rates
    df['fmr']=fmr_rates
    df['intrusions']=intrusion_rates
    
    if(model=="eucli"):
        plt.rcParams['figure.figsize']=(15,10)
        g = sns.lineplot(data=df,marker="o")
        g.set(xticks=np.arange(10),xticklabels=threshold)    
        g.set(xlabel='Model_threshold', ylabel='Rates', title= 'euclidean distance')    
        g.figure.savefig("euclidean.pdf")
    
    elif(model=="cos"):
        plt.rcParams['figure.figsize']=(15,10)
        h = sns.lineplot(data=df,marker="o")
        h.set(xticks=np.arange(10),xticklabels=threshold)    
        h.set(xlabel='Model_threshold', ylabel='Rates', title= 'cosine distance')
        h.figure.savefig("cosine.pdf")
    
    elif(model=="manhat"):
        plt.rcParams['figure.figsize']=(15,10)
        j = sns.lineplot(data=df,marker="o")
        j.set(xticks=np.arange(10),xticklabels=threshold)
        j.set(xlabel='Model_threshold', ylabel='Rates', title= 'manhatten distance')
        j.figure.savefig("manhatten.pdf")
    


