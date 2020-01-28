import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np

#models=['eucli','cos','manhat']
xyz=[]

model="cos"

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
df.reset_index(level=0, inplace=True)

print("L1 complete")

import plotly.graph_objs as go
import plotly 
#import chart_studio.plotly as py

trace0 = go.Scatter(
    x = threshold,
    y = df['miss_rates'],
    name = 'miss_rates',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4)
)
trace1 = go.Scatter(
    x = threshold,
    y = df['fmr'],
    name = 'false_match_rate',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,
        dash = 'dash')
)
trace2 = go.Scatter(
    x = threshold,
    y = df['intrusions'],
    name = 'intrusion_rate',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4,
        dash = 'dot') # dash options include 'dash', 'dot', and 'dashdot'
)


data = [trace0, trace1, trace2]

if (model=='eucli'):
    layout = dict(title = 'euclidean_distance',
                  xaxis = dict(title = 'units same as embeddings'),
                  yaxis = dict(title = 'rates'),
                  )
elif(model=='cos'):
    layout = dict(title = 'cosine_distance',
                  xaxis = dict(title = 'radians'),
                  yaxis = dict(title = 'rates'),
                  )
elif(model=='manhat'):
    layout = dict(title = 'manhatten_distance',
                  xaxis = dict(title = 'units same as embeddings'),
                  yaxis = dict(title = 'rates'),
                  )
    
fig = dict(data=data, layout=layout)

plotly.offline.plot(fig, filename=model+'.html')

















