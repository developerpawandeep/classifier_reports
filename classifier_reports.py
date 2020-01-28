###############################################################################
import pandas as pd
import os
import cv2
import numpy as np
import math


from scipy.spatial import distance
from scipy.stats import norm
from statistics import mean

from keras.models import load_model
model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)

###############################################################################
#euclidean
def euclidean(a,b):
    #dst = distance.cdist(a, b , 'seuclidean')
    above= pow((np.std(a-b)),2)
    below= (pow((np.std(a)),2))+(pow((np.std(b)),2))
    dst= math.sqrt(above/below)
    return dst
#cosine
def cosine(a,b):
    dst=distance.cdist(a,b ,'cosine')
    #dst = sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))
    return dst
    
    # Calculate numerator of cosine similarity
    #vector1=a
    #vector2=b
    #dot = [vector1[i] * vector2[i] for i in range(vector1)]
      
    # Normalize the first vector
    #sum_vector1 = 0.0
    #sum_vector1 += sum_vector1 + (vector1[i]*vector1[i] for i in range(vector1))
    #norm_vector1 = math.sqrt(sum_vector1)
      
    # Normalize the second vector
    #sum_vector2 = 0.0
    #sum_vector2 += sum_vector2 + (vector2[i]*vector2[i] for i in range(vector2))
    #norm_vector2 = math.sqrt(sum_vector2)
      
    #return (dot/(norm_vector1*norm_vector2))
    
#manhatten
def manhatten(a,b):
    dst=distance.cdist(a,b ,'cityblock')
    #dst = abs(x_value - x_goal) + abs(y_value - y_goal)
    return dst
###############################################################################
#Train data
    
core_dir="/home/dk-tanmay/Desktop/train_classifier_data/"
data=[]
all_train_labels=[]

for i in os.listdir(core_dir):
    folder_dir=core_dir+i+"/"
    for j in os.listdir(folder_dir):
        image_dir=folder_dir+j
        img = image_dir
        data.append(img)
        all_train_labels.append(i)        

unique_name_train = []
for x in all_train_labels:
    if x not in unique_name_train:
        unique_name_train.append(x)

data_train = pd.DataFrame()

data_train['person']=all_train_labels
data_train['images']=data
###############################################################################
#Test data

core_dir="/home/dk-tanmay/Desktop/test_classifier_data/"
data=[]
all_test_labels=[]

for i in os.listdir(core_dir):
    folder_dir=core_dir+i+"/"
    for j in os.listdir(folder_dir):
        image_dir=folder_dir+j
        img = image_dir
        data.append(img)
        all_test_labels.append(i)

data_test = pd.DataFrame()
data_test['person']=all_test_labels
data_test['images']=data
###############################################################################
len_test=len(data_test)
len_train=len(data_train)
###############################################################################
#Merge embed and separate

data_full = data_train.append(data_test)


#embedder = FaceNet()
embeds=[]
images=data_full['images']
for img in images:
    image = cv2.imread(img)
    image = cv2.resize(image, (160, 160))
    #pixels = asarray(image)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    embeds.append(model.predict(image))
    #embeds.append(embedder.embeddings(image))

data_full['embeds']=embeds

train=data_full[0:len_train]
test=data_full[len_train:len_train+len_test]
###############################################################################
#Unknown finder

dummy=test.person.isin(train.person).astype(str)

count=-1
test.is_copy = False
for item in dummy:
    count+=1
    if(item=='False'):
        test['person'][count]='Unknown'

all_test_labels=test['person'].values.tolist()

unique_name_test = []
for x in all_test_labels:
    if x not in unique_name_test:
        unique_name_test.append(x)
###############################################################################
#Metrics point creation
# =============================================================================
#         
# res = list(set(unique_name_test+unique_name_train))
# 
# for i in res:
#     for j in res:
#         locals()['{}_{}'.format(i,j)] = 0 
#     
# =============================================================================
###############################################################################
# =============================================================================
#Test Module
# =============================================================================
#min_dis_list=[]
# 
# for x,value in  test.iterrows():
#     name_test = test['person'][x]
#     num1 = test['embeds'][x]
#     
#     temp_dis=[]
#     temp_name=[]
#     
#     for y,value in  train.iterrows():
#         num2 = train['embeds'][y]
#         dis = euclidean(num1,num2)
#         temp_dis.append(dis)
#         temp_name.append(train['person'][y])
#     
#     if(min(temp_dis)<20):
#         xyz=temp_dis.index(min(temp_dis))
#         min_dis_list.append(min(temp_dis))
#         name_train = temp_name[xyz]
#     else:
#         min_dis_list.append(min(temp_dis))
#         name_train = 'Unknown'
#     print(name_test,name_train)
#     locals()['{}_{}'.format(name_test,name_train)] += 1
# =============================================================================
    
###############################################################################
# =============================================================================
# Report definations:
# 
# 1. TN: when detecting unknown user as unknown user
# 2. TP: when detecting reg user as correct reg user
# 3. FP: when detecting unknown as reg user
# 4. FN: when detecting reg user as unknown
# 5. FMC: (false match count)when detecting reg user as incorrect reg user
# 6. Intrusion rate: sum of all false positives wrt. total number of unknown user
# 7. MBS rate: total miss rate of detection of our model
# 8. FM rate: total false matches of our model
# 
# Report metrics:
# 
# 1. Intrusion Rate = (sum of all FP)/(total unknown) 
# 2. MBS Rate = FN(x)/((TP(x)+FN(x)) -> (sum of all MBS Rate)/total no. of people  (x: index/name of person)
# 3. FM Rate = FMC(x)/((TP(x)+FMC(x)) -> (sum of all FM Rate)/total no. of people (x: index/name of person)
# 
# Supporting algo:
# 
# 1. if(GT = Pred) but (GT != unknown user) -> TP
# 2. if(GT = unknown user) but (Pred !=unknown user ) ->FP
# 3. if(GT != unknown user) but (Pred = unknown user) ->FN
# 4. if(GT != Pred) and (GT != unknown user and Pred != unknown user) -> FMC
# 5. if(GT = Pred) but (GT = unknown user) -> TN
# 
# We need to calculate for each (x: Person then aggregate in end)
# =============================================================================
###############################################################################
#Report metrics to find optimum threshold

# =============================================================================
#listofthreshold=[10,20,50,75,100,150,200,300,400,500]

#for alpha in listofthreshold:
#    alph = int(alpha)
# =============================================================================
# alph=20
# for x,value in  test.iterrows():
#     name_test = test['person'][x]
#     num1 = test['embeds'][x]
#     temp_dis=[]
#     temp_name=[]
#     TP=0
#     FP=0
#     FN=0
#     FMC=0
#     TN=0
#     for y,value in  train.iterrows():
#         num2 = train['embeds'][y]
#         dis = euclidean(num1,num2)
#         temp_dis.append(dis)
#         temp_name.append(train['person'][y])
#     
#     if(min(temp_dis)<alph):
#         xyz=temp_dis.index(min(temp_dis))
#         min_dis_list.append(min(temp_dis))
#         name_train = temp_name[xyz]
#     else:
#         min_dis_list.append(min(temp_dis))
#         name_train = 'Unknown'
#     GT=name_test
#     Pred=name_train
#     if(GT == Pred and GT != "Unknown"):
#         TP+=1
#     if(GT == "Unknown" and Pred != "Unknown"):
#         FP+=1
#     if(GT != "Unknown" and Pred == "Unknown"):
#         FN+=1
#     if(GT != Pred and GT != "Unknown" and Pred != "Unknown"):
#         FMC+=1
#     if(GT == Pred and GT == "Unknown"):
#         TN+=1
#     print(GT,TP,FP,FN,FMC,TN)
# 
# print(TP,FP,FN,FMC,TN)
# 
# =============================================================================
###############################################################################
#Dummy

#models=["eucli","cos","manhat"]
models=["manhat"]
#models=['cos']
for model in models:
    #model="manhat"
    
    if(model=="eucli"):
        listofthreshold=[0.05,0.11,0.16,0.21,0.27,0.33,0.38,0.44,0.49,0.55]
    
    elif(model=="cos"):
        listofthreshold=[0.00,0.01,0.03,0.04,0.05,0.07,0.09,0.13,0.14,0.15]
    
    elif(model=="manhat"):
        listofthreshold=[10,17,24,31,38,45,52,59,66,71]
    
        
    columns_name=['Person_name','Threshold','TP','FP','FN','TN','FMC','Miss_R','FMR','Intrusion','Train_count','Test_count']
    df_ = pd.DataFrame(columns=columns_name)
    ###############################################################################
    
    
    
    for alpha in listofthreshold:
        min_dis_list=[]
        alph = float(alpha)
        print(alph)
        #alph=0.9
        for x,value in  test.iterrows():
            name_test = test['person'][x]
            locals()['{}_TP'.format(name_test)] = 0
            locals()['{}_FP'.format(name_test)] = 0
            locals()['{}_FN'.format(name_test)] = 0
            locals()['{}_FMC'.format(name_test)] = 0
            locals()['{}_TN'.format(name_test)] = 0
        
        for x,value in  test.iterrows():
            name_test = test['person'][x]
            num1 = test['embeds'][x]
            temp_dis=[]
            temp_name=[]
            TP=0
            FP=0
            FN=0
            FMC=0
            TN=0
            for y,value in  train.iterrows():
                num2 = train['embeds'][y]
                if(model=="eucli"):
                    dis = euclidean(num1,num2)
                elif(model=="cos"):
                    dis = cosine(num1,num2)
                elif(model=="manhat"):
                    dis = manhatten(num1,num2)
                
                temp_dis.append(dis)
                temp_name.append(train['person'][y])
            
            if(min(temp_dis) < alph):    
                xyz=temp_dis.index(min(temp_dis))
                min_dis_list.append(min(temp_dis))
                name_train = temp_name[xyz]
            else:
                min_dis_list.append(min(temp_dis))
                name_train = 'Unknown'
            GT=name_test
            Pred=name_train
            if(GT == Pred and GT != "Unknown"):
                TP+=1
                locals()['{}_TP'.format(name_test)] +=1 
            if(GT == "Unknown" and Pred != "Unknown"):
                FP+=1
                locals()['{}_FP'.format(name_test)] +=1
            if(GT != "Unknown" and Pred == "Unknown"):
                FN+=1
                locals()['{}_FN'.format(name_test)] +=1
            if(GT != Pred and GT != "Unknown" and Pred != "Unknown"):
                FMC+=1
                locals()['{}_FMC'.format(name_test)] +=1
            if(GT == Pred and GT == "Unknown"):
                TN+=1
                locals()['{}_TN'.format(name_test)] +=1
        
        ###############################################################################
        # Reports MBS and FMR
        
        for x in unique_name_test:
            name_test = x
            if(locals()['{}_FN'.format(name_test)]==0):
                locals()['{}_MBS_Rate'.format(name_test)]=0
            else:
                locals()['{}_MBS_Rate'.format(name_test)] = (locals()['{}_FN'.format(name_test)]) / (locals()['{}_TP'.format(name_test)]+locals()['{}_FN'.format(name_test)])
            
            if(locals()['{}_FMC'.format(name_test)]==0):
                locals()['{}_FM_Rate'.format(name_test)]=0
            else:
                locals()['{}_FM_Rate'.format(name_test)] = (locals()['{}_FMC'.format(name_test)]) / (locals()['{}_TP'.format(name_test)]+locals()['{}_FMC'.format(name_test)])
            (print(name_test,locals()['{}_MBS_Rate'.format(name_test)]*100,locals()['{}_FM_Rate'.format(name_test)]*100))
        
        ###############################################################################
        # Report Intrusion Rate
        count_of_unk=0
        
        for x,value in  test.iterrows():
            name_test = test['person'][x]
            if(name_test == 'Unknown'):
                count_of_unk +=1 
        
        Intrusion_Rate= Unknown_FP/count_of_unk
        
        print(Intrusion_Rate*100)
        ###############################################################################
        
        print(unique_name_test)
        
        for x in sorted(unique_name_test):
            name_test = x
            num_train=all_train_labels.count(name_test)
            num_test=all_test_labels.count(name_test)
            df_=df_.append(pd.Series([name_test,str(alph),str(locals()['{}_TP'.format(name_test)]),str(locals()['{}_FP'.format(name_test)]),
                                      str(locals()['{}_FN'.format(name_test)]),str(locals()['{}_TN'.format(name_test)]),str(locals()['{}_FMC'.format(name_test)]),str(locals()['{}_MBS_Rate'.format(name_test)]*100),
                                      str(locals()['{}_FM_Rate'.format(name_test)]*100),str(Intrusion_Rate*100),num_train,num_test],index=df_.columns),ignore_index=True)
    
    
    if(model=="eucli"):
        df_.to_csv("euclidean.csv",index=False)
    elif(model=="cos"):
        df_.to_csv("cosine.csv",index=False)
    elif(model=="manhat"):
        df_.to_csv("manhatten.csv",index=False)
    
        
    
# =============================================================================
#     print(min(min_dis_list))
#     print(max(min_dis_list))
#     
#     print(min(temp_dis))
#     print(max(temp_dis))
#            
# =============================================================================
###############################################################################            
        