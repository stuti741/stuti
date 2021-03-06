import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#Import Library Pandas
df = pd.read_csv("C:\\Users\\dell\\newsend.csv")
x=df.iloc[:,0]
x=x.values
print(x)
y=df.iloc[:,1]
y=y.values
print(y)
z=np.sqrt(x*x+y*y)
q=np.absolute(z)

r=df.iloc[:,2]
#print(q)
#plt.plot(q)
j=0
i=0
labels=[];
newlis=[];

while(j<2000):
    labels.append(1)
    j=i+100; 
    newlis.append([np.mean(q[i:j]),np.std(q[i:j])])
    i=j
print(len(newlis))

#print(newlis)

start=2000
end=0
lis=[]

while(start>=2000 and end<4000):
    end=start+100
    lis.append([np.mean(q[start:end]),np.std(q[start:end])])
    start=end
    labels.append(2)
print(len(lis))
#print(lis)



m=4000
n=0
lis3=[]
while(m>=4000 and n<6000):
    n=m+100
    lis3.append([np.mean(q[m:n]),np.std(q[m:n])])
    labels.append(3)
    m=n
print(len(lis3))


x4=6001
y4=0
lis4=[]
while(x4>=6000 and y4<8000):
    y4=x4+100
    lis4.append([np.mean(q[x4:y4]),np.std(q[x4:y4])])
    x4=y4
    labels.append(4)
print(len(lis4))

x5=8000
y5=0
lis5=[]
while(x5>=8000 and y5<10000):
    y5=x5+100
    lis5.append([np.mean(q[x5:y5]),np.std(q[x5:y5])])
    x5=y5
    labels.append(1)
    
print(len(lis5))


x6=10000
y6=0
lis6=[]
while(x6>=10000 and y6<11000):
    y6=x6+100
    lis6.append([np.mean(q[x6:y6]),np.std(q[x6:y6])])
    x6=y6
    labels.append(5)
    
print(np.shape(lis6))

x7=11000
y7=0
lis7=[]
while(x7>=11000 and y7<14000):
    y7=x7+100
    lis7.append([np.mean(q[x7:y7]),np.std(q[x7:y7])])
    x7=y7
    labels.append(6)
    
print(np.shape(lis7))

x8=14000
y8=0
lis8=[]

while(x8>=14000 and y8<16000):
    y8=x8+100
    lis8.append([np.mean(q[x8:y8]),np.std(q[x8:y8])])
    x8=y8
    labels.append(7)
print(np.shape(lis8))

x9=16000
y9=0
lis9=[]
while(x9>=16000 and y9<18000):
    y9=x9+100
    lis9.append([np.mean(q[x9:y9]),np.std(q[x9:y9])])
    x9=y9
    labels.append(8)
print(np.shape(lis9))

x10=18000
y10=0
lis10=[]
while(x10>=18000 and y10<20000):
    y10=x10+100
    lis10.append([np.mean(q[x10:y10]),np.std(q[x10:y10])])
    x10=y10
    labels.append(7)
    
print(np.shape(lis10))

x11=20000
y11=0
lis11=[]

while(x11>=20000 and y11<21000):
    y11=x11+100
    lis11.append([np.mean(q[x11:y11]),np.std(q[x11:y11])])
    x11=y11
    labels.append(9)
    
    
print(np.shape(lis11))

x12=21000
y12=0
lis12=[]
while(x12>=21000 and y12<22000):
    y12=x12+100
    lis12.append([np.mean(q[x12:y12]),np.std(q[x12:y12])])
    x12=y12
    labels.append(10)
print(np.shape(lis12))

x13=22000
y13=0
lis13=[]
while(x13>=22000 and y13<23000):
    y13=x13+100
    lis13.append([np.mean(q[x13:y13]),np.std(q[x13:y13])])
    x13=y13
    labels.append(11)
    
print(np.shape(lis13))

x14=23000
y14=0
lis14=[]

while(x14>=23000 and y14<24000):
    y14=x14+100
    lis14.append([np.mean(q[x14:y14]),np.std(q[x14:y14])])
    x14=y14
    labels.append(7)

print(np.shape(lis14))

x15=24000
y15=0
lis15=[]
while(x15>=24000 and y15<25000):
    y15=x15+100
    lis15.append([np.mean(q[x15:y15]),np.std(q[x15:y15])])
    x15=y15
    labels.append(1)
    
print(np.shape(lis15))

#feature training
print ("-----------------------------")
lisadd=newlis+lis+lis3+lis4+lis5+lis6+lis7+lis8+lis9+lis10+lis11+lis12+lis13+lis14+lis15
print(np.shape(lisadd))

print(np.shape(labels))
print ("-------------------------- check point----------------")


i=0
lisfeaturetest=[]
lislabeltest=[]
lisfeaturetrain=[]
lislabeltrain=[]

while(i<250):
    
    if i%12==0 and i!=0:
        print(i)
        lisfeaturetest.append(lisadd[i])
        lislabeltest.append(labels[i])
    else:
        lisfeaturetrain.append(lisadd[i])
        lislabeltrain.append(labels[i])
        
        
       
    i=i+1


print ("-------------------------- check point----------------")



lisfeaturetrain=np.asarray(lisfeaturetrain)
lisfeaturetrain=lisfeaturetrain.reshape(lisfeaturetrain.shape[0],2)
lislabeltrain=np.asarray(lislabeltrain)

lisfeaturetest=np.asarray(lisfeaturetest)
lisfeaturetest=lisfeaturetest.reshape(lisfeaturetest.shape[0],2)
lislabeltest=np.asarray(lislabeltest)


clf= DecisionTreeClassifier(criterion = 'entropy').fit(lisfeaturetrain,lislabeltrain)

print ("-------------------------- check point----------------")

predicted_values=clf.predict(lisfeaturetest)

print(predicted_values)
print ("**************")
print (lislabeltest)

### Checking the accuracy of the current model 

w=clf.score(lisfeaturetest,lislabeltest)
print("The prediction accuracy is: ",w*100,"%")
