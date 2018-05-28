import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from time import time
from sklearn.linear_model import LogisticRegression
import pickle
import math
from sklearn.ensemble import RandomForestClassifier

df1 = pd.read_csv("Test1655_240518.CSV") 
df2= pd.read_csv("Test1734_240518.CSV")

df2.describe()

x_values=df1.iloc[:,0].values
y_values=df1.iloc[:,1].values



x_values2=df2.iloc[:,0].values
y_values2=df2.iloc[:,1].values

diff_x_values2=np.diff(x_values2)
diff_y_values2=np.diff(y_values2)


diff_x_values=np.diff(x_values)
diff_y_values=np.diff(y_values)



j=0
i=0
labels=[];
feature1=[];
feature2=[];
feature3=[];
feature4=[];
count=0
end=0
start=5930;

while(start>=5930 and end<17930):
    #FAN SPEED 1 and FAN SPEED 2
    count=count+1
    labels.append(1)
    end=start+100; 
    feature1.append(np.mean(diff_x_values[start:end]))
    feature2.append(np.std(diff_x_values[start:end])) 
    feature3.append(np.mean(diff_y_values[start:end]))
    feature4.append(np.std(diff_y_values[start:end])) 
    start=end
 


print count

# end=0;
# start=3000;

# while(start>=3000 and end<9000):
#     #FAN SPEED 3
#     count=count+1
#     labels.append(1)
#     end=start+100; 
#     feature1.append(np.mean(diff_x_values2[start:end]))
#     feature2.append(np.std(diff_x_values2[start:end])) 
#     feature3.append(np.mean(diff_y_values2[start:end]))
#     feature4.append(np.std(diff_y_values2[start:end])) 
#     start=end



start=20930
end=0

while(start>=20930 and end<23930):
    count=count+1
    #POWDER
    end=start+100
    feature1.append(np.mean(diff_x_values[start:end]))
    feature2.append(np.std(diff_x_values[start:end]))
    feature3.append(np.mean(diff_y_values[start:end]))
    feature4.append(np.std(diff_y_values[start:end]))
    start=end
 
    labels.append(2)


    

print count
m=26930
n=0


while(m>=26930 and n<29930):
    ##Spray
    count=count+1
    n=m+100
    feature1.append(np.mean(diff_x_values[m:n]))
    feature2.append(np.std(diff_x_values[m:n]))
    feature3.append(np.mean(diff_y_values[m:n]))
    feature4.append(np.std(diff_y_values[m:n])) 
    labels.append(3)
    m=n


# m1=32930
# n1=0



# while(m1>=32930 and n1<35930):
#     ##Dhoop
#     count=count+1
#     n1=m1+100
#     feature1.append(np.mean(diff_x_values[m1:n1]))
#     feature2.append(np.std(diff_x_values[m1:n1]))
#     feature3.append(np.mean(diff_y_values[m1:n1]))
#     feature4.append(np.std(diff_y_values[m1:n1])) 
#     labels.append(4)
#     m1=n1
    
# print(count)

#Dhoop 


# m2=36930
# n2=0

# while(m2>=36930 and n2<45930):
#     ##Dhoop speed1, speed2, speed3
#     count=count+1
#     n2=m2+100
#     feature1.append(np.mean(diff_x_values[m2:n2]))
#     feature2.append(np.std(diff_x_values[m2:n2]))
#     feature3.append(np.mean(diff_y_values[m2:n2]))
#     feature4.append(np.std(diff_y_values[m2:n2])) 
#     labels.append(5)
#     m2=n2

    
print count

feature1=(np.asarray(feature1)).reshape(180,1)
feature2=(np.asarray(feature2)).reshape(180,1)
feature3=(np.asarray(feature3)).reshape(feature1.shape[0],1)
feature4=(np.asarray(feature4)).reshape(feature1.shape[0],1)

mag_diff1=(np.sqrt(feature1**2+feature3**2)).reshape(feature1.shape[0],1);
dir_diff1=(np.arctan(feature3/feature1)).reshape(feature1.shape[0],1);


mag_diff2=(np.sqrt(feature2**2+feature4**2)).reshape(feature1.shape[0],1);
dir_diff2=(np.arctan(feature4/feature2)).reshape(feature1.shape[0],1);

# Training features

training=np.concatenate((mag_diff1,dir_diff1,mag_diff2,dir_diff2), axis=1)
testing=np.asarray(labels)
print training.shape
print testing.shape

 
#TRAINING THE MODEL
clf = RandomForestClassifier(random_state=1)
clf.fit(training,testing)


with open('Random_Forest_MODE_flowchart2.pickle','wb') as f:
    pickle.dump(clf,f)


print "MODEL SAVED"
