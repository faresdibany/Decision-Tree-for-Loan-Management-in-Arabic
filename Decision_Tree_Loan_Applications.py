# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:27:41 2020

@author: Dell
"""
from tensorflow import keras 
import pandas as pd
import arabic_reshaper
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Softmax
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Adamax
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import externals
import six
from six import StringIO
import sys
import numpy as np
# numpy.set_printoptions(threshold=sys.maxsize)
# load dataset
# dataframe = read_csv("new_set_random_one_nashat.csv",encoding= 'utf8')
# # dt = dataframe.astype('string')
# # split into input (X) and output (Y) variables
# X = dataframe[[ 'النشاط رئيسي','الحالة الاجتماعية','عدد الاعالة','النوع','ضامن']].values
# # X = dataframe[:,1:10]
# Y = dataframe[['الدفع']].values

#Converting the categorical data to numerical for the decision tree
# le = preprocessing.LabelEncoder()
# le.fit(['متعسر', 'متعسر جدا', 'منتظم'])
# Y = le_sex.transform(Y) 
def loantree(X):
    le = preprocessing.LabelEncoder()
    le.fit(['أرمل','أعزب','أعزب ويعول','متزوج','متزوج ويعول','مطلق','غير متاح'])
    X[:,1] = le.transform(X[:,1]) 
    
    le = preprocessing.LabelEncoder()
    le.fit(['Male', 'female'])
    X[:,3] = le.transform(X[:,3]) 
    
    le = preprocessing.LabelEncoder()
    le.fit(['yes', 'no'])
    X[:,4] = le.transform(X[:,4]) 
    
    le = preprocessing.LabelEncoder()
    le.fit(['القطاع الخدمى','القطاع التجارى','القطاع الزراعى','القطاع الصناعى','لا يوجد نشاط'])
    X[:,0] = le.transform(X[:,0]) 
    
    # le = preprocessing.LabelEncoder()
    # le.fit(['اخرى','ادوات كهربائيه','ادوات نظافه','اراضى زراعية','ارانب وطيور','استرجى','اصلاح احذية','اصلاح دراجات و موتوسيكلات','الات زراعية بسيطة','التسمين','الحرفيين','الحلابة','انشطه تجاريه متنوعه','بقاله','بلاستيك','بويات و حدايد','بيع اجهزة و اكسسوار محمول','بيع ادوات صحية','بيع ادوات منزلية','بيع اسماك','بيع اغنام و مواشى','بيع اكسسوار حريمى','بيع غلال وحبوب وعلافة وعطارة','بيع فاكهة و خضار','بيع لحوم - جزارة','بيع مجمدات','بيع مفروشات','بيع منتجات البان','بيع منتجات بترولية','بيع منتجات بلاستيك','بيع منتجات ورقية','بيع موبيليات','تجارة اقمشة و مانيفاتورة','تجارة خرده','تجارة مواد بناء','تجارة و بيع احذيه و منتجات جلديه','تربية اغنام','تربية مواشى','تريية مواشى','تعبئة فحوم - توزيع و بيع بوتاجاز','تعبئة و تجهيز بويات و شحومات','جراج سيارات - محطة بنزين','جلديه','حضانه ومعاهد تعليمية و مكاتب','خدمات متنوعه','خردوات و لوازم خياطيه','خشبيه','دوكو سيارات','رفا و تنظيف و دراى كلين و مكوجى','زجاج','ستوديو تصوير و طبع و تحميض افلام','سجاير و حلويات','سروجى','سمكرى سيارات','سنترالات و خدمات تليفونيه','شحن بطاريات','صيادلة ولوازم صيدليات','صيانة و اصلاح اجهزة كهربائيه','طباعه','غذائية','قطع غيار سيارات و خلافه','كهربائى سيارات','كوافيير و تزيين عرائس و حلاق','مركب صيد','مركبات النقل','مشاتل','مطاعم وقهاوى شعبية','معادن','مقاولات عامه','مكتبة وادوات مكتبية','ملابس جاهزه بيع','ملابس و صناعات نسيجيه','منتجات شمع','مواد بناء','ميكانيكى سيارات','نجارة اخشاب','اصلاح ابواب و شبابيك السيارات', 'لا يوجد نشاط', 'بيع و تأجير شرائط فيديو'])
    # X[:,1] = le.transform(X[:,1]) 
    
    #splitting the data
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
    print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
    print('Shape of X testing set {}'.format(X_testset.shape),'&',' Size of Y testing set {}'.format(y_testset.shape))
    
    loanTree = DecisionTreeClassifier(criterion="entropy", splitter='best')
    loanTree # it shows the default parameters
    
    loanTree.fit(X_trainset,y_trainset)
    
    predTree = loanTree.predict(X_testset)
    return predTree

e = read_csv("Bookss1.csv",encoding= 'utf8')
X = e[[ 'النشاط رئيسي','الحالة الاجتماعية','عدد الاعالة','النوع','ضامن']].values
# X = dataframe[:,1:10]
Y = e[['الدفع']].values
u = loantree(X)


# print (predTree)
# print (y_testset)

# boolarr = (predTree==y_testset)
# result = np.where(boolarr)
# print(result)

# boolarr1 = (predTree!=y_testset)
# result2 = np.where(boolarr1)
# result1 = np.where(predTree)

# AB = np.stack([predTree], axis=-1)
# XY = np.stack([y_testset], axis=-1)

# eq = AB[:, np.newaxis, :] == XY[np.newaxis, :, :]
# eq = np.logical_and.reduce(eq, axis=-1)

# indAB, = np.where(np.logical_or.reduce(eq, axis=1))
# indXY, = np.where(np.logical_or.reduce(eq, axis=0))

# print("indAB", indAB)
# print("indXY", indXY)

# from sklearn import metrics
# import matplotlib.pyplot as plt
# print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# import pydotplus
# import matplotlib.image as mpimg
# from sklearn import tree


# dot_data = StringIO()
# filename = "loantree.png"
# featureNames = dataframe[["قيمة القرض ","عدد الأقساط ", "قيمة القسط ", "النشاط رئيسي", "النشاط فرعي","الحالة الاجتماعية","عدد الاعالة","النوع","ضامن"]]
# targetNames = dataframe["الدفع"]
# out=tree.export_graphviz(loanTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png(filename)
# img = mpimg.imread(filename)
# plt.figure(figsize=(100, 200))
# plt.imshow(img,interpolation='nearest')

# def undummify(df, prefix_sep="_"):
#     cols2collapse = {
#         item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
#     }
#     series_list = []
#     for col, needs_to_collapse in cols2collapse.items():
#         if needs_to_collapse:
#             undummified = (
#                 df.filter(like=col)
#                 .idxmax(axis=1)
#                 .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
#                 .rename(col)
#             )
#             series_list.append(undummified)
#         else:
#             series_list.append(df[col])
#     undummified_df = pd.concat(series_list, axis=1)
#     return undummified_df

