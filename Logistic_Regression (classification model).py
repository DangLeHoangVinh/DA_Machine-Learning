# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:07:03 2024

@author: Dang Le Hoang Vinh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#============= Import thư viện pandas ============
url='https://drive.google.com/file/d/134juqX_tF0S_YWw-o5g5q3Go7zX__QM2/view?usp=drive_link'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url)

#del df['Unnamed: 0']
df.head()

#=================== Understanding ==============

df.head()

df['loan_status'].value_counts()

df.info()

#================= chỉnh dữ liệu, effective_date + due_date==============
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['due_date'] = pd.to_datetime(df['due_date'])

#=============== chỉnh ngày tháng ================================
# extract day of week people get the  _ ngày trong tuần mà họ băt đầu loan
df['dayofweek'] = df['effective_date'].dt.dayofweek

# drop columns
df = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'effective_date', 'due_date'], axis=1)


#=================chỉnh Male/female =============================

df['Gender'] = df['Gender'].map({'male':0,'female':1})

#========================================================
df = pd.get_dummies(df, columns=["education", "dayofweek"], drop_first=False) # chuyển education thành biến có thể phân tích_ dạng 

#=================== đồng bộ số lượng dữ liệu bằng pp over sampling ===================
df['loan_status'].value_counts() # 

X = df.drop(columns=['loan_status']) # X là biến dữ liệu độc lập phục vụ biến dự đoán
y = df['loan_status'] # biến dự 

# khai báo thêm thư viện
from imblearn.over_sampling import SMOTE

# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# hàm đếm số ký tự không nằm trong data frame, check kết quả sau khi đồng 
from collections import Counter
counter = Counter(y)
print(counter) # data set loan_status đã về 260 - 260


#========================= train model ================================
# thêm thư viện:
    from sklearn.model_selection import train_test_split
# train test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0) # tách 

# fit logistic regression

#khai báo model 
from sklearn.linear_model import LogisticRegression

logres = LogisticRegression()

# train model với tập train 
logres.fit(X_train, y_train)

#===================== = Tạo Predict cho tập test ================================
#predict thử kết quả với tập test 
#2 dạng predict
# dạng 1: predict ra kết quả luôn 
y_pred_log = logres.predict(X_valid)

#dạng 2: predict ra # probability và 1- probability
y_pred_proba = logres.predict_proba(X_valid) # trả về dạng [{p, 1-p}]

#================= xuất bảng kết quả =============================
result = X_valid
result['predict'] = y_pred_log


# ========================= KIỂM TRA HIỆU QUẢ MODEL ====================

#Khai báo thư viện
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# confusion matrix
print(confusion_matrix(y_pred_log, y_valid)) # là một bảng có 2 cột, xem cách đọc google 
import seaborn as sns
sns.heatmap(confusion_matrix(y_pred_log, y_valid), annot=True)

#classification_report
print(classification_report(y_pred_log, y_valid))

# logistic accuracy
print ('Accuracy: ', accuracy_score(y_pred_log, y_valid))










































