#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:47:56 2018

@author: srikantswamy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:33:16 2018

@author: srikantswamy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

import requests
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

import sys

train_year=sys.argv[1]
test_year=sys.argv[2]
user_id=sys.argv[3]
pass_word=sys.argv[4]
rec_count=sys.argv[5]



def load_file(file, cnt):
    count=int(cnt)
    df=pd.read_csv(file, sep="|", names=['loan_seq_nbr',
                                            'mnth_rep_dt',
                                                                        'curr_upb',
                                                                        'loan_delq_stat',
                                                                        'loan_age',
                                                                        'rem_month',
                                                                        'repurch_flg',
                                                                        'modn_flg',
                                                                        'zero_bal_cd',
                                                                        'zero_bal_efftv_dt',
                                                                        'curr_int_rate',
                                                                        'curr_deff_upb',
                                                                        'due_dt_lpi',
                                                                        'mi_recv',
                                                                        'net_sales',
                                                                        'non_mi_recv',
                                                                        'expens',
                                                                        'legal_cost',
                                                                        'preserv_cost',
                                                                        'taxes',
                                                                        'misc_expens',
                                                                        'actual_loss',
                                                                        'modn_cost',
                                                                        'stp_modn_flg',
                                                                        'def_pay_modn',
                                                                        'est_loan_to_val'], nrows=count)
    return df
    

def frame_transform(frame):
    frame['loan_delq_stat'] = [ 0 if x=='XX' else x for x in (frame['loan_delq_stat'].apply(lambda x: x))]
    frame['loan_delq_stat'] = [ 999 if x=='R' else x for x in (frame['loan_delq_stat'].apply(lambda x: x))]
    frame['modn_flg'] = [ 0 if x=='N' else 1 for x in (frame['modn_flg'].apply(lambda x: x))]
    frame[['net_sales']] = [ 0 if x=='U' else x for x in (frame['net_sales'].apply(lambda x: x))]
    frame[['net_sales']] = [ 999 if x=='C' else x for x in (frame['net_sales'].apply(lambda x: x))]
    frame[['def_pay_modn']] = [ 'Z' if x==' ' else x for x in (frame['def_pay_modn'].apply(lambda x: x))]
    # frame[['net_sales']] = [ frame['curr_deff_upb'] if x=='C' else x for x in (frame['net_sales'].apply(lambda x: x)]
    # frame[['net_sales']] = [ frame['curr_deff_upb'] if x=='C' else x for x in (frame['net_sales'].apply(lambda x: x)) ]
    return frame
                                      


def handle_missing(frame):
    frame['loan_delq_stat']=frame['loan_delq_stat'].fillna(0)
    frame['repurch_flg']=frame['repurch_flg'].fillna('Z')
    frame['modn_flg']=frame['modn_flg'].fillna('N')
    frame['stp_modn_flg']=frame['stp_modn_flg'].fillna('Z')
    frame['zero_bal_cd']=frame['zero_bal_cd'].fillna(0)
    frame['curr_deff_upb']=frame['curr_deff_upb'].fillna(0)
    frame['mi_recv']=frame['mi_recv'].fillna(0)
    frame['net_sales']=frame['net_sales'].fillna(0)
    frame['non_mi_recv']=frame['non_mi_recv'].fillna(0)
    frame['expens']=frame['expens'].fillna(0)
    frame['legal_cost']=frame['legal_cost'].fillna(0)
    frame['preserv_cost']=frame['preserv_cost'].fillna(0)
    frame['misc_expens']=frame['misc_expens'].fillna(0)
    frame['actual_loss']=frame['actual_loss'].fillna(0)
    frame['modn_cost']=frame['modn_cost'].fillna(0)
    # df['zero_bal_efftv_dt']=df['zero_bal_efftv_dt'].fillna(0)
    # df['due_dt_lpi']=df['due_dt_lpi'].fillna(0)
    # df['taxes']=df['taxes'].fillna(0)
    # df['est_loan_to_val']=df['est_loan_to_val'].fillna(0)
    return frame


def handle_dtypes(frame):
    frame[['loan_seq_nbr','repurch_flg','modn_flg','stp_modn_flg','def_pay_modn']] = frame[['loan_seq_nbr','repurch_flg','modn_flg','stp_modn_flg','def_pay_modn']].astype(str)
    frame[['curr_upb','curr_int_rate','mi_recv','non_mi_recv','expens','legal_cost','preserv_cost','taxes','misc_expens','actual_loss','modn_cost','est_loan_to_val']] = frame[['curr_upb','curr_int_rate','mi_recv','non_mi_recv','expens','legal_cost','preserv_cost','taxes','misc_expens','actual_loss','modn_cost','est_loan_to_val']].astype(float)
    frame[['loan_age','rem_month','zero_bal_cd','curr_deff_upb','loan_delq_stat']] = frame[['loan_age','rem_month','zero_bal_cd','curr_deff_upb','loan_delq_stat']].astype(int)
    return frame


def feature_engg(frame):
    frame['delinquency'] = (frame.loan_delq_stat > 0).astype(int)
    frame = frame.drop('loan_delq_stat', axis = 1)
    return frame

def feature_encode(frame):
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le = LabelEncoder()

    frame['repurch_flg'] = le.fit_transform(frame['repurch_flg'])
    frame['stp_modn_flg'] = le.fit_transform(frame['stp_modn_flg'])
    frame['def_pay_modn'] = le.fit_transform(frame['def_pay_modn'])

    onehotencoder = OneHotEncoder()

    frame_repurch_flg = onehotencoder.fit_transform(frame.repurch_flg.values.reshape(-1,1)).toarray()
    frame_stp_modn_flg=onehotencoder.fit_transform(frame.stp_modn_flg.values.reshape(-1,1)).toarray()
    frame_def_pay_modn=onehotencoder.fit_transform(frame.def_pay_modn.values.reshape(-1,1)).toarray()

    repurch_flg_onehot = pd.DataFrame(frame_repurch_flg, columns = ["repurch_flg_"+str(int(i)) for i in range(frame_repurch_flg.shape[1])])
    frame = pd.concat([frame, repurch_flg_onehot], axis=1)

    stp_modn_flg_onehot = pd.DataFrame(frame_stp_modn_flg, columns = ["stp_modn_flg_"+str(int(i)) for i in range(frame_stp_modn_flg.shape[1])])
    frame = pd.concat([frame, stp_modn_flg_onehot], axis=1)

    def_pay_modn_onehot = pd.DataFrame(frame_def_pay_modn, columns = ["def_pay_modn"+str(int(i)) for i in range(frame_def_pay_modn.shape[1])])
    frame = pd.concat([frame, def_pay_modn_onehot], axis=1)

    frame.drop(['repurch_flg', 'stp_modn_flg','def_pay_modn'], axis=1)
    return frame
    

def feature_encode_feature_select(frame):
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    le = LabelEncoder()

    #frame['repurch_flg'] = le.fit_transform(frame['repurch_flg'])
    frame['stp_modn_flg'] = le.fit_transform(frame['stp_modn_flg'])
    frame['def_pay_modn'] = le.fit_transform(frame['def_pay_modn'])

    onehotencoder = OneHotEncoder()

    #frame_repurch_flg = onehotencoder.fit_transform(frame.repurch_flg.values.reshape(-1,1)).toarray()
    frame_stp_modn_flg=onehotencoder.fit_transform(frame.stp_modn_flg.values.reshape(-1,1)).toarray()
    frame_def_pay_modn=onehotencoder.fit_transform(frame.def_pay_modn.values.reshape(-1,1)).toarray()

    #repurch_flg_onehot = pd.DataFrame(frame_repurch_flg, columns = ["repurch_flg_"+str(int(i)) for i in range(frame_repurch_flg.shape[1])])
    #frame = pd.concat([frame, repurch_flg_onehot], axis=1)

    stp_modn_flg_onehot = pd.DataFrame(frame_stp_modn_flg, columns = ["stp_modn_flg_"+str(int(i)) for i in range(frame_stp_modn_flg.shape[1])])
    frame = pd.concat([frame, stp_modn_flg_onehot], axis=1)

    def_pay_modn_onehot = pd.DataFrame(frame_def_pay_modn, columns = ["def_pay_modn"+str(int(i)) for i in range(frame_def_pay_modn.shape[1])])
    frame = pd.concat([frame, def_pay_modn_onehot], axis=1)

    frame.drop(['stp_modn_flg','def_pay_modn'], axis=1)
    return frame
    

def scale_minmax(X):
    
    from sklearn.preprocessing import MinMaxScaler

    mm_scale_X = MinMaxScaler()

    X = mm_scale_X.fit_transform(X)

    return X

def create_X(frame):
    frame=frame.drop(['loan_seq_nbr','delinquency'], axis=1)
    
    # Removing these columns temporarily
    frame = frame.drop('zero_bal_efftv_dt', axis = 1)
    frame = frame.drop('due_dt_lpi', axis = 1)
    frame = frame.drop('taxes', axis = 1)
    frame = frame.drop('est_loan_to_val', axis = 1)
    
    return frame

def create_X_feature_select(frame,columns):
    frame=frame.drop(['loan_seq_nbr','delinquency'], axis=1)
    
    # Removing these columns temporarily
    frame = frame.drop('zero_bal_efftv_dt', axis = 1)
    frame = frame.drop('due_dt_lpi', axis = 1)
    frame = frame.drop('taxes', axis = 1)
    frame = frame.drop('est_loan_to_val', axis = 1)
    
    frame=frame[columns]
    return frame

def create_y(frame):
    
    frame=frame[['delinquency']]
    return frame

def model_metrics(model,x_training,x_testing,y_training,y_testing,y_predict,y_predict_train):

    print("Training and Testing accuracy:")
    print("\n")
    print('Random Forest Regrssion  - Score - Training: %.4f' % model.score(x_training, y_training))
    print('Random Forest Regression - Score - Testing: %.4f' % model.score(x_testing, y_testing))
    
    from sklearn import metrics

    print("Confusion Matrix:")
    print("\n")
    
    
    cm_train = metrics.confusion_matrix(y_training, y_predict_train)
    print(cm_train)
    print("\n")
    print("Confusion matrix accuracy (train): " + str((cm_train[0,0]+cm_train[1,1])/np.sum(cm_train)))
    
    print("\n")

    cm_test = metrics.confusion_matrix(y_testing, y_predict)
    print(cm_test)
    print("\n")
    print("Confusion matrix accuracy (test): " + str((cm_test[0,0]+cm_test[1,1])/np.sum(cm_test)))
    
    print("\n")
    
    
    
    print("Confusion Matrix:")
    print("\n")
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    # %matplotlib inline
    
    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm_train), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix - Train', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm_test), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix - Test', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print("ROC Graph:")
    print("\n")
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_testing, model.predict(x_testing))
    fpr, tpr, thresholds = roc_curve(y_testing, model.predict_proba(x_testing)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Classifier (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    
    # print("Model Co-efficients:" + str(model.coef_))
    
    print("Accuracy:",metrics.accuracy_score(y_testing, y_predict))
    
    return cm_test


def gen_metric_file(y_frame_test,c_matrix,quarter,reg_name):
    
    import os
    
    quart = quarter
    
    regression_name=reg_name
    
    cols = ["Quarter",
            "Name",
            "Total actual deliquent",
            "Total predicted deliquent",
            "Total records",
            "Total deliquent prop classd",
            "Total deliquent improp classd"]

    total_num_act_delq = np.count_nonzero(y_frame_test==1)
    total_num_pred_delq = (c_matrix[1][0] + c_matrix[1][1])
    total_num_records = len(y_frame_test)
    total_num_delq_class_prop = (c_matrix[1][1])
    total_num_delq_class_improp = (c_matrix[1][0])
    
    frame_file = pd.DataFrame(columns=cols)
    
    frame_temp = pd.DataFrame([[quart,
                       regression_name,
                       total_num_act_delq, 
                       total_num_pred_delq, 
                       total_num_records, 
                       total_num_delq_class_prop, 
                       total_num_delq_class_improp]], columns=cols)

    frame_file = frame_file.append(frame_temp)
    
    filenm = "Metrics.csv"
    
    writeHeader = True
    
    if os.path.exists(filenm):
        writeHeader = False
    
    if writeHeader is False:
        with open(filenm, 'a',encoding = 'utf-8', newline="") as f:
            frame_file.to_csv(f, mode='a', header = False,index = False)
    else:
        with open(filenm, 'w',encoding = 'utf-8', newline="") as f:
            frame_file.to_csv(f, mode='a', header = True,index = False)


# gen_metric_file(y_test,cm,'1999')
            
def user_cred(user, passwd):
    creds={'username': user,'password': passwd}
    return creds


def download_files(usercred, trainq, testq):
    url='https://freddiemac.embs.com/FLoan/secure/auth.php'
    postUrl='https://freddiemac.embs.com/FLoan/Data/download.php'
    
    with requests.Session() as s:
        url_init = s.post(url, data=usercred)
        accept={'action':'acceptTandC', 'accept': 'Yes','acceptSubmit':'Continue'}
        finalUrl=s.post(postUrl,accept)
        url2 =finalUrl.text
        files=BeautifulSoup(url2, "html.parser")
        lst=files.find_all('td')
        
        hist_file=[]
        
        
        for anchor in lst:
            tags=anchor.findAll('a')
            for tg in tags:
                filenm='historical_data1_'
                if (trainq in tg.text):
                    if(filenm in tg.text):
                        link = tg.get('href')
                        dirname= 'qtr_files'
                        tgtdir=str(os.getcwd())+"/"+dirname                      
                        final ='https://freddiemac.embs.com/FLoan/Data/'
                        final=final+link
                        
                        hist_file.append(final)
                elif (testq in tg.text):
                     if(filenm in tg.text):
                        link = tg.get('href')
                        dirname= 'qtr_files'
                        tgtdir=str(os.getcwd())+"/"+dirname                        
                        final ='https://freddiemac.embs.com/FLoan/Data/'
                        final=final+link
                        
                        hist_file.append(final)
                        
        for file in hist_file:   
            down = s.get(file)
            z = ZipFile(BytesIO(down.content)) 
            z.extractall(tgtdir)

            

def logistic_reg_initial(train_file,test_file,quarter,cnt):
    
    regress_name='LOGISTIC_REG_INIT'
    
    quart=quarter
    
    train=load_file(train_file,cnt)
    train=handle_missing(train)
    train=frame_transform(train)
    train=handle_dtypes(train)
    train=feature_engg(train)


    X_train=create_X(train)
    y_train=create_y(train)
    X_train=feature_encode(X_train)
    X_train=scale_minmax(X_train)
    y_train=scale_minmax(y_train)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    model=lr.fit(X_train,y_train)
    
    score = model.score(X_train, y_train)
    
    test=load_file(test_file,cnt)
    test=handle_missing(test)
    test=frame_transform(test)
    test=handle_dtypes(test)
    test=feature_engg(test)


    X_test=create_X(test)
    y_test=create_y(test)
    X_test=feature_encode(X_test)
    X_test=scale_minmax(X_test)
    y_test=scale_minmax(y_test)

    y_pred=model.predict(X_test)
    
    # For Train CM

    y_pred_train=model.predict(X_train)
    
    cm=model_metrics(model,X_train,X_test,y_train,y_test,y_pred,y_pred_train)
    
    
    gen_metric_file(y_test,cm,quart,regress_name)
    

def logistic_reg_final(train_file,test_file,quarter,cnt):
    
    regress_name='LOGISTIC_REG_FIN'
    
    quart=quarter
    
    train=load_file(train_file,cnt)
    train=handle_missing(train)
    train=frame_transform(train)
    train=handle_dtypes(train)
    train=feature_engg(train)


    X_train=create_X(train)
    y_train=create_y(train)
    X_train=feature_encode(X_train)
    X_train=scale_minmax(X_train)
    y_train=scale_minmax(y_train)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    model=lr.fit(X_train,y_train)
    
    from sklearn.feature_selection import RFE
    rfe = RFE(model, 12)
    rfe = rfe.fit(X_train, y_train.ravel())
    # print summaries for the selection of attributes
    print("Number of features: " + str(rfe.n_features_))
    print(rfe.support_)
    print("Selected Features: " + str(rfe.ranking_))
    
    selected_columns=['mnth_rep_dt',
                  'loan_age','rem_month',
                  'zero_bal_cd',
                  'curr_int_rate',
                  'mi_recv','net_sales','non_mi_recv',
                  'expens','actual_loss','stp_modn_flg','def_pay_modn']
    

    train=load_file(train_file,cnt)
    train=handle_missing(train)
    train=frame_transform(train)
    train=handle_dtypes(train)
    train=feature_engg(train)


    X_train=create_X_feature_select(train,selected_columns)
    y_train=create_y(train)
    X_train=feature_encode_feature_select(X_train)
    X_train=scale_minmax(X_train)
    y_train=scale_minmax(y_train)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    fit=model.fit(X_train,y_train)

    score = fit.score(X_train, y_train)
    

    test=load_file(test_file,cnt)
    test=handle_missing(test)
    test=frame_transform(test)
    test=handle_dtypes(test)
    test=feature_engg(test)


    X_test=create_X_feature_select(test,selected_columns)
    y_test=create_y(test)
    X_test=feature_encode_feature_select(X_test)
    X_test=scale_minmax(X_test)
    y_test=scale_minmax(y_test)


    y_pred=model.predict(X_test)

    
    # For Train CM

    y_pred_train=model.predict(X_train)
    

    cm=model_metrics(model,X_train,X_test,y_train,y_test,y_pred,y_pred_train)
    
    
    
    gen_metric_file(y_test,cm,quart,regress_name)
    
    

def random_forest(train_file,test_file,quarter,cnt):
    
    regress_name='RANDOM_FOREST'
    
    quart=quarter
    
    train=load_file(train_file,cnt)
    train=handle_missing(train)
    train=frame_transform(train)
    train=handle_dtypes(train)
    train=feature_engg(train)

    selected_columns=['mnth_rep_dt',
                      'loan_age','rem_month',
                      'zero_bal_cd',
                      'curr_int_rate',
                      'mi_recv','net_sales','non_mi_recv',
                      'expens','actual_loss','stp_modn_flg','def_pay_modn']
    
    X_train=create_X_feature_select(train,selected_columns)
    y_train=create_y(train)
    X_train=feature_encode_feature_select(X_train)
    X_train=scale_minmax(X_train)
    y_train=scale_minmax(y_train)

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Classifier
    clf=RandomForestClassifier(n_estimators=70, random_state=2)

    # Train the model using the training sets
    model_rf=clf.fit(X_train,y_train)

    score = model_rf.score(X_train, y_train)
    print(score)

    test=load_file(test_file,cnt)
    test=handle_missing(test)
    test=frame_transform(test)
    test=handle_dtypes(test)
    test=feature_engg(test)


    X_test=create_X_feature_select(test,selected_columns)
    y_test=create_y(test)
    X_test=feature_encode_feature_select(X_test)
    X_test=scale_minmax(X_test)
    y_test=scale_minmax(y_test)


    y_pred=model_rf.predict(X_test)

    # For Train CM

    y_pred_train=model_rf.predict(X_train)
    
    cm=model_metrics(model_rf,X_train,X_test,y_train,y_test,y_pred,y_pred_train)

    
    gen_metric_file(y_test,cm,quart,regress_name)



def neural_network(train_file,test_file,quarter,cnt):
    
    regress_name='NEURAL_NETWORK'
    
    quart=quarter
    
    train=load_file(train_file,cnt)
    train=handle_missing(train)
    train=frame_transform(train)
    train=handle_dtypes(train)
    train=feature_engg(train)

    selected_columns=['mnth_rep_dt',
                      'loan_age','rem_month',
                      'zero_bal_cd',
                      'curr_int_rate',
                      'mi_recv','net_sales','non_mi_recv',
                      'expens','actual_loss','stp_modn_flg','def_pay_modn']

    X_train=create_X_feature_select(train,selected_columns)
    y_train=create_y(train)
    X_train=feature_encode_feature_select(X_train)
    X_train=scale_minmax(X_train)
    y_train=scale_minmax(y_train)

    # Import MLP Neural Classifier
    from sklearn.neural_network import MLPClassifier

    # Create a Classifier
    clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,30,10), random_state=1)

    # Train the model using the training sets
    model_neu=clf.fit(X_train,y_train)

    score = model_neu.score(X_train, y_train)
    print(score)

    test=load_file(test_file,cnt)
    test=handle_missing(test)
    test=frame_transform(test)
    test=handle_dtypes(test)
    test=feature_engg(test)


    X_test=create_X_feature_select(test,selected_columns)
    y_test=create_y(test)
    X_test=feature_encode_feature_select(X_test)
    X_test=scale_minmax(X_test)
    y_test=scale_minmax(y_test)


    y_pred=model_neu.predict(X_test)
    
    # For Train CM
    y_pred_train=model_neu.predict(X_train)
    
    cm=model_metrics(model_neu,X_train,X_test,y_train,y_test,y_pred,y_pred_train)


    gen_metric_file(y_test,cm,quart,regress_name)




# **************  Main Processing Starts  *******************************

# Take inputs

# train_year=input("Enter train year : ")
# test_year=input("Enter test year : ")
# user_id=input("Enter username : ")
# pass_word=input("Enter password : ")
# rec_count=input("Enter record count : ")




def main_run(trn_yr,test_yr,u_id,p_wd,r_cnt):
# Generating path
    train_year=trn_yr
    test_year=test_yr
    user_id=u_id
    pass_word=p_wd
    rec_count=r_cnt
    
    train_file='historical_data1_time_' + train_year + '.txt'
    test_file='historical_data1_time_' + test_year + '.txt'

    qtr_year=train_year[2:6]

    qtr_file_path=str(os.getcwd())+"/"+"qtr_files"


    train_file=qtr_file_path + "/" + train_file
    test_file=qtr_file_path + "/" + test_file

    cred=user_cred(user_id, pass_word)

    # Downloading files

    # download_files(cred, train_year, test_year)

    download_files(cred, train_year, test_year)

    train_flag=os.path.isfile(train_file)

    test_flag=os.path.isfile(test_file)

    if (train_flag == True) and (test_flag == True):
        print('***** Files present **************************')
        print('***** Running Logistic Regression Init *******')
        logistic_reg_initial(train_file, test_file, qtr_year,rec_count)
        print('***** Logistic Regression Init run complete **')
    else:
    
        print ("**** Files not present **********************")
    


    train_flag=os.path.isfile(train_file)

    test_flag=os.path.isfile(test_file)

    if (train_flag == True) and (test_flag == True):
        print('***** Files present **************************')
        print('***** Running Logistic Regression ************')
        logistic_reg_final(train_file, test_file, qtr_year,rec_count)
        print('***** Logistic Regression run complete *******')
    else:
    
        print ("**** Files not present **********************")
    


    train_flag=os.path.isfile(train_file)

    test_flag=os.path.isfile(test_file)

    if (train_flag == True) and (test_flag == True):
        print('***** Files present **************************')
        print('***** Random Forest Regression ************')
        random_forest(train_file, test_file, qtr_year,rec_count)
        print('***** Random Forest run complete *******')
    else:
    
        print ("**** Files not present **********************")
    
    

    train_flag=os.path.isfile(train_file)

    test_flag=os.path.isfile(test_file)

    if (train_flag == True) and (test_flag == True):
        print('***** Files present **************************')
        print('***** Running Neural Network ************')
        neural_network(train_file, test_file, qtr_year,rec_count)
        print('***** Neural Network run complete *******')
    else:
    
        print ("**** Files not present **********************")


        
        
Year_list=[1999,2000,2001,2002,2003,
           2004,2005,2006,2007,2008,
           2009,2010,2011,2012,2013,
           2014,2015,2016]

#Year_list=[1999,2000]

for year in Year_list:
    train_year="Q1"+str(year)
    test_year="Q2"+str(year)
    main_run(train_year,test_year,user_id,pass_word,rec_count)
    
    


    
    
    
    