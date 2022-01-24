# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:41:48 2022

@author: RUVINDI
"""
#Web Development
import streamlit as st

#Numerical Computation
import numpy as np

#Data frames 
import pandas as pd

#Simulate Real time data
import time

#For visualizations
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px

#To load Pickle files
import pickle

import os
#import base64
#from wordcloud import WordCloud, STOPWORD




#Load models for Loan Approval Predictions

#Random Forest classifier
L_RF_pickle_in = open("Models\Loan-RFclassifier.pkl","rb")
L_RF_classifier =pickle.load(L_RF_pickle_in)

#Decision Tree model
L_DT_pickle_in = open("Models\Loan-DTClassifier.pkl","rb")
L_DT_classifier=pickle.load(L_DT_pickle_in)

#KNN model
#L_KNN_pickle_in = open("Models\Cred-KNNClassifier.pkl","rb")
#L_KNN_classifier=pickle.load(L_KNN_pickle_in)






#Load models for High-Risk Identification

#Random Forest classifier
D_RF_pickle_in = open("Models\Cred-RFclassifier.pkl","rb")
D_RF_classifier =pickle.load(D_RF_pickle_in)

#Decision Tree model
D_DT_pickle_in = open("Models\Cred-DTClassifier.pkl","rb")
D_DT_classifier=pickle.load(D_DT_pickle_in)

#KNN model
D_KNN_pickle_in = open("Models\Cred-KNNClassifier.pkl","rb")
D_KNN_classifier=pickle.load(D_KNN_pickle_in)



# Main Method
def main():
    
    #Title
    st.title("ABC Bank Loan Approval System 🏦")
    st.markdown("## with Identification of the High-Risk Customers")
    
    #Side Bar
    st.sidebar.title("Automatic Loan Approval System 🏦")
    st.sidebar.markdown("Real Time Bank Data Analysis")
    activities = ["Home", "Loan Approval System", "Loan Default Prediction"]
    choice = st.sidebar.selectbox("Go To", activities)
    
    
    
    #Visualization of the data
    if choice == 'Home':
        #Reading the data from Source
        df = pd.read_csv("Risk_Prediction.csv")
        
        sentiment_count = df['default'].value_counts()
        #st.write(sentiment_count)
        sentiment_count = pd.DataFrame({'default':sentiment_count.index, 'Count':sentiment_count.values})

        st.title("Number Of customers according to Default count ")
        fig = px.bar(sentiment_count, x='default' , y='Count',color='Count',height=500)
        st.plotly_chart(fig) 
        
     
     
    
    #Loan Approval System
    if choice == 'Loan Approval System':
        st.info("Predicting whether the Loan can be Approved")
        
        #Inputs
        
        loan_amnt = st.text_input("Amount of the loan:")
        funded_amnt = st.text_input("Funded Amount:")
        funded_amnt_inv = st.text_input("Funded Amount Inv:")
        term = st.text_input("Term:")
        int_rate = st.text_input(" Int Rate:")
        installment = st.text_input("Installment Amount:")
        grade = st.text_input("Grade:")
        sub_grade = st.text_input("Sub-Grade:")
        home_ownership = st.text_input("Home ownership:")
        annual_inc = st.text_input("Annual Inc:")
        verification_status = st.text_input("Verification status:")
        dti = st.text_input("DTI:")
        fico_range_low = st.text_input("lower boundary of FICO at loan origination:")
        fico_range_high = st.text_input("upper boundary of FICO at loan origination:")
        inq_last_6mths = st.text_input("inquiries in past 6 months:")
        pub_rec = st.text_input("Number of derogatory public records:")
        revol_bal = st.text_input("Total credit revolving balance:")
        revol_util  = st.text_input("Revolving line utilization rate:")
        total_pymnt = st.text_input("Total Payment:")
        total_pymnt_inv = st.text_input("Payments received to date for portion of total amount funded by investors:")
        total_rec_prncp = st.text_input("Principal received to date:")
        total_rec_late_fee = st.text_input("Late fees received to date:")
        recoveries = st.text_input("post charge off gross recovery:")
        collection_recovery_fee = st.text_input("collection recovery fees:")
        last_pymnt_amnt = st.text_input("last payment amount:")
        last_fico_range_high = st.text_input("last FICO high range:")
        last_fico_range_low = st.text_input("Last FICO low range:")
        pub_rec_bankruptcies  = st.text_input("Bankruptcies:")
        ##Add the other feild
        
        #Feature list values
        feature_list1 = [loan_amnt,funded_amnt,funded_amnt_inv,term,int_rate,installment,grade,sub_grade,home_ownership,annual_inc,verification_status,dti,
 fico_range_low,fico_range_high,inq_last_6mths,pub_rec,revol_bal,revol_util,total_pymnt,total_pymnt_inv,total_rec_prncp,total_rec_late_fee,
         recoveries,collection_recovery_fee,last_pymnt_amnt,last_fico_range_high, last_fico_range_low, pub_rec_bankruptcies]
        single_sample = np.array(feature_list1).reshape(1,-1)
        
        #Selecting the model
        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        
        st.text("")
        
        #Predicting Using the models
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
               # prediction = 'a'
                prediction = L_RF_classifier.predict(single_sample)
             #  pred_prob = L_RF_classifier.predict_proba(single_sample)
            elif model_choice == "Decision Tree Classifier":
               prediction = L_DT_classifier.predict(single_sample)
        #      pred_prob = L_DT_classifier.predict_proba(single_sample)
        #    else:
        #      prediction = L_KNN_classifier.predict(single_sample)
            #    pred_prob = L_KNN_classifier.predict_proba(single_sample)
            # prediction = 'b'   
                
            #Displaying the Predicted Outcome
            if prediction == 'TRUE' :
                st.text("")
                st.warning("This Customer is a High-Risk Customer")
               
                		
            else:
                st.text("")
                st.success("This Customer is not a High-Risk Customer")
                
                
                
                
                

    #Identify the High-Risk Customers
    if choice == 'Loan Default Prediction':
        st.info("Predicting whether the customer defaults a loan")
        
        #Inputs
        
        loan_amount = st.text_input("Amount of the loan:")
        duration = st.text_input("Loan Duration:")
        payments = st.text_input("Last Payment amount:")
        order_amount = st.text_input("Ordered Amount:")
        n_inhabitants = st.text_input(" Number of Inhabitants:")
        average_salary = st.text_input("Average Salary of the Customer:")
        entrepreneur_rate = st.text_input("Entrepreneur Rate:")
        Day_between_account_creation_and_loan_application = st.text_input("Day between account creation and loan application:(ddmmyyyy)")
        average_unemployment_rate = st.text_input("Average Unemployment Rate:")
        average_crime_rate = st.text_input("Average Crime Rate:")
        
        
        #Feature list values
        feature_list = [loan_amount,duration,payments,order_amount,n_inhabitants,average_salary,entrepreneur_rate,Day_between_account_creation_and_loan_application,average_unemployment_rate,average_crime_rate]
        single_sample = np.array(feature_list).reshape(1,-1)
        
        #Selecting the model
        model_choice = st.selectbox("Select Model",["Random Forest Classification","Decision Tree Classifier", "KNN Classifier"])

        
        st.text("")
        
        #Predicting Using the models
	
        if st.button("Predict Outcome"):
            if model_choice == "Random Forest Classification":
          #      prediction = 'TRUE'
                prediction = D_RF_classifier.predict(single_sample)
                pred_prob = D_RF_classifier.predict_proba(single_sample)
            elif model_choice == "Decision Tree Classifier":
               prediction = D_DT_classifier.predict(single_sample)
           #     pred_prob = D_DT_classifier.predict_proba(single_sample)
        #    else:
        #        prediction = D_KNN_classifier.predict(single_sample)
            #   pred_prob = D_KNN_classifier.predict_proba(single_sample)
           #  prediction = 'FALSE'   
                
            #Displaying the Predicted Outcome
            if prediction == 'TRUE' :
                st.text("")
                st.warning("This Customer is will default the loan. Customer is high risk")
               
                		
            else:
                st.text("")
                st.success("This Customer will not default the loan. Customer is not a High-Risk Customer")
                       
    
                
               
if __name__ == '__main__':
	main()