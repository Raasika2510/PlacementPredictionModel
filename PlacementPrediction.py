#Importing all the required libraries
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV
from sklearn.metrics import roc_curve,ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay

#Mapping the model with deployment framework
st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    image = Image.open('image.png')
    st.markdown("<h1 style='text-align: center; color: black;'>CAMPUS PLACEMENT PREDICTION</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font_size : 10, color: black;'>Empowering Careers with ML</h2>",
                unsafe_allow_html=True)
    st.sidebar.title("PARAMETER MANIPULATION")
    st.sidebar.header("Select the parameters range for GridSearch :")
    est = st.sidebar.slider("Range of comparison for estimator (in 100's) :", min_value=1, max_value=5)
    md = st.sidebar.slider("Range of comparison for max_depth :", min_value=2, max_value=10)
    ss = st.sidebar.slider("Range of comparison for subsample ( in 0.X ):", min_value=3, max_value=9)
    lr = st.sidebar.slider("Range of comparison for learning rate ( in 0.0X ) :", min_value=1, max_value=3)
    gm = st.sidebar.slider("Range of comparison for gamma ( in 0.X ):", min_value=5, max_value=9)
    st.image(image,  caption="Forecast the Future of your Career")
    st.header ("ENTER THE FEATURE VARIABLES BELOW :")
    gender = st.radio("Pick your Gender :",['M','F'])
    age = st.radio("Pick your Age :",[19,20,21])
    ssp_t = st.radio("Pick your Senior Secondary Board of Education:",['Central','Others'])
    hsp_t = st.radio("Pick your Higher Senior Secondary Board of Education:",['Central','Others'])
    hsp_s = st.radio("Pick your Higher Senior Secondary Stream :",['Science','Arts','Commerce'])
    degree_t = st.radio("Pick the Stream of your Degree :",['Sci&Tech','Comm&Mgmt','Others'])
    volunteerings = st.radio("Are you volunteering for any clubs ?",['Yes','No'])
    workex = st.radio("Do you have any prior Work Experience ?",['Yes','No'])
    specialisation = st.radio ('Pick your specialization :',['Mkt&Fin','SDE','Jnr.DevOps','Mkt&HR','Analyst','WebTech'])
    ssp = st.number_input("Senior Secondary Score:", min_value= 0, max_value= 100)
    hsp = st.number_input("Higher Senior Secondary Score:", min_value= 0, max_value= 100)
    attendance = st.number_input("Attendance Percentage :",min_value=0,max_value=100)
    backlogs = st.slider("Number of Current Backlogs :",min_value=0,max_value=10)
    degree_p = st.number_input("Degree Overall Academic Percentage :",min_value=0,max_value=100)
    courses = st.slider("Number of skill courses undertaken :",min_value=1,max_value=10)
    interns = st.slider("Numer of Internships attended :",min_value=0,max_value=6)
    projects = st.slider("Number of projects done :",min_value=0,max_value=10)
    etest_p = st.number_input("Campus Placement Entrance Score :",min_value=0,max_value=100)
    st.text("Access the sidebar on the top left to select the parameter range and metrics for GridSearch and click Predict to view the predictions")
    st.subheader("RESULTS:")
    st.sidebar.header("METRIC VISUALIZATION :")
    metrics = st.sidebar.multiselect("Select the required Metrics :",
                                     ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button('PREDICT'):
            #st.subheader("RESULTS:")
            data = pd.read_csv("placement_data_3.csv")
            num = [[ssp, hsp, attendance, backlogs, age, degree_p, courses, interns, projects, etest_p]]
            cat1 = [[gender, ssp_t, hsp_t, hsp_s, degree_t, volunteerings, workex, specialisation]]
            df1 = pd.DataFrame(cat1,
                               columns=['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'volunteerings', 'workex',
                                        'specialisation'])
            df2 = pd.DataFrame(num, columns=['ssc_p', 'hsc_p', 'attendance', 'backlogs', 'age', 'degree_p', 'courses',
                                             'interns', 'projects', 'etest_p'])
            datax = pd.concat([df1, df2], axis='columns')
            datap = pd.concat([data,datax],axis='rows')
            datap.drop(['sl_no', 'salary'], axis=1, inplace=True)

            cat = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'volunteerings', 'workex', 'specialisation']
            df = datap[cat]
            datap.drop(cat, axis=1, inplace=True)
            datap['status'] = datap['status'].map({'Placed': 1, 'Not Placed': 0,None:0})
            enc = OneHotEncoder(sparse=False).fit(df)
            encoded = enc.transform(df)
            encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out())
            num_cols = ['ssc_p', 'hsc_p', 'attendance', 'backlogs', 'age', 'degree_p', 'courses', 'interns', 'projects',
                        'etest_p']
            scaler = StandardScaler()
            datap[num_cols] = scaler.fit_transform(datap[num_cols])
            dfe = datap.reset_index()
            #st.subheader("Your Scaled input data :")
            #st.dataframe(encoded_df.tail(1))
            #st.dataframe(dfe.tail(1))
            new_data = pd.concat([encoded_df, dfe], axis=1)
            idf = new_data.iloc[0:2000 , :]
            x = idf.drop('status', axis=1)
            y = idf['status']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
            #XGBoost model
            xgb = XGBClassifier(objective='binary:logistic', seed=42)
            k = 5
            kfc = KFold(n_splits=k, shuffle=True, random_state=42)
            estl = [i*100 for i in range(1,est+1) ]
            mdl = [i for i in range(2,md+1)]
            ssl = [i/10 for i in range(3 ,ss+1)]
            lrl = [i/100 for i in range(1 ,lr+1)]
            gml = [i/10 for i in range(5 ,gm+1)]
            param_gridc = {'n_estimators': estl,
                           'max_depth': mdl,
                           'subsample': ssl,
                           'learning_rate': lrl,
                           'gamma': gml
                           }
            grid_searchc = GridSearchCV(estimator=xgb,
                                        param_grid=param_gridc,
                                        cv=kfc, n_jobs=-1,
                                        verbose=4, scoring="accuracy")
            grid_searchc.fit(x_train, y_train, verbose=True,
                             early_stopping_rounds=8,
                             eval_metric='auc',
                             eval_set=[(x_test, y_test)])

            def plot_metrics(metrics_list):
                if "Confusion Matrix" in metrics_list:
                    st.subheader("Confusion Matrix")
                    ConfusionMatrixDisplay.from_estimator(grid_searchc, x_test, y_test)
                    st.pyplot()
                if "ROC Curve" in metrics_list:
                    st.subheader("ROC Curve")
                    RocCurveDisplay.from_estimator(grid_searchc, x_test, y_test)
                    st.pyplot()
                if "Precision-Recall Curve" in metrics_list:
                    st.subheader("Precision-Recall Curve")
                    PrecisionRecallDisplay.from_estimator(grid_searchc, x_test, y_test)
                    st.pyplot()

            plot_metrics(metrics)
            testing = new_data.tail(1)
            prediction = grid_searchc.predict(testing.drop('status',axis=1))
            probability = grid_searchc.predict_proba(testing.drop('status', axis=1))
            #st.markdown("<h1 style='text-align: center; color: black;'>RESULTS :</h1>",
                     #   unsafe_allow_html=True)
            for i in prediction:
                if i == 1:
                    pred = "Likely to be placed"
                    prob = probability[: ,-1:]
                    pc  = round(float(prob*100),2)
                    s = str(pc)+" %"
                    st.metric(label="Prediction", value=pred, delta=s)
                    #st.markdown("<h1 style='text-align: center; color: green;'>Congrats! You are most likely to be placed!</h1>",
                              #  unsafe_allow_html=True)
                    #st.markdown("<h2 style='text-align: center; color: green;'>Prediction Probability: You are most likely to be placed!</h1>",
                                #unsafe_allow_html=True)
                    #st.text(probability)
                else:
                    pred = "Not likely to be placed"
                    prob = probability[:,:1]
                    pc = round(float(prob*100), 2)
                    s = str(pc) + " %"
                    st.metric(label="Prediction", value=pred, delta=s, delta_color= "inverse")
                    #st.markdown("<h1 style='text-align: center; color: red;'>Oops! You are not likely to be placed! Push a little harder and you'll get it right next time!</h1>",
                    #unsafe_allow_html=True)


if __name__ == "__main__":
    main()
