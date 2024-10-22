import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

# Update header and styling
st.markdown("<h1 style='text-align: center; color: #2A9D8F;'>🌟 Diabetes Prediction App 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use this app to predict the likelihood of diabetes based on patient health data.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 3px solid #E9C46A;'>", unsafe_allow_html=True)

# Sidebar header styling
st.sidebar.header('🩺 Enter Patient Data')
st.sidebar.write("Please fill in the following details for a diabetes checkup:")

def get_user_input():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age (years)', min_value=21, max_value=88, value=33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

user_data = get_user_input()

# Updated subheader and styling
st.subheader('📝 Patient Data Summary')
st.write(user_data)

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

if st.button('Predict'):
    st.markdown("<h3 style='text-align: center;'>🔄 Prediction In Progress...</h3>", unsafe_allow_html=True)
    
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)
    
    prediction = rf.predict(user_data)
    
    st.subheader('🔍 Prediction Result:')
    result = 'You are not Diabetic' if prediction[0] == 0 else 'You are Diabetic'
    st.markdown(f"<h2 style='text-align: center; color: {'#2A9D8F' if prediction[0] == 0 else '#E76F51'};'>{result}</h2>", unsafe_allow_html=True)
    
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.subheader('📊 Model Accuracy:')
    st.write(f"{accuracy:.2f}%")
else:
    st.markdown("<h3 style='text-align: center;'>🖱️ Click 'Predict' to get the result</h3>", unsafe_allow_html=True)
