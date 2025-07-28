import joblib
import streamlit as st
import pandas as pd


model = joblib.load('tuned_rf_diabetes.pkl')

# input_dict = {'age' : 20, 'sex' : 'male', 'bmi' : 20, 'children' : 2, 'smoker' : 'yes', 'region' : 'southwest'}
# input_df = pd.DataFrame([input_dict])
# predictions_df = predict_model(estimator=model, data=input_df)
# predictions = predictions_df.iloc[0]['prediction_label']
# st.markdown(predictions)

def predict(model, input_df):
    predictions = model.predict(input_df)[0]
    #prediction = int(prediction.iloc[0]['prediction_label'])
    return predictions

def run():

    # from PIL import Image
    # image = Image.open('logo.png')
    # image_hospital = Image.open('hospital.jpg')

    # st.image(image,use_column_width=False)

    

    st.title("Diabetes Prediction App")

    
    pregnancies = st.text_input("Pregnancies")
    glucose = st.text_input("Glucose")
    bloodpressure = st.text_input("BloodPressure")
    skinthickness = st.text_input("SkinThickness")
    insulin = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    dib_function = st.text_input("DiabetesPedigreeFunction")
    age = st.text_input("Age")
    
    

    output=""

    input_dict = {'Pregnancies' : pregnancies, 'Glucose' : glucose, 'BloodPressure' : bloodpressure, 'SkinThickness' : skinthickness, 'Insulin' : insulin, 'BMI' : bmi, 'DiabetesPedigreeFunction' : dib_function, 'Age' : age}
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = str(output)

    st.success('The output is {}'.format(output))

    

if __name__ == '__main__':
    run()


