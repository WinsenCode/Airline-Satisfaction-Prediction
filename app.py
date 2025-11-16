import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout = "wide", page_title = "Airline Satisfaction Prediction")

model = joblib.load("models/model_pipeline.pkl")

st.title("Airline Satisfaction Prediction")
st.markdown(" ")

st.header("Passenger and Flight Information")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.slider("Age", 0, 100, 30)

with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    flight_distance = st.number_input("Flight Distance (in Mile(s))", min_value = 0, max_value = 10000, value = 1000)
    
with col3:
    departure_delay = st.number_input("Departure Delay (in Minute(s))", 0, 2000, 0)
    arrival_delay = st.number_input("Arrival Delay (in Minute(s))", 0, 2000, 0)

st.header("Service and Facility Rating (1 - 5)")

rating_cols_1 = [
    "Inflight wifi service",
    "Food and drink",
    "Inflight entertainment",
    "Leg room service",
    "Online boarding",
    "Departure/Arrival time convenient",
    "Ease of Online booking"
]

rating_cols_2 = [
    "Seat comfort",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Gate location",
    "On-board service"
]

colA, colB = st.columns(2)

with colA:
    ratings1 = {col: st.slider(col, 1, 5, 3) for col in rating_cols_1}

with colB:
    ratings2 = {col: st.slider(col, 1, 5, 3) for col in rating_cols_2}

input_data = {
    "Gender": gender,
    "Customer Type": customer_type,
    "Age": age,
    "Type of Travel": travel_type,
    "Class": flight_class,
    "Flight Distance": flight_distance,
    "Departure Delay in Minutes": departure_delay,
    "Arrival Delay in Minutes": arrival_delay,
}

input_data.update(ratings1)
input_data.update(ratings2)

df_input = pd.DataFrame([input_data])

if st.button("Satisfaction Prediction"):
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.subheader("Prediction Result:")
    
    if prediction == 1:
        st.success(f"Customer is SATISFIED (Probability: {proba:.2f})")
    else:
        st.error(f"Customer is NOT SATISFIED (Satisfied probability: {proba:.2f})")

    st.write("Input Data:")
    st.dataframe(df_input)
