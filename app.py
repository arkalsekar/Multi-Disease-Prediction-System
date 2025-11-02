import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import googlemaps
import pandas as pd
import folium
from streamlit_folium import folium_static

# Set page configuration
st.set_page_config(page_title="Multi Disease Prediction",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# Custom Header
custom_html = """
<div class="banner">
    <img src="https://planonsoftware.com/upload_mm/4/7/1/4719_fullimage_20210309-image-dataanalytics-magnifieroverview-high-3000x1987.jpg" alt="Banner Image">
</div>
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .banner {
        width: 160%;
        height: 400px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

kidney_model = pickle.load(open(f'{working_dir}/saved_models/kidney_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Kidney Disease Prediction',
                            'Heart Disease Prediction',
                            'Locate Nearby Hospitals'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'hospital'],
                           default_index=0)


# Kidney Prediction Page
if selected == 'Kidney Disease Prediction':
    st.title('Kidney Disease Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        bp = st.text_input('Blood Pressure (mm/Hg)', value='0')

    with col2:
        al = st.text_input('Albumin Level (0–5)', value='0')

    with col3:
        su = st.text_input('Sugar Level (0–5)', value='3')

    with col1:
        bu = st.text_input('Blood Urea (mg/dl)', value='30')

    with col2:
        sc = st.text_input('Serum Creatinine (mg/dl)', value='0.5')

    with col3:
        hemo = st.text_input('Hemoglobin (gms)', value='0')

    with col1:
        pcv = st.text_input('Packed Cell Volume', value='0')

    with col2:
        wc = st.text_input('White Blood Cell Count (cells/cumm)', value='0')

    with col3:
        rc = st.text_input('Red Blood Cell Count (millions/cmm)', value='0')

    # Code for prediction
    kidney_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Kidney Disease Test Result'):

        try:
            user_input = [bp, al, su, bu, sc, hemo, pcv, wc, rc]
            user_input = [float(x) for x in user_input]

            # Making prediction
            kidney_prediction = kidney_model.predict([user_input])

            if kidney_prediction[0] == 1:
                kidney_diagnosis = 'The person is likely to have Chronic Kidney Disease'
            else:
                kidney_diagnosis = 'The person is not likely to have Chronic Kidney Disease'

        except ValueError:
            kidney_diagnosis = 'Please enter valid numerical values in all fields.'

    st.success(kidney_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age', value=21)

    with col2:
        sex = st.text_input('Sex', 1)

    with col3:
        cp = st.text_input('Chest Pain types', value=2)

    with col1:
        trestbps = st.text_input('Resting Blood Pressure', 51)

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', 12)

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl', 23)

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results', 23)

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved', 68)

    with col3:
        exang = st.text_input('Exercise Induced Angina', 45)

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', 34)

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment', 98)

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy', 23)

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', 23)

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Find Nearby Hospital Page
if selected == "Locate Nearby Hospitals":

    # Google Maps API Key (Replace with your own key)
    API_KEY = "AIzaSyDeM_jxtlMkJDBNytaVtjp88Ti1uFeb9t4"

    # Initialize Google Maps client
    gmaps = googlemaps.Client(key=API_KEY)

    # Function to get user location
    def get_location():
        try:
            geocode_result = gmaps.geolocate()
            return geocode_result['location']
        except Exception as e:
            st.error(f"Error fetching location: {e}")
            return None

    # Function to find nearby hospitals related to diabetes
    def find_nearby_diabetes_hospitals(location, radius=5000):
        try:
            places_result = gmaps.places_nearby(
                location=location,
                radius=radius,  # Search radius in meters (5km)
                keyword=["heart hospital", "kidney hospital"],  # Search specifically for diabetes-related hospitals
                type="hospital"
            )
            return places_result.get("results", [])
        except Exception as e:
            st.error(f"Error fetching hospitals: {e}")
            return []

    # Function to create a Folium map
    def create_map(user_location, hospitals):
        m = folium.Map(location=[user_location['lat'], user_location['lng']], zoom_start=12)

        # Add marker for user location
        folium.Marker(
            [user_location['lat'], user_location['lng']], 
            popup="You are here",
            icon=folium.Icon(color="blue")
        ).add_to(m)

        # Add markers for hospitals
        for hospital in hospitals:
            name = hospital.get('name', 'Unknown')
            lat = hospital['geometry']['location']['lat']
            lng = hospital['geometry']['location']['lng']
            address = hospital.get('vicinity', 'No address available')

            folium.Marker(
                [lat, lng],
                popup=f"{name}<br>{address}",
                icon=folium.Icon(color="red", icon="plus-sign")
            ).add_to(m)

        return m



    # st.sidebar.page_link("pages/nearby_hospitals.py", label="Home")

    # Streamlit UI
    st.title("Finding Nearby Heart / Kidney Hospitals")
    st.write("We are currently searching for the Nearby Heart and Kidney Hospitals / Clinics")

    # Get user location
    user_location = get_location()
    if user_location:
        st.success(f"Your Location: {user_location}")

        # Find hospitals
        hospitals = find_nearby_diabetes_hospitals(user_location)

        if hospitals:

            st.write(f"Found :blue[{len(hospitals)}] Heart / Kidney hospitals near you:")
            df = pd.DataFrame([
                {"Name": h["name"], "Address": h.get("vicinity", "Unknown")}
                for h in hospitals
            ])
            st.dataframe(df)

            # Display the map
            folium_map = create_map(user_location, hospitals)
            folium_static(folium_map)
        else:
            st.warning("No Heart / Kidney hospitals found nearby.")
    else:
        st.error("Could not determine your location. Please check API permissions or network connections")

