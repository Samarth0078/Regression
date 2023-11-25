# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Reading the dataset
df = pd.read_csv(r"C:\Users\HP\Dropbox\My PC (LAPTOP-LDEF9MDH)\Downloads\Data_Train.csv")

# """**2.Data Cleaning**"""

df.dropna(inplace=True)


# Convert the 'Date_of_Journey' column to datetime
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
# Extract the day and month components
df['Journey_Day'] = df['Date_of_Journey'].dt.day
df['Journey_Month'] = df['Date_of_Journey'].dt.month

# Drop the original 'Date_of_Journey' column
df =df.drop(['Date_of_Journey'], axis=1)

# Extracting Hours
df['dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
# Extracting Minutes
df['dep_minute'] = pd.to_datetime(df['Dep_Time']).dt.minute
# Now we drop Dep_time as it is of no use
df.drop(['Dep_Time'],axis=1,inplace=True)

# Extracting Hours
df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
# Extracting Minutes
df['Arrival_minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute
# Now we drop Dep_time as it is of no use
df.drop(['Arrival_Time'],axis=1,inplace=True)

# Extract Hours and Minutes from Duration
df['Duration_hours'] = df['Duration'].str.extract(r'(\d+)h').astype(float).fillna(0)
df['Duration_minutes'] = df['Duration'].str.extract(r'(\d+)m').astype(float).fillna(0)
# Drop the original 'Duration' column
df = df.drop('Duration', axis=1)

# df.head()

# **3.Exploratory Data Analysis(EDA)**"""

# Replace values in the 'Destination' column
df['Destination'] = df['Destination'].replace({'Delhi': 'New Delhi'})
df['Source'] = df['Source'].replace({'Delhi': 'New Delhi'})
# df



# """**Handling Categorical Data**"""

# Save the LabelEncoders during training
label_encoders = {}
for column in ['Airline', 'Source', 'Destination', 'Total_Stops','Additional_Info']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save the LabelEncoders
for column, le in label_encoders.items():
    with open(f'label_encoder_{column}.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)

# Route and Total Stops are related to each other
df.drop(['Route'],axis=1,inplace =True)

# """**5.Feature Selection**"""

#Split the dataset into features (X) and the target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Create and fit a StandardScaler to the training data
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

# """**5.Model Building and Evaluation**"""
# """**Hyper Parameter Tuning**"""
# Hyperparameter Tuning with GridSearchCV for Random Forest
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [25],
              'max_features': ['sqrt'],
              'min_samples_leaf': [1],
              'min_samples_split': [2],
              'n_estimators': [100]}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best Parameters
best_rf_model = grid_search.best_estimator_

# Predicting with the best model
y_pred_new = best_rf_model.predict(X_test)
print("R-squared after Hyperparameter Tuning:", r2_score(y_test, y_pred_new))

# Load yur pre-trained Random Forest model after hyperparameter tuning
joblib.dump(grid_search, 'tuned_random_forest_model.pkl')
# # Load the trained model
loaded_model = joblib.load('tuned_random_forest_model.pkl')
st.title("Flight Price Prediction App")

st.write("Enter the following details to predict the flight price:")

# Create input fields for user to enter data
airline = st.selectbox("Airline", ["Jet Airways","IndiGo","Air India","Multiple carriers","SpiceJet"
                                   ,"Vistara","Air Asia", "GoAir","Multiple carriers Premium economy"
                                   ,"Jet Airways Business","Vistara Premium economy","Trujet"])  # Replace with actual airline names
source = st.selectbox("Source", ['New Delhi','Kolkata','Banglore','Mumbai','Chennai'])  # Replace with actual source names
destination = st.selectbox("Destination", ['Cochin','Banglore','New Delhi','Hyderabad','Kolkata'])  # Replace with actual destination names
total_stops = st.selectbox("Total Stops", ['1 stop','non-stop','2 stops','3 stops','4 stops'])
additional_info=st.selectbox("Additional_Info",['No info','In-flight meal not included','No check-in baggage included ','1 Long layover','Change airports ','Business class','No Info','1 Short layover','Red-eye flight','2 Long layover '])

journey_day = st.number_input("Journey Day", min_value=1, max_value=31)
journey_month = st.number_input("Journey Month", min_value=3, max_value=6)
def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(https://images.unsplash.com/photo-1556388158-158ea5ccacbd?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D);
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()
#
#
# During prediction, load the LabelEncoders
label_encoders = {}
for column in ['Airline', 'Source', 'Destination', 'Total_Stops','Additional_Info']:
    with open(f'label_encoder_{column}.pkl', 'rb') as le_file:
        label_encoders[column] = pickle.load(le_file)
# Create a button to trigger the prediction
if st.button("Predict Price"):
    # Preprocess the input data (similar to what you did in your original code)
    # Handle unseen cases
    airline_encoder = label_encoders['Airline']
    source_encoder = label_encoders['Source']
    destination_encoder = label_encoders['Destination']
    total_stops_encoder = label_encoders['Total_Stops']
    additional_info_encoder = label_encoders['Additional_Info']

    airline = airline_encoder.transform([airline])[0]
    source = source_encoder.transform([source])[0]
    destination = destination_encoder.transform([destination])[0]
    total_stops = total_stops_encoder.transform([total_stops])[0]
    additional_info = additional_info_encoder.transform([additional_info])[0]

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Total_Stops': [total_stops],
        'Additional_Info':[additional_info],
        'Journey_Day': [journey_day],
        'Journey_Month': [journey_month]
    })
    columns_used_during_training=X.columns
    input_data=input_data.reindex(columns=columns_used_during_training,fill_value=0)


    # Scale the input data (using the same scaler you used in your original code)
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    predicted_price = loaded_model.predict(input_data_scaled)

    st.markdown("<h2 style='color: black;font-family: Roboto, sans-serif;'> ₹    " + str(round(predicted_price[0], 3)) + "</h2>", unsafe_allow_html=True)








