import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
import re
import matplotlib.pyplot as plt
import io
import base64

# === Load data and models ===
df = pd.read_csv("cleaned_data.csv", index_col=0, parse_dates=True)
device_cols = ['Dishwasher', 'Furnace_1', 'Furnace_2', 'Home_office', 'Fridge', 'Wine_cellar',
               'Garage_door', 'Kitchen_12', 'Kitchen_14', 'Kitchen_38', 'Barn', 'Well',
               'Microwave', 'Living_room']

lstm_model = load_model("lstm_model.h5")
arima_model = joblib.load("arima_model.pkl")
arima_preds = pd.read_csv("arima_predictions.csv", index_col=0, parse_dates=True)
arima_anomalies_thresh = pd.read_csv("arima_anomalies_threshold.csv", index_col=0, parse_dates=True)
arima_anomalies_conf = pd.read_csv("arima_anomalies_confidence.csv", index_col=0, parse_dates=True)
arima_conf_bounds = pd.read_csv("arima_confidence_bounds.csv", index_col=0, parse_dates=True)
sarimax_model = joblib.load("sarimax.pkl")

# === Auth ===
def authenticate(usn, password):
    pattern = r"^[1-5]RVU(21|22|23|24)(COM|DES|BBA)[0-9]{3}$"
    if re.match(pattern, usn):
        expected_password = f"RVU{usn[-3:]}"
        return password == expected_password
    return False

# === LSTM Prediction ===
def predict_lstm():
    sequence_length = 30
    data = df['use'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X_input = [data_scaled[i:i+sequence_length] for i in range(len(data_scaled) - sequence_length)]
    X_input = np.array(X_input)
    y_pred_scaled = lstm_model.predict(X_input)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    result = pd.DataFrame(y_pred[-30:], columns=['Predicted Use'])
    return result

# === ARIMA Plot ===
def plot_arima():
    plt.figure(figsize=(12, 4))
    plt.plot(df['use'], label='Actual', color='blue')
    plt.plot(arima_preds, label='ARIMA Prediction', color='red')
    plt.fill_between(arima_conf_bounds.index,
                     arima_conf_bounds.iloc[:, 0],
                     arima_conf_bounds.iloc[:, 1],
                     color='gray', alpha=0.3, label='Confidence')
    plt.scatter(arima_anomalies_thresh.index, arima_anomalies_thresh.values, color='black', s=10, label='Anomalies')
    plt.title("ARIMA Forecast & Anomalies")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# === SARIMAX Prediction ===
def predict_sarimax():
    pred = sarimax_model.get_prediction(start=0, end=len(df)-1)
    pred_mean = pred.predicted_mean
    return pred_mean.tail(30).to_frame("Predicted Use")

# === Device Usage ===
def get_device_usage():
    usage = df[device_cols].mean().sort_values(ascending=False)
    return usage.reset_index().rename(columns={"index": "Device", 0: "Average Usage"})

# === Gradio App ===
with gr.Blocks() as app:
    gr.Markdown("## ⚡ Wattson - Smart Home Energy Dashboard")
    gr.Markdown("**Login to access energy forecasts and device usage stats.**")

    with gr.Tab("🔐 Login"):
        usn = gr.Text(label="USN")
        password = gr.Text(label="Password", type="password")
        login_output = gr.Textbox(label="Login Status", interactive=False)
        login_btn = gr.Button("Login")

    with gr.Tab("📊 Dashboard") as main_tab:
        model_choice = gr.Radio(["LSTM", "ARIMA", "SARIMAX"], label="Choose Forecast Model")
        run_model_btn = gr.Button("Run Model")
        output_table = gr.Dataframe(label="Prediction Results")
        output_image = gr.Image(label="ARIMA Forecast Plot", visible=False)

        usage_btn = gr.Button("Show Device Usage")
        usage_out = gr.Dataframe(label="Device Avg Usage")

    def handle_login(usn_val, pw_val):
        if authenticate(usn_val, pw_val):
            return "✅ Login Successful!"
        return "❌ Invalid USN or Password"

    def handle_model_choice(choice):
        if choice == "LSTM":
            return predict_lstm(), None
        elif choice == "SARIMAX":
            return predict_sarimax(), None
        elif choice == "ARIMA":
            return None, plot_arima()
        return None, None

    login_btn.click(fn=handle_login, inputs=[usn, password], outputs=login_output)
    run_model_btn.click(fn=handle_model_choice, inputs=model_choice, outputs=[output_table, output_image])
    usage_btn.click(fn=get_device_usage, outputs=usage_out)

app.launch()
