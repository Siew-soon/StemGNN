import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV files
actual_file_path = "dataset/target(w).csv"
predicted_w_file_path = "dataset/predict(w).csv"
predicted_wo_file_path = "dataset/predict(wo).csv"
header_path = "output\EURUSD60_OHLC_MACD_MA10_MA20\data_centric_approach.csv"

df_actual = pd.read_csv(actual_file_path, header=None)
df_predicted_w = pd.read_csv(predicted_w_file_path, header=None)
df_predicted_wo = pd.read_csv(predicted_wo_file_path, header=None)
df_header = pd.read_csv(header_path)

# Initialize lists to store MAPE for each feature
mape_w_list = []
mape_wo_list = []

# Calculate MAPE for 'with' dataset
for i in range(df_predicted_w.shape[1]):
    y_true = df_actual.iloc[:, i]
    y_pred_w = df_predicted_w.iloc[:, i]

    # Calculate errors
    errors_w = y_pred_w - y_true

    # Calculate MAPE
    mape_w = np.mean(np.abs(errors_w / y_true)) * 100

    # Append the MAPE to the list
    mape_w_list.append(mape_w)

# Calculate MAPE for 'without' dataset
for i in range(df_predicted_wo.shape[1]):
    y_true = df_actual.iloc[:, i]
    y_pred_wo = df_predicted_wo.iloc[:, i]

    # Calculate errors
    errors_wo = y_pred_wo - y_true

    # Calculate MAPE
    mape_wo = np.mean(np.abs(errors_wo / y_true)) * 100

    # Append the MAPE to the list
    mape_wo_list.append(mape_wo)

# Generate feature names
feature_names_w = [df_header.columns[i] for i in range(df_predicted_w.shape[1])]
feature_names_wo = [df_header.columns[i] for i in range(df_predicted_wo.shape[1])]
index_w = np.arange(len(feature_names_w))
index_wo = np.arange(len(feature_names_wo))

# Plotting the bar graph for MAPE (with features)
plt.figure(figsize=(15, 6))
plt.bar(
    index_w - 0.2,
    mape_w_list,
    0.4,
    label="Original and Data Centric Approach Features",
    color="b",
)
plt.bar(index_wo + 0.2, mape_wo_list, 0.4, label="Original Features", color="r")

plt.xlabel("Features")
plt.ylabel("MAPE (%)")
plt.title("MAPE (Original vs Data Centric Features)")
plt.xticks(index_w, feature_names_w, rotation=90)
plt.legend()
plt.tight_layout()

plt.show()

# Filter features with MAPE higher than 50%
filtered_feature_names_w = [
    feature_names_w[i] for i in range(len(mape_w_list)) if mape_w_list[i] > 50
]
filtered_mape_w_list = [
    mape_w_list[i] for i in range(len(mape_w_list)) if mape_w_list[i] > 50
]

filtered_feature_names_wo = [
    feature_names_wo[i] for i in range(len(mape_wo_list)) if mape_wo_list[i] > 50
]
filtered_mape_wo_list = [
    mape_wo_list[i] for i in range(len(mape_wo_list)) if mape_wo_list[i] > 50
]

# Plotting the bar graph for MAPE higher than 50%
index_filtered_w = np.arange(len(filtered_feature_names_w))
index_filtered_wo = np.arange(len(filtered_feature_names_wo))

plt.figure(figsize=(15, 6))
plt.bar(
    index_filtered_w - 0.2,
    filtered_mape_w_list,
    0.4,
    label="Original and Data Centric Approach Features",
    color="b",
)
plt.bar(
    index_filtered_wo + 0.2,
    filtered_mape_wo_list,
    0.4,
    label="Original Features",
    color="r",
)

plt.xlabel("Features")
plt.ylabel("MAPE (%)")
plt.title("MAPE (Original vs Data Centric Features) - MAPE > 50%")
plt.xticks(index_filtered_w, filtered_feature_names_w, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# Calculate the mean of each feature
mean_w_list = df_predicted_w.mean().tolist()
mean_wo_list = df_predicted_wo.mean().tolist()

# Scatter plot for mean values and MAPE
plt.figure(figsize=(15, 6))

# Plot scatter for 'with' data-centric approach
plt.scatter(
    mean_w_list,
    mape_w_list,
    color="b",
    alpha=0.6,
    s=100,
    label="Original and Data Centric Approach Features",
)

# Plot scatter for 'without' data-centric approach
plt.scatter(
    mean_wo_list,
    mape_wo_list,
    color="r",
    alpha=0.6,
    s=100,
    label="Original Features",
)

plt.xlabel("Mean Value")
plt.ylabel("MAPE (%)")
plt.title("Scatter Plot of Mean Values vs MAPE (Original vs Data Centric Features)")
plt.legend()
plt.tight_layout()
plt.show()
