import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, chi2_contingency, shapiro


df = pd.read_excel("DATASET LUVCKNOW.xlsx")
df = df.dropna(subset=['AQI'])
df.fillna(df.mean(numeric_only=True), inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
fig.tight_layout(pad=6.0)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
ax = axes[0, 0]
ax.plot(df['Date'], df['AQI'], color='green')
ax.set_title("AQI Trends Over Time in Lucknow")
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.tick_params(axis='x', rotation=45)

ax = axes[0, 1]
sns.heatmap(df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']].corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap of Pollutants")

ax = axes[1, 0]
labels = df['AQI_Bucket'].value_counts().index
data = df['AQI_Bucket'].value_counts().values
ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=140)
ax.set_title("Distribution of AQI Buckets")

ax = axes[1, 1]
sns.histplot(df['NO2'], bins=20, kde=True, color='purple', ax=ax)
ax.set_title("NO2 Concentration Distribution")
ax.set_xlabel("NO2")

plt.subplots_adjust(top=0.90)
plt.suptitle("AQI Analysis Graphs - Lucknow", fontsize=18)
plt.show()

print("Summary Statistics:\n", df.describe())
print("\nCorrelation Matrix:\n", df.select_dtypes(include='number').corr())

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
z_scores = np.abs((df[pollutants] - df[pollutants].mean()) / df[pollutants].std())
print("\nOutliers per pollutant (Z > 3):\n", (z_scores > 3).sum())

shapiro_stat, shapiro_p = shapiro(df['AQI'])
print(f"Shapiro-Wilk test for AQI: stat={shapiro_stat}, p={shapiro_p}")

good = df[df['AQI_Bucket'] == 'Good']['AQI']
poor = df[df['AQI_Bucket'] == 'Poor']['AQI']
t_stat, t_p = ttest_ind(good, poor, equal_var=False, nan_policy='omit')
print(f"T-test: stat={t_stat}, p={t_p}")

pm25_category = pd.cut(df['PM2.5'], bins=[0, 30, 60, 90, 120, 250], labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
contingency = pd.crosstab(pm25_category, df['AQI_Bucket'])
chi2, chi_p, _, _ = chi2_contingency(contingency)
print(f"Chi-squared Test: chi2={chi2}, p={chi_p}")

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
ml_df = df[['AQI'] + pollutants].copy()
ml_df = ml_df.dropna(axis=1, how='all')
features = ml_df.drop('AQI', axis=1)
labels = ml_df['AQI']
features = features.fillna(features.mean())
labels = labels.fillna(labels.mean())

if features.isnull().values.any():
    print("Error: Features still contain NaN values after cleaning.")
else:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Performance:")
    print("R-squared:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("""
Recent Trends in Data Science:
- Generative AI like GPT-4 and DALL·E help in generating synthetic data and simulations.
- These tools are being used in environmental modeling to forecast pollution using hypothetical scenarios.
""")
