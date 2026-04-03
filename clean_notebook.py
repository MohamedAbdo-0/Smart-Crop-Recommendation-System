import json

notebook = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('crop_data.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop('label', axis=1)\n",
    "y = df['label']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_predictions = rf_model.predict(X_test_scaled)\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_predictions)\n",
    "print(f\"Model Accuracy: {accuracy * 100:.2f} %\\n\")\n",
    "print(classification_report(y_test, y_predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_top_5_crops(soil_data_scaled, model):\n",
    "    probabilities = model.predict_proba(soil_data_scaled)[0]\n",
    "    crop_names = model.classes_\n",
    "    \n",
    "    results = pd.DataFrame({\n",
    "        'Crop': crop_names,\n",
    "        'Probability (%)': probabilities * 100\n",
    "    })\n",
    "    \n",
    "    top_5 = results.sort_values(by='Probability (%)', ascending=False).head(5)\n",
    "    top_5['Probability (%)'] = top_5['Probability (%)'].round(1).astype(str) + \" %\"\n",
    "    return top_5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "new_farm_perfect = pd.DataFrame(\n",
    "    [[104, 18, 50, 23.6, 60.3, 6.7, 140.9]], \n",
    "    columns=X.columns\n",
    ")\n",
    "\n",
    "new_farm_scaled = scaler.transform(new_farm_perfect)\n",
    "top_crops_table = get_top_5_crops(new_farm_scaled, rf_model)\n",
    "print(top_crops_table.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "crop_characteristics = df.groupby('label').mean().round(2)\n",
    "crop_characteristics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    user_N = float(input(\"Enter Nitrogen (N) content: \"))\n",
    "    user_P = float(input(\"Enter Phosphorus (P) content: \"))\n",
    "    user_K = float(input(\"Enter Potassium (K) content: \"))\n",
    "    user_temp = float(input(\"Enter Temperature (C): \"))\n",
    "    user_hum = float(input(\"Enter Humidity (%): \"))\n",
    "    user_pH = float(input(\"Enter pH level: \"))\n",
    "    user_rain = float(input(\"Enter Rainfall (mm): \"))\n",
    "\n",
    "    user_soil_data = pd.DataFrame(\n",
    "        [[user_N, user_P, user_K, user_temp, user_hum, user_pH, user_rain]],\n",
    "        columns=['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']\n",
    "    )\n",
    "\n",
    "    user_scaled = scaler.transform(user_soil_data)\n",
    "    \n",
    "    print(\"\\nResults:\")\n",
    "    top_recommendations = get_top_5_crops(user_scaled, rf_model)\n",
    "    print(top_recommendations.to_string(index=False))\n",
    "\n",
    "except ValueError:\n",
    "    print(\"Error: Please enter valid numbers only.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(rf_model, 'crop_model.pkl')\n",
    "joblib.dump(scaler, 'crop_scaler.pkl')\n",
    "print(\"Model and Scaler exported successfully!\\n\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('model_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
