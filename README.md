# 🏠 Home Price Predictor (Linear Regression Project)

This project demonstrates how to build a **Linear Regression model** using `scikit-learn` to predict **house prices** based on multiple features such as area, number of bedrooms, and age of the property.

---

## 📂 Dataset

The dataset used is `homeprices.csv`, which contains the following columns:

* `area`: Size of the house in square feet
* `bedrooms`: Number of bedrooms (may contain missing values)
* `age`: Age of the property (in years)
* `price`: Price of the property (target variable)

---

## 📌 Features

* Handles missing values in the `bedrooms` column
* Splits the data into training and test sets
* Trains a **multiple linear regression** model
* Predicts house prices
* Visualizes model performance using scatter plots

---

## 🛠️ Technologies Used

* Python 🐍
* pandas
* numpy
* matplotlib
* scikit-learn

---

## 🚀 How It Works

### 1. **Load and Clean Data**

```python
import pandas as pd
from math import floor

df = pd.read_csv("homeprices.csv")
df.bedrooms = df.bedrooms.fillna(floor(df.bedrooms.median()))  # Handling missing values
```

### 2. **Prepare Data for Training**

```python
from sklearn.model_selection import train_test_split

x = df[["area", "bedrooms", "age"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 3. **Train the Model**

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)
```

### 4. **Make Predictions**

```python
y_predict = reg.predict(x_test)
```

### 5. **Visualize Predictions vs Actual Prices**

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_predict, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.grid(True)
plt.show()
```

---

## 📈 Sample Output

A scatter plot comparing the predicted and actual home prices will help you evaluate how well the model performs.

---

## 📄 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Contributing

Feel free to fork the repo and submit pull requests for:

* Improving model accuracy
* Adding more features
* Enhancing visualizations
