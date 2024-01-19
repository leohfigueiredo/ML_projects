# Waiter Tips Prediction Project

## Introduction
This project aims to analyze and predict waiter tips based on various features such as total bill, gender, smoking status, day of the week, time of the meal, and party size. The dataset used for this analysis is available in the `tips.csv` file.

## Dataset
To get started, download the dataset using the following link:
```bash
!wget https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv
```

## Data Analysis and Visualization
We begin by importing the necessary Python libraries and loading the dataset using pandas:

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
data = pd.read_csv("tips.csv")
print(data.head())
```

### Scatter Plots
The project includes scatter plots to visualize tips based on different factors, such as total bill, number of people, and various categorical features:

- Tips vs. Total Bill and Day:
```python
figure = px.scatter(data_frame=data, x="total_bill", y="tip", size="size", color="day", trendline="ols")
figure.show()
```

- Tips vs. Total Bill and Gender:
```python
figure = px.scatter(data_frame=data, x="total_bill", y="tip", size="size", color="sex", trendline="ols")
figure.show()
```

- Tips vs. Total Bill and Meal Time:
```python
figure = px.scatter(data_frame=data, x="total_bill", y="tip", size="size", color="time", trendline="ols")
figure.show()
```

### Pie Charts
Additionally, the project includes pie charts to visualize the distribution of tips based on different categorical features:

- Tips distribution by Day:
```python
figure = px.pie(data, values='tip', names='day', hole=0.5)
figure.show()
```

- Tips distribution by Gender:
```python
figure = px.pie(data, values='tip', names='sex', hole=0.5)
figure.show()
```

- Tips distribution by Smoking Status:
```python
figure = px.pie(data, values='tip', names='smoker', hole=0.5)
figure.show()
```

- Tips distribution by Meal Time:
```python
figure = px.pie(data, values='tip', names='time', hole=0.5)
figure.show()
```

## Waiter Tips Prediction Model
The project includes a machine learning model to predict waiter tips based on the provided features. Categorical values are transformed into numerical values before splitting the data into training and test sets:

```python
# Transformation
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})

# Split the data
x = np.array(data[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Linear Regression Model
The project utilizes a Linear Regression model to predict tips:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain, ytrain)
```

### Model Testing
Finally, the performance of the model is tested by providing sample inputs:

```python
# Test the model
features = np.array([[24.50, 1, 0, 0, 1, 4]])
predicted_tip = model.predict(features)
print(predicted_tip)
```

This project serves as a comprehensive exploration of waiter tips analysis and prediction based on various factors. Feel free to explore and adapt the code for your own analysis or prediction tasks.
