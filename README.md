---

* Technologies 

# Pandas
We used Pandas

*Importing the data.
*Creating Pandas objects with Pandas methods.
*Using Pandas methods on our dataframes.

---

```
# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
applicant_data_df = pd.read_csv(
    Path('applicants_data.csv'))

# Review the DataFrame

applicant_data_df.head(5)

# Drop the 'EIN' and 'NAME' columns from the DataFrame
applicant_data_df_droped = applicant_data_df.drop(columns=['EIN', 'NAME'])

# Create a list of categorical variables 
categorical = list(applicant_data_df_droped.dtypes[applicant_data_df_droped.dtypes=='object'].index)

```
---
https://pandas.pydata.org/
Package overview
pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis/manipulation tool available in any language. It is already well on its way toward this goal.

![Pandas](https://miro.medium.com/max/819/1*Dss7A8Z-M4x8LD9ccgw7pQ.png)

---
# TensorFlow 
We used TensorFlow  

*In creating statistical graphics 
*The heatmap to display correlations in our data. To plot detail 
*It help fortifine our inital ideas in visual form.
---
```
# Note that when using the delayed-build pattern (no input shape specified),
# the model gets built the first time you call `fit`, `eval`, or `predict`,
# or the first time you call the model on some input data.
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='sgd', loss='mse')

# This builds the model for the first time:
model.fit(x, y, batch_size=32, epochs=10)

```
---
https://www.tensorflow.org/

TensorFlow provides a collection of workflows to develop and train models using Python or JavaScript, and to easily deploy in the cloud, on-prem, in the browser, or on-device no matter what language you use. The tf. data API enables you to build complex input pipelines from simple, reusable pieces

![TensorFlow](https://www.tensorflow.org/site-assets/images/project-logos/tensorflow-quantum-logo-social.png)

---

# Scikit-learn
We used Scikit-learn 

*We used train_test_split to split our data as needed.
*We used StandardScaler to Standardize features by removing the mean and scaling to unit variance 
*We used transform to perform standardization by centering and scaling

---
```
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 1)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
X_scaler = scaler.fit(X_train)  

X_train_scaled = X_scaler.transform(X_train)

X_test_scaled = X_scaler.transform(X_test)

```
---


https://scikit-learn.org/stable/
Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python.

![Scikit](https://miro.medium.com/max/866/1*1ouD8HMkmJffNSAMfvBSkw.png)

---
