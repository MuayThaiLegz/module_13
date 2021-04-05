---

* Technologies 

# Pandas
We used Pandas

Importing the data.
Creating dataframes with Pandas methods.
Using Pandas methods on our dataframes.
---
```
balance_sheet_tanglibles.groupby(level=0).plot.bar(
    title= ('Displays[balance_sheet_tanglibles] per Company'),
    figsize=(17,13),
    rot =360, 
    grid =True, 
    fontsize = 13)
```
---
https://pandas.pydata.org/
Package overview
pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis/manipulation tool available in any language. It is already well on its way toward this goal.

![Pandas](https://miro.medium.com/max/819/1*Dss7A8Z-M4x8LD9ccgw7pQ.png)

---
# TensorFlow 
We used TensorFlow  

In creating statistical graphics 
The heatmap to display correlations in our data. To plot detail 
It help fortifine our inital ideas in visual form.
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

We used StandardScaler to scale our data as needed.
We used PCA to decompose our data down in dimensions. 
We used Kmeans as our ML Algo.
---
```
# Initialize the K-Means model

second_model4 = KMeans(n_clusters=4, random_state=1)

# Fit the model

second_model4.fit(scaled_data_pca_data)

# Predict clusters
second_model4 = second_model4.predict(scaled_data_pca_data)

# View the resulting array
second_model4
```
---


https://scikit-learn.org/stable/
Scikit-learn provides a range of supervised and unsupervised learning algorithms via a consistent interface in Python.

![Scikit](https://miro.medium.com/max/866/1*1ouD8HMkmJffNSAMfvBSkw.png)

---
