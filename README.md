#  Kyphosis Dataset - Decision Trees & Random Forest

This project explores the **Kyphosis dataset**, which contains information about spinal surgeries in children, and applies **Decision Tree** and **Random Forest classifiers** to predict the presence of **kyphosis** (a spinal deformity) after surgery.

---

## Project Structure

```
.
├── kyphosis.csv              # Dataset
├── main.py                   # Script (your code)
└── README.md                 # Project documentation
```

---

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

##  Dataset Overview

* **Target Variable**: `Kyphosis` (Yes/No)
* **Features**:

  * `Age` – Age of the patient at the time of surgery (in months)
  * `Number` – Number of vertebrae involved in the surgery
  * `Start` – The number of the first vertebra operated on

---

##  Exploratory Data Analysis (EDA)

A pairplot was generated to visualize the distribution of features with respect to the target variable:

```python
sns.pairplot(df, hue='Kyphosis')
```

This provides an overview of how **Age**, **Number**, and **Start** correlate with kyphosis outcomes.

---

##  Model Training & Evaluation

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
```

* Evaluated using **Confusion Matrix** and **Classification Report**.

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
```

* Uses an **ensemble of 200 decision trees** for improved accuracy and robustness.

---

##  Sample Results

**Decision Tree Output**

```
Confusion Matrix:
[[17  3]
 [ 2  2]]

Classification Report:
              precision    recall  f1-score   support
      No          0.89       0.85      0.87        20
     Yes          0.40       0.50      0.44         4
```

**Random Forest Output**

```
Confusion Matrix:
[[19  1]
 [ 2  2]]

Classification Report:
              precision    recall  f1-score   support
      No          0.90       0.95      0.92        20
     Yes          0.67       0.50      0.57         4
```

✔️ Random Forest performs **better overall** than a single Decision Tree, especially in reducing misclassifications.

---

##  How to Run

1. Place `kyphosis.csv` in the working directory.
2. Run the script:

   ```bash
   python main.py
   ```
3. The script outputs confusion matrices and classification reports for both models.

---

##  Future Improvements

* Apply **cross-validation** to get more robust results.
* Perform **feature importance analysis** (especially using Random Forest).
* Try **hyperparameter tuning** (e.g., tree depth, min\_samples\_split).
* Visualize decision trees with `graphviz` or `plot_tree`.

---

Would you like me to also **add code to visualize the trained Decision Tree** (so you can include a tree diagram in the README)?
