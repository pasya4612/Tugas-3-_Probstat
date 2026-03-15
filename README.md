# Naive Bayes Algorithm Implementation
**Group Number:** [12]

## Group Members
* **Muhammad Hilbran Akmal Abrar** - [5025241052]
* **Ary Pasya Fernanda** - [5025241053]
* **Anggota 3** - [NIM]

## 1. Explanation of Bayes' Theorem
Bayes' Theorem is a fundamental principle in probability theory used to determine the conditional probability of an event based on prior knowledge of conditions related to that event.

The mathematical formula is expressed as:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Components:**
* **$P(A|B)$ :** The probability of hypothesis $A$ being true given that evidence $B$ has occurred.
* **$P(B|A)$ :** The probability of evidence $B$ being observed given that hypothesis $A$ is true.
* **$P(A)$ :** The initial probability of hypothesis $A$ before seeing the evidence.
* **$P(B)$ :** The total probability of the evidence $B$ occurring.

---

# 2. Bayes Theory in Naive Bayes Algorithm

Naive Bayes is a classification algorithm based on Bayes Theorem with the assumption that all features are independent.

The algorithm calculates the probability of a data point belonging to each class and chooses the class with the highest probability.

Steps of Naive Bayes:

1. Calculate prior probability for each class.
2. Calculate likelihood of each feature for each class.
3. Multiply probabilities according to Bayes theorem.
4. Select the class with the highest posterior probability.

Because of the *independence assumption*, Naive Bayes becomes computationally efficient and works well for large datasets.

---

# 3. Dataset Used

Dataset used in this project: *Online Retail Dataset*

Dataset Source:  
https://archive.ics.uci.edu/dataset/352/online+retail

Dataset description:

The Online Retail dataset contains transactions occurring between 2010 and 2011 for a UK-based online retail store.

Important attributes include:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

For this project, the dataset was processed and used to classify purchasing behavior.

Example of dataset:

| Quantity | UnitPrice | Country | Label |
|---------|----------|--------|------|
| 6 | 2.55 | UK | Normal |
| 48 | 0.85 | UK | Bulk |
| 2 | 5.95 | France | Normal |

---

# 4. Implementation

The Naive Bayes algorithm was implemented using *Python* with the *Scikit-learn* library.

Main steps:

1. Load dataset
2. Data preprocessing
3. Train-test split
4. Model training
5. Prediction
6. Model evaluation

Example code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and label
X = data[['Quantity','UnitPrice']]
y = data['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)
