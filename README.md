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

## 3. Dataset Used

The dataset used in this project is the **SMS Spam Collection Dataset**, available on Kaggle:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Dataset Overview

The **SMS Spam Collection Dataset** is a public dataset containing **5,572 SMS messages in English**.  
Each message is labeled as either **spam** or **ham (legitimate message)**.

This dataset is widely used in **Natural Language Processing (NLP)** and **text classification tasks**, particularly for building machine learning models that detect spam messages.

---

### Dataset Structure

The dataset consists of **two main columns**:

| Column | Description |
|------|-------------|
| v1 | Label of the message (`spam` or `ham`) |
| v2 | The SMS message text |

Each row represents one SMS message.

---

### Example Data

| v1 | v2 |
|----|----|
| ham | Go until jurong point crazy.. Available only in Bugis n great world |
| spam | Free entry in 2 a wkly comp to win FA Cup final tkts |
| ham | Ok lar... Joking wif u oni |

- **ham** represents legitimate messages.
- **spam** represents unwanted promotional or scam messages.

---

### Dataset Size

- Total messages: **5,572**
- Total columns: **2**
- File format: **CSV / TXT**

---

### Class Distribution

The dataset is **imbalanced**, meaning legitimate messages appear more frequently than spam messages.

| Class | Count |
|------|------|
| Ham | 4,825 |
| Spam | 747 |

This means approximately:

- **86% Ham**
- **14% Spam**

---

### Data Preprocessing

Before applying the Naive Bayes algorithm, several preprocessing steps were performed:

1. **Text Cleaning**
   - Convert text to lowercase
   - Remove punctuation and special characters

2. **Tokenization**
   - Split text messages into individual words.

3. **Stopword Removal**
   - Remove common words such as *the, is, and*.

4. **Feature Extraction**
   - Convert text into numerical form using **Bag of Words** or **TF-IDF**.

After preprocessing, the dataset can be used to train a **Naive Bayes classifier** that predicts whether a message is spam or ham.
print("Accuracy:", accuracy)
