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


# Naive Bayes SMS Spam Classification

## Overview

Project ini mengimplementasikan algoritma **Naive Bayes** untuk melakukan klasifikasi pesan SMS menjadi dua kategori:

* **Ham** → pesan normal (bukan spam)
* **Spam** → pesan yang berisi promosi, penipuan, atau iklan tidak diinginkan

Model dilatih menggunakan dataset **SMS Spam Collection** dan dievaluasi menggunakan beberapa metrik evaluasi, salah satunya adalah **Confusion Matrix**.

---

# Confusion Matrix

Confusion Matrix adalah tabel evaluasi yang digunakan untuk melihat performa model klasifikasi dengan membandingkan **prediksi model** dengan **label sebenarnya**.

Struktur Confusion Matrix untuk klasifikasi biner adalah sebagai berikut:

| Actual / Predicted | Ham                 | Spam                |
| ------------------ | ------------------- | ------------------- |
| **Ham**            | True Negative (TN)  | False Positive (FP) |
| **Spam**           | False Negative (FN) | True Positive (TP)  |

---

# Penjelasan Setiap Komponen

### 1. True Positive (TP)

Model memprediksi **Spam**, dan pesan tersebut memang **Spam**.

Contoh:
Pesan promosi seperti:

```
Congratulations! You won a free ticket. Click here to claim.
```

Model berhasil mengenali pesan tersebut sebagai spam.

---

### 2. True Negative (TN)

Model memprediksi **Ham**, dan pesan tersebut memang **bukan spam**.

Contoh:

```
Hey, are we still meeting tonight?
```

Model berhasil mengklasifikasikan pesan normal dengan benar.

---

### 3. False Positive (FP)

Model memprediksi **Spam**, tetapi pesan sebenarnya **Ham**.

Contoh:

```
Don't forget to bring the documents tomorrow.
```

Pesan normal dianggap spam oleh model.
Kesalahan ini disebut juga **Type I Error**.

---

### 4. False Negative (FN)

Model memprediksi **Ham**, tetapi pesan sebenarnya **Spam**.

Contoh:

```
Win a brand new phone now! Limited offer!
```

Pesan spam tidak terdeteksi oleh model.
Kesalahan ini disebut **Type II Error**.

---

# Visualisasi Confusion Matrix

Confusion Matrix divisualisasikan menggunakan heatmap untuk mempermudah interpretasi.

```python
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])
```

Penjelasan visualisasi:

* Sumbu **X** → Predicted Label (hasil prediksi model)
* Sumbu **Y** → Actual Label (label sebenarnya)
* Angka pada setiap sel menunjukkan jumlah data pada kategori tersebut.

---

# Model Performance

Model Naive Bayes pada proyek ini menghasilkan:

* **Accuracy : 95.61%**

Artinya sekitar **95% pesan berhasil diklasifikasikan dengan benar** oleh model.

---

# Kesimpulan

Berdasarkan Confusion Matrix dan nilai akurasi, model **Naive Bayes** mampu melakukan klasifikasi SMS spam dengan performa yang sangat baik.

Namun masih terdapat kemungkinan kesalahan klasifikasi seperti:

* **False Positive** → pesan normal dianggap spam
* **False Negative** → pesan spam tidak terdeteksi

Evaluasi ini penting untuk memahami kelemahan model dan melakukan perbaikan di tahap selanjutnya.

---

# Tools & Libraries

Project ini menggunakan beberapa library Python berikut:

* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib

---

# Dataset

Dataset yang digunakan adalah:

**SMS Spam Collection Dataset**

Dataset ini berisi ribuan pesan SMS yang telah dilabeli sebagai **spam** atau **ham** untuk keperluan klasifikasi teks.
