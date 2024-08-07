## Project 1: British Airways Ticket Booking System

### Project Overview

This project focuses on predicting whether a customer will successfully book a ticket using various features related to customer behavior and interaction with the booking system.

### Dataset

- **Source:** The dataset for this project was downloaded from Kaggle via [The Forage Virtual Experience Program](https://www.theforage.com/simulations/british-airways/data-science-yqoz).
- **Features:** The dataset includes various attributes related to customer demographics, previous booking behaviors, and interaction with the ticket booking system.

### Tools & Libraries Used

- **Data Analysis:**
  - `Matplotlib` (Graphic plot)
  - `Groupby()`
  - `info()`
  - `duplicated()`
  - `isnull()`
  - `describe()`
- **Data Transformation:**
  - `OrdinalEncoder()` for transforming categorical variables into numerical values
- **Data Visualization:**
  - `seaborn.heatmap()` for plotting dataset correlations
- **Data Preprocessing:**
  - `sklearn MinMaxScaler` for data normalization
  - `imblearn.over_sampling Smote` for balancing the data
  - `sklearn.utils shuffle` for shuffling the data
  - `sklearn Train_test_split` for splitting the data into training and testing sets
- **Model Training:**
  - `GradientBoostingClassifier`
  - `KNeighborsClassifier`
  - `RandomForestClassifier`
- **Model Evaluation:**
  - `confusion_matrix`
  - `classification_report`
  - `precision_score`
  - `recall_score`
  - `f1_score`
  - `accuracy_score`

### Methodology

1. **Data Exploration:**
   - Conducted exploratory data analysis (EDA) to understand the dataset's structure and key patterns.
   - Identified and handled missing values, duplicates, and imbalances in the dataset.

2. **Feature Engineering:**
   - Converted categorical variables to numeric using `OrdinalEncoder`.
   - Visualized feature correlations with a heatmap to guide feature selection.

3. **Data Balancing and Normalization:**
   - Used `Smote` to address class imbalances and `MinMaxScaler` for normalizing data before model training.

4. **Model Training and Evaluation:**
   - Trained three different algorithms (`GradientBoostingClassifier`, `KNeighborsClassifier`, and `RandomForestClassifier`) to predict ticket booking success.
   - Evaluated model performance using confusion matrix, classification report, and various other metrics.

### Results

- Successfully built and evaluated models to predict ticket booking success with a high level of accuracy.
- The `RandomForestClassifier` showed the best performance across all metrics.

### Conclusion

This project provided valuable insights into customer booking behavior, helping to predict the likelihood of successful ticket purchases.

### Future Work

- Experiment with additional models and hyperparameter tuning to further improve performance.
- Consider implementing this model in a live ticket booking system to provide real-time predictions.

---

## Project 2: British Airways Customer Reviews Sentiment Analysis

### Project Overview

This project focuses on analyzing customer reviews to determine the sentiment (positive or negative) using natural language processing (NLP) techniques.

### Dataset

- **Source:** The dataset for this project was downloaded from Kaggle via [The Forage Virtual Experience Program](https://www.theforage.com/simulations/british-airways/data-science-yqoz).
- **Features:** The dataset contains customer reviews from the Airline Quality website, which includes text reviews, ratings, and other related data.

### Tools & Libraries Used

- **Web Scraping:**
  - `BeautifulSoup()` for extracting customer reviews from the Airline Quality website.
- **Data Analysis:**
  - `Matplotlib` (Graphic plot)
  - `WordCloud()` for visualizing the most common words in the reviews
  - `Counter()` for counting word frequency in the dataset
- **Data Transformation:**
  - `CountVectorizer()` for converting text documents to a matrix of token counts
- **Data Preprocessing:**
  - `sklearn Train_test_split` for splitting the data into training and testing sets
- **Model Training:**
  - `BernoulliNB` (from `sklearn.naive_bayes`) for training the sentiment analysis model
- **Model Evaluation:**
  - Tested the model's predictions using real user reviews

### Methodology

1. **Web Scraping:**
   - Extracted a large number of customer reviews using `BeautifulSoup` from the Airline Quality website.

2. **Data Preprocessing:**
   - Visualized the most frequent words in customer reviews using WordCloud and `Counter`.

3. **Text Processing:**
   - Converted the text reviews into numerical data using `CountVectorizer` to prepare for model training.

4. **Model Training and Evaluation:**
   - Trained the `BernoulliNB` model on the processed review data and tested predictions with user reviews.

### Results

- The model successfully identified the sentiment in customer reviews with a high level of accuracy.
- The analysis revealed key trends in customer sentiment towards British Airways services.

### Conclusion

This project demonstrated the use of NLP techniques for sentiment analysis, providing insights into customer satisfaction and areas for improvement.

### Future Work

- Explore more advanced NLP techniques such as LSTM or Transformer models to improve sentiment prediction accuracy.
- Apply the model to real-time review data for ongoing sentiment analysis.

