# nlp-news-sentiment-forecasting
A hybrid NLP-ML pipeline that extracts sentiment from economic news to predict inflation using Prophet and Linear Regression.

>  This is an educational project developed during my university studies,
> reflecting my hands-on experience.

In this repository, I document my hands-on experience applying natural language processing (NLP) and machine learning (ML) techniques. I started exploring sentiment analysis using NLP and moved into forecasting with ML models across various datasets. I focused on an economic dataset because of a clear pipeline for applying NLP and ML techniques.

The approach of integrating NLP review analysis with time series forecasting for economic datasets involves a multi-step process:
1. Data Collection and Preparation
2. NLP Preprocessing and Feature Extraction
3. Sentiment Analysis and Feature Engineering
4. Time Series Data Integration
5. Time Series Forecasting Model Development
6. Evaluation and Interpretation

### Step 1: Data Collection and Preparation

I began by gathering two datasets: news articles and corresponding economic indicators.

The first dataset contains over 12,000 news articles from news media. The raw data had several quality issues (duplicate entries and articles with missing body text), which I resolved through a cleaning.

The second dataset covers official inflation statistics. I used monthly consumer price index data from the statistical institute as the target variable for the forecasting pipeline.

### Step 2: NLP Preprocessing and Feature Extraction

I know that an NLP pipeline requires systematic text preprocessing. According to Tony Guida's *Big Data and Machine Learning in Quantitative Investment*, "any statisctical analysis begins with collecting data" followed by preprocessing to "clean the data and reduce noice where possible". 

I applied the following preprocessing steps:
* Used the re library to remove non-alphabetic characters and digits
* Used nltk to filter out stopwords
* Used simplemma for morphological analysis and lemmatization

Why `simplemma`? Turkish is an agglutinative language, which makes standard tools like `PorterStemmer` or basic `NLTK` stemmers ineffective. I chose simplemma because it provides high-accuracy lemmatization for Turkish (≈0.89 accuracy), allowing the model to correctly identify word roots.

Implementation Logic
```python
def preprocess_data(text):
  # Standardizing Turkish characters and lowering case
  text = str(text).replace("'", "").replace("’", "")
  text = text.replace('İ', 'i').replace('I', 'ı').lower()
  # Remove non-alphabetic characters
  text = re.sub(r'[^\w\s]', ' ', text)
  text = re.sub(r'\d+', ' ', text)

  text = text.split()
  # Lemmatization and stopword removal
  text = [simplemma.lemmatize(word, lang='tr') for word in text
                       if word not in set(stopwords.words('turkish'))]
  return " ".join(text)
```

### Step 3: Sentiment Analysis and Feature Engineering

I began by defining three sentiment labels for the news articles:
* -1 (Negative): News indicating rising inflation, currency depreciation, or economic instability;
* 0 (Neutral): Standard statistical reports or balanced economic news;
* 1 (Positive): News indicating inflation slowdown, investments, or strengthening local currency.

I manually labeled a control sample of 80 articles to train a Support Vector Machine (SVM) classifier, which then predicted sentiment scores across the entire dataset. The resulting sentiment score serves as a key engineered feature for the model.

**A note on `fit_transform()` vs `transform()`**

Through the project, I have gained a clear understanding of how to correctly use these two methods in a data pipeline:
* `fit_transform()` is used on training data — it learns the transformation parameters and applies them in one step, streamlining initial preprocessing
* `transform()` is used alone on test/validation data to prevent data leakage and ensure consistency across the pipeline

I applied `fit_transform()` to the entire corpus to capture patterns from the text data and build a transformed dataset, then used `transform()` on the manually labeled data to prepare it for the SVM classifier. This approach produced the best performance in identifying sentiment across the dataset.

### Step 4: Time Series Data Integration

At this step, I combined sentiment features with structured economic indicators. The integration process involved:
* Data Alignment: I standardized date formats in both datasets to the first day of each month;
* Temporal Aggregation: I aggregated individual news sentiment scores into a monthly sentiment index;
* Feature Merging: I combined the inflation data and sentiment index into a single dataframe.

### Step 5: Time Series Forecasting Model Development

Following the data integration step, I applied two models that incorporate the NLP-derived sentiment features:

1. Prophet model
2. Linear Regression

```python
# Feature Engineering for Linear Regression
df['y_lag'] = df['y'].shift(1) # Inflation value from the previous month
X = df[['y_lag', 'sentiment']]
y = df['y']
```

*Not. While Linear Regression is not a time series model, I included it intentionally as a baseline to highlight the difference in how each model utilizes the sentiment score.*

### Step 6: Evaluation and Interpretation

Since the dataset covers only two years, I evaluated the model on a 1-month horizon, using the officially published data for comparison. Producing reliable longer-term predictions would require a larger volume of data. The primary evaluation metric was Root Mean Square Error (RMSE).

*An interesting result came from the Linear Regression model, which achieved a highly accurate prediction with only a 0.51% RMSE error relative to the officially published inflation figure.*

## References

- Volgina, E. (2025). *Forecasting Inflation Using News Indices*. Russian Journal of Money and Finance, 84(1), pp. 26–59.
- Bravo, C., Maldonado, S., & Oskarsdottir, M. (2026). *Deep Learning in Banking*. Wiley.
- Guida, T. (2019). *Big Data and Machine Learning in Quantitative Investment*. Wiley.
- Noring, C., Jain, A., Fernandez, M., Mutlu, A., & Jaokar, A. (2024). *AI-Assisted Programming for Web and Machine Learning*. Packt Publishing.
- Hwang, Y. H., & Burtch, N. C. (2024). *Machine Learning and Generative AI for Marketing*. Packt Publishing.
- Bruce, P. C., Stephens, M. L., Shmueli, G., Anandamurthy, M., & Patel, N. R. (2023). *Machine Learning for Business Analytics* (2nd ed.). Wiley.
- Doloc, C. (2019). *Applications of Computational Intelligence in Data-Driven Trading*. Wiley.
