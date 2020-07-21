# YouTube-Engagement-Prediction
Data Science project conducted to explore business value from a dataset containing Trending YouTube videos across a period of time.
The project defines Engagement of a YouTube video across time and approaches this as a regression problem.

prepare_features.py file is used for constructing additional features to the dataset.

full-optimized-pipeline notebook performs inference using Bayesian Optimization and XGBoost on the preprocessed dataset combining numeric features and vector format features obtained from video text information. The final model beat the baseline significantly.
