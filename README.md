# Neural Network Charity Analysis

## Analysis Overview
The analysis is for the nonprofit foundation, Alphabet Soup, to help analyze the impact of their donations by predicting which applicants will be successful if they receive funding. This analysis uses deep learning neural networks with Python's TensorFlow library to analyze the input data and make predictions.

## Resources
- **Data Source:** charity_data.csv
- **Software:** Python 3.7.13 (pandas, sklearn, and tensorflow libraries), Jupyter Notebook 6.4.8

## Results

#### Data Preprocessing
- The columns `EIN` and `NAME` are identification information and removed from the input data.
- The target is `IS_SUCCESSFUL`. It contains binary data indicating if an applicant successfully used Alaphabet Soup's donation. 
- The features are the following columns: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`.
- The categorical variables were encoded using `OneHotEncoder`.

#### Compiling, Training, and Evaluating the Model
**AlphabetSoupCharity.ipynb:** <br>
- This model is made of two hidden layers with 100 and 50 neurons respectively.
- The hidden layers' activation function is `ReLU` to speed up the training process. While the output layer's activation function is `Sigmoid` because the target is binary (see model below). 
<img width="1008" alt="Screen Shot 2022-10-15 at 6 50 19 PM" src="https://user-images.githubusercontent.com/106405775/196011826-ba314484-1723-4162-9bd6-59529fd52e4d.png">
- This model's accuracy was 73% and thus did not achieve the target model performance of 75% accuracy. 
<img width="632" alt="Screen Shot 2022-10-15 at 7 03 45 PM" src="https://user-images.githubusercontent.com/106405775/196011891-b612489b-11f6-4564-9189-02e4a6b8a400.png"> 

**AlphabetSoupCharity_Optimization.ipynb:** <br>
To achieve the targeted predictive accuracy, I optimized the model by doing the following: 
- Binning the `ASK_AMT` feature,
- Added an additional layer with 100, 50, 25 neurons respectively,
- Used the `linear` activation function on two hidden layers, and
- Added additional neurons to the two hidden layers (100 and 80 respectively). <br>

All four models failed to reach the target predictive accuracy of 75%. 

## Summary
While the initial model preformed best with 73% accuracy, all five attempts of the deep learning neural network models failed to achieve the targeted predictive accuracy of 75%. 

#### Recommendation
A supervised machine learning ensemble model, like `RandomForestClassifier`, may work better by randomly sampling the preprocessed data and building several smaller, simpler decision trees. 
