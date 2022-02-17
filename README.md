# Labor_relationship_predict
This data set comes from https://www.openml.org/d/246.
It has 1000,000 observations, 16 independent variables and 1 response variable. This data set describes the relationship between the employee and the employer, and the target has two values: good and bad.

In the EDA part, I performed a comprehensive exploratory data analysis. First of all, check missing values; convert some ordinal categorical variables to numerical variables; solve the imbalance data by downsample the training data set. Then, visualize all independent variables and check constant variables and outliers. Also, I performed statistical tests. For the categorical variables, I usex Pearson's Chi-squared test to test whether two variables(the independent variable and the response variable) are in fact dependent. For the numerical variables, I used the Wilcoxon rank sum test. Also, I ploted the normal Q-Q plot and calculated their
skewness and kurtosis to check whether they are normal distributions. Then I used a random forest model to perform variable importance, show correlation between all numerical variables, and drop the variable that has high correlation with others and also has low importance. Then, use VIF to check the multicollinearity.

In the HW2, I performed 5 individual classifiers: Logistic Regression, Naive Bayes, SVM, Decision Tree, and KNN. For each individual classifier, first, I determined the performance based on confusion matrix, accuracy, precision, recall (sensitivity), FPR (false positive rate), TNR(true negative rate, or specificity), F1, ROC plot, and AUC. Second, I used bootstrap resampling to create 10 training data sets and train 10 models, and estimate the bias and variance based on their prediction on the test data set.

In the HW3, I performed 3 ensemble methods: XGBoost, Random Forest, and Cross Validation. Compare their performances with the base model Decision Tree by confusion matrix, accuracy, precision, recall, specificity, F1, ROC plot, and AUC. Also, compare bias and variance of these models.
