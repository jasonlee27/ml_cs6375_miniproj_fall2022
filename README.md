# ml_cs6375_miniproj_fall2022

# Google doc link

* Slides: https://docs.google.com/presentation/d/15_Pyh8NA429rfLNGUe9T8ih9-qHuHPmYxcsachdIJUI/edit?usp=sharing
* Doc: https://docs.google.com/document/d/10dKfPOjwR2aAfTR2q0Ur1K6BNMnV5U2uQgMQyJbbG-g/edit?usp=sharing

# Projects

## 1. onlineclass_survey:
### Task: predicting if student prefer studying online/in person (Classification problem)
### Questions for features used for data collection
1. What is your major?
2. What is your gender identity?
3. Which of the following best describes you?
4. Are you a domestic or international student?
5. What is your class standing?
6. How many hours do you study per week outside of regular class?
7. What is your primary location of study?
8. What is the average length of your commute?
9. What best describes your current living situation?
10. What is your average bedtime
11. How many student organizations are you involved with?
12. Are you currently concerned with catching / spreading COVID-19?
13. Which learning modality do you generally prefer? (Label to be predicted)
14. Where did you hear about this survey?

### ML models used
* Gradient boosting classifier
* Catboost classfier
* Xgboost classifier
* Random forest classifier
* Extra trees classifier
* Linear discriminant analysis
* Ridge classifier
* Decision tree classidier
* Naive bayes classifier
* Ada boost classifier
* K-Neighbors classifier
* SVM - RBF kernel
* Quadratic discriminant analysis
* Logistic regression

### How to run
At the project root directory ( `ml_cs6375_miniproj_fall2022/` ),
```bash
# install prerequisites
pip install -r requirements.txt

# run the experiment
./run_online.sh
```

### Results
The training/testing results are at `ml_cs6375_miniproj_fall2022/_results/online_survey` directory
In the directory, there are model evaluation files as following:
* `test_accuracy.csv`: model accuracy on test set for all models.
* `confusion_matrix.csv`: confusion matrix values for all models.
* `coef_importance_{model_name}_barplot.eps` or `feature_importance_{model_name}_barplot.eps`: barplot for feature importances for each {model_name} model.

## 2. Salary
### Task: prediction of the amount of professors annual salary in university (Regression Problem)

### Data Creation Procedure

1. Run `python google_scholar_scraper.py`. This code generates scholar.csv in the present directory. This file contains faculty names, citations5y, i10index5y, and hindex5y.

2. Run `python scraping_directory_data.py`. This code generates scraping_directory.csv in the present directory. This file contains faculty names, and department.

3. Run `python scraping_salary.py`. This code generates salary_data.csv in the present directory. This file contains faculty names, title and salary.

4. Run `python scraping_ratemyprof_data.py`. This code generates scraping_ratemyprof_data.csv in the present directory. This file contains faculty names,  overall_rating, rating_class and total_rating.

5. Use https://namesorts.com/api/ to generate and add gender feature for each name in salary.csv. Only keep name and actual prediction amongst features returned by the api.

6. Run `python formatting_salary_data.py`. This code generates salary_data_cleaned.csv in the present directory. This file contains faculty names, salary, and department.

7. Clean and standardize the name feature in each csv file to firstName and lastName to add in merging the files together.

8. Convert each of the four csv files listed above to pandas dataframes. Merge each csv file to salary_data_cleaned.csv using the pandas merge function: `salary_dataframe([other_dataframe], left_on=["firstName","lastName"])`.

9. Fill in missing values in salary_data_cleaned.csv for school using individual google searches.

10. Fill in missing values salary_data_cleaned.csv for department using google by going to each department's website.

11. Fill in remaining missing values in salary_data_cleaned.csv using average value of feature.


### Features Collected
1. Title
2. Salary (Target variable to be predicted)
3. Lname
4. Fname
5. citedby5y
6. hindex5y
7. i10index5y
8. Age
9. School
10. rating_class
11. total_ratings
12. overall_rating
13. total_courses
14. average_grade
15. percent_passing
16. gender

### ML models used
* Gradient boosting regressor
* Catboost regressor
* Xgboost regressor
* Random forest regressor
* Extra trees regressor
* Kernel ridge regressor - Linear kernel
* Decision tree regressor
* Ada boost regressor
* K-Neighbors regressor
* SVM - RBF kernel
* Linear regression

### How to run
At the project root directory ( `ml_cs6375_miniproj_fall2022/` ),
```bash
# install prerequisites
pip install -r requirements.txt

# run the experiment
./run_salary.sh
```

### Results
The training/testing results are at `ml_cs6375_miniproj_fall2022/_results/salary` directory
In the directory, there are model evaluation files as following:
* `test_accuracy.csv`: model accuracy on test set for all models.
* `coef_importance_{model_name}_barplot.eps` or `feature_importance_{model_name}_barplot.eps`: barplot for feature importances for each {model_name} model.
