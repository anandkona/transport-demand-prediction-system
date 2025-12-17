# Regression Transportation Demand Prediction

## Project Summary :

Increased rural-urban migration has drastically enhanced Nairobi's population in the recent years. The mobility needs of the population is met with mainly two types of services that are bus services and shuttle services. Mobiticket an online ticket booking service has planned to conduct an survey to estimate the number of tickets that will sold by their platform for the routes which start from one of the 14 towns and heading towards Nairobi

Using the data set created by the mobiticket I have checked for possible duplicate rows and null data points in data set. Using the feature ride id we have found the number of tickets booked on the time period which is the dependent variable. Also I have conducted an Exploratory Data Analysis to understand different features in the data set and carried out different analyses like most preferred time for travel, Number of tickets booked Vs Travel date, month wise booking analysis and weekday booking analysis and route wise booking analysis. Further, I have performed Feature Engineering by dropping unnecessary features and features that are highly correlated with each other. I have created features for time of travel, distance of travel and calculated the average speed of travel. Further, I have created a feature for waiting time for the next bus. Using the new updated data set I have conducted Feature scaling of the data using z score to deal with any outliers present in the data set and I have created a test train split of the data and implemented different regression like Linear Regression, Lasso, Ridge, Elasticnet, decision tree, gradient boosting, extreme gradient boosting and random forest regression and evaluated the performance metrics like mean squared error (MSE), Root Mean Squared error (RMSE), R2Score on train and test score to asses the performance of the regression algorithm and perform Bias- Variance Trade off and increase the performance of the regression.
After performing the performance metrics for all the regressions I have found that Random forest regression have performed the best in terms of Bias and variance when evaluated using R2 score with a R2 score on Train data with 63.46% and R2 score on test with 63.93%.

## Problem Statement

### Business Problem Overview

Ticket sales demand prediction plays a very important role in determining the pricing strategy for the logistics companies. Companies strategically plan to maximize the profits in the days with high demand in tickets and try to increase the sales on low demand days by offering promotions.

Also the companies try to predict the sales of the ride to sell other services more effectively such as micro travel insurances.

### Dataset Information

The dataset given is a dataset from Mobiticket, and we have to predict the number of tickets that are expected to be sold for each ride.

The above dataset has 51645 rows and 10 columns. There are no mising values and duplicate values in the dataset.

* **ride_id         :** Unique id for ride.
* **seat_number     :** Seat number of the ride.
* **payment_method  :** Categorical value for mode of payment. (Mpesa or cash)
* **payment_receipt :** Unique id for receipt of payment.
* **travel_date     :** Date of travel. 
* **travel_time     :** Starting time of travel.
* **travel_from     :** Ride starting location.
* **travel_to       :** Ride destination location.
* **car_type        :** Category variable for type of vehicle. (Bus or Shuttle)
* **max_capacity    :** Maximum capacity of the vehicle.

### Data Visualization and Understand the relationships between variables:

**Chart - 1 - Most prefered time of travel**

![image](https://user-images.githubusercontent.com/117559898/232096406-c5f3b080-83ff-42ba-b8d7-fa1cd00e589c.png)

From the above chart we have found that the most travelers book tickets to travel early in the morning from 5:00 to 8:00 AM. Knowing the most preferred time of travel helps the business to fix a pricing strategy with slightly higher prices to gain profits from the timings. Also it helps the company to increase the busses in the timings with maximum number of travelers

**Chart - 2 - Date of Journey Vs Ticket Count**

![image](https://user-images.githubusercontent.com/117559898/232096922-db38b89e-c637-4511-a967-e81702e24dc9.png)

The numbrer of bookings for the month of December is the highest. We can also see an increasing trend in booking from January to mid-March. January has the lowest number of tickets bookings.
Knowing the trends in bookings over time helps the business to plan over the time and plan for its pricing strategy. Especially in december pricing can be increased to make profits in the festival season. Also, from the chart we found January has the least bookings followed by an increasing trend in sales of ticket, from this we can say that promotions can be done in january month to increase present sales and attract customers from january to march.

**Chart - 3 - Weekday travel Vs Ticket Count**

![image](https://user-images.githubusercontent.com/117559898/232097433-74537321-f1c2-47cf-bd24-5a471e4809d4.png)

From the pie chart we can find that from saturday to tuesday we have less number of tickets booked with a least booking on saturday of 10.0%.

From wednesday to friday there are higher number of bookings with Friday being the highest of 16.8 %

From the graph we can clearly say that the working days have the highest number of bookings. Mobiticket can target to increase the sales from tuesday to Thursday to increase the revenue.

**Chart - 4 - Month wise booking Analysis**

![image](https://user-images.githubusercontent.com/117559898/232097733-266276f0-82b5-4d54-a218-7799774ce169.png)

From the bar chart we can find that Febuary,December and March has the highest number of bookings.

From the graph we can understand the intensity of demand in booking the tickets of each and every month. This helps in understanding the demand in those months and plan for pricing strategy.

**Chart - 5 - Starting location wise booking Analysis**

![image](https://user-images.githubusercontent.com/117559898/232097985-8d8b5f55-c1a8-4a17-a0ee-31f030192254.png)

From the above chart we found that kisii has the highest number of bookings. Also we can see that some of the routes have very low sales.

From the graphs we can see that the route starting from kisii has maximum number of sales. Also we can say that mobiticket is only having bookings in only kissi,kijauri and Rongo. There is a need of marketing in other locations to increase sales.

**Chart - 6 - Most preffered Vehicle**

![image](https://user-images.githubusercontent.com/117559898/232098311-c55bf7cb-279d-491e-9d22-bb52b7320f13.png)

From the pie chart we can find that both types of vehicles are equally contributing to the number of tickets booked. However shuttle services having a low capacity ( 11 seating capacity ) is contributing a good share in total sales.

**Chart - 7 - Correlation Heat Map**

![image](https://user-images.githubusercontent.com/117559898/232098497-66caeaa9-17fa-4e38-ab92-213e346665ef.png)

From the above correlation map we can say month and year is highly correlated with day of year. We can say that these two columns can be dropped beacause all the information present in month and year is stored in the day of month column. Also month and year are also highly correlated.

Also travel_from_distance, travel_from_time and speed are highly related because the follow the equation speed=distance/time.

## Feature Engineering & Data Pre-processing

### Handling Missing Values

![image](https://user-images.githubusercontent.com/117559898/232101192-fba25708-5bdf-4fcd-8509-2a55e5343dd9.png)

There are no missing values present in the dataset.

### Categorical Encoding

There are five categorial values however the information in the features are converted into othe features as follows

**weekday**: The days of week days (Monday to Sunday) are replaced with using numerical data 1 to 7.

**travel_date**: The information is extracted in the form of column "day of year"

**travel_from**: The information of the data is extracted in the form of distance because the distance from the source to destination is unique for all features. The essence of the data is fully extracted in this form due to high correlation between the features.

**travel_time**:Travel time is a date time feature and has a continuos valuable information. The values of hours is extracted from the data in the column "timeperiod"

## Feature Manipulation & Selection

![image](https://user-images.githubusercontent.com/117559898/232102267-f5226cd3-ef95-4115-9c96-3ede161ff31c.png)

I have droppeed Ride Id as the values in ride id is unique and doesnot have any useful information for the mmodel.

Also I have dropped year and month column because it is having high correlation with the feature day of year which is also a derived feature from the column travel_date.

Also the features travel_from_time and session is removed because travel_from_time is an object data type and the useful information is extracted to a new feature "time period". The feature session is created from time period for conducting statestical study so it shows high correlation with time period. So the feature session is removed.

![image](https://user-images.githubusercontent.com/117559898/232102468-39400c75-e12a-4aa3-8c4d-02086320d8d6.png)

From the above correlation graph we can say that travel_from_distance, timeperiod and speed are important features in predicting the model.

## Data Scaling

When you are using an algorithm that assumes your features have a similar range, you should use feature scaling.

If the ranges of your features differ much then you should use feature scaling. If the range does not vary a lot like one of them is between 0 and 2 and the other one is between -1 and 0.5 then you can leave them as it's. However, you should use feature scaling if the ranges are, for example, between -2 and 2 and between -100 and 100.

generally we use Standardization when your data follows Gaussian distribution and Normalization when your data does not follow Gaussian distribution.

So, in my data the data does not follow gaussian distribution so I have used a Normalization technique (Zscore) which is a good normalizing technique when there are outliers present in tht data set.

## Data Splitting

There are two competing concerns: with less training data, your parameter estimates have greater variance. With less testing data, your performance statistic will have greater variance. Broadly speaking you should be concerned with dividing data such that neither variance is too high, which is more to do with the absolute number of instances in each category rather than the percentage.

If you have a total of 100 instances, you're probably stuck with cross validation as no single split is going to give you satisfactory variance in your estimates. If you have 100,000 instances, it doesn't really matter whether you choose an 80:20 split or a 90:10 split (indeed you may choose to use less training data if your method is particularly computationally intensive).

You'd be surprised to find out that 80/20 is quite a commonly occurring ratio, often referred to as the Pareto principle. It's usually a safe bet if you use that ratio.

In this case the training dataset is small, that's why I have taken 70:30 ratio.

## ML Model Implementation

**Implementing Linear Regression**

I used Linear regression algorithm to create the model. As I got not so good result.

For training dataset, i found that the R2 score is 25.26% and for the test dataset,I found that R2 score is 28.76%. From the above R2 scores we can say that the model is highly biased and has low variance.

High bias in the model indicates that the the model is not able to understand the relation between predicted and actual. This can be visualized from the graph between predicted and actual.

The linear regression based algorithms are often used to reduce overfitting problem but our data is underfitting so we will use other algorithms.

**Implementing Lasso Regression**

I used Lasso regression algorithm to create the model. As I got not so good result.

For training dataset, i found that the R2 score is 25.13% and for the test dataset,I found that R2 score is 28.75%. From the above R2 scores we can say that the model is highly biased and has low variance.

High bias in the model indicates that the the model is not able to understand the relation between predicted and actual. This can be visualized from the graph between predicted and actual.

The lasso regression is a linear regression based algorithm and it is often used to reduce overfitting problem but our data is underfitting so we will use other algorithms.

**Implementing Ridge Regression**

I used Ridge regression algorithm to create the model. As I got not so good result.

For training dataset, i found that the R2 score is 25.25% and for the test dataset,I found that R2 score is 28.73%. From the above R2 scores we can say that the model is highly biased and has low variance.

High bias in the model indicates that the the model is not able to understand the relation between predicted and actual. This can be visualized from the graph between predicted and actual.

The Ridge regression is a linear regression based algorithm and it is often used to reduce overfitting problem but our data is underfitting so we will use other algorithms.

**Implementing Elasticnet Regression**

I used Elastic regression algorithm to create the model which is a hybrid of Lasso and Ridge. As I got not so good result.

For training dataset, i found that the R2 score is 25.06% and for the test dataset,I found that R2 score is 28.39%. From the above R2 scores we can say that the model is highly biased and has low variance.

High bias in the model indicates that the the model is not able to understand the relation between predicted and actual. This can be visualized from the graph between predicted and actual.

The ElasticNet regression is a linear regression based algorithm and it is often used to reduce overfitting problem but our data is underfitting so we will use other algorithms. So improve the R2 score I have used cross validation with hyper parameter tuning

**Implementing Elasticnet Regression with Cross Validation & Hyper parameter Tuning**

**Which hyperparameter optimization technique have you used and why?**

GridSearchCV which uses the Grid Search technique for finding the optimal hyperparameters to increase the model performance.

our goal should be to find the best hyperparameters values to get the perfect prediction results from our model. But the question arises, how to find these best sets of hyperparameters? One can try the Manual Search method, by using the hit and trial process and can find the best hyperparameters which would take huge time to build a single model.

For this reason, methods like Random Search, GridSearch were introduced. Grid Search uses a different combination of all the specified hyperparameters and their values and calculates the performance for each combination and selects the best value for the hyperparameters. This makes the processing time-consuming and expensive based on the number of hyperparameters involved.

In GridSearchCV, along with Grid Search, cross-validation is also performed. Cross-Validation is used while training the model.

That's why I have used GridsearCV method for hyperparameter optimization.

**Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.**

For training dataset, i found that the R2 score is 25.25% and for the test dataset,I found that R2 score is 28.73%. From the above R2 scores we can say that the model is highly biased and has low variance.

There is a minimal improvement in the R2 scores after cross validation and hyper parameter tuning. Because the Linear regression based algorithms perform better on datasets having more rows and columns. So, to improve the R2 scores we can use decission tree based algorithms.

**Decission Tree Regression**

![image](https://user-images.githubusercontent.com/117559898/232105985-05b49520-6883-4ed5-9a46-a0546d9182e5.png)

I have used Decission Tree regression algorithm to create the model. As I got not so good result on test set.

For training dataset, i found that the R2 score is 82.73% and for the test dataset,I found that R2 score is 17.20%. From the above R2 scores we can say that the model is low biased and has high variance.

High variance in the model indicates that the the model is has high variability in prediction if another data set is used for training.

The Decission tree regressior is a good learning algorithm which learns every data point in the data set including noise which makes it overfitting. In order to reduce overfitting problem we use boosting algorithms like Gradient Boosting and Extreme Gradient Boosting algorithms or use bagging algorithms like Random Forest.

**Gradient Boosing Regression**

I have used Gradient Boosting regression algorithm to create the model. I have got a good result in terms of bias and variance after applying this regressor.

For training dataset, i found that the R2 score is 51.85% and for the test dataset,I found that R2 score is 47.93%. From the above R2 scores we can say that the model is moderately biased and has low variance.

The model performed well using gradient boosting algorithm. However the bias can be reduced by using cross validation and hyper parameter tuning.

**Gradient Boosing Regression with Cross Validation and Hyper Parameter Tuning**

**Which hyperparameter optimization technique have you used and why?**
RandomSearchCV which uses the random Search technique for finding the optimal hyperparameters to increase the model performance.

our goal should be to find the best hyperparameters values to get the perfect prediction results from our model. But the question arises, how to find these best sets of hyperparameters? One can try the Manual Search method, by using the hit and trial process and can find the best hyperparameters which would take huge time to build a single model.

For this reason, methods like Random Search, GridSearch were introduced. Random Search uses some random combination of the specified hyperparameters and their values and calculates the performance for each combination and selects the best value for the hyperparameters. This makes the processing time-less consuming and less expensive based on the number of hyperparameters involved.

In RandomSearchCV, along with Random Search, cross-validation is also performed. Cross-Validation is used while training the model.

That's why I have used RandomsearhCV method for hyperparameter optimization.

**Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.**

After using Randomsearch with cross validation For training dataset, i found that the R2 score has increased from 51.85% to 53.19% (i.e approximately 2 percent increase in training data) and for the test dataset,I found that R2 score has increased from 47.93% to 48.52% (i.e, approximately 1 percent increase in test data). From the above R2 scores we can say that the model is moderately biased and has low variance.

The model performed well using gradient boosting algorithm with Cross validation and Hyper parameter tuning but we can try other algorithms to obtain better scores.

**Extreme Gradient Boosing Regression**

I have used Extreme Gradient Boosting regression algorithm to create the model. I have got a good result in terms of bias and variance after applying this regressor.

For training dataset, i found that the R2 score is 51.27% and for the test dataset,I found that R2 score is 47.96%. From the above R2 scores we can say that the model is moderately biased and has low variance.

The model performed well using Extreme gradient boosting algorithm. However the bias can be reduced by using cross validation and hyper parameter tuning.

**Extreme Gradient Boosing Regression with Cross Validation and Hyper Parameter Tuning**

**Which hyperparameter optimization technique have you used and why?**

GridSearchCV which uses the Grid Search technique for finding the optimal hyperparameters to increase the model performance.

our goal should be to find the best hyperparameters values to get the perfect prediction results from our model. But the question arises, how to find these best sets of hyperparameters? One can try the Manual Search method, by using the hit and trial process and can find the best hyperparameters which would take huge time to build a single model.

For this reason, methods like Random Search, GridSearch were introduced. Grid Search uses a different combination of all the specified hyperparameters and their values and calculates the performance for each combination and selects the best value for the hyperparameters. This makes the processing time-consuming and expensive based on the number of hyperparameters involved.

In GridSearchCV, along with Grid Search, cross-validation is also performed. Cross-Validation is used while training the model.

That's why I have used GridsearCV method for hyperparameter optimization.

**Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.**

After using Gridsearch with cross validation For training dataset, i found that the R2 score has increased from 51.27% to 56.61% (i.e approximately 5 percent increase in training data)and for the test dataset,I found that R2 score is 47.96% to 50.62% (i.e, approximately 3 percent increase in test data). From the above R2 scores we can say that the model is moderately biased and has low variance.

The model performed well using Extreme gradient boosting algorithm with Cross validation and Hyper parameter tuning but we can try bagging algorithms to obtain better scores.

**Random Forest Regression**

I have used Random Forest regression algorithm to create the model. I have got a good result in terms of bias and variance after applying this regressor.

For training dataset, i found that the R2 score is 45.37% and for the test dataset,I found that R2 score is 45.02%. From the above R2 scores we can say that the model is moderately biased and has low variance.

The model performed well using Random Forest Regressor algorithm. However the bias can be reduced by using cross validation and hyper parameter tuning.

**Random Forest Regression with Cross Validation and Hyper Parameter Tuning**

**Which hyperparameter optimization technique have you used and why?**

GridSearchCV which uses the Grid Search technique for finding the optimal hyperparameters to increase the model performance.

our goal should be to find the best hyperparameters values to get the perfect prediction results from our model. But the question arises, how to find these best sets of hyperparameters? One can try the Manual Search method, by using the hit and trial process and can find the best hyperparameters which would take huge time to build a single model.

For this reason, methods like Random Search, GridSearch were introduced. Grid Search uses a different combination of all the specified hyperparameters and their values and calculates the performance for each combination and selects the best value for the hyperparameters. This makes the processing time-consuming and expensive based on the number of hyperparameters involved.

In GridSearchCV, along with Grid Search, cross-validation is also performed. Cross-Validation is used while training the model.

That's why I have used GridsearCV method for hyperparameter optimization.

**Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.**

After using Gridsearch with cross validation For training dataset, i found that the R2 score has increased from 45.37% to 63.46 (i.e approximately percent increase in training data)and for the test dataset,I found that R2 score is 45.02% to 63.93 (i.e, approximately percent increase in test data). From the above R2 scores we can say that the model has low biased and has low variance.

The model performed well using Random Forest Regression algorithm with Cross validation and Hyper parameter tuning algorithms with better scores in terms of bias and variance.

**Choosing the best Model**

![df](https://user-images.githubusercontent.com/117559898/232105009-066d88d3-94be-438d-82de-469f2ed6c11d.PNG)

For Random Forest with Cross Validation the R2 score for train and test data are 61.1 and 61.5 respectively. This is found to be the best model. 

## Conclusion

Here are some solutions to increase the sales of tickets for the company:

* Provide periodical offers to attract customers.
* Focus more on early morning rides from 5:00 AM to 9:00AM.
* Improve services on all routes like kisii, Rongo and Kijauri which have high potential sales.
* Look at customers who travel on weekdays than on weekends.
* We can deploy the model with Random Forest Regressor with the parameters {'bootstrap': True,'max_depth': None,'max_features': 'sqrt','min_samples_leaf': 2,'min_samples_split': 12,'n_estimators': 400} with 3 fold cross validation. The model has predicted well at training. I found that for training data there is a R2 score of 63.46% which means the model has predicted well on the datapoints. Which means low bias. Also the model has a R2 score of 63.93%. The variance in model is also low. This is the best performing model I have found.
* There is no overfitting in data.
* Due to less no. of data in the dataset, the scores are around 60%. Once we get more data we can retrain our algorithm for better performance.




