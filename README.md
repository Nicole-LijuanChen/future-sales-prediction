<!-- HEADER SECTION -->

<div class='header'> 
<!-- Your header image here -->
<div class='headingImage' id='mainHeaderImage' align="center">
    <img src="https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/forecasting.jpg" width='1200' height='500' ></img>
</div>

<!-- Put your badges here, either for fun or for information -->

</br>

<!-- Brief Indented Explaination, you can choose what type you want -->
<!-- Type 1 -->
>Future Sales Prediction



<!-- TABLE OF CONTENTS SECTION -->
<!-- 
In page linkings are kind of weird and follow a specific format, it can be done in both markdown or HTML but I am sticking to markdown for this one as it is more readable. 

Example:
- [Title of Section](#title-of-section)
  - [Title of Nested Section](#title-of-nested-section)

## Title of Section

### Title of Nested Section

When linking section titles with spaces in between, you must use a '-' (dash) to indicate a space, and the reference link in parentheses must be lowercase. Formatting the actual title itself has to be in markdown as well. I suggest using two hashtags '##' to emphasize it is a section, leaving the largest heading (single #) for the project title. With nested titles, just keep going down in heading size (###, ####, ...)
-->

## Table of Contents

<!-- Overview Section -->

- [Overview](#overview)
  - [Background & Motivation](#context)
  - [Goal](#context)

<!-- Section 1 -->
- [EDA](#context)
    - [Raw data](#visualizations)
    - [Data understanding](#context)
    - [Data processing](#context)


<!-- Section 2 -->
- [Modeling](#visualizations)
    - [ARIMA](#visualizations)
    - [VAR](#visualizations)
    - [LSTM_500_output](#visualizations)
    - [LSTM_10_output](#visualizations)

<!-- Section 3 -->
- [Models Comparison](#visualizations)

<!-- Section 4 -->
- [Final Model](#context)

<!-- Section 5 -->
- [Conclusion](#context)


<!-- Section 6 -->
- [Reference](#context)




<!-- Optional Line -->
---



# Overview

## Background & Motivation
Forecasting plays an important role in many business plans and decisions. Such as: scheduling and planning the production depend on forecasting; one supermarket does or not increase the inventory of a specific product according to forecasting; transportation and storage costs are affected by the accuracy of forecasting. Even, the forecast of 3 to 5 years will affect whether a boss opens a new supermarket or builds a new production plant.


<img src='' width='800' height='auto'></img>



## Goal

In this project, I will develop a model to predict 3 months of item-level sales data at different store locations.
An accurate forecast for the next three months would greatly facilitate the management of retail stores. I hope that the model I built can be used by some retailer managers. The users can use the model I have trained with a simple operation.

## My results

<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/VAR_forecast_vs_Actuals_by_day.png' width='800' height='auto'></img>

My results are encouraging, as The final model not only performed better than other models in the next three-month forecast, but also performed stable in the sales forecast for the past year.



<!-- SECTION 1 -->
# EDA
## Raw data

<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/raw_data.png' width='600' height='auto'></img>

Raw data information:

date - every date of items sold

store - unique number of every shop(1-10)

item - unique number of every item(1-50)

sales - items sold on a particular day and a particular store

-4 years observed data, no missing values, it's good.




## Feature engineering
Before we dive into any statistical or machine learning methods for predicting future data, let's take a look at the data we already have. 

Extract some features from datetime, such as : 'year', 'month', 'day_of_week','weekend' etc.

<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/feature_engineering.png' width='600' height='auto'></img>



## Data understanding through visualizations
#### Get some intuitive sense of the trends response to stores and items

<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/correlation_matrix.png' width='600' height='auto'></img>
<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/correlation_matrix.png' width='600' height='auto'></img>

According the correlation matrix and sales trends plot, we could make some hypotheses for the sales pattern:
	1. There was an increase in the sales as the years pass by.
	2. Sales were affected by the month. Sales was higher in summer.
	3. Sales were more as compared to weekends.
    

### Take a closer look at the data by navigate different frequency

#### Yealy sales

<img src='' width='600' height='auto'></img>

#### Monthly sales

<img src='' width='600' height='auto'></img>

#### day of week sales

<img src='' width='600' height='auto'></img>

#### weekend sales

<img src='' width='600' height='auto'></img>

#### daily sales

<img src='' width='800' height='auto'></img>


<img src=''></img>


#### Navigate average sales by store

<center class="half">
    <img src="https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/Sales_distribution_for_stores.png" width="400"/>
    <img src="https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/Sales_trend_by_stores.png" width="400"/>
</center>

#### Navigate average sales by item

<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/Sales_distribution_for_items.png'></img>
<img src='https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/Sales_trend_by_items.png'></img>

The trends by item and store are similar. It looks good for model building.



#### Navigate "minimum_nights": required minimum nights stay
<center class="half">
    <img src="" width="400"/>
    <img src="" width="400"/>
</center>



<!-- SECTION 2 -->
# Modeling
### - ARIMA Model

      ARIMA is a popular and widely used statistical method for time series forecasting. But it is only used to predict univariate time series
      Pass
    
### - LSTM 500 output  Neural Networks Model

      <img src='' width='600' height='auto'></img>  
      There are a lot parameter could be tuning. 
      LSTM NN model is good at predict 1 or few output time series. 
      
      However, when the y labels increase to 500, the model performance is not so good. It shows a simple linear relationship.
### - LSTM 10 output Neural Networks Model 

      <img src='' width='600' height='auto'></img> 
      After exploring 500, 50, 20, 10 output LSTM NN models, the 10-output model is the best.
        
### - VAR Model
      
#### Vactor Autoregressions is useful for simultaneously modeling and analyzing multiple time series.
    <img src='' width='600' height='auto'></img> 
####The VAR class assumes that the passed time series are stationary. Non-stationary or trending data can often be transformed to be stationary by first-differencing or some other method. For direct analysis of non-stationary time series, a standard stable VAR(p) model is not appropriate(https://www.statsmodels.org/dev/_images/statsmodels-logo-v2-horizontal.svg).


<img src="" width='800' height='auto'></img>




<!-- SECTION 3 -->
# Models Comparison
### RMES by total forcasting (10 stores, 50 items, 3 months)

<img src="https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/3_models_RMSEs.png" width='600' height='auto'></img>   

### Performance by store-item

<img src="https://github.com/Nicole-LijuanChen/future-sales-prediction/blob/master/images/3_models_Forecast_VS_Actuals.png" width='800' height='auto'></img>

### Feature engineering again

###    Before  VS  After



<!-- SECTION 4 -->

# Final Model
### VAR Model

### Performance
    - Evaluate model By RMSE
    
    <img src='' width='600' height='auto'></img>
    <img src='' width='600' height='auto'></img>
    


Feature Importance
<img src='' width='600' height='auto'></img>
<center class="half">
    <img src="" width="400"/>
    <img src="" width="400"/>
<center>



<!-- SECTION 5 -->
# Reference

##### https://www.statsmodels.org/dev/vector_ar.html#var
##### https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
##### https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
##### https://www.kaggle.com/alexdance/store-item-combination-part-6-deep-learning
##### https://www.tensorflow.org/tutorials/structured_data/time_series






<!-- Another line -->
---
