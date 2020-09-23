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

<img src='' width='800' height='auto'></img>

My results are encouraging, as The final model not only performed better than other models in the next three-month forecast, but also performed stable in the sales forecast for the past year.



<!-- SECTION 1 -->
# EDA
## Raw data

<img src='' width='800' height='auto'></img>

raw data information:
date - every date of items sold
store - unique number of every shop(1-10)
item - unique number of every item(1-50)
sales - items sold on a particular day and a particular store
-4 years observed data, no missing values, it's good.



Scan the data infomation:

<img src='' width='800' height='auto'></img>

## Feature engineering
Before we dive into any statistical or machine learning methods for predicting future data, let's take a look at the data we already have. 

Extract some features from datetime, such as : 'year', 'month', 'day_of_week','weekend' etc.

<img src='' width='800' height='auto'></img>



## Data understanding through visualizations
#### Get some intuitive sense of the trends response to stores and items

<img src='' width='800' height='auto'></img>


According the scatter matrix, 

### Take a closer look at the data by navigate different frequency

<img src=''></img>
#### Yealy sales

<img src='' width='800' height='auto'></img>

#### Monthly sales

<img src='' width='800' height='auto'></img>

#### day of week sales

<img src='' width='800' height='auto'></img>

#### weekend sales

<img src='' width='800' height='auto'></img>

#### daily sales

<img src='' width='800' height='auto'></img>


<img src=''></img>


#### Trends comparison
<center class="half">
    <img src="" width="400"/><img src="" width="400"/>
</center>

#### Navigate average sales by store
<img src=''></img>

#### Navigate average sales by item

<img src=''></img>

Mean sales trend by item 

<img src=''></img>

#### Navigate "minimum_nights": required minimum nights stay
<center class="half">
    <img src="" width="300"/>
    <img src="" width="300"/>
    <img src="" width="300"/>
</center>

#### Navigate "name": listing name

<img src=''></img>


#### Closer look at "price" : listing price

<img src="" width='800' height='auto'/>

<center class="half">
    <img src="" height="250"/>
    <img src="" height="250"/>
</center>

#### There are just less than 0.5% listing price is greater than $1,000






<!-- SECTION 2 -->
# Feature engineering
### - Fill Nan values using the specified method.
        According to data.info, there are 4 cloumns missing some values.
        name,host_name,last_review and reviews_per_month
    
### - Convert categorical variables into numeric variables(0/1).
        According to data understanding, there are 3 important categorical features:
        neighbourhood_group,neighbourhood,room_type

### - Drop some columns that have a low correlation with "price".
     id, host_id, host_name ....

The processed data:

<img src="" width='800' height='auto'></img>




<!-- SECTION 3 -->
# Modeling
### Define models
- Try 4 regressor models

<img src="" width='800' height='auto'></img>   


- Evaluate models

<img src= width='800' height='auto'></img>

### Feature engineering again
- Drop outliers
    - price:
        The mean of price is 152.72, but the max price is 10000 that is not a reasonal price for me. 
        There are just less than 0.5% listing price is greater than $1,000.
        #### Drop the rows whose price is greater than 1000 and equal to 0

    - minimum_nights: required minimun nights stay
        The minimum night stay policy on Airbnb is the minimum number of nights that a guest can book a short-term vacation rental. 
        Short-term stays means less than 30 nights at a time.
        #### Replace the df['minimum_nights'] >30 with 30
- Drop low correlation columns
    - latitude
    - longitude
- Convert text variable into numeric variables
    - name -> neme_length
- Look at correlation again
###    Before  VS  After
<center class="half">
    <img src="" width="700"/>
    <img src="" width="700"/>
<center>



- Evaluate models again

The models performance have improved!!

<img src='' width='800' height='auto'></img>

### Try best hyperparameters


<img src="" width="600"/>
<img src="" width="600"/>
<img src="" width="600"/>


### create final model: Random Forest Regressor

<img src='' width='800' height='auto'></img>

### Evaluate model

<img src='' width='800' height='auto'></img>


<!-- SECTION 4 -->

# Bussiness insights

Feature Importance
<img src='' width='600' height='auto'></img>


### AS a guest:
    - When you travel to New York, living in Brooklyn is a good choice.
      Manhattan and Brooklyn are both close to New York’s commercial centers, and both have many listings.
      However, the average listing price in Brooklyn is 34% cheaper than those in Manhattan($118 VS S188).

    - If cost saving is your main consideration, you might choose a Private room in the room type. 
        Because the average listing price of an apartment is 2.3 times that of a private room($195 VS S85). 
        Moreover, although the average listing price of a shared room is cheapest, 
        there are fewer listings(1160 outof 48895 ), and only $17 cheaper than a Private room.


### AS a host:
    - When you can't change the area where the house is, you could 
      increase the listing price by increasing the total numbers of 
       days listing is avaliable out of 365.

    - Also, you could reduce the required minimun nights.

    - Describe your listing as specific as possible in the listing name, 
       which may help you attract customers. Such as room type, 
       neighborhood, business district name.







<!-- SECTION 5 -->
# Future Steps
Source: kaggle.com
Next, I would like to search some related dataset, such as: the ratings, the reviews, numbers of booking. And then analyze the key fetures for customers to choose a listing.







<!-- Another line -->
---
