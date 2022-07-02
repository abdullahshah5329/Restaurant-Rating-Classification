# Restaurant-Rating-Classification

## Abstract:  
For restaurants, the rating is the most important indicator. It not only reveals the quality of food and services of the restaurants but also helps to attract more customers. In this project, we focus on predicting the rating of restaurants based on their features. With data from Yelp, we use several machine learning models such as Logistic Regression, Decision Tree Classification, and Random Forest Classification to make relevant predictions. Random Forest Classification was the best performing model with an accuracy of 69.3%, followed by Decision Tree Classification with 67.6 % accuracy and Logistic Regression with an accuracy of 65%.

I.	INTRODUCTION & MOTIVATION

Most of us have probably pondered where to eat when hanging out, especially when visiting somewhere unfamiliar. Although it is fun to explore a strange suburb, town, or city, most of us will not feel happy if we visit a new restaurant and realize the food and services are not as good as we expected, or in a worse case, it is not even worth going there. 

With the development of technology nowadays, so many apps have been developed to provide restaurant ratings to the customers. Yelp is one of the most popular apps. As an excellent platform for choosing a restaurant, Yelp allows people to get a holistic view of the restaurant based on its basic information, pictures, reviews, and so on [5]. Thus, most of us nowadays often go to Yelp to check for the rating and reviews of a restaurant before deciding to go and check in that restaurant. Although Yelp is famous and is one of the largest sources for restaurant reviews, not all the restaurants are listed on Yelp, or even if the restaurant is listed, not all of them have reviews yet. What if we find a brand-new local restaurant without any review on Yelp or other restaurant rating platforms, should we take a risk to visit that restaurant, or should we switch to the other similar restaurants that already have a review on Yelp? It is hard to decide. 

With that emerging problem, in this project, we would like to build a restaurant rating predictor system to predict the restaurant's rating on Yelp based on the restaurant features such as price level, available services, opening hours, etc. This project aims to provide the customers interested in visiting a restaurant a tool to check the restaurant rating and give the business and the restaurant owner a tool to predict their restaurant rating and a suggestion on what features should be applied when applied to open a new business. 


II.	RELATED WORKS

The Yelp dataset has been a valuable resource for predicting a restaurant star rating. There are so many past research and projects that tried to use the Yelp dataset to predict the ratings of the restaurants. For example, Kong, Nguyen, and Xu [1] used the Yelp dataset to classify restaurants based on cultural categories and analyzed international restaurants' success. Several other previous papers focused on sentiment analysis with text content from Yelp, such as Xu, Wu, and Wang [2] combined the customer reviews and ratings to conduct sentiment analysis. Gingerich and Bochkov [3] mainly used matrix factorization to analyze text information and predict Yelp ratings. Linshi [4] worked on user-based text analysis on Yelp rating prediction. Guo, Lu, and Wang [5] applied user-based analysis to predict ratings and provide suggestions to Yelp restaurants. So far, most of the papers published on the internet use text features from customer reviews to predict the ratings. We decided to do it differently in this project by using non-text features to predict the restaurant ratings.


III.	DATASET

We decided to use the Yelp dataset for this project. The Yelp dataset folder has five different json files:
    
●	business.json contains business data, including location data, stars, review_count,  attributes, categories, open hours

●	review.json contains full review text data, the user_id who wrote that review, and the business_id the review is written for.

●	user.json contains all the metadata associated with the user (or customer), including user's friend

●	checkin.json contains data about the date the user check-ins on a business

●	tip.json contains data about the tips written by a user on a business
   
In this project, we only use Business.json. Initially, the business.json dataset contained 150,346 records with 114,117 different businesses. Because our project focuses on predicting restaurant ratings, we eliminated the businesses that are closed and non-restaurants or non-food-related. This left us with 28740 restaurants in 16 different states.


IV.	PREPROCESSING & FEATURE SELECTIONS

We ran a Python script to convert json to a csv file. Because the ‘categories’ column was displayed as a list (each restaurant has a list of different categories it falls into), we transformed each element of the list-like into a row. Similarly, we split and transformed each element in the ‘attributes’ column into columns and then combined them with the original dataset. After doing that, our dataset increased from 14 attributes to 53 attributes. For features that we think those are not applicable or relevant to our problems, such as 'AcceptsInsurance', 'RestaurantsCounterService', 'AgesAllowed', 'DietaryRestrictions', 'BusinessAcceptsBitcoin', 'HairSpecializesIn', etc., we removed them from our dataset. We narrowed down our final feature’s category to 30; some of them are RestauantDelivery, OutdoorSeating, WiFi, Alcohol, etc. In addition, we filled in null values and dropped the duplicates.
 
These are the features we have left after doing the preprocessing:

 
 
 
 
 


V.	OVERVIEW OF PROJECT TECHNICAL DETAILS


Loading datasets and doing data preprocessing are our first two tasks. After data preprocessing, we applied libraries such as geopandas, plotly.express, plotly.graph_objs, seaborn for creating graphs and EDAs. To make the data ready for using the machine learning model, we converted the label column (‘stars’) to binary, with the value of ‘1’ corresponding to a rating equal or greater than 4 and the value of ‘0’ corresponding to a rating below 4. We then created a feature matrix (x) and a target vector (y), doing label encoding for all the categorical features, splitting our data into training and testing sets with 80% used for training and 20% for testing, then doing feature scaling for the training set. After that, we imported scikit-learn modules for building our machine learning models. Finally, we evaluated our models by calculating the accuracy score, AUC-ROC score, and the confusion matrix.


VI.	EDAs

EDAs are helpful to identify the errors, and a better understanding of the data sets and the relationship with data set variables. The main purpose of us using the EDAs is to be able to recognize various patterns in the data. It helps us to have a better understanding of the data before making any assumptions and picking the models that we can use in the project. 

1)	Restaurant Location Map

 



The map shows all the restaurants located based on longitude and latitude data. It gives us an understanding of the locations of restaurants the data is taken for. What's helpful about this type of map is it is interactive. Meaning you can change the size of the map which allows easy navigation. It also allows you to hover over each of the restaurants to view its information, such as longitude, latitude, stars rating, the count of reviews it was provided, the category it falls into, and its postal code. Looking at the map first-hand, we can assume that there are more restaurants that its data was collected for as compared to others.

2)	Restaurant Location Map in Pennsylvania

  

The Figure 2 map shows all of the restaurants located based on longitude and latitude data related to Pennsylvania. It gives us an understanding of the Pennsylvania restaurant locations.

3) Top cities and states business listed in Yelp

 

Figure 3 is showing side-by-side horizontal bar charts. The chart on the left shows the number of restaurants in cities containing the most. The chart on the right represents a similar visualization but instead focuses on the number of restaurants per state. We found that the city of Philadelphia is top in business listings, with Tampa being the second highest and Indianapolis being the third. Pennsylvania is the top state in the list where it has the most business listings which would make sense as its city, Philadelphia, is the top city containing the most businesses.

 4) Restaurant rating distribution


 




The Figure 4 bar chart shows the distribution for restaurant counts across all ratings. You can see here that the rating of 4.0 has the highest count that was provided as ratings to restaurants, with 4.5 being the second highest. And it also appears that not all restaurants were given a rating of 1.0 (at the lowest) or 5.0 (at the highest).



5) Most popular restaurant names

 


Figure 5 chart represents the top 10 popular restaurants in our dataset. Santa Barbara Shelfish Company has the highest reviews followed by Prep & Pastry and Mr.B's Bistro.


VII.	MODELS & RESULTS

To achieve our goal of predicting restaurant ratings, we decided to create Machine Learning classification models. There are plenty we can choose from regarding classifiers, since there are so many, we resulted in choosing three different ones to work with. Before moving forward with this, we needed to decide how we wanted to go about it in terms of what we wanted to predict and if we needed to make some changes with our existing label column. Our label column contains stars ratings ranging from 1.0 to 5.0 with increments of 0.5. What we did in this situation is we re-formulated this problem as a binary classification one, by splitting the restaurants into two groups. These two groups are restaurants with ratings of 4 and above will be given a label of 1.0, and those with 3.5 and below stars will be given 0.0. The binary values 1.0 and 0.0 will replace the stars in the stars column since that column is the labeled one. The classification models we built are:

●	Logistic Regression

●	Decision Tree Classifier

●	Random Forest Classifier

Next step was to encode all the string-valued attributes to numeric, allowing the Machine Learning algorithms to accept those values. Once the values are encoded, we need to standardize the data so that the values are within a similar range compared to each other.

As the data is ready for modeling, we then instantiated our three models and split our data into two sets for training and testing. Before we trained our model on the training set, we performed hyperparameter tuning using GridSearchCV, which returns the best parameters to apply for our models.


Final Results:
	The best performing model resulted in being the Random Forest Classifier with its accuracy score being the highest.

Logistic Regression Accuracy: 0.650		

Decision Tree Accuracy: 0.676	

Random Forest Accuracy: 0.693


VIII.	OUTCOMES & IMPACTS

‘Who’ this topic affects:

●	Users interested in visiting restaurants

●	Business Owners

  > ○	The model(s) only includes features about characteristics of restaurants, such as whether or not Wi-Fi is available, and these are things that an owner can control relatively easily.

‘What’ this topic affects:

●	businesses/restaurants

  > ○	Yelp has millions of users and reviews, and if a restaurant gets a significant number of bad reviews, or even just a handful, this can have huge financial implications for not only the owners, but for businesses as well.

IX.	CONCLUSION & FUTURE WORKS

The overall goal of the project was to create a Restaurant Rating Predictor system. There were initial confusion and difficulties with this project, but we managed to come to a finalized approach. The predictor system came out to work as expected and predicted the necessary rating labels for restaurants based on their characteristics such as whether the restaurant contains Wi-Fi or not. We ended up using three models, Logistic Regression, Decision Trees and Random Forests, and Random Forests ended up being the best performing one. For the future, we can focus on either applying further classification algorithms such as Support Vector Machines for example. We can also possibly focus on specific regions other than just the entirety of all locations provided by the dataset. The reason for this is because similar restaurants might have been given different ratings due to their poor services in certain locations, but similar restaurants might be given better ratings in a different region/state, so the locations might influence predictions.


VIII. REFERENCES

* Kong, A., Nguyen, V., Xu, C. (2016). Predicting International Restaurant Success with Yelp. https://cs229.stanford.edu/proj2016spr/report/062.pdf
* Xu, Y., Wu, X., Wang, Q. (2015). Sentiment Analysis of Yelp’s Ratings Based on Text Reviews. https://cs229.stanford.edu/proj2014/Yun%20Xu,%20Xinhui%20Wu,%20Qinxia%20Wang,%20Sentiment%20Analysis%20of%20Yelp's%20Ratings%20Based%20on%20Text%20Reviews.pdf
* Gingerich, T., Bochkov, Y. (2015). Predicting Business Ratings on Yelp. https://cs229.stanford.edu/proj2015/013_report.pdf
* Linshi, J. (2014). Personalizing Yelp Star Ratings: A Semantic Topic Modelling Approach. https://www.yelp.com/html/pdf/YelpDatasetChallengeWinner_PersonalizingRatings.pdf
* Guo, Y., Lu, A., Wang, Z. (2016). Predicting Restaurants’ Rating and Popularity Based on Yelp Dataset. http://cs229.stanford.edu/proj2017/final-reports/5244334.pdf
* Data source: https://www.yelp.com/dataset

