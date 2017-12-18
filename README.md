# Recruit-Restaurent

### Problem Statemnt 

Running a thriving local restaurant isn't always as charming as first impressions appear. There are often all sorts of unexpected troubles popping up that could hurt business. One common predicament is that restaurants need to know how many customers to expect each day to effectively purchase ingredients and schedule staff members. This forecast isn't easy to make because many unpredictable factors affect restaurant attendance, like weather and local competition. It's even harder for newer restaurants with little historical data. Recruit Holdings has unique access to key datasets that could make automated future customer prediction possible. Specifically, Recruit Holdings owns Hot Pepper Gourmet (a restaurant review service), AirREGI (a restaurant point of sales service), and Restaurant Board (reservation log management software). In this competition, you're challenged to use reservation and visitation data to predict the total number of visitors to a restaurant for future dates. This information will help restaurants be much more efficient and allow them to focus on creating an enjoyable dining experience for their customers.

### Data Description

This competitionis on a time-series forecasting problem centered around restaurant visitors. The data comes from two separate sites:

- Hot Pepper Gourmet (hpg): similar to Yelp, here users can search restaurants and also make a reservation online
- AirREGI / Restaurant Board (air): similar to Square, a reservation control and cash register system

Available features include reservations, visits, and other information from these sites to forecast future restaurant visitor totals on a given date. The training data covers the dates from 2016 until April 2017. The test set covers the last week of April and May of 2017. The test set is split based on time (the public fold coming first, the private fold following the public) and covers a chosen subset of the air restaurants. Note that the test set intentionally spans a holiday week in Japan called the "Golden Week."

There are days in the test set where the restaurant were closed and had no visitors. These are ignored in scoring. The training set omits days where the restaurants were closed.

##### File Descriptions

This is a relational dataset from two systems. Each file is prefaced with the source (either air_ or hpg_) to indicate its origin. Each restaurant has a unique air_store_id and hpg_store_id. Note that not all restaurants are covered by both systems, and that you have been provided data beyond the restaurants for which you must forecast. Latitudes and Longitudes are not exact to discourage de-identification of restaurants.

###### air_reserve.csv

This file contains reservations made in the air system. Note that the reserve_datetime indicates the time when the reservation was created, whereas the visit_datetime is the time in the future where the visit will occur.

- air_store_id - the restaurant's id in the air system
- visit_datetime - the time of the reservation
- reserve_datetime - the time the reservation was made
- reserve_visitors - the number of visitors for that reservation

###### hpg_reserve.csv

This file contains reservations made in the hpg system.

- hpg_store_id - the restaurant's id in the hpg system
- visit_datetime - the time of the reservation
- reserve_datetime - the time the reservation was made
- reserve_visitors - the number of visitors for that reservation

###### air_store_info.csv

This file contains information about select air restaurants. Column names and contents are self-explanatory.

- air_store_id
- air_genre_name
- air_area_name
- latitude
- longitude

Note: latitude and longitude are the latitude and longitude of the area to which the store belongs

###### hpg_store_info.csv

This file contains information about select hpg restaurants. Column names and contents are self-explanatory.

- hpg_store_id
- hpg_genre_name
- hpg_area_name
- latitude
- longitude

Note: latitude and longitude are the latitude and longitude of the area to which the store belongs

###### store_id_relation.csv

This file allows you to join select restaurants that have both the air and hpg system.

- hpg_store_id
- air_store_id

###### air_visit_data.csv

This file contains historical visit data for the air restaurants.

- air_store_id
- visit_date - the date
- visitors - the number of visitors to the restaurant on the date

###### sample_submission.csv

This file shows a submission in the correct format, including the days for which you must forecast.

- id - the id is formed by concatenating the air_store_id and visit_date with an underscore
- visitors- the number of visitors forecasted for the store and date combination

###### date_info.csv

This file gives basic information about the calendar dates in the dataset.

- calendar_date
- day_of_week
- holiday_flg - is the day a holiday in Japan

### Data Analysis:

I used the EDA kernel [Be My Guest](https://www.kaggle.com/headsortails/be-my-guest-recruit-restaurant-eda) developed by [Heads or Tails](https://www.kaggle.com/headsortails). 

##### Conclusion of major results:

There is an interesting long-term step structure in the overall time series due to new restaurants being added to the data base. A periodic pattern that most likely corresponds to a weekly cycle exists, with Friday and the weekend appear to be the most popular days and Monday and Tuesday the least. December appears to be most popular month and March - May is consistently busy.

In Air, the time between making a reservation and visiting the restaurant follow a nice 24-hour pattern. The most popular strategy is to reserve a couple of hours before the visit, but if the reservation is made more in advance then it seems to be common to book a table in the evening for one of the next evenings. Very long time gaps between reservation and visit are not uncommon. Those are the most extreme values for the air data, up to more than a year in advance. In HPG, the visits after reservation follow a more orderly pattern, with a clear spike in Dec 2016. As above for the air data, we also see reservation visits dropping off as we get closer to the end of the time frame.We see another nice 24-hour pattern for making these reservations. It’s worth noting that here the last few hours before the visit don’t see more volume than the 24 or 48 hours before. This is in stark constrast to the air data.

### Submitted Model

Random Forest Regressors seemed to work best for me. I tried ARMA models, LSTMs and MLP models. I am currently working on a Prophet model which ideall should yeild better results. The RSMLE of this model is 0.489. I optimised the features as much as possible.

### Future things to look into

ARIMA forecasting models are very viable for this problem. They have been tried in a few kernels yielding promising results. Due a lack of resources, my LSTM model and MLP model could not be trained fully. Thus, with appropriate hardware, they should work best due to the introduction of non linearity and sequence predition properties of RNNs. 
