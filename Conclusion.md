---
title: Conclusion and Future Work
notebook: Conclusion.ipynb
nav_include: 6
---

## Contents
{:.no_toc}
*  
{: toc}


## I. Final Model

For our final model, we select the hybrid model from combining our five individual models with weighted sums. The five models are our baseline model (recommending the 500 most popular songs), meta-playlist collaborative filtering model, advanced collaborative filtering model (with filtered songs and playlists), emotion recognition model, and audio feature model. For each round of recommendation, we feed the models with the same input and receive five sets of ordered recommendations. After that, we combine the recommendations by first normalizing the scores and then adding them up while giving the best-performing model more weight. 

Among all the models that we tested on the validation set, our model has the highest R-precision (3.6%), high NDCG (8.3%), and low Recommended Songs Clicks (4.9).

## II. Conclusion

In this project, our goal was to generate personalized playlists and recommend them to Spotify's users based on their music preferences and listening histories. We started this project by reading the literature on recommender systems and doing some data exploratory analysis. We decided to do both content-based recommendation and collaborative filtering, which are two major approaches in recommender systems. We also challenged ourselves by performing model stacking on all the models we got.
There were a few challenges that we managed to overcome during the process:

1) Our **data size was massive**. We used the data from MPD, MSD, Lyrics Wiki, and Spotify Web API, each of which contains gigabytes of data. Given the limited timeframe of our project, we must balance between the number of data we wanted to use and the computing power we currently had. To get the most use out of our data in MPD, we filtered the playlists through some criteria, such as the number of followers and number of tracks contained. We also combined some playlists into the so-called 'meta-playlists', which condensed the stored information and saved the computing time. In order to utilize our computing powers efficiently, each member on our team was responsible for training one model, either user-based or content-based, using a different set of data. This strategy made sure that we used as many data as possible while finishing on time. In the end, we utilized about 306 Gigabytes of data (derived from the raw data) to obtain several models that can recommend playlists.

2) At first, it was **hard for us to determine how to evaluate our models**. After consulting with our TF Rashmi Banthia and other pieces of literature, we were able to choose some metrics, including precision, NDCG, and clicks, to measure the quality of our models. We purposefully selected our validation set so that the results can be easily interpreted and we would be able to fairly compare our models. We had some pretty results. Our final hybrid model has its precision over 3.5%, NDCG over 8%, and number of recommended songs clicks below 5. These numbers look great on the [Spotify RecSys Challenge 2018 Final Leaderboard](https://recsys-challenge.spotify.com/static/final_main_leaderboard.html).

3) All recommender systems face **the Cold Start Problem**, that is how to recommend items to the new users who have shown little about their preferences. We solved this problem in both approaches. In user-based collaborative filtering models, if we were given a new user who had absolutely zero music listening history, we would randomly select some popular tracks to recommend. If the new user had listened to at least one or two tracks, we then recommended tracks based on the similarities between this new user and all of our old users. In content-based models, we drew similarities between the input songs and all the other songs and then selected the most similar ones.

## III. Future Work

There were some aspects in our project that we could improve upon. One of the most important aspects is incorporating more data when we train our models. This could bring up our precision a lot, but it could also slow down the model training process. Therefore, it is better for us to use GPU in the future rather than our laptops. We could also incorporate playlist names as a direct input when we generate recommendations. They would provide important information on the users' taste and preference, and utilizing them would most likely greatly boost the performance of our recommendation system. Another improvement we could make is incorporating the idea of exploration into our evaluation metrics. A well-recommended playlist should contain not only the tracks that the users listened to before (exploitation) but also a variety of new tracks which encompass the possibilities that the users like them (exploration). Finally, there is plenty of room for us to improve our precision, NDCG, and clicks. We should try to improve the current models' metrics and also try using other models that bring in new advantages.
