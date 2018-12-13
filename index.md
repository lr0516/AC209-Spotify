---
title: Building a Hybrid MRS for Spotify
---

## AC209A Data Science Final Project

Group 9. Chuqiao Yuan, Feiyu Chen, Queena Ziyi Zhou, Rong Liu

### Motivation and Overview

There are many leading music streaming services in the market, including Spotify, Pandora, and Apple Music. Thus, a robust Music Recommender System (MRS) is in high demand to improve user experience and increase user engagement. Many researchers have focused their work on traditional recommender systems, but MSR differs from these recommender systems because the duration of a song is short (3~5 minutes), the total number of existing songs is large (tens of millions), and songs are consumed in sessions, among which the same user’s taste may change.


In the early stage, we read a wide range of papers on recommendation systems, which introduce to us the popular approaches in this field, including Collaborative Filtering. More detail will be discussed in the later 'Related Work' section. Moreover, our EDA results lead us to utilize playlist names, song emotions, and song genres in our model.

- **Goal**: ​Our goal is to build a robust MRS to recommend songs to the users given a set of songs the users have been listened to.
- **Approach**: W​e would like to hybridize two models: 1) Collaborative filtering, recommending songs based on similar users’ past preferences; 2) Content-based recommendation, using NLP techniques on song titles and lyrics to detect mood, and analyzing sound features to detect genre. We combine the models into one hybrid model through stacking with Logistic Regression and computing weighted sums.


### Related Work

In recommendation systems, a very popular algorithm nowadays is collaborative filtering. Koren, Bell & Volinsky (2009) $^{[1]}$ introduce a very specfic collaborative filtering technique (matrix factorization technique) which has good performance in the Netflix Prize Competition. For music recommendation problems specifically, Berenzweig, Logan, Ellis & Whitman (2003)$^{[2]}$ propose to use acoustic similarities to recommend songs. Schedl, Zamani, Chen, Deldjoo & Elahi (2018)$^{[3]}$ have pointed out that to solve the cold start problem and playlist continuation problem in music recommendation system, context-based models can be taken into consideration.

Inspired by the works mentioned above, we use collaborative filtering techniques and content-based filtering techniques in our final model. Here are some related works that guide us to design and implement our project. We would like to give special thanks to them.

For collaborative filtering, we build up our implicit feedback model based on the library developed by Maciej (2017)$^{[4]}$.

For content-based filtering, we build up 3 models, one is based on emotions passed by lyrics, one is based on acoustic features and one is based on genres.

For the lyrics-based filtering model, we learn from Kim, Schmidt, Migneco, Morton, Richardson, Scott, Speck & Turnbull (2010)$^{[5]}$ that recognizing musical mood is a multiclass-multilabel classification or regression problem, where we try to annotate each music piece with a set of emotions. We can represent moods as multi-dimensional vector. In addition, we can directly use the existing music database, which has been manually annotated, to train our model.

We want to associate song lyrics with musical mood and theme. Namely, we want to classify a song's mood based on its lyrics. This website (https://xindizhao19931.wixsite.com/spotify2) teaches us to use both unsupervised method, Word2Vec, and supervised method, Long Short-Term Memory (LSTM), for language processing, sentiment analysis and predictive modeling.

And in Chapter 9 of the textbook by Leskovec, Rajaraman and Ullman (2014)$^{[6]}$, we learn to construct for each item a profile, which is a record or collection of records representing important characteristics of that item. Each profile can be viewed as a vector, whose entries are paired with a set of features. We also need to create vectors with the same components that describe the user’s preferences. Then we can just calculate the cosine distance between two sets to measure the similarity of two playlists.

Logan, Kositsky & Moreno (2004)$^{[7]}$ show us that we can use latent semantic analysis to compare the similarities among lyrics. However, as the paper points out, "similarity based on lyrics was found to be better than random but inferior to acoustic similarity, at least for the ground truth used."

Finally, we learn how to prepare our training dataset more appropriately from the work of Yang, Jeong, Choi & Lee (2018)$^{[8]}$. We would like to filter out songs that are not so popular, and therefore appear in only one or two playlists.

[1] Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009).

[2] Berenzweig, Adam, Beth Logan, Daniel P.W. Ellis and Brian Whitman. A Large-Scale Evaluation of Acoustic and Subjective Music Similarity Measures. Proceedings of the ISMIR International Conference on Music Information Retrieval (Baltimore, MD), 2003, (pp. 99-105).

[3] Shedl, M. et al, Current Challenges and Visions in Music Recommender Systems Research, https://arxiv.org/pdf/1710.03208.pdf.

[4] Kula Maciej. "Spotlight." (2017)
https://github.com/maciejkula/spotlight

[5] Kim, Y.E., Schmidt, E.M., Migneco, R., Morton, B.G., Richardson, P., Scott, J., Speck, J.A. and Turnbull, D., 2010, August. Music emotion recognition: A state of the art review. In Proc. ISMIR (pp. 255-266).

[6] Leskovec, J., Rajaraman, A. and Ullman, J.D., 2014. Mining of massive datasets. Cambridge university press.

[7] Logan, B., Kositsky, A. and Moreno, P., 2004, June. Semantic analysis of song lyrics. In Multimedia and Expo, 2004. ICME'04. 2004 IEEE International Conference on (Vol. 2, pp. 827-830). IEEE.

[8] Yang, H., Jeong, Y., Choi, M. and Lee, J., 2018, October. MMCF: Multimodal Collaborative Filtering for Automatic Playlist Continuation. In Proceedings of the ACM Recommender Systems Challenge 2018 (p. 11). ACM.
