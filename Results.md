---
title: Results
notebook: Results.ipynb
nav_include: 5
---

## Contents
{:.no_toc}
*  
{: toc}




```python
import pandas as pd
import numpy as np
import os
import re
import json
import random
import math
```


## I. Metrics Definition

We will evaluate our models based on R-Precision, Normalized Discoutned Cumulative Gain (NDCG), and Recommended Songs Clicks. In order to clearly define our metrics, we use $G$ to denote the ordered ground truth list of songs that the user would like to listen to, and we use R to denote the ordered recommendations produced by our model. We use $\mid \cdot \mid$ to indicate the length of a list, and we use $R_i$ to refer to the i-th song in our recommendation. Furthermore, we say a song in our recommentation is relavent if it also exists in the ground truth list. We then define $r_i = 1$ if $R_i$ is relavent and $r_i = 0$ if otherwise.

### 1. R-Precision

R-Precision measures the overlap between the ground truth set and our recommendation. Its value is simply the number of relavent songs in our model's  first $\mid G \mid$ recommendations divided by the length of the ground truth set.
<center>$$ \text{R-Precision} = \frac{\sum_{1}^{\mid G \mid} r_i}{\mid G \mid}$$</center>

### 2. NDCG

Normalized Discoutned Cumulative Gain (NDCG) further measures the quality of order in our recommendation. It gives more credit when a relavent song is placed higher in our recommendation. DCG is a score on our recommendation, and IDCG is the ideal DCG value is all of our top $\mid G \mid$ recommended songs are relavent. By dividing the two, NDCG gives us a normalized score.
<center>$$ \text{DCG} = r_1 + \sum_{2}^{\mid R \mid} \frac{r_i}{log_2(i+1)}$$</center>
<center>$$ \text{IDCG} = 1 + \sum_{2}^{\mid G \mid} \frac{1}{log_2(i+1)}$$</center>
<center>$$ \text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$</center>

### 3. Recommended Songs Clicks 

Recommended Songs Clicks is a special metric targeted for Spotify. Spotify has a feature that generates ten songs in a round. The Recommended Songs Clicks is the minimal number of refreshes required to get the first relavent song. 
<center>$$ \text{Clicks} = \left \lfloor \frac{argmin_i\{R_i:R_i \subset G\}}{10} \right \rfloor +1$$</center>
When there are more songs in R than in G, we only consider the first $\mid G \mid$ songs in R. If none of the recommended songs is relavent, the value of the Recommended Songs Clicks would be $1 + \frac{|R|}{10}$, which is one more than the maximal number of rounds possible.



```python
def R_precision(rec, Y):
    count = 0
    for song in rec:
        if song in Y[:len(rec)]:
            count += 1 
    return count/len(rec)

def NDCG(rec, Y):
    IDCG = 0
    for i in range(0,len(rec)):
        if i == 0: IDCG += 1
        else: IDCG += 1/math.log((i+2),2)
    DCG = 0
    for i in range(0,len(Y)):
        if i == 0 and Y[i] in rec: DCG += 1
        elif i > 0 and Y[i] in rec: DCG += 1/math.log((i+2),2)     
    return DCG/IDCG

def clicks(rec, Y):
    found_at = -1
    find = 0
    while found_at == -1 and find < len(Y):
        if rec[find] in Y: found_at = find
        else: find += 1
    if found_at == -1:
        return len(Y)//10 +1
    else:
        return found_at//10 +1

def TEST_ALL(recs, Ys):
    R_precision_scores = []
    NDCG_scores = []
    clicks_scores = []
    for i in range(len(Ys)):
        rec = recs[i]
        Y = Ys[i]
        R_precision_scores.append(R_precision(rec,Y))
        NDCG_scores.append(NDCG(rec,Y))
        clicks_scores.append(clicks(rec,Y))
    return R_precision_scores,NDCG_scores, clicks_scores
```


## II. Model Evaluation

During the hybridization process, we have chosen 5,000 playlists and divided them into input and output parts. The inputs have lengths from 0 to 150, distributed roughly evenly, and the outputs all have lengths of 100. We feed our models with the validation input, and each model produces 500 ordered song recommendations. We then calculate the three metrics for each of the models.

Because we generated our validation set from MPD, and MPD does not provide information on a user's preference among the songs within a single playlist, we make the assumption that the position of a song indicates the user's preference. That is to say, we consider that users prefer songs that are placed in the front of the playlists, and we calculate NDCG based on this assumption. 

### 1. Baseline Model

### 2. Single Models

### 3. Hybrid Models
