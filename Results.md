---
title: Results
notebook: Results.ipynb
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
<center>$$ \text{Clicks} = \left \lfloor \frac{argmin_i\{R_i:R_i \subset G\}-1}{10} \right \rfloor$$</center>
When there are more songs in R than in G, we only consider the first $\mid G \mid$ songs in R. If none of the recommended songs is relavent, the value of the Recommended Songs Clicks would be $ \frac{|R|}{10}$, which is one more than the maximal number of rounds possible.



```python
def R_precision(rec, Y):
    count = 0
    for song in Y:
        if song in rec[:len(Y)]:
            count += 1 
    return count/len(Y)

def NDCG(rec, Y):
    IDCG = 0
    for i in range(0,len(Y)):
        if i == 0: IDCG += 1
        else: IDCG += 1/math.log((i+2),2)
    DCG = 0
    for i in range(0,len(rec)):
        if i == 0 and rec[i] in Y: DCG += 1
        elif i > 0 and rec[i] in Y: DCG += 1/math.log((i+2),2)     
    return DCG/IDCG

def clicks(rec, Y):
    found_at = -1
    find = 0
    while found_at == -1 and find < len(Y):
        if rec[find] in Y: found_at = find
        else: find += 1
    if found_at == -1:
        return len(Y)//10
    else:
        return found_at//10

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




```python
def R_precision(rec, Y):
    count = 0
    for song in Y:
        if song in rec[:len(Y)]:
            count += 1 
    return count/len(Y)

def NDCG(rec, Y):
    IDCG = 0
    for i in range(0,len(Y)):
        if i == 0: IDCG += 1
        else: IDCG += 1/math.log((i+2),2)
    DCG = 0
    for i in range(0,len(rec)):
        if i == 0 and rec[i] in Y: DCG += 1
        elif i > 0 and rec[i] in Y: DCG += 1/math.log((i+2),2)     
    return DCG/IDCG

def clicks(rec, Y):
    found_at = -1
    find = 0
    while found_at == -1 and find < len(Y):
        if rec[find] in Y: found_at = find
        else: find += 1
    if found_at == -1:
        return len(Y)//10
    else:
        return found_at//10

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
    return R_precision_scores,NDCG_scores, clicks_scoresdef test_recs(fn):
    with open(fn) as json_file: 
         rec = json.load(json_file)
    with open('validation/val_Y.json') as json_file: 
         val_Y = json.load(json_file)  

    empty = []
    for i in range(len(rec)):
        if len(rec[i])==0: empty.append(i)
    for i in reversed(sorted(empty)): 
        del rec[i]
        del val_Y[i]

    R_precision_score, NDCG_score, clicks_score = TEST_ALL(rec,val_Y)
    score1 = np.mean(R_precision_score)
    score2 = np.mean(NDCG_score)
    score3 = np.mean(clicks_score)
    print(f'R_precision: {score1}')
    print(f'NDCG: {score2}')
    print(f'#clicks: {score3}')
    return score1, score2, score3
    
def test_scores(fn):
    with open(fn) as json_file: 
         scores = json.load(json_file)
    with open('validation/val_Y.json') as json_file: 
         val_Y = json.load(json_file)  
    rec = [list(single_score.keys()) for single_score in scores]

    empty = []
    for i in range(len(rec)):
        if len(rec[i])==0: empty.append(i)     
    for i in reversed(sorted(empty)): 
        del rec[i]
        del val_Y[i]

    R_precision_score, NDCG_score, clicks_score = TEST_ALL(rec,val_Y)
    score1 = np.mean(R_precision_score)
    score2 = np.mean(NDCG_score)
    score3 = np.mean(clicks_score)
    print(f'R_precision: {score1}')
    print(f'NDCG: {score2}')
    print(f'#clicks: {score3}')
    return score1, score2, score3
```


## II. Model Evaluation

During the hybridization process, we have chosen 5,000 playlists and divided them into input and output parts. The inputs have lengths from 0 to 150, distributed roughly evenly, and the outputs all have lengths of 100. We feed our models with the validation input, and each model produces 500 ordered song recommendations. We then calculate the three metrics for each of the models.

Because we generated our validation set from MPD, and MPD does not provide information on a user's preference among the songs within a single playlist, we make the assumption that the position of a song indicates the user's preference. That is to say, we consider that users prefer songs that are placed in the front of the playlists, and we calculate NDCG based on this assumption. 

### 1. Baseline Model

**Top 500 Popular Songs**



```python
R_bl, N_bl, C_bl = test_recs('validation/val_Y_top500.json')
```


    R_precision: 0.035814
    NDCG: 0.08360293878734794
    #clicks: 5.0066


#### Summary



```python
df_bl = pd.DataFrame([R_bl, N_bl, C_bl]).T
df_bl.columns = ['R-Precision','NDCG','Recommended Songs Clicks']
df_bl.index = ['Baseline - Top 500']
display(df_bl)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R-Precision</th>
      <th>NDCG</th>
      <th>Recommended Songs Clicks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Baseline - Top 500</th>
      <td>0.035814</td>
      <td>0.083603</td>
      <td>5.0066</td>
    </tr>
  </tbody>
</table>
</div>


### 2. Collaborative Filtering Models



```python
R_cf_list,N_cf_list,C_cf_list = [],[],[]
```


**Baseline (50000 Playlists)**



```python
R, N, C = test_scores('validation/score_baseline_CF.json')
R_cf_list.append(R)
N_cf_list.append(N)
C_cf_list.append(C)
```


    R_precision: 0.022812
    NDCG: 0.06110045476489474
    #clicks: 5.5702


**Meta-Playlist**



```python
R, N, C = test_scores('validation/score_metaplaylist.json')
R_cf_list.append(R)
N_cf_list.append(N)
C_cf_list.append(C)
```


    R_precision: 0.017968
    NDCG: 0.053416332195018505
    #clicks: 5.2572


**Advanced (Filtered Songs and Playlists)**



```python
R, N, C = test_scores('validation/score_advanced_CF.json')
R_cf_list.append(R)
N_cf_list.append(N)
C_cf_list.append(C)
```


    R_precision: 0.020694000000000004
    NDCG: 0.06297452664793372
    #clicks: 6.12


#### Summary



```python
df_cf = pd.DataFrame([R_cf_list,N_cf_list,C_cf_list]).T
df_cf.columns = ['R-Precision','NDCG','Recommended Songs Clicks']
df_cf.index = ['Baseline CF','Meta-Playlist CF','Advanced CF']
display(df_cf)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R-Precision</th>
      <th>NDCG</th>
      <th>Recommended Songs Clicks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Baseline CF</th>
      <td>0.022812</td>
      <td>0.061100</td>
      <td>5.5702</td>
    </tr>
    <tr>
      <th>Meta-Playlist CF</th>
      <td>0.017968</td>
      <td>0.053416</td>
      <td>5.2572</td>
    </tr>
    <tr>
      <th>Advanced CF</th>
      <td>0.020694</td>
      <td>0.062975</td>
      <td>6.1200</td>
    </tr>
  </tbody>
</table>
</div>


### 3. Content Based Models



```python
R_cb_list,N_cb_list,C_cb_list = [],[],[]
```


**Clustering - Emotion**



```python
R, N, C = test_scores('validation/val_Y_lyric_score_c.json')
R_cb_list.append(R)
N_cb_list.append(N)
C_cb_list.append(C)
```


    R_precision: 0.0006313815378203792
    NDCG: 0.0020591344073271735
    #clicks: 9.656386747239008


**Clustering - Genre**



```python
R, N, C = test_scores('validation/val_Y_genre_score_c.json')
R_cb_list.append(R)
N_cb_list.append(N)
C_cb_list.append(C)
```


    R_precision: 7.052186177715091e-05
    NDCG: 0.00025659978471954096
    #clicks: 9.962522667741286


**Clustering - Audio Feature**



```python
R, N, C = test_scores('validation/val_Y_audio_score_c.json')
R_cb_list.append(R)
N_cb_list.append(N)
C_cb_list.append(C)
```


    R_precision: 0.0004755188394116462
    NDCG: 0.0013314987133503497
    #clicks: 9.748942172073344


**No Clustering - Emotion**



```python
R, N, C = test_scores('validation/val_Y_lyric_score_a.json')
R_cb_list.append(R)
N_cb_list.append(N)
C_cb_list.append(C)
```


    R_precision: 0.0008689310272973536
    NDCG: 0.002719010744180856
    #clicks: 9.550531360700147


**No Clustering - Genre**



```python
R, N, C = test_scores('validation/val_Y_genre_score_a.json')
R_cb_list.append(R)
N_cb_list.append(N)
C_cb_list.append(C)
```


    R_precision: 0.00010276042716099134
    NDCG: 0.0003291407777316353
    #clicks: 9.948418295385855


**No Clustering - Audio Feature**



```python
R, N, C = test_scores('validation/val_Y_audio_score_a.json')
R_cb_list.append(R)
N_cb_list.append(N)
C_cb_list.append(C)
```


    R_precision: 0.0003788031432601249
    NDCG: 0.001110149776504973
    #clicks: 9.795688091879912


#### Summary



```python
df_cb = pd.DataFrame([R_cb_list,N_cb_list,C_cb_list]).T
df_cb.columns = ['R-Precision','NDCG','Recommended Songs Clicks']
df_cb.index = ['Clustering - Emotion','Clustering - Genre', 'Clustering - Audio Feature',
              'No Clustering - Emotion','No Clustering - Genre', 'No Clustering - Audio Feature']
display(df_cb)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R-Precision</th>
      <th>NDCG</th>
      <th>Recommended Songs Clicks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Clustering - Emotion</th>
      <td>0.000631</td>
      <td>0.002059</td>
      <td>9.656387</td>
    </tr>
    <tr>
      <th>Clustering - Genre</th>
      <td>0.000071</td>
      <td>0.000257</td>
      <td>9.962523</td>
    </tr>
    <tr>
      <th>Clustering - Audio Feature</th>
      <td>0.000476</td>
      <td>0.001331</td>
      <td>9.748942</td>
    </tr>
    <tr>
      <th>No Clustering - Emotion</th>
      <td>0.000869</td>
      <td>0.002719</td>
      <td>9.550531</td>
    </tr>
    <tr>
      <th>No Clustering - Genre</th>
      <td>0.000103</td>
      <td>0.000329</td>
      <td>9.948418</td>
    </tr>
    <tr>
      <th>No Clustering - Audio Feature</th>
      <td>0.000379</td>
      <td>0.001110</td>
      <td>9.795688</td>
    </tr>
  </tbody>
</table>
</div>


### 4. Hybrid Models

From the results above, we find that collaborative filtering models perform better than the content-based models. Therefore, in the hybridization process, we will focus on combining collaborative filtering models with other models. Based on the performance of the various content-based models, we will use the emotion model without clustering and the audio feature model with clustering. Lastly, because the training dataset for baseline content baseline model may have a few overlaps with the validation set, we exclude it from our final hybrid model.

**Stacking with Logistic Regression CV**



```python
R_stack_list,N_stack_list,C_stack_list = [],[],[]
val_Y_files = ['validation/hybridize_BL2CF2CB.json',
               'validation/hybridize_2CF2CB.json',
               'validation/hybridize_2CFs.json',
               'validation/hybridize_BLmeta.json']
for file in val_Y_files:
    R, N, C = test_recs(file)
    print()
    R_stack_list.append(R)
    N_stack_list.append(N)
    C_stack_list.append(C)
```


    R_precision: 0.035660000000000004
    NDCG: 0.08328080372461233
    #clicks: 5.0124
    
    R_precision: 0.020246
    NDCG: 0.06229288799130191
    #clicks: 6.1938
    
    R_precision: 0.018852
    NDCG: 0.061758758863905396
    #clicks: 6.3528
    
    R_precision: 0.035814
    NDCG: 0.08360293878734794
    #clicks: 5.0066
    


#### Summary



```python
df_stack = pd.DataFrame([R_stack_list,N_stack_list,C_stack_list]).T
df_stack.columns = ['R-Precision','NDCG','Recommended Songs Clicks']
df_stack.index = ['Top500 & CF-MetaPlaylist & CF-Advanced & CB-Emotion & CB-Audio',
                 'CF-MetaPlaylist & CF-Advanced & CB-Emotion & CB-Audio',
                 'CF-MetaPlaylist & CF-Advanced',
                 'Top500 & CF-MetaPlaylist']
display(df_stack)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R-Precision</th>
      <th>NDCG</th>
      <th>Recommended Songs Clicks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Top500 &amp; CF-MetaPlaylist &amp; CF-Advanced &amp; CB-Emotion &amp; CB-Audio</th>
      <td>0.035660</td>
      <td>0.083281</td>
      <td>5.0124</td>
    </tr>
    <tr>
      <th>CF-MetaPlaylist &amp; CF-Advanced &amp; CB-Emotion &amp; CB-Audio</th>
      <td>0.020246</td>
      <td>0.062293</td>
      <td>6.1938</td>
    </tr>
    <tr>
      <th>CF-MetaPlaylist &amp; CF-Advanced</th>
      <td>0.018852</td>
      <td>0.061759</td>
      <td>6.3528</td>
    </tr>
    <tr>
      <th>Top500 &amp; CF-MetaPlaylist</th>
      <td>0.035814</td>
      <td>0.083603</td>
      <td>5.0066</td>
    </tr>
  </tbody>
</table>
</div>


**Combining with Assigned Weights**



```python
R_comb_list,N_comb_list,C_comb_list = [],[],[]
val_Y_files = ['validation/weightedsum_2CFs.json']
for file in val_Y_files:
    R, N, C = test_recs(file)
    R_comb_list.append(R)
    N_comb_list.append(N)
    C_comb_list.append(C)
```


    R_precision: 0.018852
    NDCG: 0.061758758863905396
    #clicks: 6.3528

