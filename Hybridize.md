---
title: Hybrid Models
notebook: Hybridize.ipynb
nav_include: 4
---

## Contents
{:.no_toc}
*  
{: toc}




```python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


## I. Validation Set Selection

Before moving on to building the hybrid model, we select a validation set of 5,000 playlists. We will divide each playlist in the validation set to an input part and an output part, and we will use them for training in the hybridization process as well as model selection at the end. We set several criteria in selecting this validation set.

- First, each playlist should not contain duplicates of songs. Otherwise, repeated songs may harm the performance measurements of our model, as our models never recommendent existing songs. 

- Second, we want all the outputs to be of length 100, so we filtered our playlists that are not long enough (contain fewer than 100 unique songs), and we randomly sample from the remaining pool.

- In order to better simulate the music consumption process, we take the last 100 songs from each playlist as our output, and take the rest as our input. Notice that the order of the last 100 songs is stored, in order to calculate Normalized Discounted Cumulative Gain (NDCG).

- Lastly, in order to satisfy different users, we try to keep the input length evenly distributed between 0 and 150. In our final validation set, 100 input lists have two or fewer songs, among which 37 inputs are totally empty.



```python
'''Read MPD: store songs in each playlist, removing duplicates'''
data_path = os.path.join(os.getcwd(),'millionplaylist','data')
playlist_fn = os.listdir(data_path)

pl_dict = {}
```




```python
for fn in sorted(playlist_fn):
    print(fn)
    with open(os.path.join(data_path,fn)) as f:
        data = json.load(f)

    playlists = data['playlists']
    
    for playlist in playlists:
        # get data
        pid = playlist['pid']
        #num_followers = playlist['num_followers']
        unique_tracks = set()
        tracks = []
        for song in playlist['tracks']:
            track_uri = song['track_uri'].split(':')[2]
            tracks.append(track_uri)
            unique_tracks.add(track_uri)
        num_tracks = len(tracks)
        num_unique_tracks = len(unique_tracks)
        # store data
        pl_dict[pid] = {'tracks': tracks,
                        #'num_followers': num_followers, 
                        'num_tracks': num_tracks,'num_unique_tracks': num_unique_tracks}
```




```python
pl_df = pd.DataFrame.from_dict(pl_dict,orient='index')
pl_df.insert(0,'pid',pl_df.index)
pl_df.head()
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
      <th>pid</th>
      <th>tracks</th>
      <th>num_tracks</th>
      <th>num_unique_tracks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[0UaMYEvWZi0ZqiDOoHU3YI, 6I9VzXrHxO9rA9A5euc8A...</td>
      <td>52</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[2HHtWyy5CgaQbC7XSoOb0e, 1MYYt7h6amcrauCOoso3G...</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[74tqql9zP6JjF5hjkHHUXp, 4erhEGuOGQgjv3p1bccnp...</td>
      <td>64</td>
      <td>64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[4WJ7UMD4i6DOPzyXU5pZSz, 1Kzxd1kkjaGX4JZz2CYsX...</td>
      <td>126</td>
      <td>126</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[4iCGSi1RonREsPtfEKYj5b, 5qqabIl2vWzo9ApSC317s...</td>
      <td>17</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>





```python
'''filter out playlists that have fewer than 100 songs'''
pl_pool = pl_df.loc[pl_df['num_unique_tracks']>=100]\
.loc[pl_df['num_unique_tracks']==pl_df['num_tracks']]
```




```python
'''sample playlists based on length'''
pl_selected = pd.DataFrame(columns = pl_pool.columns)
for i in np.arange(100,250,3):
    start = i
    end = i+2
    if end == 249: end = 250
    select = pl_pool.loc[pl_pool['num_tracks']>=start]\
    .loc[pl_pool['num_tracks']<=end].sample(n=100,random_state=42)
    pl_selected = pd.concat([pl_selected,select])
```




```python
'''Find number of completely cold start problems'''
num_coldstart = len(pl_selected.loc[pl_selected['num_tracks'] == 100])
print(f'number of total cold start problems: {num_coldstart}')
```


    number of total cold start problems: 37




```python
'''Divide each playlist into input and output, and save to json'''
selected_tracks = pl_selected.tracks.values

val_X = []
val_Y = []
for tracks in selected_tracks:
    val_X.append(tracks[:-100])
    val_Y.append(tracks[-100:])
```




```python
with open("val_X.json", "w") as f:
    data_json = json.dump(val_X,f)

with open("val_Y.json", "w") as f:
    data_json = json.dump(val_Y,f)

with open("val_pid.json", "w") as f:
    data_json = json.dump(list(pl_selected.pid.values),f)
```


## II. Model Hybridization: Stacking with Logistic Regression CV

So far, we have constructed five models that we consider combining in our final hybrid model. They are: 1) Collaborative Filtering with filtered data, 2) meta-playlist Collaborative Filtering, 3) lyric-emotion clustering, 4) audio-genre clustering, and 5) audio feature clustering. Moreover, we have selected a validation set, splited into a list of 5,000 inputs and a list 5,000 outputs. For each input list, each model outputs 500 song recommendations, with corresponding scores used to order the recommendations.

Therefore, we can transform the model hybridization problem to a stacking problem. Regarding each recommended song track for each input list, we have multiple score values from the different models. We consider this set of values to be one observation. Moreover, we can assign the output of this observation to be one if this song track actually exists in the true validation output corresponding to this particular input, and we assign zero otherwise. Since the stacking problem outputs only take up values one or zero, we use Logistic Regression to achieve stacking. In order not to overfit to this validation set, we apply Cross Validation when we train our Logistic Regression model.

If we want to combine all the five models, they will generate 2,500 songs in total. Although there will be some overlaps, the number of unique song recommendations is still much larger than the number of true outputs, which is 100. This would result in an unbalanced model, which classifies more songs as not recommended than it should. To combat this issue, when we train our Logistic Regression CV model, we keep all the observations whose output is 1, and randomly sample observations whose output is 0, so that the resulting stacking model is balanced.



```python
with open('val_Y.json', 'r') as f:
    val_Y = json.load(f)
output = []
for lst in val_Y:
    dic = {}
    for i in lst:
        dic[i] = 1
    output.append(dic)
```




```python
def hybridize(files):
    train_x = []
    train_y = []
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            data_list.append(json.load(f))

    predict = [0] * len(data_list[0])
    track_record = [0] * len(data_list[0]) 

    for i in range(len(data_list[0])):
        y_val = output[i]

        ## combine tracks
        dics = []
        for data in data_list:
            dics.append(data[i])
        tracks=[]
        for dic in dics:
            tracks += list(dic.keys())
        track_record[i] = tracks
        
        ## assign scores
        predict[i] = []
        for track in tracks:
            scores = []
            for dic in dics:
                if track in dic.keys(): scores.append(dic[track])
                else: scores.append(0)
            predict[i].append(scores)
            
            try: y_val[track]
            except KeyError: y = 0
            else: y = 1
            
            if y == 1:
                train_x.append(scores)
                train_y.append(y)
            else:
                p = random.random()
                if p < 0.005:
                    train_x.append(scores)
                    train_y.append(y)
    model = LogisticRegressionCV(cv = 5)
    model.fit(train_x, train_y)
    print(model.coef_)

    rec = []
    for i in range(len(data_list[0])):
        print('rec: ', i)
        score = {}
        for j in range(len(predict[i])):
            score[track_record[i][j]] = model.predict_proba([list(predict[i][j])])[0,1]
        s = sorted(score.items(), key=lambda item:item[1])
        final_s = list(np.array(s[-1:-501:-1])[:,0])
        rec.append(final_s)
    with open('hybridize.json', 'w') as f:
        json.dump(rec, f)
    return rec
```

