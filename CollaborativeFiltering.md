---
title: Collaborative Filtering Models
notebook: CollaborativeFiltering.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}


It is intuitive that people who listen to the same songs have similar tastes in music, and this is the basic assumption of collaborative filtering. Interactions between a bunch of users and a bunch of items can be represented by a utility matrix with a lot of blanks (since some of the interactions are absent). Collaborative filtering techniques fill in the blanks of the utility matrix by giving predictions based on existing interactions. We build up the collaborative filtering models based on Million Playlist Dataset, assuming that users with similar past behaviors are likely to listen to the same songs in the future.

To build up the collborative filtering models, we use matrix factorization technique. The idea behind matrix factorization technique is that users actually respond to some very specific features of items and these features could be captured by decomposing the $n*m$ utility matrix into two matrices whose dimensions are $n*d$ and $d*m$. (Here d is the number of features characterized by the specific decomposition.) When people listen to music, they may be attracted by some very specific characteristics such as artist, genre and emotions passed by the music. So matrix factorization technique is suitable for music recommendation system.


We use the [Spotlight library](https://maciejkula.github.io/spotlight/index.html) which is based on PyTorch to build up matrix factorization models to get prediction utility matrix for all the users on all the items included in the training data set. We use the implicit feedback models as users do not give explicit ratings for the songs in the playlists. Only presence and absence of events can be captured from the Million Playlist Dataset.

Now that utility matrix is determined, for each user in the test dataset, we predict his/her ratings on the tracks by calculating the weighted sum of training dataset users’ ratings on the tracks. The weight of users in the training dataset is defined by cosine similarity between users in the training dataset and the specific tested user.

We have three collaborative filtering models which are:
    - Baseline Model
    - Filtered-Data Model
    - Meta-Playlist Model
 These three models differ the most in their training datasets.

## I. Baseline Model

We build up the baseline model by simply feeding original (playlist, song) pairs from 50,000 playlists to the Implicit Feedback Factorization model. 5% playlists from the original MDP dataset are used to train the model, which include 50,000 playlists, 450,000 songs and 3,320,000 (playlist, song) pairs. 



```python
import numpy as np
import json
import os

data_path = os.getcwd()+'/millionplaylist/data/'
playlist_fn = os.listdir(data_path)

CF_list = []

for fn_index in range(50): #range(len(playlist_fn)):
    with open(data_path+playlist_fn[fn_index]) as f:
        data = json.load(f)

    playlists = data['playlists']

    for playlist in playlists:
        pid = playlist['pid']
        for song in playlist['tracks']:
            track_uri = song['track_uri'].split(':')[2]
            CF_list.append([pid,track_uri])

import json
with open("CF_lists_50000.json", "w") as f:
    data_json = json.dump(CF_list,f)
```




```python
from spotlight.interactions import Interactions
from spotlight.evaluation import rmse_score
from spotlight.factorization.implicit import ImplicitFactorizationModel


with open('CF_lists_50000.json', 'r') as f:
    data = json.load(f)

user = [int(u) for u, i in data]
item = [i for u,i  in data]


count_u = {}
for u in user:
    if u not in count_u.keys():
        count_u[u] = 1
    else:
        count_u[u] = count_u[u] + 1

user_id = {}
id = 0
for u in count_u.keys():
    user_id[u] = id
    id = id+1

user_processed = []
for u in user:
    user_processed.append(user_id[u])


count = {}
for i in item:
    if i not in count.keys():
        count[i] = 1
    else:
        count[i] = count[i] + 1

item_id = {}
id = 0
for i in count.keys():
    item_id[i] = id
    id = id+1


len(item_id.keys())

item_processed = []
for i in item:
    item_processed.append(item_id[i])

id_to_track = [0] * len(item_id.keys())
for item in item_id.keys():
    id_to_track[item_id[item]] = item


rating = np.ones(len(item))

data = Interactions(np.array(user_processed), np.array(item_processed), rating)
```




```python

model = ImplicitFactorizationModel(n_iter = 1)
model.fit(data, verbose = 1)
torch.save(model, 'baseline_model')
model = torch.load('baseline_model')
```




```python

with open('Val_X.json', 'r') as f:
    validation = json.load(f)

def top_500_dic(inp):
    dic = {}
    score = np.array([0] * len(model.predict(1)))
    sum_w = 0
    users = np.array(range(50000))
    random.shuffle(users)
    users = users[0:99]
    for i in users:
        s = model.predict(i, np.array(inp))
        w = sum(s) / (np.linalg.norm(s) * np.sqrt(len(inp)))
        sum_w = sum_w + w
        score = w * model.predict(i) + score
    score = score / sum_w
    for index in np.argsort(score)[-1:-501:-1]:
        dic[id_to_track[index]] = score[index]
    return dic


rec = []
for i in range(len(validation)):
    each_input = validation[i]
    input_id = []
    for j in range(len(each_input)):
        it = each_input[j]
        if it in item_id.keys():
            input_id.append(item_id[it]) 
    if(len(input_id) > 1):
        rec.append(top_500_dic(input_id))
    else:
        rec.append(popular_dic)
    print('Rec completed', i)

with open('Recommend_songs_baseline_CFmodel.json', 'w') as f:
    json.dump(rec, f)
```


A weakness of the baseline model is that some of the songs in the training datasets are included in only one playlist and some of the playlists in the training datasets are too short to describe the similarities between songs. Another problem is that since it takes too long to train the Implicit Feedback Factorization models, only 5% of the MDP data were used. To overcome these problems, we further build up Filtered-Data Model and Meta-Playlist Model by training the Implicit Feedback Factorization model on processed data.

## II. Filtered-Data Model

For Filtered-Data Model, we filter the MDP data by only maintaining songs which appear in 10 or more playlists, which results in 190,000 playlists with more than 50 songs. Then we sample from the 190,000 playlists to get 50,000 playlists as the training dataset, which includes 50,000 playlists, 320,000 songs and 22,900,000 (playlist, song) pairs. 



```python

song_df = pd.read_csv('song_df_orig.csv')

pid_lists = song_df['pid'].values
pid_lists = [list(set(pid_list.split('[')[1].split(']')[0].split(', '))) for pid_list in pid_lists]
num_pid = [len(pid_list) for pid_list in pid_lists]

song_df['num_pid'] = num_pid
song_df['pid'] = pid_lists

song_df.columns = ['track_uri']+list(song_df.columns)[1:]

song_df_filtered = song_df.loc[song_df['num_pid']>=10]

df_filtered = song_df_filtered[['pid']]
df_filtered.index = song_df_filtered['track_uri']
dict_filtered = df_filtered.to_dict()['pid']

CF_filtered = []
for uri in dict_filtered:
    pids = dict_filtered[uri].split('\'')
    for pid in pids:
        try: CF_filtered.append([int(pid),uri])
        except ValueError: pass
        
user = [u for u,i in CF_filtered]
user_count = Counter(user)
length_count = Counter(np.array(sorted(user_count.items()))[:,1])
length_count = np.array(sorted(length_count.items()))
print(sum(length_count[99:,1]))

with open("val_pid.json", "r") as f:
    filter_pid = json.load(f)

user_count = np.array(sorted(user_count.items()))
for pid,count in user_count:
    if count < 100: filter_pid.append(pid)

full_pid = list(set(np.array(CF_filtered)[:,0]))
full_pid = [int(i) for i in full_pid]
keep_pid = list(set(full_pid) - set(filter_pid))
keep_dict = {i:1 for i in keep_pid}

import random
keep_pid_2 = random.sample(keep_pid,50000)
keep_dict = {i:1 for i in keep_pid_2}

CF_filtered2 = []
for uri in dict_filtered:
    for pid in dict_filtered[uri].split('\''):
        try: 
            current_pid = int(pid)
            try:
                if keep_dict[current_pid]: CF_filtered2.append([current_pid,uri]) 
            except KeyError: pass
        except ValueError: pass

with open("CF_lists_filtered.json", "w") as f:
    data_json = json.dump(CF_filtered2,f)
```




```python

import numpy as np
import json
from spotlight.interactions import Interactions
from spotlight.evaluation import rmse_score
from spotlight.factorization.implicit import ImplicitFactorizationModel


with open('CF_lists_filtered.json', 'r') as f:
    data = json.load(f)

user = [int(u) for u, i in data]
item = [i for u,i  in data]


count_u = {}
for u in user:
    if u not in count_u.keys():
        count_u[u] = 1
    else:
        count_u[u] = count_u[u] + 1

user_id = {}
id = 0
for u in count_u.keys():
    user_id[u] = id
    id = id+1

user_processed = []
for u in user:
    user_processed.append(user_id[u])


count = {}
for i in item:
    if i not in count.keys():
        count[i] = 1
    else:
        count[i] = count[i] + 1

item_id = {}
id = 0
for i in count.keys():
    item_id[i] = id
    id = id+1


len(item_id.keys())

item_processed = []
for i in item:
    item_processed.append(item_id[i])

id_to_track = [0] * len(item_id.keys())
for item in item_id.keys():
    id_to_track[item_id[item]] = item


rating = np.ones(len(item))

data = Interactions(np.array(user_processed), np.array(item_processed), rating)
```




```python

model = ImplicitFactorizationModel(n_iter = 1)
model.fit(data, verbose = 1)
torch.save(model, 'advanced_model')
model = torch.load('advanced_model.dms')
```




```python

with open('Val_X.json', 'r') as f:
    validation = json.load(f)

def top_500_dic(inp):
    dic = {}
    score = np.array([0] * len(model.predict(1)))
    sum_w = 0
    users = np.array(range(50000))
    random.shuffle(users)
    users = users[0:99]
    for i in users:
        s = model.predict(i, np.array(inp))
        w = sum(s) / (np.linalg.norm(s) * np.sqrt(len(inp)))
        sum_w = sum_w + w
        score = w * model.predict(i) + score
    score = score / sum_w
    for index in np.argsort(score)[-1:-501:-1]:
        dic[id_to_track[index]] = score[index]
    return dic

rec = []
for i in range(len(validation)):
    each_input = validation[i]
    input_id = []
    for j in range(len(each_input)):
        it = each_input[j]
        if it in item_id.keys():
            input_id.append(item_id[it]) 
    if(len(input_id) > 1):
        rec.append(top_500_dic(input_id))
    else:
        rec.append(popular_dic)
    print('Rec completed', i)

with open('Recommend_songs_advanced_CFmodel.json', 'w') as f:
    json.dump(rec, f)
```


## III. Meta-Playlist Model

To use as many data as possible, we created these ‘meta-playlist’. We observed that the
MPD dataset contains many sets of playlists with shared titles, the top five of which being ‘country’,
‘chill’, ‘rap’, ‘workout’, and ‘oldies’. We combined each set of playlists with the same title into a
meta-playlist. Each meta-playlist contains all the songs from the sub-playlists and keeps track of the
number of times each song appears in all the sub-playlists.
Then we fitted a collaborative filtering model on the meta-playlists. This method contains fewer rows
and more columns compared to the previous model described.

Currently, we selected the top 100 common titles to be the titles of our 100 meta-playlists.
This includes 272,584 playlists from MPD in total and 939,760 distinct songs, which give us 4,620,153
(playlist, song, rating) pairs. We treated each meta-playlist as one user, whose music preference is clearly
stated in the title. The number of appearance of each song was treated as the song’s rating by the user.
The final model would give a score to every song included in the training dataset for every playlist
in the test dataset. For every playlist in the test dataset, we could generate the songs to which the model
thinks that the playlist will give rather high scores.

We built up an implicit feedback model and use matrix factorization techniques for this
problem . We fed the model with (playlist, song, rating) interactions. 80% of the data were used for
training, the rest was used for validation. 10 epoches were run to give an idea on this model.



```python
import re
from collections import Counter

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

data_path = os.path.join(os.getcwd(),'millionplaylist','data')
filenames = os.listdir(data_path)

names = []
for filename in sorted(filenames):
    with open(os.path.join(data_path,filename)) as f:
        file_data = json.load(f)

    for playlist in file_data['playlists']:
        names.append(normalize_name(playlist['name']))

name_count = Counter(names).most_common(100)
print(name_count)

top_names = [pair[0] for pair in name_count]
CF_dict = {name:{} for name in top_names}

progress_count = 0
for filename in filenames:
    with open(os.path.join(data_path,filename)) as f:
        file_data = json.load(f)
    
    for playlist in file_data['playlists']:
        progress_count += 1
        print(progress_count)
        name_norm = normalize_name(playlist['name'])
        if name_norm in top_names:
            for song in playlist['tracks']:
                try: CF_dict[name_norm][song['track_uri'].split(':')[2]] += 1
                except KeyError: CF_dict[name_norm][song['track_uri'].split(':')[2]] = 1

CF_list = []
for name in CF_dict:
    songs = CF_dict[name]
    for uri in songs:
        CF_list.append([name,uri,songs[uri]])

with open("CF_matrix_condensed.json", "w") as f:
    data_json = json.dump(CF_list,f)
```




```python
import numpy as np

from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score
from spotlight.evaluation import mrr_score
from spotlight.evaluation import precision_recall_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel

from enum import Enum
```




```python
import json
with open('CF_matrix_condensed.json', 'r') as f:
    data = json.load(f)
    
user = [u for u, i, r in data]
item = [i for u, i, r in data]
rating = [r for u, i, r in data]
```




```python
User_id = Enum(value = 'User_id', names = list(set(user)))
```




```python
for i in range(len(user)):
    u = user[i]
    user[i] = User_id[u].value
```




```python
a = list(set(item))

item_id = {}
id = 1
for i in range(len(a)):
    item_id[a[i]] = id
    id = id+1

for i in range(len(item)):
    it = item[i]
    item[i] = item_id[it]    
```




```python
data = Interactions(user_ids=np.array(user), item_ids=np.array(item), ratings=np.array(rating))
train, test = random_train_test_split(data)
model_full = ImplicitFactorizationModel()
model_full.fit(train, verbose=1)

```




```python
import torch
torch.save(model_full, 'meta_playlist_full')
```




```python
with open('Val_X.json', 'r') as f:
    validation = json.load(f)
```




```python
reverse_item_id = {}
for key, value in item_id.items():
    reverse_item_id[value] = key
    
def top_500_dic(inp):
    dic = {}
    score = np.array([0] * len(model.predict(1)))
    sum_w = 0
    for i in range(1,101):
        s = model.predict(i, np.array(inp))
        w = sum(s) / (np.linalg.norm(s) * np.sqrt(len(inp)))
        sum_w += w
        score = w * model.predict(i) + score
    score = (score/sum_w)[1:]
    for index in np.argsort(score)[-1:-501:-1]:
        dic[reverse_item_id[index + 1]] = score[index] # map the song id in string format to numeric score
    return dic

rec_playlist = []
for each_input in validation:
    input_id = []
    for i in range(len(each_input)):
        it = each_input[i]
        try: 
            input_id.append(item_id[it]) # Handle songs not in item_id
        except:
            pass
    if len(input_id) == 1: # Model has to predict on more than one item
        input_id.append(input_id[0])
    if len(input_id) == 0: #If no known input, randomly select songs
        input_id.append(1)
        input_id.append(2)
    rec_playlist.append(top_500_dic(input_id))
```




```python
with open('score_500_full.json', 'w') as outfile:
    json.dump(rec_playlist, outfile)
```

