---
title: Building a Hybrid MRS for Spotify
---

## AC209A Data Science Final Project

Group 9. Chuqiao Yuan, Feiyu Chen, Queena Ziyi Zhou, Rong Liu

### Overview

There are many leading music streaming services in the market, including Spotify, Pandora, and Apple Music. Thus, a robust Music Recommender System (MRS) is in high demand to improve user experience and increase user engagement. Many researchers have focused their work on traditional recommender systems, but MSR differs from these recommender systems because the duration of a song is short (3~5 minutes), the total number of existing songs is large (tens of millions), and songs are consumed in sessions, among which the same user’s taste may change.
- **Goal**: ​Our goal is to build a robust MRS to recommend songs to the users given a set of songs the users have been listened to.
- **Approach**: W​ e would like to hybridize two models: 1) Collaborative filtering, recommending songs based on similar users’ past preferences; 2) Content-based recommendation, using NLP techniques on song titles and lyrics to detect mood, and analyzing sound features to detect genre.

### Motivation

### Literature Review