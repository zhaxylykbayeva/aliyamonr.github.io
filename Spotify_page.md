## Spotify Project

**Project description:** The goal of this project is to predict how popular a song would be, based on its features. Spotify might be interested in popularity prediction to decide which songs to recommend to their users. Moreover, this analysis would help them make data-driven decisions when deciding things like how much to pay for song licenses.

**Dataset Description:**
The dataset I will be using for this analysis is from Kaggle and can be accessed here: https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db
 
Most of the columns are self-explanatory, and represent a feature of a song (e.g. genre, name, loudness, acousticness, duration, etc.) Popularity column is an integer number between 0 and 100.



### 1. Which features have the most effect on popularity?

As I  ran two random forests and CV Grid Search on the entire dataset, I realized that the most logical approach would be to analyze this dataset splitting it up by genre. (People who are into Electronic Dance Music probably care about danceability the most, while people who are into classical music put more importance on instrumentalness.)

```javascript
genre=np.unique(songs['genre'])
rmse2=[]
for i in genre:
    temp=songs[songs.genre == i]
    y = temp['popularity']
    x = temp[['danceability','duration_ms','energy','instrumentalness','loudness','liveness','speechiness','tempo', 'valence']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    randomforest = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=200, n_jobs=-1)
    randomforest.fit(x_train, y_train)
    yhat_test=randomforest.predict(x_test)
    res_hat = yhat_test-y_test
    rmse2.append((round(np.sqrt(sum(res_hat**2)/len(yhat_test)),3)))
    
}
```

### 2. Here are the results of root mean squared error by genre:

<img src="/predictions by genre.png" alt="hi" class="inline"/>

Even with less data in each training set, when analyzing by genre, the testing error (RMSE) was lower than the overall error. While some genres like Children's Music or Reggaeton are harder to predict with high accuracy, this experiment has proven that it is very important to take descriptive features like genre into account.

### 3. Now, I'd like to ask another interesting question - what is the effect of each song characteristic on its popularity on average? 

To perform that analysis, I'll use linear regression:

```javascript
import statsmodels.api as sm

# Add important features
x=songs[['danceability','duration_ms','energy','instrumentalness','loudness','liveness','speechiness','tempo', 'valence']]
x['duration']=x['duration_ms'].div(60000)
# Popularity is likely nonlinear in duration, so I added a column with squared values of duration.
x['duration_sq']= np.square(x['duration'])
x=x.drop(axis=1,columns='duration_ms')

y=songs['popularity']

# Train and fit linear regression
lm=sm.OLS(exog=x, endog=y, hasconst=True)
lm_res = lm.fit()
lm_res.summary()
} 
```

### 4. Conclusion

From linear regression analysis I performed above estimates that **danceability** had the largest per unit effect on popularity, followed by **energy**. 

Speechiness and valence, on the other hand, had the biggest negative effect on popularity. Longer songs are more popular, but there's a diminishing return, indicated by the negative quadratic term (duration_sq).

<img src="images/linear regression results.png" alt="linearregression" width="600" height="600" class="inline"/>


You can check out the entire project linked <a href="Spotify Project.ipynb" title="here.">here.</a>
