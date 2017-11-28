X_train0, X_cv, y_train0, y_cv = train_test_split( X_train, y_train, test_size=0.20, random_state=42)
print "Ratio in training", str(sum(y_train0==1.0) * 100 /len(y_train0))
print "Ratio in test", str(sum(y_cv==1.0) * 100 /len(y_cv))


y_train0 = pd.DataFrame(y_train0)
X_train0 = pd.DataFrame(X_train0)

df_ranked

X_cv_sorted = X_cv[[df_ranked]]

for k in range(1,len(df_ranked)):

    xmat_filt = X_cv_sorted.loc[:,0:k]

    logit = sm.Logit(y_cv, xmat_filt)

    results = logit.fit()

    yhat = pd.DataFrame(results.predict(xmat_filt),columns=['predict'])

    yhat['predict'] = yhat['predict'].apply(lambda x: 0.0 if x < 0.5 else 1.0 )

    match =0.0
    for m in range(0,len(yhat)):
        if yhat.iloc[m].values == y_train.iloc[m].values : match += 1

    match = match/len(yhat)

    #logr_rs.append(results.prsquared)

    logr_rs.append(match)