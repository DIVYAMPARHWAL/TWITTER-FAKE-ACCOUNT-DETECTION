#!/usr/bin/env python
# coding: utf-8

# # Detecting Twitter Fake Accounts

# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import time
from sklearn import metrics
mpl.rcParams['patch.force_edgecolor'] = True
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


file= 'training_data.csv'
training_data = pd.read_csv(file)


# In[62]:


training_data.head()


# In[63]:


# printing all columns
count = 1
for col in training_data.columns:
  print(f'{count} -> {col}')
  count += 1


# In[64]:


training_data.info()


# In[65]:


# Statistical details of the data

training_data.describe()


# In[66]:


# Returns True if there are any NaN values and Returns False otherwise

training_data.isnull().values.any()


# In[67]:


# Total Number of NaN values

training_data.isnull().sum().sum()


# In[68]:


bots = training_data[training_data.bot==1]
nonbots = training_data[training_data.bot==0]


# #### Number of Real Accounts vs Number of Fake Accounts

# In[69]:


# value counts of humans and bots

human_bots = training_data.bot
human_bots.value_counts()


# In[70]:


count_real = human_bots.value_counts()[0]
count_fake = human_bots.value_counts()[1]


# In[71]:


plt.figure(figsize=(6, 6))
sns.barplot(x = ['Real Account', 'Fake Account'], y = [count_real, count_fake], palette = sns.color_palette('magma'))
plt.title("Number of Entries by Account Type", fontname="Times New Roman", fontsize=21, fontweight="bold")
sns.despine()


# #### Identifying Missing data

# In[72]:


# Column wise NaN value count

training_data.isnull().sum()


# In[73]:


def get_heatmap(df):
    #This function gives heatmap of all NaN values
    plt.figure(figsize=(29,21))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='plasma')
    sns.set(font_scale=15.0)
    plt.title('Heatmap of all NaN values', fontname="Times New Roman", fontsize=50, fontweight="bold")
    return plt.show()

get_heatmap(training_data)


# In[74]:


sns.set(font_scale=1)


# In[75]:


# Returns list of columns with NaN values

training_data.columns[training_data.isna().any()].tolist()


# #### ✈ We won't be using 'description', 'lang', 'profile_background_image_url', 'profile_image_url' in our analysis. So, we can directly convert the NaN to string directly.
# 
# #### ✈ But for 'location' -> NaN values can be changed to 'unknown'

# In[76]:


# checking the dataframe for values for 'location' as NaN

training_data[pd.isnull(training_data['location'])]


# In[77]:


# Most accounts have not disclosed their location. So, unknown is the mode (i.e. the account didn't disclose it for privacy reasons)
training_data['location'].value_counts().keys()[0]

# mode of column 'location' == 'unknown'
training_data['location'].mode()

# we replace NaN values with mode of the column 'location'
training_data.fillna({'location': training_data['location'].mode()}, inplace=True)


# In[78]:


bots.friends_count/bots.followers_count

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.title('Fake Accounts Friends vs Followers', fontname="Times New Roman", fontsize=24, fontweight="bold")
sns.regplot(bots.friends_count, bots.followers_count, color='purple', label='Fake Accounts')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()

plt.subplot(2,1,2)
plt.title('Real Accounts Friends vs Followers', fontname="Times New Roman", fontsize=24, fontweight="bold")
sns.regplot(nonbots.friends_count, nonbots.followers_count, color='blue', label='Real Accounts')
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()


# #### Identifying Imbalance in the data

# In[79]:


bots['friends_by_followers'] = bots.friends_count/bots.followers_count
bots[bots.friends_by_followers<1].shape


# In[80]:


nonbots['friends_by_followers'] = nonbots.friends_count/nonbots.followers_count
nonbots[nonbots.friends_by_followers<1].shape


# In[81]:


plt.figure(figsize=(10,5))
plt.plot(bots.listed_count, color='purple', label='Fake Accounts')
plt.plot(nonbots.listed_count, color='blue', label='Real Accounts')
plt.legend(loc='upper left')
plt.ylim(10000,20000)
print(bots[(bots.listed_count<5)].shape)


# In[82]:


bots_listed_count_df = bots[bots.listed_count<16000]
nonbots_listed_count_df = nonbots[nonbots.listed_count<16000]

bots_verified_df = bots_listed_count_df[bots_listed_count_df.verified==False]
bots_screenname_has_bot_df_ = bots_verified_df[(bots_verified_df.screen_name.str.contains("bot", case=False)==True)].shape


# In[83]:


plt.figure(figsize=(12,7))

plt.subplot(2,1,1)
plt.plot(bots_listed_count_df.friends_count, color='purple', label='Fake Accounts Friends')
plt.plot(nonbots_listed_count_df.friends_count, color='blue', label='Real Accounts Friends')
plt.legend(loc='upper left')

plt.subplot(2,1,2)
plt.plot(bots_listed_count_df.followers_count, color='purple', label='Fake Accounts Followers')
plt.plot(nonbots_listed_count_df.followers_count, color='blue', label='Real Accounts Followers')
plt.legend(loc='upper left')


# In[84]:


#bots[bots.listedcount>10000]
condition = (bots.screen_name.str.contains("bot", case=False)==True)|(bots.description.str.contains("bot", case=False)==True)|(bots.location.isnull())|(bots.verified==False)

bots['screen_name_binary'] = (bots.screen_name.str.contains("bot", case=False)==True)
bots['location_binary'] = (bots.location.isnull())
bots['verified_binary'] = (bots.verified==False)
bots.shape


# In[85]:


condition = (nonbots.screen_name.str.contains("bot", case=False)==False)| (nonbots.description.str.contains("bot", case=False)==False) |(nonbots.location.isnull()==False)|(nonbots.verified==True)

nonbots['screen_name_binary'] = (nonbots.screen_name.str.contains("bot", case=False)==False)
nonbots['location_binary'] = (nonbots.location.isnull()==False)
nonbots['verified_binary'] = (nonbots.verified==True)

nonbots.shape


# In[86]:


df = pd.concat([bots, nonbots])
df.shape


# ### Feature Independence using Spearman correlation

# In[87]:


df.corr(method='spearman')


# In[88]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(method='spearman'), cmap='RdBu_r', annot=True)
plt.tight_layout()
plt.show()


# Result:
# - There is no correlation between **id, statuses_count, default_profile, default_profile_image** and target variable.
# - There is strong correlation between **verified, listed_count, friends_count, followers_count** and target variable.
# - We cannot perform correlation for categorical attributes. So we will take **screen_name, name, description, status** into feature engineering. While use **verified, listed_count** for feature extraction.

# #### Performing Feature Engineering

# In[89]:


file= open('training_data.csv', mode='r', encoding='utf-8', errors='ignore')

training_data = pd.read_csv(file)

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget'                     r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon'                     r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb'                     r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
            
training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)


# #### Performing Feature Extraction

# In[90]:


training_data['listed_count_binary'] = (training_data.listed_count>20000)==False
features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']


# ## Implementing Different Models

# ## Decision Tree Classifier

# In[91]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dt = dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

# Confusion matrix
train_cm = confusion_matrix(y_train, y_pred_train)
test_cm = confusion_matrix(y_test, y_pred_test)
print("Train Confusion Matrix:\n", train_cm,"\n")
print("Test Confusion Matrix:\n", test_cm,"\n")

# Getting performance metrics
train_precision_decisiontree = precision_score(y_train, y_pred_train)
test_precision_decisiontree = precision_score(y_test, y_pred_test)
train_recall_decisiontree = recall_score(y_train, y_pred_train)
test_recall_decisiontree = recall_score(y_test, y_pred_test)
train_f1_score_decisiontree = f1_score(y_train, y_pred_train)
test_f1_score_decisiontree = f1_score(y_test, y_pred_test)

# Print performance metrics
print(f'Train precision: {train_precision_decisiontree:.5f}')
print(f'Test precision: {test_precision_decisiontree:.5f}\n')
print(f'Train recall: {train_recall_decisiontree:.5f}')
print(f'Test recall: {test_recall_decisiontree:.5f}\n')
print(f'Train F1-score: {train_f1_score_decisiontree:.5f}')
print(f'Test F1-score: {test_f1_score_decisiontree:.5f}\n')

# Getting and Printing the train and test accuracy of the classifier
train_accuracy_decisiontree = accuracy_score(y_train, y_pred_train)
test_accuracy_decisiontree = accuracy_score(y_test, y_pred_test)
print("Trainig Accuracy: %.5f" %train_accuracy_decisiontree)
print("Test Accuracy: %.5f" %test_accuracy_decisiontree)


# In[92]:


sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})

scores_train = dt.predict_proba(X_train)
scores_test = dt.predict_proba(X_test)

y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])

for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])
    
fpr_dt_train, tpr_dt_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_dt_test, tpr_dt_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

plt.plot(fpr_dt_train, tpr_dt_train, color='black', label='Train AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
plt.plot(fpr_dt_test, tpr_dt_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
plt.title("Decision Tree ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')


# #### Result: Decision Tree gives very good performance and generalizes well. But it may be overfitting as AUC is 0.93, so we will try other models.

# ## Multinomial Naive Bayes Classifier

# In[93]:


from sklearn.naive_bayes import MultinomialNB

X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

mnb = MultinomialNB(alpha=0.0009)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

mnb = mnb.fit(X_train, y_train)
y_pred_train = mnb.predict(X_train)
y_pred_test = mnb.predict(X_test)

# Confusion matrix
train_cm = confusion_matrix(y_train, y_pred_train)
test_cm = confusion_matrix(y_test, y_pred_test)
print("Train Confusion Matrix:\n", train_cm,"\n")
print("Test Confusion Matrix:\n", test_cm,"\n")

# Getting performance metrics
train_precision_naive_bayes = precision_score(y_train, y_pred_train)
test_precision_naive_bayes = precision_score(y_test, y_pred_test)
train_recall_naive_bayes = recall_score(y_train, y_pred_train)
test_recall_naive_bayes = recall_score(y_test, y_pred_test)
train_f1_score_naive_bayes = f1_score(y_train, y_pred_train)
test_f1_score_naive_bayes = f1_score(y_test, y_pred_test)

# Print performance metrics
print(f'Train precision: {train_precision_naive_bayes:.5f}')
print(f'Test precision: {test_precision_naive_bayes:.5f}\n')
print(f'Train recall: {train_recall_naive_bayes:.5f}')
print(f'Test recall: {test_recall_naive_bayes:.5f}\n')
print(f'Train F1-score: {train_f1_score_naive_bayes:.5f}')
print(f'Test F1-score: {test_f1_score_naive_bayes:.5f}\n')

# Getting and Printing the train and test accuracy of the classifier
train_accuracy_naive_bayes = accuracy_score(y_train, y_pred_train)
test_accuracy_naive_bayes = accuracy_score(y_test, y_pred_test)
print("Trainig Accuracy: %.5f" %train_accuracy_naive_bayes)
print("Test Accuracy: %.5f" %test_accuracy_naive_bayes)


# In[94]:


sns.set_style("whitegrid", {'axes.grid' : False})

scores_train = mnb.predict_proba(X_train)
scores_test = mnb.predict_proba(X_test)

y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])

for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])
    
fpr_mnb_train, tpr_mnb_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_mnb_test, tpr_mnb_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

plt.plot(fpr_mnb_train, tpr_mnb_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_mnb_train, tpr_mnb_train))
plt.plot(fpr_mnb_test, tpr_mnb_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_mnb_test, tpr_mnb_test))
plt.title("Multinomial Naive Bayes ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')


# #### Result: Clearly, Multinomial Niave Bayes peforms poorly and is not a good choice as the Train AUC is just 0.556 and Test is 0.555.

# # Random Forest Classifier

# In[95]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

rf = rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Confusion matrix
train_cm = confusion_matrix(y_train, y_pred_train)
test_cm = confusion_matrix(y_test, y_pred_test)
print("Train Confusion Matrix:\n", train_cm,"\n")
print("Test Confusion Matrix:\n", test_cm,"\n")

# Print performance metrics
train_precision_random_forest = precision_score(y_train, y_pred_train)
test_precision_random_forest = precision_score(y_test, y_pred_test)
train_recall_random_forest = recall_score(y_train, y_pred_train)
test_recall_random_forest = recall_score(y_test, y_pred_test)
train_f1_score_random_forest = f1_score(y_train, y_pred_train)
test_f1_score_random_forest = f1_score(y_test, y_pred_test)

# Print performance metrics
print(f'Train precision: {train_precision_random_forest:.5f}')
print(f'Test precision: {test_precision_random_forest:.5f}\n')
print(f'Train recall: {train_recall_random_forest:.5f}')
print(f'Test recall: {test_recall_random_forest:.5f}\n')
print(f'Train F1-score: {train_f1_score_random_forest:.5f}')
print(f'Test F1-score: {test_f1_score_random_forest:.5f}\n')

# Getting and Printing the train and test accuracy of the classifier
train_accuracy_random_forest = accuracy_score(y_train, y_pred_train)
test_accuracy_random_forest = accuracy_score(y_test, y_pred_test)
print("Trainig Accuracy: %.5f" %train_accuracy_random_forest)
print("Test Accuracy: %.5f" %test_accuracy_random_forest)


# In[96]:


sns.set_style("whitegrid", {'axes.grid' : False})

scores_train = rf.predict_proba(X_train)
scores_test = rf.predict_proba(X_test)

y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])

for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])
    
fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)

plt.plot(fpr_rf_train, tpr_rf_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_rf_train, tpr_rf_train))
plt.plot(fpr_rf_test, tpr_rf_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_rf_test, tpr_rf_test))
plt.title("Random ForestROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')


# # Own Classifier

# In[97]:


class twitter_bot(object):
    def __init__(self):
        pass

    def perform_train_test_split(df):
        msk = np.random.rand(len(df)) < 0.75
        train, test = df[msk], df[~msk]
        X_train, y_train = train, train.iloc[:,-1]
        X_test, y_test = test, test.iloc[:, -1]
        return (X_train, y_train, X_test, y_test)

    def get_heatmap(df):
        # This function gives heatmap of all NaN values
        plt.figure(figsize=(29,21))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='plasma')
        plt.tight_layout()
        return plt.show()

    def bot_prediction_algorithm(df):
        # creating copy of dataframe
        train_df = df.copy()
        # performing feature engineering on id and verfied columns
        # converting id to int
        train_df['id'] = train_df.id.apply(lambda x: int(x))
        #train_df['friends_count'] = train_df.friends_count.apply(lambda x: int(x))
        train_df['followers_count'] = train_df.followers_count.apply(lambda x: 0 if x=='None' else int(x))
        train_df['friends_count'] = train_df.friends_count.apply(lambda x: 0 if x=='None' else int(x))
        #We created two bag of words because more bow is stringent on test data, so on all small dataset we check less
        if train_df.shape[0]>600:
            #bag_of_words_for_bot
            bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget'                            r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon'                            r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb'                            r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
        else:
            # bag_of_words_for_bot
            bag_of_words_bot = r'bot|b0t|cannabis|mishear|updates every'

        # converting verified into vectors
        train_df['verified'] = train_df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)

        # check if the name contains bot or screenname contains b0t
        condition = ((train_df.name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.description.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.status.str.contains(bag_of_words_bot, case=False, na=False))
                     )  # these all are bots
        predicted_df = train_df[condition]  # these all are bots
        predicted_df.bot = 1
        predicted_df = predicted_df[['id', 'bot']]

        # check if the user is verified
        verified_df = train_df[~condition]
        condition = (verified_df.verified == 1)  # these all are nonbots
        predicted_df1 = verified_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if description contains buzzfeed
        buzzfeed_df = verified_df[~condition]
        condition = (buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False))  # these all are nonbots
        predicted_df1 = buzzfeed_df[buzzfeed_df.description.str.contains("buzzfeed", case=False, na=False)][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        # check if listed_count>16000
        listed_count_df = buzzfeed_df[~condition]
        listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: 0 if x == 'None' else x)
        listed_count_df.listed_count = listed_count_df.listed_count.apply(lambda x: int(x))
        condition = (listed_count_df.listed_count > 16000)  # these all are nonbots
        predicted_df1 = listed_count_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1])

        #remaining
        predicted_df1 = listed_count_df[~condition][['id', 'bot']]
        predicted_df1.bot = 0 # these all are nonbots
        predicted_df = pd.concat([predicted_df, predicted_df1])
        return predicted_df

    def get_predicted_and_true_values(features, target):
        y_pred, y_true = twitter_bot.bot_prediction_algorithm(features).bot.tolist(), target.tolist()
        return (y_pred, y_true)
    
    def get_confusion_matrix(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        train_cm = confusion_matrix(y_train, y_pred_train)
        test_cm = confusion_matrix(y_test, y_pred_test)
        print("Train Confusion Matrix:\n", train_cm,"\n")
        print("Test Confusion Matrix:\n", test_cm,"\n")
        
    def get_performance_metrics(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        
        # Getting performance metrics
        train_precision = precision_score(y_train, y_pred_train)
        test_precision = precision_score(y_test, y_pred_test)
        train_recall = recall_score(y_train, y_pred_train)
        test_recall = recall_score(y_test, y_pred_test)
        train_f1_score = f1_score(y_train, y_pred_train)
        test_f1_score = f1_score(y_test, y_pred_test)
        return train_precision, test_precision, train_recall, test_recall, train_f1_score, test_f1_score

    def get_accuracy_score(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        # predictions on training data
        y_pred_train, y_true_train = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        train_acc = metrics.accuracy_score(y_pred_train, y_true_train)
        #predictions on test data
        y_pred_test, y_true_test = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        test_acc = metrics.accuracy_score(y_pred_test, y_true_test)
        return (train_acc, test_acc)

    def plot_roc_curve(df):
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid", {'axes.grid': False})
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        # Train ROC
        y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        scores = np.linspace(start=0.01, stop=0.9, num=len(y_true))
        fpr_train, tpr_train, threshold = metrics.roc_curve(y_pred_train, scores, pos_label=0)
        plt.plot(fpr_train, tpr_train, label='Train AUC: %5f' % metrics.auc(fpr_train, tpr_train), color='darkblue')
        #Test ROC
        y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        scores = np.linspace(start=0.01, stop=0.9, num=len(y_true))
        fpr_test, tpr_test, threshold = metrics.roc_curve(y_pred_test, scores, pos_label=0)
        plt.plot(fpr_test,tpr_test, label='Test AUC: %5f' %metrics.auc(fpr_test,tpr_test), ls='--', color='red')
        #Misc
        plt.xlim([-0.1,1])
        plt.title("Reciever Operating Characteristic (ROC)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right')
        plt.show()


# In[98]:


if __name__ == '__main__':
    start = time.time()
    print("Training the Bot detection classifier. Please wait a few seconds.\n")
    train_df = pd.read_csv('training_data.csv')
    test_df = pd.read_csv('test_data.csv', sep='\t', encoding='ISO-8859-1')
    twitter_bot.get_confusion_matrix(train_df)
    train_precision, test_precision, train_recall, test_recall, train_f1_score, test_f1_score=twitter_bot.get_performance_metrics(train_df)
    print(f'Train precision: {train_precision:.5f}')
    print(f'Test precision: {test_precision:.5f}\n')
    print(f'Train recall: {train_recall:.5f}')
    print(f'Test recall: {test_recall:.5f}\n')
    print(f'Train F1-score: {train_f1_score:.5f}')
    print(f'Test F1-score: {test_f1_score:.5f}\n')
    train_accuracy=twitter_bot.get_accuracy_score(train_df)[0]
    test_accuracy=twitter_bot.get_accuracy_score(train_df)[1]
    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy, "\n")

    #predicting test data results
    predicted_df = twitter_bot.bot_prediction_algorithm(test_df)
    # preparing subission file
    predicted_df.to_csv('submission.csv', index=False)
    print("Predicted results are saved to submission.csv. File shape: {}".format(predicted_df.shape))
    #ssss: " ,twitter_bot.get_bot_predictions(test_df))
    #plotting the ROC curve
    twitter_bot.plot_roc_curve(train_df)
    print("Time duration: {} seconds.".format(time.time()-start))


# ## ROC Comparison after tuning the baseline model 

# In[99]:


plt.figure(figsize=(21,16))
(X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)

#Train ROC
y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
scores = np.linspace(start=0, stop=1, num=len(y_true))
fpr_botc_train, tpr_botc_train, threshold = metrics.roc_curve(y_pred_train, scores, pos_label=0)

#Train ROC
plt.subplot(2,2,1)
plt.plot(fpr_botc_train, tpr_botc_train, label='Our Classifier AUC: %5f' % metrics.auc(fpr_botc_train,tpr_botc_train), color='darkblue')
plt.plot(fpr_rf_train, tpr_rf_train, label='Random Forest AUC: %5f' %auc(fpr_rf_train, tpr_rf_train))
plt.plot(fpr_dt_train, tpr_dt_train, label='Decision Tree AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
plt.plot(fpr_mnb_train, tpr_mnb_train, label='Multinomial Naive Bayes AUC: %5f' %auc(fpr_mnb_train, tpr_mnb_train))
plt.title("Training Set ROC Curve", fontname="Times New Roman", fontsize=29, fontweight="bold")
plt.xlabel("False Positive Rate (FPR)", fontname="Times New Roman", fontsize=19, fontweight="bold")
plt.ylabel("True Positive Rate (TPR)", fontname="Times New Roman", fontsize=19, fontweight="bold")
plt.legend(loc='lower right')


# In[100]:


plt.figure(figsize=(21,16))
(X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)

#Test ROC
y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
scores = np.linspace(start=0, stop=1, num=len(y_true))
fpr_botc_test, tpr_botc_test, threshold = metrics.roc_curve(y_pred_test, scores, pos_label=0)

#Test ROC
plt.subplot(2,2,1)
plt.plot(fpr_botc_test,tpr_botc_test, label='Our Classifier AUC: %5f' %metrics.auc(fpr_botc_test,tpr_botc_test), color='darkblue')
plt.plot(fpr_rf_test, tpr_rf_test, label='Random Forest AUC: %5f' %auc(fpr_rf_test, tpr_rf_test))
plt.plot(fpr_dt_test, tpr_dt_test, label='Decision Tree AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
plt.plot(fpr_mnb_test, tpr_mnb_test, label='Multinomial Naive Bayes AUC: %5f' %auc(fpr_mnb_test, tpr_mnb_test))
plt.title("Test Set ROC Curve", fontname="Times New Roman", fontsize=29, fontweight="bold")
plt.xlabel("False Positive Rate (FPR)", fontname="Times New Roman", fontsize=19, fontweight="bold")
plt.ylabel("True Positive Rate (TPR)", fontname="Times New Roman", fontsize=19, fontweight="bold")
plt.legend(loc='lower right')


# # COMPARISON CHART OF ALL CLASSIFIERS

# In[126]:


# set width of bar
barWidth = 0.2
fig = plt.subplots(figsize =(50,26))

# set height of bar
Our_Classifier = [train_accuracy, train_precision, train_recall, train_f1_score]
Random_Forest = [train_accuracy_random_forest, train_precision_random_forest, train_recall_random_forest, train_f1_score_random_forest]
Decision_Tree = [train_accuracy_decisiontree, train_precision_decisiontree, train_recall_decisiontree, train_f1_score_decisiontree]
Naive_Bayes = [train_accuracy_naive_bayes, train_precision_naive_bayes, train_recall_naive_bayes, train_f1_score_naive_bayes]

# Set position of bar on X axis
br1 = np.arange(len(Our_Classifier))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, Our_Classifier, color ='r', width = barWidth, edgecolor ='grey', label ='Our Classifier')
plt.bar(br2, Random_Forest, color ='g', width = barWidth, edgecolor ='grey', label ='Random Forest')
plt.bar(br3, Decision_Tree, color ='b', width = barWidth, edgecolor ='grey', label ='Decision Tree')
plt.bar(br4, Naive_Bayes, color ='k', width = barWidth, edgecolor ='grey', label ='Mutinomial Naive Bayes')

# Adding x-axis and y-axis labels, xticks, yticks and title
plt.xlabel('\nEvaluation Metrics', fontname="Times New Roman", fontsize=88, fontweight="bold")
plt.ylabel('Score\n', fontname="Times New Roman", fontsize=88, fontweight="bold")
plt.title('Training Performance Metrics Comparison of All the Models\n', fontname="Times New Roman", fontsize=111, fontweight="bold")
plt.yticks(fontsize=44, fontweight="bold")
plt.xticks([r + barWidth for r in range(len(Our_Classifier))], ['Accuracy', 'Precision', 'Recall', 'f1_score'], fontname="Times New Roman", fontsize=46, fontweight="bold")

# Adding value labels above each bar
for i, v in enumerate(Our_Classifier):
    plt.text(r1[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")
for i, v in enumerate(Random_Forest):
    plt.text(r2[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")
for i, v in enumerate(Decision_Tree):
    plt.text(r3[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")
for i, v in enumerate(Naive_Bayes):
    plt.text(r4[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")

# Adding legend
plt.legend(loc='upper right', fontsize=36)

# Adjust the spacing between the legend and bars
plt.subplots_adjust(top=2.1)

# Display the chart

plt.show()


# In[127]:


# set width of bar
barWidth = 0.2
fig = plt.subplots(figsize =(50,26))

# set height of bar
Our_Classifier = [test_accuracy, test_precision, test_recall, test_f1_score]
Random_Forest = [test_accuracy_random_forest, test_precision_random_forest, test_recall_random_forest, test_f1_score_random_forest]
Decision_Tree = [test_accuracy_decisiontree, test_precision_decisiontree, test_recall_decisiontree, test_f1_score_decisiontree]
Naive_Bayes = [test_accuracy_naive_bayes, test_precision_naive_bayes, test_recall_naive_bayes, test_f1_score_naive_bayes]

# Set position of bar on X axis
br1 = np.arange(len(Our_Classifier))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, Our_Classifier, color ='r', width = barWidth, edgecolor ='grey', label ='Our Classifier')
plt.bar(br2, Random_Forest, color ='g', width = barWidth, edgecolor ='grey', label ='Random Forest')
plt.bar(br3, Decision_Tree, color ='b', width = barWidth, edgecolor ='grey', label ='Decision Tree')
plt.bar(br4, Naive_Bayes, color ='k', width = barWidth, edgecolor ='grey', label ='Mutinomial Naive Bayes')

# Adding x-axis and y-axis labels, xticks, yticks and title
plt.xlabel('\nEvaluation Metrics', fontname="Times New Roman", fontsize=88, fontweight="bold")
plt.ylabel('Score\n', fontname="Times New Roman", fontsize=88, fontweight="bold")
plt.title('Testing Performance Metrics Comparison of All the Models\n', fontname="Times New Roman", fontsize=111, fontweight="bold")
plt.yticks(fontsize=44, fontweight="bold")
plt.xticks([r + barWidth for r in range(len(Our_Classifier))], ['Accuracy', 'Precision', 'Recall', 'f1_score'], fontname="Times New Roman", fontsize=46, fontweight="bold")

# Adding value labels above each bar
for i, v in enumerate(Our_Classifier):
    plt.text(r1[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")
for i, v in enumerate(Random_Forest):
    plt.text(r2[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")
for i, v in enumerate(Decision_Tree):
    plt.text(r3[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")
for i, v in enumerate(Naive_Bayes):
    plt.text(r4[i], v + 0.01, f'{v:.3f}', color='black', ha='center', fontname="Times New Roman", fontsize=40, fontweight="bold")

# Adding legend
plt.legend(loc='upper right', fontsize=36)

# Adjust the spacing between the legend and bars
plt.subplots_adjust(top=2.1)

# Display the chart

plt.show()


# # THANKYOU !!!!
