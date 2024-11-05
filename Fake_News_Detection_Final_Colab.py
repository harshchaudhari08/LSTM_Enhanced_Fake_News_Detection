#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP libraries to clean the text data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Vectorization technique TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# For Splitting the dataset
from sklearn.model_selection import train_test_split

# Model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Accuracy measuring library
from sklearn.metrics import accuracy_score


# In[3]:


data = pd.read_csv('data.csv')


# In[4]:


data.shape #Returns the number of rows and columns present in the dataset


# In[5]:


data.head()  # Returns the first 5 rows of the dataset


# In[6]:


data.columns # Returns the column headings


# In[7]:


data.isnull().sum() #To check the null values in the dataset, if any


# In[8]:


df = data.copy() #Creating a copy of my data, I will be working on this Dataframe


# In[9]:


df['Body'] = df['Body'].fillna('')   # As Body is empty, just filled with an empty space


# In[10]:


df.isnull().sum()  # No null values found


# In[11]:


df['News'] = df['Headline']+df['Body']


# In[12]:


df.head()


# In[13]:


df.columns


# In[14]:


features_dropped = ['URLs','Headline','Body']
df = df.drop(features_dropped, axis =1)


# In[15]:


ps = PorterStemmer()
def wordopt(text):
    text = re.sub('[^a-zA-Z]', ' ',text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text


# In[16]:


# from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# In[17]:


df['News'] = df['News'].apply(wordopt) #Applying the text processing techniques onto every row data


# In[18]:


X = df['News']
Y = df['Label']

#Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# In[19]:


#Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[20]:


#1. Logistic Regression - used because this model is best suited for binary classification
LR_model = LogisticRegression()

#Fitting training set to the model
LR_model.fit(xv_train,y_train)

#Predicting the test set results based on the model
lr_y_pred = LR_model.predict(xv_test)

#Calculate the accurracy of this model
score = accuracy_score(y_test,lr_y_pred)
print('Accuracy of LR model is ', score)


# In[21]:


#2. Support Vector Machine(SVM) - SVM works relatively well when there is a clear margin of separation between classes.
svm_model = SVC(kernel='linear')

#Fitting training set to the model
svm_model.fit(xv_train,y_train)

#Predicting the test set results based on the model
svm_y_pred = svm_model.predict(xv_test)

#Calculate the accuracy score of this model
score = accuracy_score(y_test,svm_y_pred)
print('Accuracy of SVM model is ', score)
from sklearn.metrics import f1_score, recall_score

# Calculate F1-score
f1 = f1_score(y_test, svm_y_pred)

# Calculate Recall
recall = recall_score(y_test, svm_y_pred)

# Print the F1-score and Recall
print('F1 Score of SVM model is', f1)
print('Recall of SVM model is', recall)


# In[22]:


#3. Random Forest Classifier
RFC_model = RandomForestClassifier(random_state=0)

#Fitting training set to the model
RFC_model.fit(xv_train, y_train)

#Predicting the test set results based on the model
rfc_y_pred = RFC_model.predict(xv_test)

#Calculate the accuracy score of this model
score = accuracy_score(y_test,rfc_y_pred)
print('Accuracy of RFC model is ', score)


# In[23]:


# As SVM is able to provide best results - SVM will be used to check the news liability

def fake_news_det(news):
    input_data = {"text":[news]}
    new_def_test = pd.DataFrame(input_data)
    # new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    #print(new_x_test)
    vectorized_input_data = vectorization.transform(new_x_test)
    prediction = svm_model.predict(vectorized_input_data)

    if prediction == 1:
        print("Not a Fake News")
    else:
        print("Fake News")


# In[33]:


fake_news_det("""(Reuters) - The Weinstein Co has fired co-Chairman Harvey Weinstein, effective immediately, following reports of sexual harassment allegations against the executive, who was one of Hollywood’s biggest power brokers, the film production company said on Sunday.
The departure leaves Weinstein’s brother Bob, a co-chairman, and chief operating officer David Glasser at the helm of the company.
The board of directors made the decision “in light of new information about misconduct by Harvey Weinstein that has emerged in the past few days,” the company said in an emailed statement, adding that he had been notified.
A spokeswoman for the executive did not immediately respond to a request for comment.
The company said on Friday that Weinstein, 65, was taking an indefinite leave of absence after the New York Times reported that he had made eight settlements with women who had accused him of unwanted physical contact and sexual harassment over three decades.
Weinstein has produced or distributed Oscar-winning movies including “Shakespeare in Love” and “Chicago.” He was a prominent donor to Democrats during the 2016 general election campaign.
The company also said it was conducting its own internal investigation.
Reporting by Sangameswaran S in Bengaluru; Additional reporting and writing by Hilary Russ in New York; Editing by Richard Chang""")


# In[25]:


fake_news_det("WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a â€œfiscal conservativeâ€ on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBSâ€™ â€œFace the Nation,â€ drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense â€œdiscretionaryâ€ spending on programs that support education, scientific research, infrastructure, public health and environmental protection. â€œThe (Trump) administration has already been willing to say: â€˜Weâ€™re going to increase non-defense discretionary spending ... by about 7 percent,â€™â€ Meadows, chairman of the small but influential House Freedom Caucus, said on the program. â€œNow, Democrats are saying thatâ€™s not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I donâ€™t see where the rationale is. ... Eventually you run out of other peopleâ€™s money,â€ he said. Meadows was among Republicans who voted in late December for their partyâ€™s debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. â€œItâ€™s interesting to hear Mark talk about fiscal responsibility,â€ Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. â€œThis is one of the least ... fiscally responsible bills weâ€™ve ever seen passed in the history of the House of Representatives. I think weâ€™re going to be paying for this for many, many years to come,â€ Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or â€œentitlement reform,â€ as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, â€œentitlementâ€ programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryanâ€™s early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the â€œDreamers,â€ people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. â€œWe need to do DACA clean,â€ she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid")


# In[26]:


# Implementing Long Short Term memory(LSTM)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['News'])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(df['News'])
input_sequences = pad_sequences(input_sequences)


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(input_sequences, Y, test_size=0.25)


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(total_words, 100, input_length=input_sequences.shape[1]))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# In[29]:


scores = model.evaluate(x_test, y_test)
print('LSTM Model Accuracy: %.2f%%' % (scores[1] * 100))


# In[30]:


def fake_news_det_lstm(news):
    input_data = {"text":[news]}
    new_def_test = pd.DataFrame(input_data)
    new_x_test = new_def_test["text"]

    # Tokenize and pad the input sequence
    input_sequence = tokenizer.texts_to_sequences(new_x_test)
    input_sequence = pad_sequences(input_sequence, maxlen=input_sequences.shape[1])

    # Predict using the LSTM model
    prediction = model.predict(input_sequence)

    if prediction > 0.5:
        print("Not a Fake News")
    else:
        print("Fake News")


# In[37]:


news_content = """When Pulisic tore open the left side of the Panama defense 11 minutes later, slotting a cross that Altidore simply poked into the net, the game — 19 minutes old — seemed over. The cherry on top — an Altidore penalty kick, on a nervy chip after Bobby Wood had been cut down on a run into the area in the 42nd minute — erased all doubt.
The day had been saved.
After Wood added a fourth in the 63rd minute, the worries, the fears of the past few days, and the past few months, were gone. It had gone so well, in fact, that United States Coach Bruce Arena could even feign the tiniest bit of disappointment afterward.
“On the night, we actually didn’t finish well,” he said. “We could have scored a lot more goals.”
Panama Coach Hernán Darío Gómez would hardly disagree. “They could have had 10 goals,” he said in declaring the Americans “immensely superior” in every way in the match, and his team’s play as “muy, muy mal” — very, very bad.
The only concern of the Americans afterward, then, was for Pulisic, who was battered several times and then, finally, so badly that he required treatment after a particularly rough foul at midfield just after halftime.
Newsletter Sign Up Continue reading the main story Please verify you're not a robot by clicking the box. Invalid email address. Please re-enter. You must select a newsletter to subscribe to. Sign Up You agree to receive occasional updates and special offers for The New York Times's products and services. Thank you for subscribing. An error has occurred. Please try again later. View all New York Times newsletters.
Arena, realizing Pulisic’s unique value, and the fact that he may need a similar performance to close the deal on Tuesday, had seen enough. He substituted for Pulisic a few minutes later.
“He took a few shots,” Arena said. “And so we thought it was smart to get him off the field.”
Pulisic acknowledged the beating, a fact of his life that has become common in games against Concacaf opposition, and shrugged it off. “I got kicked a few times, but I’ll be fine,” he said.
Advertisement Continue reading the main story
That may or may not be the case. After answering about a dozen questions from reporters, he slipped away while teammates took their turns with the news media. Moments later, far away from the scrum, Pulisic was limping noticeably.
He had done enough, though, to nearly push his team over the qualifying line.
The decisive 90 minutes will come on Tuesday, when the United States faces Trinidad and Tobago on the road. The Soca Warriors sit last in the group, their hopes long dashed and their pride potentially bruised depending on the result of a game at the group leader Mexico later Friday night. They would seem to have little motivation at this point other than to play spoiler.
That can be an attractive bit of motivation for small Concacaf teams, however, especially when there is a chance to ding the mighty United States. The Americans know that, and they know the stakes.
And if they have learned anything on their long qualifying road — one that began with two humbling losses and the firing of their last coach, Jurgen Klinsmann — it is that nothing is certain, and that even a country accustomed to a quadrennial World Cup trip has to earn its place on the field before it can pack its bags.
“The job is half-done,” defender Matt Besler said. “Now we have to go down there and do the rest."""
fake_news_det_lstm(news_content)

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load your dataset
data = pd.read_csv('data.csv')

# Data preprocessing
# ...

# Combine 'Headline' and 'Body' into 'News'
data['News'] = data['Headline'] + data['Body']

# Drop unnecessary columns
features_dropped = ['URLs', 'Headline', 'Body']
data = data.drop(features_dropped, axis=1)

# Text processing
# ...

# Split the dataset
X = data['News']
Y = data['Label']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Base Models
lr_model = LogisticRegression()
svm_model = SVC(kernel='linear', probability=True)
rfc_model = RandomForestClassifier(random_state=0)

# LSTM Model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['News'])
total_words = len(tokenizer.word_index) + 1
input_sequences = tokenizer.texts_to_sequences(data['News'])
input_sequences = pad_sequences(input_sequences)
x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm = train_test_split(input_sequences, Y, test_size=0.25)
lstm_model = Sequential()
lstm_model.add(Embedding(total_words, 100, input_length=input_sequences.shape[1]))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(x_train_lstm, y_train_lstm, epochs=5)

# Ensemble Model - Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('svm', svm_model),
        ('rfc', rfc_model),
        ('lstm', lstm_model)
    ],
    voting='soft'  # Use 'soft' for probability voting
)

# Fit the ensemble model
voting_classifier.fit(xv_train, y_train)

# Predictions
y_pred_ensemble = voting_classifier.predict(xv_test)

# Evaluate accuracy
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Model Accuracy: {accuracy_ensemble * 100:.2f}%')



# In[ ]:




