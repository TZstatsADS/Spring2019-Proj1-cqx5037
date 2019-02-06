
# coding: utf-8

# # Happy Moment
#                     By Caihui Xiao (cx2225)

# In[9]:


from IPython.display import Image
Image(filename="./Happy.jpg")


# HappyDB is a corpus of 100,000 crowd-sourced happy moments. The goal of the corpus is to advance the state of the art of understanding the causes of happiness that can be gleaned from text.
# 
# We will focus on what make people have different type of happiness category.

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud


# In[4]:


data = pd.read_csv("processed_moments.csv")
data1 = pd.read_csv("demographic.csv")
data.count(axis=0) 


# We can see that there is no missing value so we do not worry about that.
# 
# We will do analysis about the relationship between sentences and Happiness category label

# In[11]:


combine_data = pd.merge(data,data1,left_on='wid',right_on='wid')
data_affection = data.loc[data['predicted_category']=='affection']
data_nature = data.loc[data['predicted_category']=='nature']
data_bonding = data.loc[data['predicted_category']=='bonding']
data_enjoy_the_moment = data.loc[data['predicted_category']=='enjoy_the_moment']
data_leisure = data.loc[data['predicted_category']=='leisure']
data_exercise = data.loc[data['predicted_category']=='exercise']
data_achievement = data.loc[data['predicted_category']=='achievement']
data_boxplot = pd.DataFrame({'affection':data_affection['num_sentence'],'nature':data_nature['num_sentence'],'bonding':data_bonding['num_sentence'],'leisure':data_leisure['num_sentence'],'exercise':data_exercise['num_sentence'],'achievement':data_achievement['num_sentence'],'enjoy the moment' : data_enjoy_the_moment['num_sentence']})
data_boxplot.boxplot()  
plt.show()


# From the boxplot, we can say that most of them only need few sentences. But for the affection category, the range is bigger than others. We will check what make this happen.

# In[12]:


com_affection =  combine_data.loc[combine_data['predicted_category']=='affection']

com_affection_marital_single = com_affection.loc[com_affection['marital'] =='single' ]
com_affection_marital_married = com_affection.loc[com_affection['marital'] =='married' ]
com_affection_marital_widowed = com_affection.loc[com_affection['marital'] =='widowed' ]
com_affection_marital_divorced = com_affection.loc[com_affection['marital'] =='divorced' ]

com_boxplot = pd.DataFrame({'single':com_affection_marital_single['num_sentence'],'married':com_affection_marital_married['num_sentence'],'widowed':com_affection_marital_widowed['num_sentence'],'divorced':com_affection_marital_divorced['num_sentence']})
com_boxplot.boxplot()  
plt.show()


# From the plot, we can say that if  people are in married, they are willing to spend more time to describe their happiness moment in affection than others. If they are single, maybe they also spend more time than others. I think they want to share their experiences about their relationship.

# In[13]:


com_affection_male = com_affection.loc[com_affection['gender'] =='m' ]
com_affection_female = com_affection.loc[com_affection['gender'] =='f' ]

com_boxplot1 = pd.DataFrame({'m':com_affection_male['num_sentence'],'f':com_affection_female['num_sentence']})
com_boxplot1.boxplot()  
plt.show()


# From this plot, we can say that man spend more time and write more sentences than womens. Maybe in reality, men do not want to show their feelings. They want to do it online.
# 
# In the next, we will check the frequency of important word for each catogory. 
# 
# We will do the affection at first. 

# In[15]:


tokenizer = RegexpTokenizer(r'\w+')

token1 = []
for i in data_affection['text']:
    words = tokenizer.tokenize(text=i)
    for word in words:
        token1.append(word.lower())

fdist1=FreqDist(token1)
fdist1.most_common(50)


# We can say that in this type, these words are most frequency. They are more care about family and want to share experience about important events like birthday.
# 
# We can say these in the following word cloud.

# In[16]:


wordcloud = WordCloud().generate(str(fdist1.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In the following, we will analysis the category of nature 

# In[17]:


token2 = []
for i in data_nature['text']:
    words = tokenizer.tokenize(text=i)
    for word in words:
        token2.append(word.lower())

fdist2=FreqDist(token2)
fdist2.most_common(50)


# In this type, we can say that people more care about weather.
# We can say these in the following word cloud.

# In[18]:


wordcloud = WordCloud().generate(str(fdist2.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In the following, we will analysis the category of bonding. 

# In[20]:


token3 = []
for i in data_bonding['text']:
    words = tokenizer.tokenize(text=i)
    for word in words:
        token3.append(word.lower())

fdist3=FreqDist(token2)
fdist3.most_common(50)


# In this type, we can say that people more care about weather and maybe go outside.We can say these in the following word cloud.
# 

# In[21]:


wordcloud = WordCloud().generate(str(fdist3.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In the following, we will analysis the category of enjoy_the_moment. 

# In[22]:


token4 = []
for i in data_enjoy_the_moment['text']:
    words = tokenizer.tokenize(text=i)
    for word in words:
        token4.append(word.lower())

fdist4=FreqDist(token4)
fdist4.most_common(50)


# In this type, we can say that people more care about the time with others.We can say these in the following word cloud.
# 

# In[23]:


wordcloud = WordCloud().generate(str(fdist4.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[35]:


In the following, we will analysis the category of leisure.


# In[33]:


token5 = []
for i in data_leisure['text']:
    words = tokenizer.tokenize(text=str(i))
    for word in words:
        token5.append(word.lower())

fdist5=FreqDist(token5)
fdist5.most_common(50)


# In this type, we can say that people more care about games or vedios.We can say these in the following word cloud.

# In[34]:



wordcloud = WordCloud().generate(str(fdist5.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In the following, we will analysis the category of exercise.

# In[28]:


token6 = []
for i in data_exercise['text']:
    words = tokenizer.tokenize(text=i)
    for word in words:
        token6.append(word.lower())

fdist6=FreqDist(token6)
fdist6.most_common(50)


# In this type, we can say that people more care about exercise like gym,yoga or other things.
# 
# We can say these in the following word cloud.

# In[29]:


wordcloud = WordCloud().generate(str(fdist6.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In the following, we will analysis the category of achievement.

# In[30]:



token7 = []
for i in data_achievement['text']:
    words = tokenizer.tokenize(text=i)
    for word in words:
        token7.append(word.lower())

fdist7=FreqDist(token7)
fdist7.most_common(50)


# In this type, we can say that people more care about jobs or study. We can see there are exam project. Maybe it is more care their study or career.
# 
# We can say these in the following word cloud.

# In[37]:


wordcloud = WordCloud().generate(str(fdist7.most_common(50)))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# # Conclusion

# We have analysis the relationship about sentences and category of happiness moment. In the category of affection, people like use more sentencs to describe their happiness experience. And if they are married or single, they would like to spend more time to write more sentences to describe their moment. And men wrote more sentences than women. Then, we found most frequency words in different categories. I think in different categories, they focus on different things. We can use this information to do future analysis. 
# 
