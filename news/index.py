import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import nltk as nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
from random import sample

nltk.download()
# nltk.download('punkt')

class PreProcessMessages:
    def __init__(self):
        self._stopwords = set(stopwords.words("english") + list(punctuation) + ["AT_USER", "URL"])

    def processMessages(self, list_of_messages):
        processedMessages = []
        for message in list_of_messages["status_message"]:
            processedMessages.append((self._processMessage(message)))
        
        return processedMessages

    def _processMessage (self, message):
        message = message.lower()
        message = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', message)
        message = re.sub('@[^\s]+', 'AT_USER', message)
        message = re.sub(r'#([^\s]+)', r'\1', message)
        # message = word_tokenize(message)
        for i in message.split():
            return i




iolNews = pd.read_csv("./data/citypress.co.za_facebook_statuses.csv")

status_messages = iolNews["status_message"]

# Remove null and duplicates and format table
formatted = iolNews[pd.notnull(iolNews["status_message"])].drop_duplicates(["status_message"])
removeAndSort = formatted.drop(["link_name", "status_type",  "status_id", "num_comments", "num_shares"], 1).sort_values(by=["num_likes"], ascending=False)


# print (removeAndSort)
# Init Class
processMessages = PreProcessMessages().processMessages(removeAndSort)
# print(processMessages)




#remove columns that arent needed and sort by date published




#Random sample set

# rindex = np.array(sample(xrange(len(removeAndSort)), 5))

#Get 5 random rows from removeAndSort
# randomSample = removeAndSort.ix(rindex)
# print (randomSample)

#Plots in matplotlib reside within a figure object, use plt.figure to create new figure
# figure = plt.figure()

# #Create one or more subplots using add_subplot, because you can't create blank figure
# axis = figure.add_subplot(1,1,1)


# axis.hist(removeAndSort["status_message"], bins=5)

# #labels and 
# plt.title("Message Popularity")
# plt.xlabel("status_message")
# plt.ylabel("num_likes")
# plt.show()

