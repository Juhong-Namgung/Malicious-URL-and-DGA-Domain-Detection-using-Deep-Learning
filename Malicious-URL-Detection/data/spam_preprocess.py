import pandas as pd
from sklearn.utils import shuffle

df_spam = pd.read_csv("./spam/dom-bl-base.txt", header=None)

print(df_spam)

spam_labels = []
spam_labels_str = []
spam_urls = []
df_spam_to_list = df_spam[0].tolist()
z = 0

for x in df_spam[0].tolist():
    if len(df_spam_to_list[z].split(';')[0]) < 101:
        spam_labels.append(2)
        spam_labels_str.append('spam')
        spam_urls.append(df_spam_to_list[z].split(';')[0])
    z = z + 1


# Data columns("url", "label", "class")
spam_archive = pd.DataFrame(columns=['url'])
spam_archive['url'] = spam_urls
spam_archive['label'] = spam_labels_str
spam_archive['class'] = spam_labels

spam_archive.to_csv("./spam_label.csv", mode='w', index=False)

print(spam_archive)