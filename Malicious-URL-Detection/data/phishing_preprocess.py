import pandas as pd
from sklearn.utils import shuffle

df_open_phish = pd.read_csv("./phishing/feed.txt", header=None, delimiter='\n')
df_phish_tank = pd.read_csv("./phishing/online-valid.csv", delimiter=',')


print(df_open_phish)
print(df_phish_tank)


phishing_labels = []
phishing_labels_str = []
phishing_urls = []
df_open_phish_to_list = df_open_phish[0].tolist()
df_phish_tank_to_list = df_phish_tank['url'].tolist()

z = 0

for x in df_open_phish[0].tolist():
    if len(df_open_phish_to_list[z]) < 101:
        phishing_labels.append(3)
        phishing_labels_str.append('phishing')
        phishing_urls.append(df_open_phish_to_list[z])
    z = z + 1

z = 0

for x in df_phish_tank['url'].tolist():
    if len(df_phish_tank_to_list[z]) < 101:
        phishing_labels.append(3)
        phishing_labels_str.append('phishing')
        phishing_urls.append(df_phish_tank_to_list[z])
    z = z + 1


# Data columns("url", "label", "class")
phishing_archive = pd.DataFrame(columns=['url'])
phishing_archive['url'] = phishing_urls
phishing_archive['label'] = phishing_labels_str
phishing_archive['class'] = phishing_labels

phishing_archive.to_csv("./phishing_label.csv", mode='w', index=False)

print(phishing_archive)