import pandas as pd
from sklearn.utils import shuffle

df_benign_train = pd.read_csv("./benign/Webpages_Classification_train_data.csv")
#df_benign_train = pd.read_csv("./benign/benign_sample.csv")

benign_labels = []
benign_labels_str = []
benign_urls = []
df_benign_train_to_list = df_benign_train['url'].tolist()
z = 0

for x in df_benign_train['label'].tolist():
    if x == 'good':
        if len(df_benign_train_to_list[z]) < 101:
            benign_urls.append(df_benign_train_to_list[z])
            benign_labels.append(0)
            benign_labels_str.append('benign')
    z = z + 1

# Data columns("url", "label", "class")
benign_archive = pd.DataFrame(columns=['url'])
benign_archive['url'] = benign_urls
benign_archive['label'] = benign_labels_str
benign_archive['class'] = benign_labels

benign_archive.to_csv("./benign_label.csv", mode='w', index=False)

print(benign_archive)