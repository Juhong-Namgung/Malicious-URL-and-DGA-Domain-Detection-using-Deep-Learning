import pandas as pd
from sklearn.utils import shuffle

df_benign = pd.read_csv("./benign_label.csv")
df_malware = pd.read_csv("./malware_label.csv")
df_spam = pd.read_csv("./spam_label.csv")
df_phishing = pd.read_csv("./phishing_label.csv")



# Combine DGA and non-DGA data
result = pd.concat([df_benign, df_malware, df_spam, df_phishing])

# Shuffle and save the data
result = shuffle(result, random_state=33)
result.to_csv("./url_label.csv",mode='w',index=False)