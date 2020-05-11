import pandas as pd
from sklearn.utils import shuffle

df_dga = pd.read_csv("./dga_feed.csv")
df_non_dga = pd.read_csv("./majestic_million.csv")

# DGA labels(include non-DGA)
dga_labels_dict = {'majestic':0, 'banjori':1, 'tinba':2, 'Post':3, 'ramnit':4, 'qakbot':5, 'necurs':6, 'murofet':7, 'shiotob/urlzone/bebloh':8, 'simda':9,
                   'ranbyus':10, 'pykspa':11, 'dyre':12, 'kraken':13, 'Cryptolocker':14, 'nymaim':15, 'locky':16, 'vawtrak':17, 'shifu':18,
                   'ramdo':19, 'P2P':20 }

# Process DGA data
# Select top 19 DGA data(전체 DGA 데이터 중 상위 19개만 선택)
dga_labels= []
dga_labels_str = []
dga_domains = []

df_dga_source_to_list = df_dga['source'].tolist()
df_dga_domain_to_list = df_dga['domain'].tolist()
z = 0

for x in df_dga['source'].tolist():
    if x in dga_labels_dict:
        dga_labels.append(dga_labels_dict[x])
        dga_labels_str.append(df_dga_source_to_list[z])
        dga_domains.append(df_dga_domain_to_list[z])
    z = z + 1

# Data columns("domain", "source", "class")
dga_archive = pd.DataFrame(columns=['domain'])
dga_archive['domain'] = dga_domains
dga_archive['source'] = dga_labels_str
dga_archive['class'] = dga_labels

# Process non-DGA(majestic) data
non_dga_domains = df_non_dga['Domain'].tolist()
non_dga_labels = []
non_dga_labels_str = []

for x in non_dga_domains:
    non_dga_labels.append(0)
    non_dga_labels_str.append("majestic")

non_dga_archive = pd.DataFrame(columns=['domain'])
non_dga_archive['domain'] = non_dga_domains
non_dga_archive['source'] = non_dga_labels_str
non_dga_archive['class'] = non_dga_labels

# Combine DGA and non-DGA data
result = pd.concat([dga_archive, non_dga_archive])

# Shuffle and save the data
result = shuffle(result, random_state=33)
result.to_csv("./dga_label.csv",mode='w',index=False)
