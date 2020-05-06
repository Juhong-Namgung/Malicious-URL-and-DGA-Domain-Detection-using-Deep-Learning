import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

df = pd.read_csv("./dga_label.csv")

'''
### 1. data distribution analysis
'''

# count num of data in class
print(df['class'].value_counts())

# num of data in class visualization(클래스별 데이터 수 그래프 시각화)
ax = sns.countplot(x="class", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center")

plt.title("DGA 클래스별 데이터 수")
plt.savefig('DGA_class_distribution.png')
plt.show()

'''
### 2. domain length analysis
'''
# calculate domain length
domain_length = []
for domain in df['domain']:
    domain_length.append(len(domain))
df['length'] = domain_length

# count domain length
domain_count = []
for i in range(min(domain_length), max(domain_length)+1):
    domain_count.append(domain_length.count(i))
print('min len= ' + str(min(domain_length)))
print('max len= ' + str(max(domain_length)))
print(domain_count)

# visualization
ax = sns.countplot(x='length', data=df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center")

plt.title("domain length")
plt.savefig('domain_length_distribution.png')
plt.show()

