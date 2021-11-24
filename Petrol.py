# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:52:42 2021

@author: AN20157679
"""


path = r'C:\Ankit\Personal\WOrk\assesment\ACE\PreScreen_r3\\'

filepath = path + 'ingredient.csv'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

df =  pd.read_csv(filepath)
df.head()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# percentile list
perc =[.20, .40, .60, .80]
  
# list of dtypes to include
include =['object', 'float', 'int']
  
# calling describe method
df.describe(percentiles = perc, include = include)


import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.set(style="whitegrid")
plt.figure(figsize=(10,8))


ls = ['a','b','c','d','e','f']
for i in ls :
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,8))
    #ax = sns.distplot(x=i, data=df, orient="v").set_title('Additive' + i.upper())
    ax = sns.distplot(df[i]).set_title('Additive' + i.upper())
    
    ax.get_figure().savefig(i.upper() +'.png')



print(df.corr())
  
# plotting correlation heatmap
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
  
# displaying heatmap
mp.show()



sns.lineplot(data = df.iloc[:,4:5])


#ANOVA

# reshape the d dataframe suitable for statsmodels package 
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['a', 'b', 'c', 'd','e','f'])
# replace column names
df_melt.columns = ['index', 'Additive', 'value']

# generate a boxplot to see the data distribution by treatments. Using boxplot, we can 
# easily detect the differences between different treatments
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.boxplot(x='Additive', y='value', data=df_melt, color='#99c2a2')
ax = sns.swarmplot(x="Additive", y="value", data=df_melt, color='#7d0013')
plt.show()

#https://www.reneshbedre.com/blog/anova.html




import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(df['a'], df['b'], df['c'], df['d'])
print(fvalue, pvalue)


# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ordinary Least Squares (OLS) model
model = ols('value ~ C(Additive)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table



from bioinfokit.analys import stat
# perform multiple pairwise comparison (Tukey's HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
res = stat()
res.tukey_hsd(df=df_melt, res_var='value', xfac_var='Additive', anova_model='value ~ C(Additive)')
res.tukey_summary



# QQ-plot
import statsmodels.api as sm
import matplotlib.pyplot as plt
# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()




plt.scatter(df['a'],df['g'])
#plt.xlim(-180,180)
#plt.ylim(-90,90)
plt.title('G and A Relationship')
plt.show()




mms = MinMaxScaler()
mms.fit(df)
data = mms.transform(df)




#Elbow curve - 

wcss=[]
for i in range(1,14):
    kmeans = KMeans(i)
    kmeans.fit(data)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,14)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# Silhouette Score for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()

#Elbow Curve
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(data)        # Fit data to visualizer
visualizer.show()  

# silhouette
visualizer = KElbowVisualizer(model, k=(2,10),metric='silhouette', timings= True)
visualizer.fit(data)        # Fit the data to the visualizer
visualizer.show() 


# calinski_harabasz
visualizer = KElbowVisualizer(model, k=(2,30),metric='calinski_harabasz', timings= True)
visualizer.fit(data)        # Fit the data to the visualizer
visualizer.show()   


#Check Dendrograms

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(data)
data_with_clusters = pd.DataFrame(data.copy())
data_with_clusters['H_Clusters'] = y_hc 

plt.scatter(data_with_clusters[0],data_with_clusters[1],c=data_with_clusters['Clusters'],cmap='rainbow')

#clustering
k = 5
mod = KMeans(k)
mod.fit(data)

pred_clu = mod.fit_predict(data)


data_with_clusters = pd.DataFrame(data.copy())
data_with_clusters['kmean_Clusters'] = pred_clu 
plt.scatter(data_with_clusters[0],data_with_clusters[1],c=data_with_clusters['Clusters'],cmap='rainbow')


#Spectral 
from sklearn.cluster import SpectralClustering

spectral_cluster_model= SpectralClustering(
    n_clusters=5, 
    random_state=25, 
    n_neighbors=8, 
    affinity='nearest_neighbors'
)

data_with_clusters['Spec_clusters'] = spectral_cluster_model.fit_predict(data)

plt.scatter(data_with_clusters['e'],data_with_clusters['b'],c=data_with_clusters['Clusters'],cmap='rainbow')

data_with_clusters.to_csv(path+'Output.csv')




# Gap Statistic for K means
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
# Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_
# Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
        return (gaps.argmax() + 1, resultsdf)
    

score_g, df = optimalK(data, nrefs=5, maxClusters=30)
plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. K');
