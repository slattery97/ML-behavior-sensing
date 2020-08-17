#Author: Ryan Slattery

import pandas
import numpy as np 
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import random as r
import scipy.signal as sig
import webbrowser


#Create simulated data function
def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    with open('GPSdataSimTrial2.txt', 'w') as t1:
        # Create random data and store in feature matrix X and response vector y.
        X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                    cluster_std=clusterDeviation)

        #Create lat/long data with noise
        #Input starting lat/long point
        X[0] = 40.4253284, -86.904772
        #Create random movement points
        for i in range(1,len(X)):
            X[i] = X[i-1,0] + r.uniform(-0.015824036,0.015824036), X[i-1,1] + r.uniform(-0.015785045,0.01530454)

        for i in range(len(X)):
            #remove brackets and add new line
            dataVal1 = str(X[i]) + str('\n')
            dataVal1 = dataVal1.replace(']','')
            dataVal1 = dataVal1.replace('[','')
            #write lat/long data to file
            t1.write(dataVal1)
        
        # Standardize features by removing the mean and scaling to unit variance
        X = StandardScaler().fit_transform(X)
        return X, y

#Read GPS data file function
def getDataPoints():
    with open('IE431 User1 Data.txt') as dataFile:
        #get list of lines as strings
        X2 = dataFile.readlines()
        #split and convert data to float type
        for i in range(len(X2)):
            #split user text file
            X1 = X2[i].split('\t')
            #split simulated text file
            #X1 = X2[i].split()
            if X1[0] != '':
                X2[i] = X1
                X2[i][1] = float(X2[i][1].strip('\n'))
                X2[i][0] = float(X2[i][0])
            else:
                continue
        #remove \t\n data points from list and get copy of lat/long list
        X = []
        X_old = []
        for i in range(len(X2)):
            if X2[i] != '\t\n':
                X.append(X2[i])
                X_old.append(X2[i])

        
        #convert lat/long to standard scale for data visualization
        X = StandardScaler().fit_transform(X)
        return X, X_old


#MODELING

#DBSCAN
def dbscan():
    epsilon = 0.4 #radius if includes enough points, is considered dense area
    minimumSamples = 6 #minimum data points in an area to define a cluster
    db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
    labels = db.labels_

    #Distinguish outliers
    #creates an array of booleans using the labels from db.
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    core_samples_mask

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_clusters_

    # Remove repetition in labels by turning it into a set.
    unique_labels = set(labels)
    unique_labels

    #get number of clusters
    numClusters1 = int(len(unique_labels))-1

    #DATA VISUALIZATION

    # Create colors for the clusters.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot the points with colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        # Plot the datapoints that are clustered
        xy = X[class_member_mask & core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

        # Plot the outliers
        xy = X[class_member_mask & ~core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

        #Set title
        plt.title('DBSCAN')
        
    return numClusters1, labels
    
#KMeans
# init: initialization method of the centroids
# n_clusters: number of clusters to form/centroids to generate
# n_init: number of time the algorithm will be run with different centroid seeds

def kmeans():
    # initialize KMeans with parameters
    k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

    #fit model with feature matrix
    k_means.fit(X)

    #get labels for each point in model
    k_means_labels = k_means.labels_

    #get coordinates of cluster centers
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_cluster_centers

    # DATA VISUALIZATION

    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(6, 4))

    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    #get number of clusters
    numClusters2 = int(len(set(k_means_labels)))

    # Create a plot
    ax = fig.add_subplot(1, 1, 1)

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)
        
        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]
        
        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        
        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

    # Title of the plot
    ax.set_title('KMeans')

    # Remove x-axis ticks
    ax.set_xticks(())

    # Remove y-axis ticks
    ax.set_yticks(())

    return numClusters2, k_means_labels
    

#FUNCTION CALLS

#Generate simulated data
#Note: 4 weeks of GPS data, recorded every 15 minutes: 2688 samples
#X, y = createDataPoints([[0,0]], 672, 0.5)

#Get data points
X, X_old = getDataPoints()

#Plot with DBSCAN
dbscan()

#Plot with KMeans
kmeans()

#Show plot
plt.show()

#CALCULATE METRICS

#get list of latitude coordinates
lat = []
for i in range(len(X_old)):
    lat.append(X_old[i][0])
    
#get list longitude coordinates
long = []
for i in range(len(X_old)):
    long.append(X_old[i][1])


#GET NUMBER OF CLUSTERS AND DATA CLUSTER LABELS
#for DBSCAN:
numClusters1, labels1 = dbscan()
print('The number of DBSCAN clusters is:', numClusters1)
#for KMeans:
numClusters2, labels2 = kmeans()
print('The number of KMeans clusters is:', numClusters2)

#CALCULATE ENTROPY
#for DBSCAN:
#intialize cluster lists
dl0 = []
dl1 = []
dl2 = []
#append data points to their cluster list
for i in range(len(labels1)):
    if labels1[i] == 0:
        dl0.append(labels1[i])
    elif labels1[i] == 1:
        dl1.append(labels1[i])
    elif labels1[i] == 2:
        dl2.append(labels1[i])  
#calculate percentage of time spent in each cluster
dt0 = len(dl0)/len(X)
dt1 = len(dl1)/len(X)
dt2 = len(dl2)/len(X)
#calculate entropy
ent1 = -(dt0*np.log(dt0) + dt1*np.log(dt1) + dt2*np.log(dt2))
print('The (DBSCAN) entropy is:', format(ent1, '.3f'))

#for KMeans:
#intialize cluster lists
kl0 = []
kl1 = []
kl2 = []
#append data points to their cluster list
for i in range(len(labels2)):
    if labels2[i] == 0:
        kl0.append(labels2[i])
    elif labels2[i] == 1:
        kl1.append(labels2[i])
    elif labels2[i] == 2:
        kl2.append(labels2[i])  
#calculate percentage of time spent in each cluster
kt0 = len(kl0)/len(X)
kt1 = len(kl1)/len(X)
kt2 = len(kl2)/len(X)
#calculate entropy
ent2 = -(kt0*np.log(kt0) + kt1*np.log(kt1) + kt2*np.log(kt2))
print('The (KMeans) entropy is:', format(ent2, '.3f'))

#CALCULATE LOCATION VARIANCE
#get standard deviation of lat/long
stdLat = np.std(lat)
stdLong = np.std(long)
#get variance of lat/long
varLat = stdLat**2
varLong = stdLong**2
#calculate location variance
locVar = np.log(varLat + varLong)
print('The location variance is:', format(locVar,'.3f'))

#CALCULATE HOMESTAY PERCENTAGE
#for DBSCAN:
#get number of points in each cluster
d0points = len(dl0)
d1points = len(dl1)
d2points = len(dl2)
#sort list from greatest to least
dblist = [d0points, d1points, d2points]
dblist.sort(reverse=True)
#get number of points in home cluster
dbhome = dblist[0]
#calculate homestay percentage
hstay1 = dbhome/(d0points + d1points + d2points)
print('The (DBSCAN) homestay percentage is:', format(hstay1, '.3f'))

#for KMeans:
#get number of points in each cluster
k0points = len(kl0)
k1points = len(kl1)
k2points = len(kl2)
#sort list from greatest to least
kmlist = [k0points, k1points, k2points]
kmlist.sort(reverse=True)
#get number of points in home cluster
kmhome = kmlist[0]
#calculate homestay percentage
hstay2 = kmhome/(k0points + k1points + k2points)
print('The (KMeans) homestay percentage is:', format(hstay2, '.3f'))

#CALCULATE CIRCADIAN MOVEMENT
#get list of hours since start of data collection
hrs = []
inc=0
while inc*4 < len(lat):
    hrs.append(inc)
    hrs.append(inc)
    hrs.append(inc)
    hrs.append(inc)
    inc += 1
hrs.pop(hrs[-1])
hrs.pop(hrs[-1])
#get lomb scargle periodogram for lat/long
latlsp = sig.lombscargle(lat,hrs, range(24,25), normalize=True)
longlsp = sig.lombscargle(long,hrs, range(24,25), normalize=True)
#calculate spectrum (E) for lat/long
Elat = 0
for i in range(len(latlsp)):
    Elat += latlsp[i]
Elong = 0
for i in range(len(longlsp)):
    Elong += longlsp[i]
#calculate circadian movement
circMov = np.log(Elat + Elong)
print('The circadian movement is:', format(circMov, '.3f'))
################################################################
#comparison table of dbscan and kmeans
def printtable(htmlfile):
    #basic format for html page
    html = "<!DOCTYPE>\n<html>\n<head></head>\n<style>table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%;}td, th {border: 1px solid #dddddd;text-align: left;padding: 8px;}\ntr:nth-child(even) {background-color: #dddddd;}\n</style>\n</head>\n<body>\n"
    #format for a table
    table = "<table>\n <tr>\n<th>Functions</th>\n<th>DBScan</th>\n<th>Kmeans</th>\n</tr>\n"

    #row for number of clusters
    cluster_db = str(numClusters1)
    cluster_kmeans = str(numClusters2)
    table += '<tr>\n<td>Number of Clusters</td>\n<td>' + cluster_db + '</td>\n<td>'+ cluster_kmeans + '</td>\n</tr>\n'
    # row for entropy
    entropy_db = str(format(ent1,'.3f'))
    entropy_k = str(format(ent2, '.3f'))
    table += '<tr>\n<td>Entropy</td>\n <td>' + entropy_db + '</td>\n <td>' + entropy_k + '</td>\n</tr>\n'

    # row for homestay percentage
    homestay_db = str(format(hstay1, '.3f'))
    homestay_k = str(format(hstay2, '.3f'))
    table += '<tr>\n<td>Homestay Percentage</td>\n<td>'+ homestay_db + '</td>\n<td>' + homestay_k + '</td>\n</tr>\n'

    # row for location variance
    location = str(format(locVar, '.3f'))
    table += '<tr>\n<td>Location Variance</td>\n<td>' + location + '</td>\n<td>' + location +'</td>\n</tr>\n'

    # row for number of clusters
    circad = str(format(circMov, '.3f'))
    table += '<tr>\n<td>Circadian Movement</td>\n<td>' + circad + '</td>\n<td>' + circad + '</td>\n</tr>\n'

    html += table
    with open(htmlfile, 'w') as f:
        f.write(html + "\n</table>\n</body>\n</html>")
        
        
printtable('table.html')
webbrowser.open_new_tab('table.html')



    
























