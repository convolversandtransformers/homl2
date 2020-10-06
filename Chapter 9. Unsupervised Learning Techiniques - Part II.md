# Chapter 9. Unsupervised Learning Techniques - Part II

**Keywords**: _k-means_, _dbscan_, _gmms_

> If intelligence was a cake, unsupervised learning would be the cake (bread), supervised learning would be the icing on the cake, and reinforcement learning would be the cherry on the cake. 	- Yan LeCunn

Most data available to us is un-labelled. One way to exploit this huge corpus of un-labelled data is through unsupervised learning. There's a lot of untapped potential with un-labelled datasets, that we've barely tapped into. 

Take for example, any **classification** problem. If you want to classify pictures of something you'd need to sit and tediously label all the images for training, depending on the variance in images  this could be anywhere from thousands to millions of labelled images. 

Wouldn't it be great to somehow reduce or even eliminate the time it requires to label or even train such dataset. Enter the **cake**, a.k.a unsupervised learning.

**Unsupervised Techniques: ** _Clustering, Anomaly Detection and Density Estimation._

Each of the techniques above can be used for data analysis, customer segmentation, recommender system, image segmentation, semi-supervised learning, anomaly detection and more. 

## Clustering

**Definition**: Identifying and assigning instances to clusters or groups containing similar instances.

Clustering has many use-cases. Here's some of them discussed briefly:

- **Customer Segmentation.** This refers to clustering your users based on their web activity (preferences and other attributes) and into _segments_. You can then adapt your products or marketing campaigns to each such segment, allowing you to recommend products/content that might be useful to people in each segment.
- **Data Analysis.** Quickly running clustering on a (labelled) dataset to figure out whether there's an intrinsic pattern in each cluster.
- **Dimensionality Reduction.** You represent each instance by the vector of it's affinity values to each cluster. This means, for $k$ clusters, you get a $k$ dimensional vector representing the instance, which is usually much smaller than original vector.
- **Anomaly Detection.** Any instance with low affinity with any (major) clusters can be considered an anomaly.
- **Active Learning.** Propagating labels from a few learned/annotated labels to the rest of the instances in the same cluster. Very useful for labelling jobs and creating datasets for supervised algorithm.
- **Search Engine.** Given a sample, identify it's cluster and return similar samples from the same cluster. Very useful for indexing and search platforms for different content type.
- **Image Segmentation.** Clustering pixels in an image and replacing the pixel values with the mean pixel value of a cluster. Very useful for tracking contour of objects.

### K-Means

For example, take the following un-labelled dataset.

<img src="assets/ch9/k_means_data.png" alt="image-20201004171905278" style="zoom:80%;" />

The K-Means algorithm is pretty straightforward. 

1. You choose the no. of clusters or blobs you want. In this case, we'll choose 5 (pretty obvious from the data). 
2. K-Means will then try to iteratively find the blob's center in the dataset and assign each instance to the blob, based on it's proximity until it converges to the best solution.

If you trained a K-Means cluster on this dataset you'll get an output similar to this.

<img src="assets/ch9/voronoi_tessellation.png" alt="image-20201004172030076" style="zoom:80%;" />

**Note:** 

1. In this case the no. of clusters is chosen was a good fit for the data. This is not always the case.
2. Doesn't behave very well with clusters of varying size _(check the boundary, some instances along the boundary are assigned to other clusters)_, since the algorithm only looks at L2 norm for clustering.

Instead of performing _hard-clustering_, i.e. assigning instances to cluster with closest centroid, we can represent each instance by the vectors of it's distance/similarity score to each cluster, which is called _soft-clustering_.

Using _soft-clustering_, we can represent a $n$-d instance as a $k$-d (# of clusters) vector. **This can be a very effective non-linear dimensionality reduction technique.**

#### K-Means Algorithm

As already mentioned the algorithm is actually straightforward. So here, we'll just be covering the specs of the algorithm and other techniques.

**Steps:**

1. Randomly initialize $k$ centroids.
2. Calculate distance of each instance from the centroids.
3. Update the label of each instance based on proximity or centroid score.
4. Update the centroids from labelled instances.
5. Repeat 2-4 till convergence.

**Computational complexity**: $O(n,m,k)$; 

- m - # of instances
- n - # of features/attributes
- k - # of clusters

The algorithm is guaranteed to converge, but might converge into a local optima, subject to centroid initialization done.

**Centroid Initialization Strategies:** To avoid bad local optima, you can do one of two things.

1. Use previous best results from different clustering runs to initialize the centroids.
2. Run K-Means multiple times till you hit the right solution.

The `sklearn` library already does this by running the algorithm `n_init` times, which defaults to 10.

#### K-Means Improvements

There's been a bunch of improvements to the K-means algorithm (founded in 1967) all the way to 2016. We'll be discussing a bunch of these next. All of these improvements have actually been baked into the `sklearn` library, so there's no special implementation required.

##### Centroid Initialization: [K-Means++](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

The K-Means++ paper suggests the following algorithm

1. Choose a centroid $c^{(1)}$ uniformly at random from the dataset.
2. Choose a new centroid $c^{(i)}$ with the following probability: $\frac{D(x)^2}{\sum_{x {\epsilon} X}{D(x)}^2}$, where $D(x)$ is the distance between the instance $x$ and the closest chosen center $c$.  
3. Repeat 2 till $k$ centroids are chosen.

**Put simply**, this formula means, farther points will have *higher probability* of being chosen than closer points. This way we don't randomly choose centers which are very close together and end up splitting a single cluster into two.

##### Accelerated K-Means and mini-batch K-Means

[Accelerated K-Means](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf): Charles Elkan (2003)

[Mini-batch K-Means](https://dl.acm.org/doi/abs/10.1145/1772690.1772862): David Sculley (2010)

For the mini-batch version of K-Means you'll have to switch from `from sklearn.clusters import KMeans` to `from sklearn.clusters import MiniBatchKMeans`.

For the mini-batch version, the `inertia` is slightly worse, however, the speedup more than enough makes up for it.

![image-20201004182120364](assets/ch9/mini_batch_k_means.png)

#### Finding Optimal # of Clusters

As we saw earlier, the sample data used was easy to dissect into 5 clusters. However, in reality, that would be far from it. So the question is, what's a good way to choose the no. of clusters?

There may be more, but in here we'll discuss two major ways:

1. Using `inertia`: Not recommended. **TL;DR** - Jump to silhouette
2. Using `silhouette_score`: Recommended.

##### Inertia:

Usually the K-Means algorithm, uses the `inertia` value for deciding which initialized model (using `n_init`) to return as the best performer. 
$$
inertia = \sum^{n}_{i=0}{min_{ \mu_j {\epsilon} C}(||x_i - {\mu}_j||^2)}
$$
This translates to sum of square of distances of each instance from the closest cluster centre.

This value is usually used as a metric for evaluating different hyper-parameters for K-Means (# of clusters being one of them). 

Inertia decreases with increase in $k$. So, it's usually a **poor** metric for finding the no. of clusters. However, there's a coarse way to use inertia to solve this, by choosing the value of $k$ corresponding to greatest change in inertia. In the below example, this corresponds to **4.** 

![image-20201004185411909](assets/ch9/inertia_vs_num_clusters.png)

But, in general it's a poor metric. When in doubt switch to **silhouette**.

##### Silhouette Score:











