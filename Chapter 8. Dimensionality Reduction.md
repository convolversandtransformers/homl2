# **Chapter 8. Dimensionality Reduction**
**Main Techniques Learnt:** PCA and its variants, LLE

## Setup

 -   Speeding up of training
 -   Data Visualization
 -   Curse of Dimensionality    
 -   The behaviour of certain functions are dependent on dimensionality. Example a random point in a normal 2D unit square has 0.4% chance of being on the edge versus 99.999% in a 10000D unit        
 -   High dimensional datasets are usually very sparse, hence predictions are much less reliable than in lower dimensional counterparts        
 -   More the dimensions, greater the chances of overfitting
   
## **Main Approaches**
### *Projection:*
 - Usually the training instances are not spread out uniformly in all dimensions. All training instances lie within (or close to) a much lower-dimensional subspace of the high-dimensional space. In this technique, every training instance is _projected_ perpendicularly onto a subspace 

    >  **Projection is not always the best approach to dimensionality reduction. In many cases the subspace may twist and turn, such as in the Swiss roll toy dataset**

    ​    
### *Manifold Learning*

 - A d-dimensional manifold is a part of an n-dimensional space *(where d< n)* that locally resembles a d-dimensional hyperplane
 - In the case of the Swiss roll, *d = 2* and *n = 3:* it locally resembles a 2D plane,but it is rolled in the third dimension
 - Manifold Learning relies on the *manifold assumption*, also called the *manifold hypothesis*, which holds that most real-world high-dimensional datasets lie close to a much lower-dimensional manifold. This assumption is very often empirically observed
 - Another implicit assumption is that that the task at hand (e.g., classification or regression) will be simpler if expressed in the lower-dimensional space of the manifold which may not be always true


## **Approach-1: Principal Component Analysis (PCA)**

### Methodology
-   Main idea behind PCA is to find the axis (principal component) that accounts for the largest amount of variance in the training set    
-   Solved using the standard matrix factorization technique called Singular Value Decomposition
- The training set matrix $X$ is decomposed into the matrix multiplication of three matrices $U\sum V_T$ , where V contains the unit vectors that define all the principal components
> **PCA assumes that the dataset is centered around the origin. Scikit-Learn’s PCA classes take care of centering the data.**

### PCA Variants
_Randomized Principal Component Analysis_
-   Stochastic algorithm, quickly finds an approximation of the first d principal components
-   Faster than svd and is default option in Sckit-learn
- PCA needs full data to be in memory

_Incremental PCA_
-   Incremental PCA (IPCA) allows to split the training set into mini-batches and feed an IPCA algorithm one mini-batch at a time. Useful for large training sets and for applying PCA online (i.e., on the fly, as new instances arrive)
-   Use of _partial_fit()_ and _fit()_ if using numpy’s _array_split_ method and _memmap_ class respectively while using IPCA.

_Kernel PCA_

-   Kernel PCA (kPCA) is often good at preserving clusters of instances after projection, or sometimes even unrolling datasets that lie close to a twisted manifold
-   Selecting a kernel can be using GridSearchCv or selecting the kernel and hyperparameters that yield the lowest reconstruction error. Detailed info reconstruction error based method in “Selecting a Kernel and Tuning Hyperparameters” in the chapter

## **Approach-2: Locally Linear Embedding (LLE)**
 -   Manifold Learning technique that does not rely on projections
 -   Nonlinear dimensionality reduction *(NLDR)* technique
 -  LLE is two stepped process
	-  **first,** measures how each training instance linearly relates to its closest neighbors *(basically find suitable weight matrix)*
	-  **next**, look for a low-dimensional representation of the training set where these local relationships are best preserved *(keeping the weights - representation of the linear relationship - fixed **and** find the optimal position of the instances images in the low-dimensional space)*.

 LLE MATH:
 - *first step* : $\widehat{W} = argmin_w \sum_{i=1}^{m}(x^{i}-\sum_{i=1}^{m}w_{i,j}x^{(j)})^{2}$ , encode the local linear relationships between the training instances subject to the constraints that $w_{i,j}=0$ if  $x^{j}$ is not a neighbour **and** $\sum_{i=1}^{m}w_{i,j}=1$
 - *second step*:  if $z_i$ is the image of $x_i$ in the lower *d*-dimensional space, then calculate the matrix $Z$ where $\widehat{Z} = argmin_z \sum_{i=1}^{m}(z^{i}-\sum_{i=1}^{m}\widehat{w_{i,j}}z^{(j)})^{2}$ .$Z$ is the matrix containing all $z_i$.

>  **LLE scales poorly to large datasets!**


## **Other techniques**

* t-Distributed Stochastic Neighbor Embedding (t-SNE) *(mostly for data viz)*
* Linear Discriminant Analysis (LDA) _(classification algorithm, but during training it learns the most discriminative axes between the classes, and these axes can then be used to define a hyperplane onto which to project the data)_


## **Answer excerpts from excercises worth being noted**


    Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have

Trick question: it depends on the dataset. Suppose the dataset is composed of points that are almost perfectly aligned. In this case, PCA can reduce the dataset down to just one dimension while still preserving 95% of the variance. If dataset is composed of perfectly random points, scattered all around the 1,000 dimensions, roughly 950 dimensions are required to preserve 95% of the variance. **Plotting the explained variance as a function of the number of dimensions is one way to get a rough idea of the dataset’s intrinsic dimensionality.**

    When should different PCA techniques be used?

In a nutshell, **Regular PCA** (actually randomized PCA since svd_solver="randomized" and not full by default) works only if the dataset fits in memory. **Incremental PCA** is useful for large datasets that don’t fit in memory, Incremental PCA and for online tasks, when you need to apply PCA on the fly, every time a new instance arrives. Finally, **Kernel PCA** is useful for nonlinear datasets.

## **Library Usage**

   Normal PCA

```python3
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.components_.T[:, 0] #The *components* attribute in this example defines the first principal component
pca.explained_variance_ratio_ #ratio indicates the proportion of the dataset’s variance that lies along each principal component.
pca = PCA(n_components=0.95) #indicates the ratio of variance that is to be preserved

##PCA for Compression
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced) 
```

  Randomized PCA  

    rnd_pca = PCA(n_components=154, svd_solver="randomized") #svd_solver="full" for normal PCA
    X_reduced = rnd_pca.fit_transform(X_train)
   IPCA array_split()

    from sklearn.decomposition import IncrementalPCA
    
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)
        X_reduced = inc_pca.transform(X_train)

   IPCA memmap

    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))
    batch_size = m // n_batches
    inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
    inc_pca.fit(X_mm)

 kPCA
    
    from sklearn.decomposition import KernelPCA
    
    rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
    X_reduced = rbf_pca.fit_transform(X)
    
    #Reconstruction error for kPCA check
    #By default, fit_inverse_transform=False and KernelPCA has no inverse_transform() method.
    #This method only gets created when you set fit_inverse_transform=True .
    
    rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,fit_inverse_transform=True)
    X_reduced = rbf_pca.fit_transform(X)
    X_preimage = rbf_pca.inverse_transform(X_reduced)
    mean_squared_error(X, X_preimage)

 LLE
    from sklearn.manifold import LocallyLinearEmbedding
    
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    X_reduced = lle.fit_transform(X)

## **Others**
**DISCLAIMER**: Might be irrelevant to the chapter too

 - fit() vs fit_transform()
To put it simply, use the fit_transform() method on the training set, as it is required to both fit and transform the data, use the fit() method on the training dataset to get the value, and later transform() test data with it.
 - Nothing on PCA here, but I like the structure used [here](https://www.kaggle.com/evanmiller/pipelines-gridsearch-awesome-ml-pipelines)
 - Also good pipeline with PCA [here](https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html)