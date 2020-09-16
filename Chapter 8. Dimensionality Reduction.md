# Chapter 8. Dimensionality Reduction

**Keywords**: *curse of dimesionality, pca, lle, kernel pca*

Dimensionality reduction is useful in a couple of ways:

1. Compression of feature space; and hence improved training speed
2. Improved feature space (Higher SNR)
3. Data visualization

Main dimensionality reduction approaches discussed here:

1. PCA
2. Kernel PCA
3. LLE

Main approaches to dimensionality reduction:

1. Manifold Learning
2. Projection

## The Curse of Dimensionality

- Math in high dimensional space is counterintuitive.
- High dimensional datasets are at risk of being very sparse. Hence, a model trained on the sparse dataset, will most likely overfit.
- The sparsity in datasets can be counteracted by increasing the number of samples. Unfortunately, that's an intractable solution for most if any high dimensional dataset. As the no. of such samples required to improve the density grows exponentially with the no. of features. 

## Main Approaches for Dimensionality Reduction

### Projection:

- Empirically, most real-world training datasets don't vary uniformly across all dimensions, but actually vary along a much lower-dimensional subspace of the high-dimensional space.

- PCA is a very good example. However, **linear projection** is a not always suited for dimensionality reduction. A very good example of this is the **Swiss-roll** dataset.

  <img src="C:\Users\Jagan\AppData\Roaming\Typora\typora-user-images\image-20200913000928002.png" alt="image-20200913000928002" style="zoom:50%;" />

### Manifold:

**Keywords:** *manifold*

- The swiss-roll is an example of a 2D manifold in a 3D space. 
- **Fun fact:** Earth is a 2D manifold as well.
- *Eh! What's a manifold?* The textbook definition of manifold is spot on. However, as an aside, here's a [link](https://www.quora.com/What-is-the-best-way-to-explain-the-concept-of-manifold-to-a-novice) that explains manifolds pretty well.  (P.S Don't read if you're a flat-earther!)
- **TLDR:** Simply put, a manifold is a structure whose local topological surface behaves like N-dimensional Euclidian space (think coordinate geometry), but it's global structure is much different.
- Another small aside, a manifold need not be a straight forward shape (like the swiss roll above) or an N-dimensional plane curved/twisted in a higher dimension. Manifolds can vary in shape and sizes and different datasets will have different degrees of freedom resulting in arbitrary manifolds.

There are two main assumptions at play here:

1. More often than not, most data lies closer to, if not on a lower-dimensional manifold of their original feature space.
2. The task of interpreting the data will be simpler if expressed in a lower-dimensional space of the manifold, with a caveat. **The caveat being that it needn't always be true.** If the decision boundary doesn't lie on the plane of the manifold, then a simple unrolling won't help. In the same case, it might be easier to handle the dataset in it's original or higher-dimensional feature space than unroll it.

## PCA

Click **[here]( https://www.youtube.com/watch?v=ciCieHQ1l1Y&list=PL8_xPU5epJdcBqm0mgFoY52yywOHmOI7y&index=43)** for a short and interesting lecture on PCA.

The **PCA** algorithm works by repeatedly finding *orthogonal directions* that maximizes the variance of the dataset along the said directions. In essence, it works like a linear transformation, followed by compression and rotation (along the eigen vectors, part of the linear transformation) to a lower-dimensional space. 

**Preserving the Variance:** By choosing an axis that maximizes the variance (i.e. minimizes the MSE between the original dataset and it's projection) we lose the least amount of data.

**Principal Components**: The orthogonal axes that PCA identifies for maximizing the variance of the dataset, are called the principal components. 

***How to find the principal components?*** 

The recipe usually involves finding the covariance matrix, followed by the projection matrix and using that to project N-dimensional data to D dimensions (N >> D).

However, there's a more efficient method in the form of **Singular Value Decomposition**. I won't go into details here, but as the name suggests you're breaking down the feature vector into 3 components. **U** is the Unitary matrix, $\Sigma$ is diagonal matrix and $V^T$ is the matrix of singular vectors for principal components. Here's a [link](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv) for explanation grounded in data science. I'd explain it, but it'd would get away from me, since I don't understand it intuitively enough. The [math](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca#:~:text=Principal%20component%20analysis%20(PCA)%20is,of%20the%20data%20matrix%20X.) is pretty interesting, IMO.

**Note:** It's important that the features are centered before projection.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<**TODO:** Flesh out the math >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Projecting down to D-dimensions:** $X_{d} = X.V_{d}$, where $V_d$ is the d-dimensional subset of $V$

**Explained Variance Ratio:** The ratio indicates the proportion of the dataset variance along each principal component.

**Choosing the Right Number of Dimensions:** Use the $d$ value for which the explained variance ratio sums up to a good portion. You can use plotting to find the *elbow* in the curve where the growth in variance across dimensions plateaus.

<img src="C:\Users\Jagan\AppData\Roaming\Typora\typora-user-images\image-20200912191101760.png" alt="image-20200912191101760" style="zoom:50%;" />

**PCA for Compression:** After compression using PCA, MNIST reduces in size to less than 20% of it's original. This speeds up heavy algorithms such as SVM, etc. 

**Reconstruction:** $X = X_d.V_{d}^T$

**[Randomized PCA](https://www.quora.com/What-is-randomized-PCA):** By default, Scikit-Learn uses randomized `svd_solver`, whose computational time is much smaller than standard SVD.

**Incremental PCA:** Useful for large datasets and  applying PCA online (instances arriving on the fly). 

More interestingly `memmap` from Numpy can be used for loading the array in batches from a file on the system. ***Nifty!***

```python
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```

## Kernel PCA

Click **[here]( https://www.youtube.com/watch?v=HbDHohXPLnU&list=PL8_xPU5epJdcBqm0mgFoY52yywOHmOI7y&index=45)** for a short and interesting lecture on kPCA.

**Kernel Trick:** The general idea is that similar to SVM, a non-linear complex boundary in original feature space can be transformed into a linear decision boundary in a higher-dimensional space. Same trick can be applied to PCA to perform nonlinear projections for dimensionality reduction, i.e. **kPCA**.

You can use kPCA using different kernels; rbf, linear, etc.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<**TODO:** Add more info on kernel trick>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

**Selecting a Kernel and Tuning Hyperparameters:** kPCA is an unsupervised learning algorithm. So there's no direct method for finetuning the hyper-parameters. Finetuning can be done in two ways:

1. **Supervised**: You can create a pipeline (sklearn API) with kPCA for dimensionality reduction followed by the supervised learning task, and use grid search over the pipeline for figuring out the best parameters.
2. **Unsupervised:** Select the set of parameters that yields the lowest reconstruction error. 

Let's talk more on the reconstruction here. 



## Locally Linear Embedding (LLE)

- A manifold learning technique where it tries to look for a lower-dimensional representation while preserving the local relationships of the data points.
- It's a ***non-linear*** dimensionality reduction method. Very good at unrolling twisted manifolds in a noiseless dataset.

**Algorithm:**

1. 



