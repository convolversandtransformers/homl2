# Chapter 7: Ensemble Learning and Random Forests

**Keywords:** Bagging, Boosting, Stacking

## Voting Classifier

### Hard Voting Classifier: 

**Keywords:** Weak & Strong Learners

- Enough weak learners with sufficient diversity can be stacked together, the voting classifier becomes a strong learner.
- Works on the **law of large numbers**; more predictors more steadier outcome.
- However, each weak learner should learn **independent** priors for this to happen.
- It's impossible since all predictors are trained on the same data.
- How to achieve different priors on same data? Try different algo.

### Soft Voting Classifier:

Average predicted logits/probabilities across all models. This way highly confident classes will be weighted higher.

## Bagging and Pasting

**Keywords: ** bagging, pasting, bootstrap, out-of-bag, random patches and random sub-spaces

- Bagging: sampling w/ replacement, also called **bootstrap aggregating**

- Pasting: sampling w/o replacement

- Bagging **better** than pasting, since aggregating leads to lower variance and bias.

- Bagging Classifier (DT) vs Random Forest Classifier: API + Optimization differences

- ```bash
  # Self Note:
  The trade-off between increasing the performance on dataset by 1-5% vs trying to build a continuously learning pipeline.
  
  Viz is important, but not always possible on high-dimensional dataset.
  ```

### Out-of-bag Evaluation

During evaluation using bagging approach, only 63% (only for a sufficiently high number of instances) on average is seen during training by each predictor. Hence an `oob` instances can function as validation set for validating model performance.

```bash
# Self Note:
In absence of enough samples (when training a bagging model), oob score can be used as a substitute for validation/hold-out set performance.	
```

### Random Patches and Random Subspaces

- Random Patches: Sampling input samples and feature
- Random Subspaces: Taking all training instances but bootstrapping the features (ex: RF)

### Random Forests

- Ensemble of DT
- Bagging and Pasting

| Random Forest                                | Bagging Classifier                                |
| -------------------------------------------- | ------------------------------------------------- |
| Trained on all samples                       | Subsampling can be done                           |
| Bootstrapping features is handled by default | Bootstrapping features and samples can be handled |

### Extra Trees

| Random Forest                                       | Extra Trees                                                  |
| --------------------------------------------------- | ------------------------------------------------------------ |
| A random subset of features considered at each node | Have random thresholds set for each feature                  |
| DTs optimize the best threshold for each split      | Faster than RF because it doesn't have to calculate threshold values |

- Is Extra Trees better than Random Forest? Answer: No free lunch. 

### Feature Importance

- A weighted average across the nodes in all trees, computed by looking at how much impurity was reduced by the tree nodes using that particular feature.
- Nodes weight is calculated by how many features are associated with it.

## Boosting

- Combines weak learners sequentially to create strong learners

### AdaBoost

- Sequentially train predictors, iteratively improving on the mis-classified instances of the previous predictor by updating the instance weights and training the next predictor with the weighted instances.
- Hyper-parameter: Learning Rate: 1 vs 0.5 (Adaptive or not **?**)
- The update procedure is similar to gradient descent but the update doesn't back propagate, instead new learners are added to make the ensemble predictions better.
- **Not Parallelizable**

**Algorithm:**

```
TODO: Add math here
```

- sklearn uses **SAMME**.
- For binary predictions AdaBoost is the same as **SAMME**. 
- For softer, probabilities based prediction sklearn also has **SAMME.R** algorithm which works on prediction probabilities.



### Gradient Boosting

Keywords: GBRT (Gradient Boosting Regression Trees)

- Doesn't tweak instance weights like AdaBoost.
- Fits the new predictor on the residual learning error of the previous predictor.
- Prediction is the **sum** of prediction by all models.
- Hyper-paramter tweaking:
  - LR is low; increase the no. of estimators.
  - LR is high; decrease LR and/or decrease the no. of estimators
  - Add some other regularization techniques
- sklearn API: **staged_predict**
- You can use early-stopping (w/ **warm_start = True**)
- Subsampling here is referred to as Stochastic Gradient Boosting and as usual reduces variance at the cost of increasing bias.
- Preferred library: XGBoost

## Stacking

- Instead of using averaging or some other algorithm to decide the output of the ensemble of predictors, why not use a predictor for averaging (weighting the predictions) itself?

- The idea behind stacking is to split the data into subsets equal to the no. of layers in the stack.

- For a 2-layer stack (one predictor layer and one blender layer on top); the data is split into two subsets, A and B.

  - The initial layer (ensemble of predictors) is trained on the subset_A. After the training, the subset_B which acts as a hold-out set is used and predictions are generated using the initial layer.

  - The predictions with the respective ground-truths combined forms the training data for the blender layer.

  - Finally after optimizing the layers on separate data subsets, the models/layers of models are stacked together to form a stacked ensemble.

  - This logic is extendable to multiple layers. However, if you want to create a stack of 20 or so layers/models put together with enough data to train, then why not just use an optimized neural network instead of all this unnecessary work. Be logical :P

    
