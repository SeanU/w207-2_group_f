# Initial feature hacking

I started with some fairly basic dummy coding of features:

1. Whether or not the animal has a name
2. Whether or not the animal is a dog
3. Age (in years) 
  - For animals with no age specified, I set the age to the mean of all animals in the set with an age specified.
  - (It may be better to create a separate HasAge, code missing ages to 0, and interact them.)
  - Alternatively, should at least segregate by species in order to assign ages.
  - I did not normalize. Ages are in range [0,22].
4. Whether or not it is a mix (specified by presence of either 'mix' or '/' in the breed string)
5. Sex (Female, Male, 0 for both if NaN)
6. Intact (0 for NaN, too).
  - This should really be re-coded as separate varibles for Intact and Fixed, with both 0 for NaN.
7. Date of outcome:
  - Month
  - Day of month
  - Day of week
  - Hour of day

# Initial model evaluation

Using these features, I did some initial evaluation of logistic regression and SVMs. I used GridSearchCV for all of this. I have not tried submitting results to Kaggle for any of these yet.

Based on the results so far, I think that logistic regression is the way to go for now, and I'm going to go ahead and focus on feature engineering. This is for two reasons:
- The SVMs take a *long* time to train.
- None of them produced significantly better results than LR.

I used a common set of C values in all cases: [.01, .05, .1, .5, 1, 5, 10, 50, 100]

I also set the `tol` to .01 for faster convergence.  I might even try setting it to .1 if I do more of this, and save finer tolerances for fine tuning.

## Logistic Regression

Performs reasonably well. Successive runs produce quite variable results for the best C value, but typically it's peaking at just under **65% accuracy with C somewhere between .01 and 10**. There's a plateau in this range, so I'm not sure the particular C value really matters a lot.

```
Best parameters: {'C': 0.5}

All parameters:
	mean: 0.62052, std: 0.00571, params: {'C': 0.01}
	mean: 0.64675, std: 0.00237, params: {'C': 0.05}
	mean: 0.64862, std: 0.00171, params: {'C': 0.1}
	mean: 0.64982, std: 0.00187, params: {'C': 0.5}
	mean: 0.64982, std: 0.00187, params: {'C': 1}
	mean: 0.64933, std: 0.00141, params: {'C': 5}
	mean: 0.64967, std: 0.00166, params: {'C': 10}
	mean: 0.64948, std: 0.00121, params: {'C': 50}
	mean: 0.64971, std: 0.00136, params: {'C': 100}
```

## SVMs (SVC class)

I coded up grid searches all 4 kinds of kernel. These all take a *long* time. I let them run overnight, and so far only three have finished. I'll report results on those.

### Linear kernel

Using same C's as for logistic regression, I get a peak of about **64.4% at C=5**. The scores for all C values in the range hovered around 64%, so I'm not sure there's a whole lot of value in exploring further, given how long it took to train. I'm guessing it's performing poorly because there isn't clear separation among cases.

```
Best parameters: {'C': 5}

All parameters:
	mean: 0.63938, std: 0.00320, params: {'C': 0.01}
	mean: 0.64290, std: 0.00322, params: {'C': 0.05}
	mean: 0.64323, std: 0.00277, params: {'C': 0.1}
	mean: 0.64413, std: 0.00195, params: {'C': 0.5}
	mean: 0.64391, std: 0.00190, params: {'C': 1}
	mean: 0.64443, std: 0.00144, params: {'C': 5}
	mean: 0.64436, std: 0.00156, params: {'C': 10}
	mean: 0.64417, std: 0.00173, params: {'C': 50}
	mean: 0.64436, std: 0.00163, params: {'C': 100}
```

### RBF kernel

In addition to the C's, I tried gammas in [.01, .05, .1, .5, 1, 5, 10, 50, 100]. 

Peak was 68% at C=5, gamma=.1.  This is a fair bit better than LR, but the grid search still took quite a while. For all C's but one, the best gamma was .1. For that one, .5 was very slightly better.

```
Best parameters: {'C': 5, 'gamma': 0.1}

All parameters:
	mean: 0.53133, std: 0.00247, params: {'C': 0.01, 'gamma': 0.01}
	mean: 0.61031, std: 0.00475, params: {'C': 0.01, 'gamma': 0.05}
	mean: 0.61476, std: 0.00566, params: {'C': 0.01, 'gamma': 0.1}
	mean: 0.40290, std: 0.00005, params: {'C': 0.01, 'gamma': 0.5}
	mean: 0.40290, std: 0.00005, params: {'C': 0.01, 'gamma': 1}
	mean: 0.40290, std: 0.00005, params: {'C': 0.01, 'gamma': 5}
	mean: 0.40290, std: 0.00005, params: {'C': 0.01, 'gamma': 10}
	mean: 0.40290, std: 0.00005, params: {'C': 0.01, 'gamma': 50}
	mean: 0.40290, std: 0.00005, params: {'C': 0.01, 'gamma': 100}
	mean: 0.60852, std: 0.00519, params: {'C': 0.05, 'gamma': 0.01}
	mean: 0.62730, std: 0.00432, params: {'C': 0.05, 'gamma': 0.05}
	mean: 0.63021, std: 0.00478, params: {'C': 0.05, 'gamma': 0.1}
	mean: 0.59228, std: 0.00161, params: {'C': 0.05, 'gamma': 0.5}
	mean: 0.40290, std: 0.00005, params: {'C': 0.05, 'gamma': 1}
	mean: 0.40290, std: 0.00005, params: {'C': 0.05, 'gamma': 5}
	mean: 0.40290, std: 0.00005, params: {'C': 0.05, 'gamma': 10}
	mean: 0.40290, std: 0.00005, params: {'C': 0.05, 'gamma': 50}
	mean: 0.40290, std: 0.00005, params: {'C': 0.05, 'gamma': 100}
	mean: 0.61761, std: 0.00617, params: {'C': 0.1, 'gamma': 0.01}
	mean: 0.63482, std: 0.00482, params: {'C': 0.1, 'gamma': 0.05}
	mean: 0.63953, std: 0.00514, params: {'C': 0.1, 'gamma': 0.1}
	mean: 0.60477, std: 0.00058, params: {'C': 0.1, 'gamma': 0.5}
	mean: 0.40327, std: 0.00010, params: {'C': 0.1, 'gamma': 1}
	mean: 0.40290, std: 0.00005, params: {'C': 0.1, 'gamma': 5}
	mean: 0.40290, std: 0.00005, params: {'C': 0.1, 'gamma': 10}
	mean: 0.40290, std: 0.00005, params: {'C': 0.1, 'gamma': 50}
	mean: 0.40290, std: 0.00005, params: {'C': 0.1, 'gamma': 100}
	mean: 0.64439, std: 0.00457, params: {'C': 0.5, 'gamma': 0.01}
	mean: 0.66179, std: 0.00348, params: {'C': 0.5, 'gamma': 0.05}
	mean: 0.66579, std: 0.00352, params: {'C': 0.5, 'gamma': 0.1}
	mean: 0.66774, std: 0.00300, params: {'C': 0.5, 'gamma': 0.5}
	mean: 0.60578, std: 0.00036, params: {'C': 0.5, 'gamma': 1}
	mean: 0.44379, std: 0.00009, params: {'C': 0.5, 'gamma': 5}
	mean: 0.44222, std: 0.00030, params: {'C': 0.5, 'gamma': 10}
	mean: 0.44005, std: 0.00027, params: {'C': 0.5, 'gamma': 50}
	mean: 0.43877, std: 0.00089, params: {'C': 0.5, 'gamma': 100}
	mean: 0.65165, std: 0.00307, params: {'C': 1, 'gamma': 0.01}
	mean: 0.66677, std: 0.00275, params: {'C': 1, 'gamma': 0.05}
	mean: 0.67313, std: 0.00214, params: {'C': 1, 'gamma': 0.1}
	mean: 0.68057, std: 0.00234, params: {'C': 1, 'gamma': 0.5}
	mean: 0.64166, std: 0.00236, params: {'C': 1, 'gamma': 1}
	mean: 0.50398, std: 0.00163, params: {'C': 1, 'gamma': 5}
	mean: 0.50286, std: 0.00140, params: {'C': 1, 'gamma': 10}
	mean: 0.49949, std: 0.00149, params: {'C': 1, 'gamma': 50}
	mean: 0.49620, std: 0.00199, params: {'C': 1, 'gamma': 100}
	mean: 0.66086, std: 0.00285, params: {'C': 5, 'gamma': 0.01}
	mean: 0.67668, std: 0.00148, params: {'C': 5, 'gamma': 0.05}
	mean: 0.68136, std: 0.00242, params: {'C': 5, 'gamma': 0.1}
	mean: 0.66815, std: 0.00110, params: {'C': 5, 'gamma': 0.5}
	mean: 0.64769, std: 0.00280, params: {'C': 5, 'gamma': 1}
	mean: 0.50425, std: 0.00177, params: {'C': 5, 'gamma': 5}
	mean: 0.50294, std: 0.00154, params: {'C': 5, 'gamma': 10}
	mean: 0.49949, std: 0.00154, params: {'C': 5, 'gamma': 50}
	mean: 0.49617, std: 0.00194, params: {'C': 5, 'gamma': 100}
	mean: 0.66546, std: 0.00274, params: {'C': 10, 'gamma': 0.01}
	mean: 0.67836, std: 0.00190, params: {'C': 10, 'gamma': 0.05}
	mean: 0.67069, std: 0.00329, params: {'C': 10, 'gamma': 0.1}
	mean: 0.66804, std: 0.00127, params: {'C': 10, 'gamma': 0.5}
	mean: 0.64757, std: 0.00282, params: {'C': 10, 'gamma': 1}
	mean: 0.50425, std: 0.00177, params: {'C': 10, 'gamma': 5}
	mean: 0.50286, std: 0.00163, params: {'C': 10, 'gamma': 10}
	mean: 0.49949, std: 0.00154, params: {'C': 10, 'gamma': 50}
	mean: 0.49617, std: 0.00194, params: {'C': 10, 'gamma': 100}
	mean: 0.67373, std: 0.00190, params: {'C': 50, 'gamma': 0.01}
	mean: 0.66303, std: 0.00327, params: {'C': 50, 'gamma': 0.05}
	mean: 0.64716, std: 0.00210, params: {'C': 50, 'gamma': 0.1}
	mean: 0.66766, std: 0.00183, params: {'C': 50, 'gamma': 0.5}
	mean: 0.64656, std: 0.00208, params: {'C': 50, 'gamma': 1}
	mean: 0.50417, std: 0.00186, params: {'C': 50, 'gamma': 5}
	mean: 0.50286, std: 0.00163, params: {'C': 50, 'gamma': 10}
	mean: 0.49953, std: 0.00154, params: {'C': 50, 'gamma': 50}
	mean: 0.49620, std: 0.00193, params: {'C': 50, 'gamma': 100}
	mean: 0.67679, std: 0.00191, params: {'C': 100, 'gamma': 0.01}
	mean: 0.65551, std: 0.00322, params: {'C': 100, 'gamma': 0.05}
	mean: 0.64335, std: 0.00312, params: {'C': 100, 'gamma': 0.1}
	mean: 0.66609, std: 0.00273, params: {'C': 100, 'gamma': 0.5}
	mean: 0.64608, std: 0.00179, params: {'C': 100, 'gamma': 1}
	mean: 0.50417, std: 0.00186, params: {'C': 100, 'gamma': 5}
	mean: 0.50286, std: 0.00163, params: {'C': 100, 'gamma': 10}
	mean: 0.49953, std: 0.00154, params: {'C': 100, 'gamma': 50}
	mean: 0.49620, std: 0.00193, params: {'C': 100, 'gamma': 100}
```

### Sigmoid kernel

In addition to the C's, I tried R's in [.01, .05, .1, .5, 1, 5, 10, 50, 100].

I might be doing something wrong here. Mean was .4029 for all configurations. In any case, I probably won't toy with this one any further.

I won't bother pasting the result since they were all the same.

### Polynomial kernel

For time expediency's sake, I reduced the search space on this one. I only looked at degree 2, and [.01, .1, 1, 10] for both C and R.  It peaks at just under **66% accuracy for C=10 and R=10**, which is on the boundary of the search space, so I'm not sure I've got a good estimate of how well it can perform.  Will probably want to come back to this one later.

```
Best parameters: {'C': 10, 'coef0': 10, 'degree': 2}

All parameters:
	mean: 0.41412, std: 0.00229, params: {'C': 0.01, 'coef0': 0.01, 'degree': 2}
	mean: 0.41461, std: 0.00192, params: {'C': 0.01, 'coef0': 0.1, 'degree': 2}
	mean: 0.60144, std: 0.00474, params: {'C': 0.01, 'coef0': 1, 'degree': 2}
	mean: 0.62842, std: 0.00536, params: {'C': 0.01, 'coef0': 10, 'degree': 2}
	mean: 0.44177, std: 0.00730, params: {'C': 0.1, 'coef0': 0.01, 'degree': 2}
	mean: 0.61536, std: 0.00542, params: {'C': 0.1, 'coef0': 0.1, 'degree': 2}
	mean: 0.63418, std: 0.00453, params: {'C': 0.1, 'coef0': 1, 'degree': 2}
	mean: 0.64731, std: 0.00259, params: {'C': 0.1, 'coef0': 10, 'degree': 2}
	mean: 0.62939, std: 0.00480, params: {'C': 1, 'coef0': 0.01, 'degree': 2}
	mean: 0.64432, std: 0.00445, params: {'C': 1, 'coef0': 0.1, 'degree': 2}
	mean: 0.65169, std: 0.00242, params: {'C': 1, 'coef0': 1, 'degree': 2}
	mean: 0.65296, std: 0.00193, params: {'C': 1, 'coef0': 10, 'degree': 2}
	mean: 0.65801, std: 0.00361, params: {'C': 10, 'coef0': 0.01, 'degree': 2}
	mean: 0.65779, std: 0.00433, params: {'C': 10, 'coef0': 0.1, 'degree': 2}
	mean: 0.65917, std: 0.00323, params: {'C': 10, 'coef0': 1, 'degree': 2}
	mean: 0.65966, std: 0.00262, params: {'C': 10, 'coef0': 10, 'degree': 2}
```

## SVM (LinearSVC)

I looked at the full suite of C's from above (from .01 to 100), and both hinge and squared hinge loss. I only examined L2 regularization.

