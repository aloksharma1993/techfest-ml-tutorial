# Classification

In classification, the outcome is one of two or more discrete *classes*. For example, a message in your inbox can be classified as *email* or *spam*. A handwritten digit recognizer has 10 classes, one for each digit (0-9).

In this example, we will look at the dataset from the 2012 PhysioNet/Computing in Cardiology challenge: [http://physionet.org/challenge/2012/](http://physionet.org/challenge/2012/). The challenge is to predict mortality of ICU patients. The dataset contains records of ICU stays of more than 48 hours. At the start of the stay, several attributes such as age, gender and height are recorded for the patient. Then over the 48 hours, various medically relevant (term?) measurements are made. The outcome we want to predict is whether the patient survived or died in-hospital.

## Loading the Dataset

The raw dataset can be found on the challenge website linked above. However, for this tutorial, we did a little preprocessing for you so we spend less time on data munging. If you want to know more about what we did, see the section at the end (TODO: put section name here).

We provide a training set and a test set. The data is stored as a CSV. Let's load the training set.

    import csv

    data = []
    with open('train-a.csv') as csvfile:
        reader = csv.reader(csvfile)
        # Read header
        header = reader.next()
        for row in reader:
             data.append(row)

What kind of attributes do we have? Let's print the list of attributes:

     print header

We have fields like `Age`, `Gender`, `ICUType` and `Height` which only have one value, and then we have a bunch of fields with `FIELDNAME_min`, `FIELDNAME_max`, `FIELDNAME_mean`, etc. This was how we decided to handle the time series measurements. For each medical metric, we computed the min, max, mean, first value, last value and difference between first and last values as a way to represent measurements over time.

The outcome field is `In-hospital_death`. 1 means the patient died in the hospital. 0 means the patient survived.

Let's separate the features from the outcome. We'll be using scikit-learn's machine learning tools in this tutorial and numpy is a useful library for scientific computing.

    import numpy
    features = numpy.array([f[0:-1] for f in data])
    labels = numpy.array([f[-1] for f in data])

#### Missing Values
Not all of the metrics were recorded for each patient. Missing values are represented by -1. All recorded values are nonnegative.

## Understanding the Data

How do we begin understanding our dataset? A good place to start is to measure the proportion of positive outcomes. Since we put our data into numpy arrays, we can use some [very useful methods on arrays](http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html).

    # Proportion of in-hospital deaths.
    labels.mean()
    >>> 0.143

The proportion is fairly low (as would hopefully be expected given modern medicine!), which means we need to be careful about how we measure prediction accuracy. A model which predicts survival for each patient would still be 86% accurate! A better metric would be *recall* or *sensitivity* (of the patients who died, how many did we predict?) and *positive predictivity* (of the ones we predicted positive, what fraction was correct?).

Next, let's see if there are any correlations with age or ICUType.??

## Building a Model

Scikit-learn uses a common API for all their machine learning models, which makes it really easy try different models. Let's start with a simple logistic regression.

A logistic regression is a linear model that is then passed through the logit function. Because it's a linear model, it won't handle missing values represented by -1 very well. Scikit-learn has an `Imputer` class that provides basic strategies for filling in, or imputing, missing values. Let's use the mean.

    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values=-1, strategy='mean')
    features_imputed = imp.fit_transform(features)

Linear models are also sensitive to different ranges for input features. For example, if one input features has a range of -1 to 1 and another has a range of 0 to 100, the feature with the larger range will disproportionately influence the model. (TODO: verify?) To combat this, we scale all the features to **zero mean and unit variance**, using scikit-learn's `StandardScaler` class.

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = preprocessing.scale(features)

Note: We filled in missing values before scaling, otherwise the number used to represent missing values would distort the scaling.

Ok, now we're ready to train the model. Scikit-learn's models all have a `fit()` function to fit the model to the training data and a `predict()` function to predict the outcome on new input data.

    from sklearn import linear_model
    model = linear_model.LogisticRegression()
    model.fit(features, labels)

## Evaluating a Model
Now that we have a model, how do we tell how good it is? This is where the test set comes in. TODO: talk about various metrics for evaluating models.

    accuracy = model.score(test_features, test_labels)
    print 'Accuracy:' accuracy

Models have a `score()` function that returns the mean accuracy. As expected, the accuracy is around 86%. However, as we mentioned before, this is not that meaningful of a metric.

Instead, we can plot the precision-recall curve. TODO: explain precision/recall.

    import matplotlib.pyplot as plt
    predicted_probs = model.predict_proba(test_features)
    precision, recall, thresholds = (
        precision_recall_curve(y_test, predicted_probs[:,1]))
    plt.plot(precision, recall, 's-', lw=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()

## Model Selection

#### Cross Validation

Scikit-learn has a library of utilities for cross-validation and performance evaluation in the `sklearn.cross-validation module`. It has several classes automatically generate different splits of the training set. We will be using the `StratifiedKFold` iterator, which splits the data into *n* folds. *n - 1* folds are used for training, and the *nth* fold is used for test. A stratified K-fold maintains approximately the same percentage of each outcome class as in the complete set.

    from sklearn.cross_validation import StratifiedKFold
    skf = StratifiedKFold(labels, 10)
    for train, test in skf:
        X_train = features[train]
        y_train = labels[train]
        X_test = features[test]
        y_test = labels[test]

## Feature Engineering

Raw data is not a nice csv. How did we preprocess the data?

We only used the training set from the challenge because that's the only labelled dataset available to the public.

TODO: insert code for processing of raw dataset from Physionet website.
