#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

LAMBDA = 0.0005

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


def sigmoid(k):
    """
    :param k: float, linear function value
    :return: float, probability of the linear function value
    """
    return 1 / (1+math.exp(-k))

############################################################
# Milestone 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = defaultdict(int)

    for line in x.split():
        d[line] += 1 if line in d else 1

    return d


############################################################
# Milestone 4: Sentiment Classification


def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    for epoch in range(numEpochs):
        for x, y in trainExamples:
            y = 0 if y < 0 else 1
            phi = featureExtractor(x)
            h = sigmoid(dotProduct(weights, phi))
            # w = w - alpha*(h-y) * feature
            increment(weights, -alpha*(h-y), phi)

        # predictor: a function that takes an x and returns a predicted y.
        def predictor(str):
            return 1 if dotProduct(weights, featureExtractor(str)) > 0 else -1

        # print(f'Train error ({epoch} epoch): {evaluatePredictor(trainExamples, predictor)}')
        # print(f'Validation error ({epoch} epoch): {evaluatePredictor(validationExamples, predictor)}')
    # print(weights)

    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and value is exactly 1.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        for key, value in weights.items():
            if len(phi) == random.randint(1, len(weights)):
                break
            phi[key] = 1
        y = 1 if dotProduct(phi, weights) > 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        d = {}
        x = x.replace(' ', '')
        for i in range(len(x)-n+1):
            if x[i:i+n] in d:
                d[x[i:i+n]] += 1
            else:
                d[x[i:i+n]] = 1

        return d
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

