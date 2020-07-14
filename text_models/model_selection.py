# Copyright 2020 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from EvoMSA.base import EvoMSA
from queue import LifoQueue
from microtc.utils import save_model
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from .utils import macro_f1
import numpy as np
import os


class Node(object):
    """
    Base class to perform model selection on the first-stage models

    :param model: Models used in the node - List of keys from :py:attr:`models`
    :type model: list
    :param models: Dictionary of pairs (see :py:attr:`EvoMSA.base.EvoMSA.models`)
    :type model: dict
    :param metric: Performance metric, e.g., accuracy
    :type metric: function
    :param split_dataset: Iterator to split dataset in training and validation
    :type split_dataset: instance
    :param aggregate: :math:`\\text{aggregate}: \\mathbb R^d \\rightarrow \\mathbb R`
    :type aggregate: function
    :param cache: Store the output of text models
    :type cache: str
    :param TR: EvoMSA's default model
    :type TR: bool
    :param stacked_method: Classifier or regressor used to ensemble the outputs of :attr:`EvoMSA.models`
    :type stacked_method: str or class

    """

    def __init__(self, model, models=None,
                 metric=None,
                 split_dataset=None,
                 aggregate=None,
                 cache=None,
                 TR=False,
                 stacked_method="sklearn.naive_bayes.GaussianNB",
                 **kwargs):
        assert metric is not None
        assert split_dataset is not None and hasattr(split_dataset, "split")
        assert aggregate is not None
        assert cache is not None
        self._models = models
        self._model = [x for x in model]
        _ = self._model.copy()
        _.sort()
        self._repr = "-".join(map(str, _))
        self._metric = metric
        self._split_dataset = split_dataset
        self._aggregate = aggregate
        self._cache = cache
        self._TR = TR
        self._kwargs = kwargs
        self._kwargs.update(dict(stacked_method=stacked_method))

    def __repr__(self):
        return self._repr

    def __eq__(self, other):

        return isinstance(other, Node) and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        variables = set(self._models.keys())
        model = self._model
        for x in variables - set(model):
            yield self.__class__(model + [x],
                                 models=self._models,
                                 metric=self._metric,
                                 split_dataset=self._split_dataset,
                                 aggregate=self._aggregate,
                                 cache=self._cache,
                                 TR=self._TR,
                                 **self._kwargs)

    @property
    def model(self):
        """Models as received by :py:class:`EvoMSA.base.EvoMSA`"""

        models = self._models
        return [models[x] for x in self._model]

    def _fit(self, X, y, cache):
        """Create an EvoMSA's instance

        :param X: Training set - independent variables
        :type X: list
        :param y: Training set - dependent variable
        :type y: list or np.array
        :param TR: EvoMSA's default model
        :type TR: bool
        :param test_set: Dataset to perform transductive learning
        :type test_set: list
        :rtype: self
        """

        return EvoMSA(TR=self._TR, models=self.model,
                      cache=cache,
                      **self._kwargs).fit(X, y)

    @property
    def perf(self):
        """Performance"""
        return self._perf

    def performance(self, X, y):
        """Compute the performance on the dataset

        :param X: Test set - independent variables
        :type X: list
        :param y: Test set - dependent variable
        :type y: list or np.array
        :rtype: float
        """

        try:
            return self._perf
        except AttributeError:
            perf = []
            cache = self._cache
            for index, (tr, vs) in enumerate(self._split_dataset.split(X)):
                evo = self._fit([X[x] for x in tr],
                                [y[x] for x in tr],
                                cache=cache + "-tr-" + str(index))
                hy = evo.predict([X[x] for x in vs],
                                 cache=cache + "-vs-" + str(index))
                perf.append(self._metric([y[x] for x in vs], hy))
            self._perf = self._aggregate(perf)
        return self._perf

 #   def __cmp__(self, other):
 #       x = self.perf
 #       y = other.perf
 #       return (x > y) - (x < y)

    def __gt__(self, other):
        return self.perf > other.perf


class ForwardSelection(object):
    """Forward Selection on the models

    >>> from EvoMSA import base
    >>> from EvoMSA.utils import download
    >>> from text_models.model_selection import ForwardSelection
    >>> from microtc.utils import tweet_iterator
    >>> import os

    Read the dataset

    >>> tweets = os.path.join(os.path.dirname(base.__file__), 'tests', 'tweets.json')
    >>> D = [[x['text'], x['klass']] for x in tweet_iterator(tweets)]
    
    Models

    >>> models = dict()
    >>> models[0] = [download("emo_Es.tm"), "sklearn.svm.LinearSVC"]
    >>> models[1] = ["EvoMSA.model.AggressivenessEs", "sklearn.svm.LinearSVC"]
    >>> models[2] = [download("b4msa_Es.tm"), "sklearn.svm.LinearSVC"]
    >>> X = [x for x, y in D]
    >>> y = [y for x, y in D]
    >>> fwdSel = ForwardSelection(models)
    >>> best = fwdSel.run(X, y)

    :param models: Dictionary of pairs (see :py:attr:`EvoMSA.base.EvoMSA.models`)
    :type models: dict
    :param node: Node use to perform the search
    :type node: :py:class:`text_models.model_selection.Node`
    :param output: Filename to store intermediate models
    :type output: str
    :param verbose: Level to inform the user
    :type verbose: int
    :param metric: Performance metric
    :type metric: function
    :param split_dataset: Iterator to split dataset in training and validation
    :type split_dataset: instance
    :param aggregate: :math:`\\text{aggregate}: \\mathbb R^d \\rightarrow \\mathbb R`
    :type aggregate: function
    :param cache: Store the output of text models
    :type cache: str

    """

    def __init__(self, models, node=Node,
                 output=None, verbose=logging.INFO,
                 metric=macro_f1,
                 split_dataset=KFold(n_splits=3, random_state=1, shuffle=True),
                 aggregate=np.median,
                 cache=os.path.join("cache", "fw"),
                 **kwargs):
        self._models = models
        self._nodes = [node([k], models=models,
                            metric=metric,
                            split_dataset=split_dataset,
                            aggregate=aggregate,
                            cache=cache,
                            **kwargs) for k in models.keys()]
        self._output = output
        self._logger = logging.getLogger("text_models.model_selection")
        self._logger.setLevel(verbose)

    def run(self, X, y):
        """Perform the search using X and y to guide it

        :param X: Dataset set - independent variables
        :type X: list
        :param y: Dataset set - dependent variable
        :type y: list or np.array
        :rtype: :py:class:`EvoMSA.model_selection.Node`
        """

        self._logger.info("Starting the search")
        r = [(node.performance(X, y), node) for node in self._nodes]
        node = max(r, key=lambda x: x[0])[1]
        while True:
            self._logger.info("Model: %s perf: %0.4f" % (node, node.perf))
            nodes = list(node)
            if len(nodes) == 0:
                if self._output:
                    save_model(node, self._output)
                return node
            r = [(xx.performance(X, y), xx) for xx in nodes]
            perf, comp = max(r, key=lambda x: x[0])
            if perf < node.perf:
                break
            node = comp
        if self._output:
            save_model(node, self._output)
        return node


class BeamSelection(ForwardSelection):
    """
    Select the models using Beam Search.
    """

    def run(self, X, y, early_stopping=1000):
        """

        :param early_stopping: Number of rounds to perform early stopping
        :type early_stopping: int
        :rtype: :py:class:`text_models.model_selection.Node`
        """

        visited = [(node.performance(X, y), node) for
                   node in self._nodes]
        _ = max(visited, key=lambda x: x[0])[1]
        best = None
        nodes = LifoQueue()
        nodes.put(_)
        index = len(visited)
        visited = set([x[1] for x in visited])
        while not nodes.empty() and (len(visited) - index) < early_stopping:
            node = nodes.get()
            if best is None or node > best:
                index = len(visited)
                best = node
                if self._output:
                    save_model(best, self._output)
            self._logger.info("Model: %s perf: %0.4f " % (best, best.perf) +
                              "visited: %s " % len(visited) +
                              "size: %s " % nodes.qsize() +
                              "Rounds: %s" % (len(visited) - index))
            nn = [(xx, xx.performance(X, y)) for
                  xx in node if xx not in visited]
            [visited.add(x) for x, _ in nn]
            nn = [xx for xx, perf in nn if perf >= node.perf]
            if len(nn) == 0:
                continue
            nn.sort()
            [nodes.put(x) for x in nn]
        return best
