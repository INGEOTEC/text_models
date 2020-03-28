.. text_models documentation master file.

Text Models
==================================
.. image:: https://travis-ci.org/INGEOTEC/text_models.svg?branch=master
	   :target: https://travis-ci.org/INGEOTEC/text_models

.. image:: https://coveralls.io/repos/github/INGEOTEC/text_models/badge.svg?branch=master
	   :target: https://coveralls.io/github/INGEOTEC/text_models?branch=master

.. image:: https://badge.fury.io/py/text_models.svg
	   :target: https://badge.fury.io/py/text_models

.. image:: https://readthedocs.org/projects/text-models/badge/?version=latest
      :target: https://text-models.readthedocs.io/en/latest/?badge=latest
      :alt: Documentation Status		    		       


INGEOTEC Text Model package deals with the creation of labeled
datasets using a self-supervised approach. It includes different text
models developed for Arabic, English, Spanish languages, as well as
the classes used to create them. Finally, it a modified Beam selection
technique, tailored to text classification, that chooses those text
models that are more suitable for a given text classification problem.

Text models are functions that transform a text into a vector, i.e.,
:math:`text \rightarrow R^d`. Different methods have been proposed to represent
text in a vector space. Perhaps, one of the most popular ones is the
Bag of Word model.

BoW model and a dataset of labeled texts can be served as a building
block to create other text models. The approach is to define the text
model as a composition of BoW and a classifier trained on the labeled
dataset.

The labeled dataset can be created using self-supervised learning.


:mod:`text_models.model_selection`
==================================

.. automodule:: text_models.model_selection
   :members:

:mod:`text_models.dataset`
==================================

.. automodule:: text_models.dataset
   :members:   

:mod:`text_models.vocabulary`
==================================

.. automodule:: text_models.vocabulary
   :members:  