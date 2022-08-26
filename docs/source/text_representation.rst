.. _text_representation:

Text Representation
==========================================

.. image:: https://github.com/INGEOTEC/text_models/actions/workflows/test.yaml/badge.svg
	   :target: https://github.com/INGEOTEC/text_models/actions/workflows/test.yaml

.. image:: https://badge.fury.io/py/text-models.svg
	  :target: https://badge.fury.io/py/text-models

.. image:: https://coveralls.io/repos/github/INGEOTEC/text_models/badge.svg?branch=develop
	  :target: https://coveralls.io/github/INGEOTEC/text_models?branch=develop

.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/text_models-feedstock?branchName=main
	  :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=16894&branchName=main

.. image:: https://img.shields.io/conda/vn/conda-forge/text_models.svg
	  :target: https://anaconda.org/conda-forge/text_models

.. image:: https://img.shields.io/conda/pn/conda-forge/text_models.svg
	  :target: https://anaconda.org/conda-forge/text_models

.. image:: https://readthedocs.org/projects/text-models/badge/?version=latest
      :target: https://text-models.readthedocs.io/en/latest/?badge=latest
      :alt: Documentation Status

Solving a text categorization problem usually starts by deciding which text transformation 
to use; the traditional approach would be to decide on a :ref:`bow`. 
A BoW representation using q-grams of character and n-grams of words produces 
a satisfactory baseline (with term-frequency inverse document-frequency as its 
weighting scheme and linear support vector machine as the classifier, e.g.,
`B4MSA <https://b4msa.readthedocs.io/en/latest>`_). 
However, it cannot encode more information than the one used 
to learn the BoW and train the classifier. 

Techniques in semi-supervised learning have been used to incorporate information 
into text representations. The basic idea is to convert a dataset into a labeled 
dataset where the labels are automatically identified. 
The :ref:`emoji` developed here followed this idea. 
The dataset used to train these representations is a set of tweets collected 
from the Twitter open stream in Arabic, Chinese, English, French, Portuguese, 
Russian, and Spanish.

.. _bow:

Bag of Word (BoW) Representation
--------------------------------------

Once the dataset is ready, it is time to develop the representations. 
The first step is to transform the text into a suitable representation. 
Particularly, it is used a BoW representation with q-grams of 
characters (2, 3, and 4) and words; the constraint is that the q-grams are 
only computed on the words, and consequently, there are no q-grams between words. 
The exception is Chinese, which only uses q-grams of 1, 2, and 3. 
The BoW was learned from 524,288 tweets randomly selected from the 
`text_models <https://text-models.readthedocs.io/en/latest/>`_ collection.

The BoW model is implemented with `microTC <https://microtc.readthedocs.io/en/latest/>`_; 
the only particular characteristic is that only the 16,384 more frequent tokens 
were kept in the representation. The BoW models for the different languages are found in:

* `Arabic <https://github.com/INGEOTEC/text_models/releases/download/models/ar_2.4.2.microtc>`_
* `Chinese <https://github.com/INGEOTEC/text_models/releases/download/models/zh_2.4.2.microtc>`_ 
* `English <https://github.com/INGEOTEC/text_models/releases/download/models/en_2.4.2.microtc>`_
* `French <https://github.com/INGEOTEC/text_models/releases/download/models/fr_2.4.2.microtc>`_
* `Portuguese <https://github.com/INGEOTEC/text_models/releases/download/models/pt_2.4.2.microtc>`_
* `Russian <https://github.com/INGEOTEC/text_models/releases/download/models/ru_2.4.2.microtc>`_
* `Spanish <https://github.com/INGEOTEC/text_models/releases/download/models/es_2.4.2.microtc>`_


These representations can be used as follows:

>>> from text_models.utils import load_bow
>>> bow = load_bow(lang='es')
>>> X = bow.transform(['Hola', 'Está funcionando'])
>>> X.shape
(2, 16384)

where the text `Hola` (`Hi` in English) and `Está funcionando` (`It is working`) 
are transformed into matrix :math:`\mathbb R^{2 \times 16384}`.

.. _emoji:

Emoji Text Representation
--------------------------------

Transforming the dataset into a labeled dataset is similar to the one used in 
`deepmoji <https://aclanthology.org/D17-1169/>`_. 
The idea is to use the emoji in the text as the labels. 
The process selects and removes the emojis in the tweets and keeps them as the 
labels of the text. The emojis kept are the ones that appear at least 1024 times 
alone, i.e., the text contains only one emoji. 
