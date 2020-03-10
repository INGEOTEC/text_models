.. text_models documentation master file.

Text Models
==================================
.. image:: https://travis-ci.org/INGEOTEC/EvoMSA.svg?branch=master
	   :target: https://travis-ci.org/INGEOTEC/EvoMSA

.. image:: https://ci.appveyor.com/api/projects/status/wg01w00evm7pb8po?svg=true
	   :target: https://ci.appveyor.com/project/mgraffg/evomsa

.. image:: https://coveralls.io/repos/github/INGEOTEC/EvoMSA/badge.svg?branch=master	    
	   :target: https://coveralls.io/github/INGEOTEC/EvoMSA?branch=master

.. image:: https://anaconda.org/ingeotec/evomsa/badges/version.svg
	   :target: https://anaconda.org/ingeotec/evomsa

.. image:: https://badge.fury.io/py/EvoMSA.svg
	   :target: https://badge.fury.io/py/EvoMSA

.. image:: https://readthedocs.org/projects/evomsa/badge/?version=latest
	   :target: https://evomsa.readthedocs.io/en/latest/?badge=latest


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
