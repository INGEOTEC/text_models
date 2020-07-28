.. text_models documentation master file.

Text Models
==================================
.. image:: https://travis-ci.org/INGEOTEC/text_models.svg?branch=master
	   :target: https://travis-ci.org/INGEOTEC/text_models

.. image:: https://coveralls.io/repos/github/INGEOTEC/text_models/badge.svg?branch=master
	   :target: https://coveralls.io/github/INGEOTEC/text_models?branch=master

.. image:: https://badge.fury.io/py/text-models.svg
	   :target: https://badge.fury.io/py/text-models

.. image:: https://readthedocs.org/projects/text-models/badge/?version=latest
      :target: https://text-models.readthedocs.io/en/latest/?badge=latest
      :alt: Documentation Status		    		       


INGEOTEC Text Model package deals with the creation of text models 
for Arabic, English, and Spanish. Text models are functions that 
transform a text into a vector, i.e., :math:`text \rightarrow R^d`.

The text models created are based on data collected from Twitter. 
The simplest of these models is a Bag of Word (BoW) model using 
:py:class:`b4msa.textmodel.TextModel`  
that it is used as a building block to create more complex models. 
We decided to make public the data (tokens and their frequency) 
used to established B4MSA models and included, as well, different 
methods that allow the analysis of this information (see :ref:`voc`).

B4MSA's models are used to develop self-supervised models. 
The starting point, of a self-supervised approach, is to transform 
a corpus into a labeled dataset. This package includes the classes 
used to create labeled datasets. Finally, it consists of a modified 
Beam selection technique, tailored to text classification, that chooses 
those text models that are more suitable for a given text classification 
problem.

Table of Contents
==================

.. toctree::
   :maxdepth: 1
   :titlesonly:

   voc
   place
