.. text_models documentation master file.

Text Models
==================================
.. image:: https://github.com/INGEOTEC/text_models/actions/workflows/test.yaml/badge.svg
	   :target: https://github.com/INGEOTEC/text_models/actions/workflows/test.yaml

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

The data collected has geographic information as well as a unique user 
identifier that can be used for another purpose different from creating 
the text models. In particular, the :ref:`place` module uses the geographic 
information collected to estimate the mobility of different regions.

Table of Contents
==================

.. toctree::
   :maxdepth: 1
   :titlesonly:

   voc
   place
