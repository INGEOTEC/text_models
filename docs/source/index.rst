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


Twitter is perhaps the social media more amenable for research. 
It requires only a few steps to obtain information, and there are 
plenty of libraries that can help in this regard. Nonetheless, 
knowing whether a particular event is expressed on Twitter is a 
challenging task that requires a considerable collection of tweets. 
This library aims to facilitate, to a researcher interested, the process 
of mining events on Twitter by opening a collection of processed 
information taken from Twitter since December 2015. The events could be 
related to natural disasters, health issues, and people's mobility, 
among other studies that can be pursued with the library proposed. 
In summary, the Python library retrieves a plethora of information in 
terms of frequencies by day of words and bi-grams of words for Arabic, 
English, Spanish, and Russian languages (see :ref:`voc`). 
As well as mobility information related to the number of travels 
among locations for more than 200 countries or territories (see :ref:`place`).

The library is described in 
`A Python library for exploratory data analysis on twitter data based on tokens and aggregated origin–destination information <https://www.sciencedirect.com/science/article/pii/S0098300421002946>`_.
Mario Graff, Daniela Moctezuma, Sabino Miranda-Jiménez, Eric S.Tellez. Computers & Geosciences
Volume 159, February 2022.

Citing
======

If you find text\_models useful for any academic/scientific purpose, we
would appreciate citations to the following reference:

.. code:: bibtex
	
	  @article{GRAFF2022105012,
        title = {A Python library for exploratory data analysis on twitter data based on tokens and aggregated originâdestination information},
        journal = {Computers & Geosciences},
        volume = {159},
        pages = {105012},
        year = {2022},
        issn = {0098-3004},
        doi = {https://doi.org/10.1016/j.cageo.2021.105012},
        url = {https://www.sciencedirect.com/science/article/pii/S0098300421002946},
        author = {Mario Graff and Daniela Moctezuma and Sabino Miranda-JimÃ©nez and Eric S. Tellez},
        keywords = {Twitter exploratory analysis, Mobility patterns, Open-source Python library},
        }


Table of Contents
==================

.. toctree::
   :maxdepth: 1
   :titlesonly:

   voc
   place
