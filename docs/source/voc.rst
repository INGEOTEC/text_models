.. _voc:

Vocabulary
==================================
This module deals with the data of tokens and their frequency obtained 
from collected tweets per day. It can be used to replicate 
:py:attr:`EvoMSA.base.EvoMSA(B4MSA=True)` pre-trained model, 
to develop text models. and to analyze the tokens used in a period. 

B4MSA's pre-trained model
-----------------------------------

Let us start describing the process to replicate the 
pre-trained text models used on EvoMSA for English. 

.. note::
    B4MSA pre-trained model on English was built 
    with the information of 1349 days. 


>>> from text_models.utils import download
>>> from microtc.utils import tweet_iterator
>>> from text_models.vocabulary import Vocabulary
>>> conf = tweet_iterator(download("config.json", cache=False))
>>> data = [x for x in conf if "b4msa_En" in x][0]["b4msa_En"]
>>> voc = Vocabulary(data, lang="En")
>>> tm = voc.create_text_model()

Analyzing the vocabulary (i.e., tokens) on a day 
-----------------------------------

Vocabulary class can also be used to analyze the tokens 
produced on a particular day or period. 
In the next example, let us examine the February 14th, 2020.

>>> from text_models.vocabulary import Vocabulary
>>> voc = Vocabulary("200214.voc", lang="En")
>>> voc.voc.most_common()[:3]
[('q:e~', 2348830), ('q:~~', 2196546), ('q:s~', 2077611)]

The result is that the three most common tokens are q-grams, 
as can be seen, this is not very informative; 
let us remove all the qgrams, and see the result.

>>> voc.remove_qgrams()
>>> voc.voc.most_common()[:3]
[('the', 873889), ('to', 754882), ('i', 684353)]

As can be seen, the result is not informative about 
the events that occurred in the day. Perhaps by removing 
common words would produce an acceptable representation. 

>>> voc.remove(voc.common_words())
>>> voc.voc.most_common()[:3]
[('valentine’s', 63762), ('valentine’s~day', 57285), ('valentines', 53351)]


By removing common words that correspond to the vocabulary used on 
B4MSA pre-trained model, it is obtained an acceptable out of the event 
of the day. It is possible to keep removing words in order to see different 
views of the information produced on that particular day.

Removing the words used in the previous February 14th produces the following output.

>>> voc.remove(voc.day_words())
>>> voc.voc.most_common()[:3]
[('🥺', 11571), ('🥰', 11057), ('#valentinesday2020', 9530)]


:mod:`text_models.vocabulary`
------------------------------------

.. automodule:: text_models.vocabulary
   :members:  