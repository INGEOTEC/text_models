.. _perf_dataset_repr:

Performance Dataset Representation
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


Spanish
==========================

.. list-table:: MeTwo
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - DOUBTFUL
	  - :math:`0.3854 \pm 0.0456`
	  - :math:`0.8605 \pm 0.0553`
	  - :math:`0.2483 \pm 0.0361`
	* - NON_SEXIST
	  - :math:`0.7827 \pm 0.0152`
	  - :math:`0.7372 \pm 0.0193`
	  - :math:`0.8342 \pm 0.0198`
	* - SEXIST
	  - :math:`0.6501 \pm 0.0231`
	  - :math:`0.7409 \pm 0.0292`
	  - :math:`0.5791 \pm 0.0261`

.. list-table:: davincis2022_1
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.7235 \pm 0.0208`
	  - :math:`0.7247 \pm 0.0260`
	  - :math:`0.7224 \pm 0.0253`

.. list-table:: delitos_ingeotec
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.7727 \pm 0.0380`
	  - :math:`0.6711 \pm 0.0517`
	  - :math:`0.9107 \pm 0.0392`

.. list-table:: detests2022_task1
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.6224 \pm 0.0314`
	  - :math:`0.5393 \pm 0.0363`
	  - :math:`0.7357 \pm 0.0366`

.. list-table:: exist2021_task1
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - sexist
	  - :math:`0.7408 \pm 0.0183`
	  - :math:`0.7735 \pm 0.0227`
	  - :math:`0.7108 \pm 0.0236`

.. list-table:: haha2018
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.7459 \pm 0.0087`
	  - :math:`0.7004 \pm 0.0107`
	  - :math:`0.7977 \pm 0.0113`

.. list-table:: meoffendes2021_task1
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - NO
	  - :math:`0.9144 \pm 0.0038`
	  - :math:`0.8815 \pm 0.0059`
	  - :math:`0.9500 \pm 0.0041`
	* - NOM
	  - :math:`0.5000 \pm 0.0228`
	  - :math:`0.8066 \pm 0.0263`
	  - :math:`0.3623 \pm 0.0214`
	* - OFG
	  - :math:`0.0838 \pm 0.0159`
	  - :math:`0.6757 \pm 0.0822`
	  - :math:`0.0446 \pm 0.0088`
	* - OFP
	  - :math:`0.5327 \pm 0.0172`
	  - :math:`0.7990 \pm 0.0207`
	  - :math:`0.3995 \pm 0.0170`

.. list-table:: meoffendes2021_task3
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.5566 \pm 0.0259`
	  - :math:`0.4866 \pm 0.0286`
	  - :math:`0.6502 \pm 0.0322`

.. list-table:: mexa3t2018_aggress
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.6866 \pm 0.0133`
	  - :math:`0.6455 \pm 0.0166`
	  - :math:`0.7333 \pm 0.0156`

.. list-table:: misoginia
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.7741 \pm 0.0177`
	  - :math:`0.7859 \pm 0.0227`
	  - :math:`0.7626 \pm 0.0238`

.. list-table:: misogyny_centrogeo
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.8882 \pm 0.0088`
	  - :math:`0.8925 \pm 0.0110`
	  - :math:`0.8840 \pm 0.0114`

.. list-table:: semeval2018_anger
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.5646 \pm 0.0288`
	  - :math:`0.6243 \pm 0.0348`
	  - :math:`0.5153 \pm 0.0328`
	* - 1
	  - :math:`0.4453 \pm 0.0278`
	  - :math:`0.6073 \pm 0.0361`
	  - :math:`0.3515 \pm 0.0269`
	* - 2
	  - :math:`0.4131 \pm 0.0292`
	  - :math:`0.7163 \pm 0.0388`
	  - :math:`0.2902 \pm 0.0254`
	* - 3
	  - :math:`0.4023 \pm 0.0339`
	  - :math:`0.6509 \pm 0.0444`
	  - :math:`0.2911 \pm 0.0302`

.. list-table:: semeval2018_fear
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.6876 \pm 0.0240`
	  - :math:`0.7225 \pm 0.0293`
	  - :math:`0.6560 \pm 0.0297`
	* - 1
	  - :math:`0.4364 \pm 0.0314`
	  - :math:`0.5934 \pm 0.0405`
	  - :math:`0.3450 \pm 0.0298`
	* - 2
	  - :math:`0.4141 \pm 0.0315`
	  - :math:`0.6560 \pm 0.0453`
	  - :math:`0.3026 \pm 0.0278`
	* - 3
	  - :math:`0.4600 \pm 0.0350`
	  - :math:`0.8214 \pm 0.0424`
	  - :math:`0.3194 \pm 0.0307`

.. list-table:: semeval2018_joy
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.6922 \pm 0.0208`
	  - :math:`0.7986 \pm 0.0239`
	  - :math:`0.6108 \pm 0.0265`
	* - 1
	  - :math:`0.4170 \pm 0.0251`
	  - :math:`0.5765 \pm 0.0351`
	  - :math:`0.3266 \pm 0.0235`
	* - 2
	  - :math:`0.4795 \pm 0.0289`
	  - :math:`0.7115 \pm 0.0368`
	  - :math:`0.3616 \pm 0.0281`
	* - 3
	  - :math:`0.3853 \pm 0.0351`
	  - :math:`0.6632 \pm 0.0498`
	  - :math:`0.2716 \pm 0.0301`

.. list-table:: semeval2018_sadness
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.6331 \pm 0.0269`
	  - :math:`0.7104 \pm 0.0315`
	  - :math:`0.5709 \pm 0.0319`
	* - 1
	  - :math:`0.4510 \pm 0.0277`
	  - :math:`0.5693 \pm 0.0341`
	  - :math:`0.3734 \pm 0.0283`
	* - 2
	  - :math:`0.3946 \pm 0.0294`
	  - :math:`0.6541 \pm 0.0421`
	  - :math:`0.2825 \pm 0.0256`
	* - 3
	  - :math:`0.4563 \pm 0.0403`
	  - :math:`0.7059 \pm 0.0521`
	  - :math:`0.3371 \pm 0.0368`

.. list-table:: semeval2018_valence
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - -3
	  - :math:`0.3755 \pm 0.0404`
	  - :math:`0.6500 \pm 0.0547`
	  - :math:`0.2640 \pm 0.0343`
	* - -2
	  - :math:`0.3610 \pm 0.0301`
	  - :math:`0.6847 \pm 0.0458`
	  - :math:`0.2452 \pm 0.0242`
	* - -1
	  - :math:`0.4222 \pm 0.0292`
	  - :math:`0.6597 \pm 0.0387`
	  - :math:`0.3105 \pm 0.0266`
	* - 0
	  - :math:`0.3463 \pm 0.0285`
	  - :math:`0.5594 \pm 0.0403`
	  - :math:`0.2508 \pm 0.0247`
	* - 1
	  - :math:`0.2609 \pm 0.0321`
	  - :math:`0.6176 \pm 0.0582`
	  - :math:`0.1654 \pm 0.0233`
	* - 2
	  - :math:`0.2435 \pm 0.0327`
	  - :math:`0.6471 \pm 0.0667`
	  - :math:`0.1500 \pm 0.0230`
	* - 3
	  - :math:`0.4095 \pm 0.0451`
	  - :math:`0.8431 \pm 0.0510`
	  - :math:`0.2704 \pm 0.0369`

.. list-table:: tass2016
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - N
	  - :math:`0.6250 \pm 0.0028`
	  - :math:`0.8317 \pm 0.0029`
	  - :math:`0.5006 \pm 0.0031`
	* - NEU
	  - :math:`0.0743 \pm 0.0022`
	  - :math:`0.7946 \pm 0.0112`
	  - :math:`0.0390 \pm 0.0012`
	* - NONE
	  - :math:`0.5923 \pm 0.0028`
	  - :math:`0.5876 \pm 0.0034`
	  - :math:`0.5971 \pm 0.0033`
	* - P
	  - :math:`0.6952 \pm 0.0024`
	  - :math:`0.7496 \pm 0.0030`
	  - :math:`0.6482 \pm 0.0030`

.. list-table:: tass2017
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - N
	  - :math:`0.6460 \pm 0.0264`
	  - :math:`0.6667 \pm 0.0324`
	  - :math:`0.6266 \pm 0.0317`
	* - NEU
	  - :math:`0.2555 \pm 0.0324`
	  - :math:`0.5942 \pm 0.0585`
	  - :math:`0.1627 \pm 0.0235`
	* - NONE
	  - :math:`0.2960 \pm 0.0340`
	  - :math:`0.6613 \pm 0.0581`
	  - :math:`0.1907 \pm 0.0253`
	* - P
	  - :math:`0.5691 \pm 0.0316`
	  - :math:`0.6859 \pm 0.0389`
	  - :math:`0.4864 \pm 0.0337`

.. list-table:: tass2018_s1_l1
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - UNSAFE
	  - :math:`0.8013 \pm 0.0173`
	  - :math:`0.8322 \pm 0.0206`
	  - :math:`0.7726 \pm 0.0228`

.. list-table:: tass2018_s1_l2
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - UNSAFE
	  - :math:`0.8390 \pm 0.0031`
	  - :math:`0.8329 \pm 0.0039`
	  - :math:`0.8453 \pm 0.0040`

.. list-table:: tass2018_s2
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - UNSAFE
	  - :math:`0.7776 \pm 0.0189`
	  - :math:`0.8845 \pm 0.0198`
	  - :math:`0.6937 \pm 0.0254`

English
=======================

.. list-table:: SCv1
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.6086 \pm 0.0175`
	  - :math:`0.6148 \pm 0.0207`
	  - :math:`0.6025 \pm 0.0205`

.. list-table:: SCv2-GEN
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.6881 \pm 0.0105`
	  - :math:`0.6681 \pm 0.0125`
	  - :math:`0.7093 \pm 0.0133`

.. list-table:: SS-Twitter
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.7824 \pm 0.0122`
	  - :math:`0.8230 \pm 0.0146`
	  - :math:`0.7455 \pm 0.0157`

.. list-table:: SS-Youtube
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 1
	  - :math:`0.8782 \pm 0.0088`
	  - :math:`0.9219 \pm 0.0096`
	  - :math:`0.8385 \pm 0.0126`

.. list-table:: business
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - david_leonhardt
	  - :math:`0.8000 \pm 0.0841`
	  - :math:`0.8000 \pm 0.1070`
	  - :math:`0.8000 \pm 0.1053`
	* - david_segal
	  - :math:`0.4262 \pm 0.0830`
	  - :math:`0.8667 \pm 0.0887`
	  - :math:`0.2826 \pm 0.0701`
	* - david_streitfeld
	  - :math:`0.7895 \pm 0.0765`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6522 \pm 0.1029`
	* - james_glanz
	  - :math:`0.8387 \pm 0.0794`
	  - :math:`0.8667 \pm 0.0948`
	  - :math:`0.8125 \pm 0.1029`
	* - javier_c_hernandez
	  - :math:`0.8750 \pm 0.0657`
	  - :math:`0.9333 \pm 0.0708`
	  - :math:`0.8235 \pm 0.0937`
	* - louise_story
	  - :math:`0.8485 \pm 0.0751`
	  - :math:`0.9333 \pm 0.0628`
	  - :math:`0.7778 \pm 0.1050`

.. list-table:: ccat
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - AlanCrosby
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`1.0000 \pm 0.0000`
	* - AlexanderSmith
	  - :math:`0.8197 \pm 0.0393`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6944 \pm 0.0561`
	* - BenjaminKangLim
	  - :math:`0.5119 \pm 0.0441`
	  - :math:`0.8600 \pm 0.0466`
	  - :math:`0.3644 \pm 0.0418`
	* - DavidLawder
	  - :math:`0.6250 \pm 0.0530`
	  - :math:`0.7000 \pm 0.0651`
	  - :math:`0.5645 \pm 0.0633`
	* - JaneMacartney
	  - :math:`0.5786 \pm 0.0471`
	  - :math:`0.9200 \pm 0.0387`
	  - :math:`0.4220 \pm 0.0475`
	* - JimGilchrist
	  - :math:`0.9800 \pm 0.0146`
	  - :math:`0.9800 \pm 0.0219`
	  - :math:`0.9800 \pm 0.0195`
	* - MarcelMichelson
	  - :math:`0.9375 \pm 0.0260`
	  - :math:`0.9000 \pm 0.0436`
	  - :math:`0.9783 \pm 0.0217`
	* - MureDickie
	  - :math:`0.5217 \pm 0.0449`
	  - :math:`0.9600 \pm 0.0291`
	  - :math:`0.3582 \pm 0.0415`
	* - RobinSidel
	  - :math:`0.8909 \pm 0.0329`
	  - :math:`0.9800 \pm 0.0201`
	  - :math:`0.8167 \pm 0.0508`
	* - ToddNissen
	  - :math:`0.5938 \pm 0.0514`
	  - :math:`0.7600 \pm 0.0589`
	  - :math:`0.4872 \pm 0.0557`

.. list-table:: cricket
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - PeterRoebuck
	  - :math:`0.7895 \pm 0.0808`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6522 \pm 0.1074`
	* - SambitBal
	  - :math:`0.8387 \pm 0.0787`
	  - :math:`0.8667 \pm 0.0902`
	  - :math:`0.8125 \pm 0.1047`
	* - dileep_premachandran
	  - :math:`0.8966 \pm 0.0622`
	  - :math:`0.8667 \pm 0.0836`
	  - :math:`0.9286 \pm 0.0739`
	* - ian_chappel
	  - :math:`0.9375 \pm 0.0470`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.8824 \pm 0.0804`

.. list-table:: news20c
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - alt.atheism
	  - :math:`0.5464 \pm 0.0207`
	  - :math:`0.8025 \pm 0.0235`
	  - :math:`0.4142 \pm 0.0205`
	* - comp.graphics
	  - :math:`0.4499 \pm 0.0157`
	  - :math:`0.8946 \pm 0.0160`
	  - :math:`0.3005 \pm 0.0134`
	* - comp.os.ms-windows.misc
	  - :math:`0.5441 \pm 0.0166`
	  - :math:`0.8223 \pm 0.0187`
	  - :math:`0.4065 \pm 0.0167`
	* - comp.sys.ibm.pc.hardware
	  - :math:`0.4506 \pm 0.0164`
	  - :math:`0.8776 \pm 0.0174`
	  - :math:`0.3031 \pm 0.0140`
	* - comp.sys.mac.hardware
	  - :math:`0.5231 \pm 0.0168`
	  - :math:`0.9247 \pm 0.0135`
	  - :math:`0.3648 \pm 0.0157`
	* - comp.windows.x
	  - :math:`0.6461 \pm 0.0167`
	  - :math:`0.9266 \pm 0.0136`
	  - :math:`0.4959 \pm 0.0187`
	* - misc.forsale
	  - :math:`0.6237 \pm 0.0158`
	  - :math:`0.9564 \pm 0.0099`
	  - :math:`0.4628 \pm 0.0169`
	* - rec.autos
	  - :math:`0.5905 \pm 0.0166`
	  - :math:`0.9066 \pm 0.0139`
	  - :math:`0.4378 \pm 0.0172`
	* - rec.motorcycles
	  - :math:`0.7206 \pm 0.0164`
	  - :math:`0.9070 \pm 0.0147`
	  - :math:`0.5977 \pm 0.0200`
	* - rec.sport.baseball
	  - :math:`0.6600 \pm 0.0157`
	  - :math:`0.9093 \pm 0.0144`
	  - :math:`0.5179 \pm 0.0180`
	* - rec.sport.hockey
	  - :math:`0.7894 \pm 0.0149`
	  - :math:`0.9298 \pm 0.0125`
	  - :math:`0.6858 \pm 0.0208`
	* - sci.crypt
	  - :math:`0.8543 \pm 0.0135`
	  - :math:`0.8737 \pm 0.0155`
	  - :math:`0.8357 \pm 0.0190`
	* - sci.electronics
	  - :math:`0.4357 \pm 0.0165`
	  - :math:`0.8015 \pm 0.0204`
	  - :math:`0.2991 \pm 0.0142`
	* - sci.med
	  - :math:`0.6932 \pm 0.0183`
	  - :math:`0.8131 \pm 0.0207`
	  - :math:`0.6041 \pm 0.0221`
	* - sci.space
	  - :math:`0.7950 \pm 0.0152`
	  - :math:`0.8909 \pm 0.0160`
	  - :math:`0.7178 \pm 0.0204`
	* - soc.religion.christian
	  - :math:`0.6757 \pm 0.0163`
	  - :math:`0.9347 \pm 0.0123`
	  - :math:`0.5292 \pm 0.0188`
	* - talk.politics.guns
	  - :math:`0.6286 \pm 0.0175`
	  - :math:`0.8929 \pm 0.0160`
	  - :math:`0.4851 \pm 0.0193`
	* - talk.politics.mideast
	  - :math:`0.8916 \pm 0.0116`
	  - :math:`0.8856 \pm 0.0163`
	  - :math:`0.8976 \pm 0.0151`
	* - talk.politics.misc
	  - :math:`0.4055 \pm 0.0202`
	  - :math:`0.7097 \pm 0.0259`
	  - :math:`0.2839 \pm 0.0173`
	* - talk.religion.misc
	  - :math:`0.3058 \pm 0.0166`
	  - :math:`0.7729 \pm 0.0272`
	  - :math:`0.1906 \pm 0.0122`

.. list-table:: news4c
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - comp
	  - :math:`0.9596 \pm 0.0032`
	  - :math:`0.9652 \pm 0.0042`
	  - :math:`0.9540 \pm 0.0045`
	* - politics
	  - :math:`0.8709 \pm 0.0082`
	  - :math:`0.9029 \pm 0.0094`
	  - :math:`0.8412 \pm 0.0111`
	* - rec
	  - :math:`0.9392 \pm 0.0044`
	  - :math:`0.9572 \pm 0.0054`
	  - :math:`0.9219 \pm 0.0065`
	* - religion
	  - :math:`0.8638 \pm 0.0084`
	  - :math:`0.9205 \pm 0.0086`
	  - :math:`0.8137 \pm 0.0122`

.. list-table:: nfl
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - joe_lapointe
	  - :math:`0.8485 \pm 0.0700`
	  - :math:`0.9333 \pm 0.0632`
	  - :math:`0.7778 \pm 0.0994`
	* - judy_battista
	  - :math:`0.8750 \pm 0.0630`
	  - :math:`0.9333 \pm 0.0650`
	  - :math:`0.8235 \pm 0.0899`
	* - pete_thamel
	  - :math:`0.6957 \pm 0.1148`
	  - :math:`0.5333 \pm 0.1308`
	  - :math:`1.0000 \pm 0.0000`

.. list-table:: offenseval2019_A
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - OFF
	  - :math:`0.5829 \pm 0.0293`
	  - :math:`0.4833 \pm 0.0324`
	  - :math:`0.7342 \pm 0.0352`

.. list-table:: offenseval2019_B
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - UNT
	  - :math:`0.2857 \pm 0.0991`
	  - :math:`0.1852 \pm 0.0729`
	  - :math:`0.6250 \pm 0.1958`

.. list-table:: offenseval2019_C
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - GRP
	  - :math:`0.6556 \pm 0.0391`
	  - :math:`0.7564 \pm 0.0484`
	  - :math:`0.5784 \pm 0.0466`
	* - IND
	  - :math:`0.6872 \pm 0.0400`
	  - :math:`0.6700 \pm 0.0490`
	  - :math:`0.7053 \pm 0.0466`
	* - OTH
	  - :math:`0.3497 \pm 0.0506`
	  - :math:`0.7143 \pm 0.0821`
	  - :math:`0.2315 \pm 0.0395`

.. list-table:: poetry
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - abbey
	  - :math:`0.4545 \pm 0.1374`
	  - :math:`0.5000 \pm 0.1701`
	  - :math:`0.4167 \pm 0.1496`
	* - benet
	  - :math:`0.7143 \pm 0.1030`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.5556 \pm 0.1183`
	* - eliot
	  - :math:`0.6897 \pm 0.1011`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.5263 \pm 0.1149`
	* - hardy
	  - :math:`0.6429 \pm 0.1113`
	  - :math:`0.9000 \pm 0.0947`
	  - :math:`0.5000 \pm 0.1212`
	* - wilde
	  - :math:`0.3125 \pm 0.1058`
	  - :math:`0.5000 \pm 0.1690`
	  - :math:`0.2273 \pm 0.0901`
	* - wordsworth
	  - :math:`0.4706 \pm 0.1500`
	  - :math:`0.8000 \pm 0.2056`
	  - :math:`0.3333 \pm 0.1352`

.. list-table:: r10
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - acq
	  - :math:`0.9744 \pm 0.0043`
	  - :math:`0.9856 \pm 0.0047`
	  - :math:`0.9635 \pm 0.0071`
	* - coffee
	  - :math:`0.9778 \pm 0.0253`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.9565 \pm 0.0469`
	* - crude
	  - :math:`0.8958 \pm 0.0205`
	  - :math:`0.9587 \pm 0.0182`
	  - :math:`0.8406 \pm 0.0318`
	* - earn
	  - :math:`0.9875 \pm 0.0023`
	  - :math:`0.9871 \pm 0.0034`
	  - :math:`0.9880 \pm 0.0032`
	* - interest
	  - :math:`0.7560 \pm 0.0343`
	  - :math:`0.9753 \pm 0.0183`
	  - :math:`0.6172 \pm 0.0436`
	* - money-fx
	  - :math:`0.6537 \pm 0.0355`
	  - :math:`0.9655 \pm 0.0209`
	  - :math:`0.4941 \pm 0.0393`
	* - money-supply
	  - :math:`0.4779 \pm 0.0609`
	  - :math:`0.9643 \pm 0.0346`
	  - :math:`0.3176 \pm 0.0526`
	* - ship
	  - :math:`0.6195 \pm 0.0529`
	  - :math:`0.9722 \pm 0.0277`
	  - :math:`0.4545 \pm 0.0564`
	* - sugar
	  - :math:`0.9412 \pm 0.0337`
	  - :math:`0.9600 \pm 0.0397`
	  - :math:`0.9231 \pm 0.0499`
	* - trade
	  - :math:`0.7150 \pm 0.0358`
	  - :math:`0.9867 \pm 0.0143`
	  - :math:`0.5606 \pm 0.0431`

.. list-table:: r52
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - acq
	  - :math:`0.9539 \pm 0.0058`
	  - :math:`0.9813 \pm 0.0054`
	  - :math:`0.9280 \pm 0.0096`
	* - alum
	  - :math:`0.6531 \pm 0.0791`
	  - :math:`0.8421 \pm 0.0816`
	  - :math:`0.5333 \pm 0.0894`
	* - bop
	  - :math:`0.2857 \pm 0.0752`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.1667 \pm 0.0510`
	* - carcass
	  - :math:`0.0199 \pm 0.0083`
	  - :math:`1.0000 \pm 0.0772`
	  - :math:`0.0101 \pm 0.0042`
	* - cocoa
	  - :math:`0.9032 \pm 0.0619`
	  - :math:`0.9333 \pm 0.0676`
	  - :math:`0.8750 \pm 0.0874`
	* - coffee
	  - :math:`0.9362 \pm 0.0365`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.8800 \pm 0.0631`
	* - copper
	  - :math:`0.8966 \pm 0.0608`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.8125 \pm 0.0961`
	* - cotton
	  - :math:`0.8182 \pm 0.0938`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6923 \pm 0.1294`
	* - cpi
	  - :math:`0.5263 \pm 0.0805`
	  - :math:`0.8824 \pm 0.0815`
	  - :math:`0.3750 \pm 0.0766`
	* - cpu
	  - :math:`0.0204 \pm 0.0207`
	  - :math:`1.0000 \pm 0.4828`
	  - :math:`0.0103 \pm 0.0107`
	* - crude
	  - :math:`0.8227 \pm 0.0259`
	  - :math:`0.9587 \pm 0.0188`
	  - :math:`0.7205 \pm 0.0368`
	* - dlr
	  - :math:`0.2105 \pm 0.1336`
	  - :math:`0.6667 \pm 0.3406`
	  - :math:`0.1250 \pm 0.0916`
	* - earn
	  - :math:`0.9862 \pm 0.0025`
	  - :math:`0.9880 \pm 0.0032`
	  - :math:`0.9844 \pm 0.0039`
	* - fuel
	  - :math:`0.1818 \pm 0.0871`
	  - :math:`0.4286 \pm 0.2060`
	  - :math:`0.1154 \pm 0.0610`
	* - gas
	  - :math:`0.0185 \pm 0.0084`
	  - :math:`0.6250 \pm 0.1903`
	  - :math:`0.0094 \pm 0.0043`
	* - gnp
	  - :math:`0.3000 \pm 0.0589`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.1765 \pm 0.0409`
	* - gold
	  - :math:`0.8163 \pm 0.0588`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6897 \pm 0.0823`
	* - grain
	  - :math:`0.0615 \pm 0.0193`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.0317 \pm 0.0103`
	* - heat
	  - :math:`0.0845 \pm 0.0437`
	  - :math:`0.7500 \pm 0.2465`
	  - :math:`0.0448 \pm 0.0243`
	* - housing
	  - :math:`0.2353 \pm 0.1316`
	  - :math:`1.0000 \pm 0.3026`
	  - :math:`0.1333 \pm 0.0874`
	* - income
	  - :math:`0.1860 \pm 0.0813`
	  - :math:`1.0000 \pm 0.1089`
	  - :math:`0.1026 \pm 0.0500`
	* - instal-debt
	  - :math:`0.0513 \pm 0.0489`
	  - :math:`1.0000 \pm 0.4844`
	  - :math:`0.0263 \pm 0.0263`
	* - interest
	  - :math:`0.7817 \pm 0.0333`
	  - :math:`0.9506 \pm 0.0256`
	  - :math:`0.6638 \pm 0.0447`
	* - ipi
	  - :math:`0.4074 \pm 0.0854`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.2558 \pm 0.0669`
	* - iron-steel
	  - :math:`0.1727 \pm 0.0427`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.0945 \pm 0.0256`
	* - jet
	  - :math:`0.0000 \pm 0.0000`
	  - :math:`0.0000 \pm 0.0000`
	  - :math:`0.0000 \pm 0.0000`
	* - jobs
	  - :math:`0.7742 \pm 0.0906`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6316 \pm 0.1145`
	* - lead
	  - :math:`0.0485 \pm 0.0227`
	  - :math:`1.0000 \pm 0.1467`
	  - :math:`0.0248 \pm 0.0120`
	* - lei
	  - :math:`0.2609 \pm 0.1250`
	  - :math:`1.0000 \pm 0.2551`
	  - :math:`0.1500 \pm 0.0828`
	* - livestock
	  - :math:`0.0353 \pm 0.0152`
	  - :math:`1.0000 \pm 0.0631`
	  - :math:`0.0180 \pm 0.0079`
	* - lumber
	  - :math:`0.0100 \pm 0.0050`
	  - :math:`1.0000 \pm 0.1530`
	  - :math:`0.0050 \pm 0.0025`
	* - meal-feed
	  - :math:`0.0015 \pm 0.0015`
	  - :math:`1.0000 \pm 0.4800`
	  - :math:`0.0008 \pm 0.0007`
	* - money-fx
	  - :math:`0.6667 \pm 0.0345`
	  - :math:`0.9540 \pm 0.0226`
	  - :math:`0.5123 \pm 0.0393`
	* - money-supply
	  - :math:`0.4091 \pm 0.0531`
	  - :math:`0.9643 \pm 0.0350`
	  - :math:`0.2596 \pm 0.0421`
	* - nat-gas
	  - :math:`0.2353 \pm 0.0558`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.1333 \pm 0.0361`
	* - nickel
	  - :math:`0.0021 \pm 0.0022`
	  - :math:`1.0000 \pm 0.4782`
	  - :math:`0.0011 \pm 0.0011`
	* - orange
	  - :math:`0.4737 \pm 0.1066`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.3103 \pm 0.0909`
	* - pet-chem
	  - :math:`0.0227 \pm 0.0089`
	  - :math:`1.0000 \pm 0.0631`
	  - :math:`0.0115 \pm 0.0046`
	* - platinum
	  - :math:`0.0024 \pm 0.0016`
	  - :math:`1.0000 \pm 0.2918`
	  - :math:`0.0012 \pm 0.0008`
	* - potato
	  - :math:`0.0328 \pm 0.0180`
	  - :math:`1.0000 \pm 0.1812`
	  - :math:`0.0167 \pm 0.0093`
	* - reserves
	  - :math:`0.2526 \pm 0.0593`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.1446 \pm 0.0387`
	* - retail
	  - :math:`0.0870 \pm 0.0819`
	  - :math:`1.0000 \pm 0.4859`
	  - :math:`0.0455 \pm 0.0466`
	* - rubber
	  - :math:`0.5161 \pm 0.1092`
	  - :math:`0.8889 \pm 0.1134`
	  - :math:`0.3636 \pm 0.1024`
	* - ship
	  - :math:`0.5528 \pm 0.0557`
	  - :math:`0.9444 \pm 0.0369`
	  - :math:`0.3908 \pm 0.0542`
	* - strategic-metal
	  - :math:`0.0214 \pm 0.0090`
	  - :math:`0.8333 \pm 0.1807`
	  - :math:`0.0108 \pm 0.0046`
	* - sugar
	  - :math:`0.8846 \pm 0.0502`
	  - :math:`0.9200 \pm 0.0541`
	  - :math:`0.8519 \pm 0.0700`
	* - tea
	  - :math:`0.0072 \pm 0.0041`
	  - :math:`1.0000 \pm 0.2551`
	  - :math:`0.0036 \pm 0.0021`
	* - tin
	  - :math:`0.0706 \pm 0.0215`
	  - :math:`0.9000 \pm 0.1107`
	  - :math:`0.0367 \pm 0.0116`
	* - trade
	  - :math:`0.6577 \pm 0.0356`
	  - :math:`0.9733 \pm 0.0180`
	  - :math:`0.4966 \pm 0.0396`
	* - veg-oil
	  - :math:`0.2136 \pm 0.0546`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.1196 \pm 0.0341`
	* - wpi
	  - :math:`0.6207 \pm 0.1090`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.4500 \pm 0.1116`
	* - zinc
	  - :math:`0.0249 \pm 0.0114`
	  - :math:`1.0000 \pm 0.1089`
	  - :math:`0.0126 \pm 0.0059`

.. list-table:: r8
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - acq
	  - :math:`0.9752 \pm 0.0040`
	  - :math:`0.9871 \pm 0.0040`
	  - :math:`0.9635 \pm 0.0070`
	* - crude
	  - :math:`0.8712 \pm 0.0229`
	  - :math:`0.9504 \pm 0.0197`
	  - :math:`0.8042 \pm 0.0345`
	* - earn
	  - :math:`0.9875 \pm 0.0024`
	  - :math:`0.9871 \pm 0.0033`
	  - :math:`0.9880 \pm 0.0033`
	* - grain
	  - :math:`0.1513 \pm 0.0444`
	  - :math:`0.9000 \pm 0.0982`
	  - :math:`0.0826 \pm 0.0264`
	* - interest
	  - :math:`0.8000 \pm 0.0301`
	  - :math:`0.9877 \pm 0.0119`
	  - :math:`0.6723 \pm 0.0419`
	* - money-fx
	  - :math:`0.7414 \pm 0.0330`
	  - :math:`0.9885 \pm 0.0114`
	  - :math:`0.5931 \pm 0.0413`
	* - ship
	  - :math:`0.4242 \pm 0.0471`
	  - :math:`0.9722 \pm 0.0294`
	  - :math:`0.2713 \pm 0.0382`
	* - trade
	  - :math:`0.7813 \pm 0.0326`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.6410 \pm 0.0438`

.. list-table:: semeval2017
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - negative
	  - :math:`0.6153 \pm 0.0054`
	  - :math:`0.8200 \pm 0.0062`
	  - :math:`0.4924 \pm 0.0058`
	* - neutral
	  - :math:`0.6034 \pm 0.0053`
	  - :math:`0.6069 \pm 0.0065`
	  - :math:`0.5999 \pm 0.0061`
	* - positive
	  - :math:`0.5592 \pm 0.0079`
	  - :math:`0.6884 \pm 0.0094`
	  - :math:`0.4708 \pm 0.0088`

.. list-table:: semeval2018_anger
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.6560 \pm 0.0182`
	  - :math:`0.6624 \pm 0.0219`
	  - :math:`0.6498 \pm 0.0223`
	* - 1
	  - :math:`0.2529 \pm 0.0229`
	  - :math:`0.5135 \pm 0.0410`
	  - :math:`0.1678 \pm 0.0172`
	* - 2
	  - :math:`0.3647 \pm 0.0242`
	  - :math:`0.5185 \pm 0.0332`
	  - :math:`0.2812 \pm 0.0219`
	* - 3
	  - :math:`0.4584 \pm 0.0289`
	  - :math:`0.6986 \pm 0.0381`
	  - :math:`0.3411 \pm 0.0266`

.. list-table:: semeval2018_fear
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.7536 \pm 0.0132`
	  - :math:`0.7393 \pm 0.0172`
	  - :math:`0.7685 \pm 0.0170`
	* - 1
	  - :math:`0.2122 \pm 0.0243`
	  - :math:`0.4758 \pm 0.0452`
	  - :math:`0.1366 \pm 0.0174`
	* - 2
	  - :math:`0.2900 \pm 0.0248`
	  - :math:`0.4873 \pm 0.0402`
	  - :math:`0.2064 \pm 0.0198`
	* - 3
	  - :math:`0.3297 \pm 0.0375`
	  - :math:`0.6479 \pm 0.0578`
	  - :math:`0.2212 \pm 0.0296`

.. list-table:: semeval2018_joy
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.4585 \pm 0.0276`
	  - :math:`0.6546 \pm 0.0350`
	  - :math:`0.3528 \pm 0.0267`
	* - 1
	  - :math:`0.4160 \pm 0.0228`
	  - :math:`0.4985 \pm 0.0273`
	  - :math:`0.3570 \pm 0.0231`
	* - 2
	  - :math:`0.4642 \pm 0.0213`
	  - :math:`0.6222 \pm 0.0276`
	  - :math:`0.3702 \pm 0.0208`
	* - 3
	  - :math:`0.4684 \pm 0.0236`
	  - :math:`0.7661 \pm 0.0288`
	  - :math:`0.3374 \pm 0.0218`

.. list-table:: semeval2018_sadness
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.6737 \pm 0.0188`
	  - :math:`0.7236 \pm 0.0236`
	  - :math:`0.6302 \pm 0.0223`
	* - 1
	  - :math:`0.3173 \pm 0.0244`
	  - :math:`0.5285 \pm 0.0382`
	  - :math:`0.2267 \pm 0.0198`
	* - 2
	  - :math:`0.3944 \pm 0.0230`
	  - :math:`0.5569 \pm 0.0315`
	  - :math:`0.3054 \pm 0.0211`
	* - 3
	  - :math:`0.4141 \pm 0.0283`
	  - :math:`0.6822 \pm 0.0431`
	  - :math:`0.2973 \pm 0.0245`

.. list-table:: semeval2018_valence
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - -3
	  - :math:`0.3280 \pm 0.0314`
	  - :math:`0.6667 \pm 0.0496`
	  - :math:`0.2175 \pm 0.0244`
	* - -2
	  - :math:`0.4034 \pm 0.0248`
	  - :math:`0.7066 \pm 0.0341`
	  - :math:`0.2823 \pm 0.0214`
	* - -1
	  - :math:`0.1600 \pm 0.0207`
	  - :math:`0.5250 \pm 0.0542`
	  - :math:`0.0944 \pm 0.0133`
	* - 0
	  - :math:`0.4591 \pm 0.0231`
	  - :math:`0.5992 \pm 0.0307`
	  - :math:`0.3720 \pm 0.0227`
	* - 1
	  - :math:`0.2389 \pm 0.0256`
	  - :math:`0.5794 \pm 0.0483`
	  - :math:`0.1505 \pm 0.0181`
	* - 2
	  - :math:`0.2667 \pm 0.0271`
	  - :math:`0.6154 \pm 0.0503`
	  - :math:`0.1702 \pm 0.0199`
	* - 3
	  - :math:`0.5037 \pm 0.0311`
	  - :math:`0.7445 \pm 0.0375`
	  - :math:`0.3806 \pm 0.0305`

.. list-table:: travel
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - jeff_bailey
	  - :math:`0.8966 \pm 0.0605`
	  - :math:`0.8667 \pm 0.0889`
	  - :math:`0.9286 \pm 0.0696`
	* - matthew_wald
	  - :math:`0.9091 \pm 0.0561`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.8333 \pm 0.0913`
	* - micheline_maynard
	  - :math:`0.5714 \pm 0.0846`
	  - :math:`0.8000 \pm 0.1038`
	  - :math:`0.4444 \pm 0.0866`
	* - michelle_higgins
	  - :math:`0.8333 \pm 0.0679`
	  - :math:`1.0000 \pm 0.0000`
	  - :math:`0.7143 \pm 0.0982`

Arabic
====================

.. list-table:: semeval2017
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - negative
	  - :math:`0.5977 \pm 0.0076`
	  - :math:`0.7570 \pm 0.0090`
	  - :math:`0.4938 \pm 0.0085`
	* - neutral
	  - :math:`0.4803 \pm 0.0092`
	  - :math:`0.4670 \pm 0.0103`
	  - :math:`0.4944 \pm 0.0106`
	* - positive
	  - :math:`0.4505 \pm 0.0101`
	  - :math:`0.5594 \pm 0.0129`
	  - :math:`0.3771 \pm 0.0105`

.. list-table:: semeval2017_taskBD
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - positive
	  - :math:`0.7391 \pm 0.0087`
	  - :math:`0.7322 \pm 0.0113`
	  - :math:`0.7461 \pm 0.0109`

.. list-table:: semeval2018_anger
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.4475 \pm 0.0402`
	  - :math:`0.6622 \pm 0.0534`
	  - :math:`0.3379 \pm 0.0380`
	* - 1
	  - :math:`0.4437 \pm 0.0353`
	  - :math:`0.5462 \pm 0.0453`
	  - :math:`0.3736 \pm 0.0351`
	* - 2
	  - :math:`0.2179 \pm 0.0336`
	  - :math:`0.4667 \pm 0.0648`
	  - :math:`0.1421 \pm 0.0240`
	* - 3
	  - :math:`0.5741 \pm 0.0338`
	  - :math:`0.7583 \pm 0.0395`
	  - :math:`0.4619 \pm 0.0354`

.. list-table:: semeval2018_fear
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.5447 \pm 0.0380`
	  - :math:`0.6837 \pm 0.0469`
	  - :math:`0.4527 \pm 0.0399`
	* - 1
	  - :math:`0.4030 \pm 0.0407`
	  - :math:`0.6000 \pm 0.0564`
	  - :math:`0.3034 \pm 0.0358`
	* - 2
	  - :math:`0.4953 \pm 0.0351`
	  - :math:`0.5852 \pm 0.0429`
	  - :math:`0.4293 \pm 0.0371`
	* - 3
	  - :math:`0.3368 \pm 0.0463`
	  - :math:`0.6531 \pm 0.0708`
	  - :math:`0.2270 \pm 0.0373`

.. list-table:: semeval2018_joy
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.4615 \pm 0.0376`
	  - :math:`0.7600 \pm 0.0479`
	  - :math:`0.3314 \pm 0.0341`
	* - 1
	  - :math:`0.4255 \pm 0.0332`
	  - :math:`0.5385 \pm 0.0412`
	  - :math:`0.3518 \pm 0.0334`
	* - 2
	  - :math:`0.5860 \pm 0.0277`
	  - :math:`0.6429 \pm 0.0336`
	  - :math:`0.5385 \pm 0.0327`
	* - 3
	  - :math:`0.3711 \pm 0.0424`
	  - :math:`0.7660 \pm 0.0644`
	  - :math:`0.2449 \pm 0.0339`

.. list-table:: semeval2018_sadness
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - 0
	  - :math:`0.6099 \pm 0.0337`
	  - :math:`0.6719 \pm 0.0406`
	  - :math:`0.5584 \pm 0.0398`
	* - 1
	  - :math:`0.2414 \pm 0.0359`
	  - :math:`0.4912 \pm 0.0633`
	  - :math:`0.1600 \pm 0.0269`
	* - 2
	  - :math:`0.3478 \pm 0.0359`
	  - :math:`0.5333 \pm 0.0500`
	  - :math:`0.2581 \pm 0.0312`
	* - 3
	  - :math:`0.5714 \pm 0.0386`
	  - :math:`0.6737 \pm 0.0478`
	  - :math:`0.4961 \pm 0.0431`

.. list-table:: semeval2018_valence
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - -3
	  - :math:`0.2932 \pm 0.0293`
	  - :math:`0.7000 \pm 0.0520`
	  - :math:`0.1854 \pm 0.0214`
	* - -2
	  - :math:`0.4885 \pm 0.0268`
	  - :math:`0.6648 \pm 0.0340`
	  - :math:`0.3861 \pm 0.0270`
	* - -1
	  - :math:`0.2362 \pm 0.0273`
	  - :math:`0.6716 \pm 0.0570`
	  - :math:`0.1433 \pm 0.0187`
	* - 0
	  - :math:`0.2681 \pm 0.0259`
	  - :math:`0.5676 \pm 0.0453`
	  - :math:`0.1755 \pm 0.0194`
	* - 1
	  - :math:`0.2607 \pm 0.0279`
	  - :math:`0.5556 \pm 0.0493`
	  - :math:`0.1703 \pm 0.0208`
	* - 2
	  - :math:`0.4578 \pm 0.0316`
	  - :math:`0.7000 \pm 0.0395`
	  - :math:`0.3401 \pm 0.0294`
	* - 3
	  - :math:`0.4318 \pm 0.0391`
	  - :math:`0.7403 \pm 0.0515`
	  - :math:`0.3048 \pm 0.0349`

Chinese
=======================

.. list-table:: NLPCC2013_emotion
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - Anger
	  - :math:`0.3404 \pm 0.0212`
	  - :math:`0.6716 \pm 0.0323`
	  - :math:`0.2280 \pm 0.0170`
	* - Disgust
	  - :math:`0.4972 \pm 0.0176`
	  - :math:`0.7356 \pm 0.0217`
	  - :math:`0.3755 \pm 0.0171`
	* - Fear
	  - :math:`0.1219 \pm 0.0192`
	  - :math:`0.8043 \pm 0.0608`
	  - :math:`0.0660 \pm 0.0111`
	* - Happiness
	  - :math:`0.5850 \pm 0.0176`
	  - :math:`0.7348 \pm 0.0208`
	  - :math:`0.4859 \pm 0.0191`
	* - Like
	  - :math:`0.5991 \pm 0.0161`
	  - :math:`0.7289 \pm 0.0190`
	  - :math:`0.5086 \pm 0.0182`
	* - Sadness
	  - :math:`0.5292 \pm 0.0192`
	  - :math:`0.7674 \pm 0.0233`
	  - :math:`0.4038 \pm 0.0194`
	* - Surprise
	  - :math:`0.1735 \pm 0.0193`
	  - :math:`0.6782 \pm 0.0539`
	  - :math:`0.0995 \pm 0.0120`

.. list-table:: NLPCC2013_opinion
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - Y
	  - :math:`0.8968 \pm 0.0114`
	  - :math:`0.9288 \pm 0.0140`
	  - :math:`0.8670 \pm 0.0168`

.. list-table:: NLPCC2013_polarity
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - NEG
	  - :math:`0.7850 \pm 0.0242`
	  - :math:`0.8025 \pm 0.0309`
	  - :math:`0.7683 \pm 0.0320`
	* - NEU
	  - :math:`0.0420 \pm 0.0239`
	  - :math:`0.5000 \pm 0.2212`
	  - :math:`0.0219 \pm 0.0128`
	* - OTHER
	  - :math:`0.0787 \pm 0.0326`
	  - :math:`0.5556 \pm 0.1606`
	  - :math:`0.0424 \pm 0.0184`
	* - POS
	  - :math:`0.7628 \pm 0.0262`
	  - :math:`0.7605 \pm 0.0331`
	  - :math:`0.7651 \pm 0.0329`

.. list-table:: online_shopping_polarity
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - POS
	  - :math:`0.9222 \pm 0.0025`
	  - :math:`0.9200 \pm 0.0035`
	  - :math:`0.9245 \pm 0.0031`

.. list-table:: simplifyweibo_4_moods
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - Anger
	  - :math:`0.4039 \pm 0.0033`
	  - :math:`0.6628 \pm 0.0048`
	  - :math:`0.2905 \pm 0.0029`
	* - Happiness
	  - :math:`0.7612 \pm 0.0018`
	  - :math:`0.7150 \pm 0.0022`
	  - :math:`0.8138 \pm 0.0022`
	* - Sadness
	  - :math:`0.4311 \pm 0.0034`
	  - :math:`0.6696 \pm 0.0047`
	  - :math:`0.3178 \pm 0.0031`

.. list-table:: waimai_polarity
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - POS
	  - :math:`0.9030 \pm 0.0054`
	  - :math:`0.9070 \pm 0.0073`
	  - :math:`0.8991 \pm 0.0078`

.. list-table:: weibo_senti_100k_polarity
	:header-rows: 1

	* - label
	  - f1
	  - recall
	  - precision
	* - POS
	  - :math:`0.9056 \pm 0.0019`
	  - :math:`0.9289 \pm 0.0023`
	  - :math:`0.8834 \pm 0.0028`

