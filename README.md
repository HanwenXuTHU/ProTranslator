# ProTranslator

## Section 1: Introduction
We provided the codes of ProTranslator here, including zero shot, few shot, annotating genesset to pathways and generation of texts from geneset.

ProTranslator redefines protein function prediction as a machine translation problem, which translates the description word sequence of a function to the amino acid sequence of a protein. We can then transfer annotations from functions that have similar textual description to annotate a novel function. 

## Section 2: Installation Tutorial
### Section 2.1: System Requirement
ProTranslator is implemented using Python 3.7 in LINUX. ProTranslator requires torch==1.7.1+cu110, torchvision==0.8.2+cu110, numpy, pandas, sklearn, transformers, networkx, seaborn, tokenizers and so on.
### Section 2.2: How to use our codes
We provided options.py file as the user interface to specify the data files, model and training parameters. You can define your own class of data_loading and model_config, as long as preserving the varible names constant.

In each folder, you can run ProTanslator_main.py to train the model and generate results in the "results/" file. You can perform the prediction based on the ProTranslator_inference.py file we provided. In the few shot setting, we also proviede the ProTranslator_with_DiamondScore.py file for combining the sequence similarity based predictions.
