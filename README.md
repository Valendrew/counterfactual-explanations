# AI in Industry Project
This repository contains the project code for the *AI in Industry* exam, at the University of Bologna.

The work concerns the development of a model combining *Machine Learning* and *Optimization* for **Counterfactual Explanations** using [OMLT](https://github.com/cog-imperial/OMLT) and [DiCE](https://github.com/interpretml/DiCE) Python packages. The main goal of the project is to generate counterfactuals with different techniques and to evaluate the strenghts and the weaknesses of each method. 

A counterfactual explanation describes a causal situation in the form: "If X had not occurred, Y would not have occurred", a typical example is "if one or more features were different I would have got the loan from the bank".
In interpretable machine learning, counterfactual explanations can be used to explain predictions of individual instances. 

<div align="center">
    <img src=https://www.microsoft.com/en-us/research/uploads/prod/2020/01/MSR-Amit_1400x788-v3-1blog.gif width="65%" />
    <p style="font-size:0.8rem" align="center">
        <em>Image taken from the Microsoft <a href="https://www.microsoft.com/en-us/research/blog/open-source-library-provides-explanation-for-machine-learning-through-diverse-counterfactuals/"> blog </a> about DiCE </em>
    </p>
</div>

First of all we needed to train a machine learning model to make it guess if a given device belongs to the low, medium or high price range. After that the model is trained we can use it to generate counterfactual explanations of the desired class, in our case we considered only changes of labels of 1 (i.e. from low to medium price range, from medium to low price range, ...). In the case of samples that were misclassified by our model, we generated a counterfactual to change the wrong label in the actual one, such that we can observe which features the optimization model would change to correct the model classification.

***Notice***: you can run all the notebooks using jupyter notebook and showing them as a slideshow, after having installed the requirements, everything should work as expected.

You can also find an implementation of a GUI interface that uses our model based on DiCE to generate counterfactuals for some devices. You can find the repository with the code and the instructions to launch the demo at [this link](https://github.com/Valendrew/counterfactual-demo).

## Dataset
<div align="center">
    <img src="/images/gsmarena_logo.png" width="40%" />
</div>

We decided to use the [GSM arena dataset](https://www.kaggle.com/datasets/msainani/gsmarena-mobile-devices) taken from Kaggle, that contains information about different characteristics of the smartphones (e.g. RAM, ROM, display size, ...), but in order to simplify the work and to get better results we had to perform a lot of data preprocessing and to further integrate the data, adding some more information provided directly by the [GSM arena website](https://www.gsmarena.com/).

After the preprocessing we restricted the number of features from 86 to 22, and the total number of samples is 1911. A detailed explanation of the preprocessing and data exploration can be found in the ```data_preprocessing.ipynb``` and ```data_exploration.ipynb``` notebooks.


## Counterfactual explanations
As explained before, we implemented 2 solutions with different packages that work in a different way. 
- ***OMLT***: using this package we need to encode by ourselves the constraints, objective functions and whatever we want to embed in the model. The package is based on [Pyomo](https://github.com/Pyomo/pyomo) and it allows to encode trees, neural networks and some other machine learning models as Pyomo variables and constraints. We used [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) as a solver for the optimization problem that we formulate in order to generate the counterfactuals.
- ***DiCE***: this package is a creation of Microsoft and it's very easy to use. You simply have to pass all the needed parameters and a machine learning model and the tool is able to generate how many counterfactuals as you need (if it finds a possible solution). 
**WARNING**: in order to use the 'genetic' method to generate the counterfactuals we had to slightly modify the source code of the DiCE package, otherwise the genetic method raised errors when using it with neural networks. Thus, the link in the requirements makes you install the fork that we created from the original project.

A detailed explanation of how we implemented our work on top of these packages can be found in the ```model_optimization.ipynb``` notebook. The evaluation of the results and the comparisons are in the ```model_evaluation.ipynb``` notebook.

## References

- Counterfactual explanations: literature review and benchmarking - [LINK](https://link.springer.com/article/10.1007/s10618-022-00831-6)
- Interpretable Credit Application Predictions With Counterfactual Explanations - [LINK](https://arxiv.org/abs/1811.05245)
- OMLT: Optimization & Machine Learning Toolkit - [LINK](https://arxiv.org/abs/2202.02414)
- Counterfactual Explanations in Interpretable Machine Learning - [LINK](https://christophm.github.io/interpretable-ml-book/counterfactual.html)
- Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations (DiCE) - [LINK](https://arxiv.org/abs/1905.07697)
## Team

[Daniele Morotti](https://github.com/DanieleMorotti) & [Andrea Valente](https://github.com/Valendrew)

