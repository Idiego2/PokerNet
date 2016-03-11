# PokerNet

## Predicting poker hand's strength with artificial neural networks in Python

Based off the similarly titled study ["Predicting Poker Hand's Strength With Artificial Neural Networks"](bit.ly/1RecknH) by Gökay Dişken from Adana STU, PokerNet is a Python-based and undergraduate-led near identical study and final Artificial Intelligence project.

## Abstract

Based off the similarly titled study “Predicting Poker Hand's Strength With Artificial Neural Networks” by Gökay Dişken from Adana STU, PokerNet is a Python-based variant which investigates the effectiveness of neural network poker hand classification using different supervised learning methods and their associated parameters. We also compare and contrast our results with those presented in the original MATLAB-implemented study. In both studies, artificial neural networks are used to classify five cards drawn from a standard deck of 52 by suits and ranks of each card. The network classifies these cards into their respective poker hands, e.g. pairs, straight, flush, according to the poker game. Python machine learning library PyBrain (https://github.com/pybrain/pybrain) is used to create, train and evaluate the network, using data taken from the UCI data archive page (https://archive.ics.uci.edu/ml/datasets/Poker+Hand). Gradient descent method and resilient back-propagation are selected as training methods, similar to the original study, although the original also included the second order scaled conjugate gradient method which was missing from the machine learning library we chose. Each method is used several times for different numbers of neurons, learning rates, maximum epoch numbers, and hidden transfer functions. Purelin and tansig functions are applied to the hidden layer while the softmax function is applied to the output layer in order to “squash” the weights. Our results were closely aligned with the results published in the original study. We saw that resilient back-propagation achieved the best performance, achieving a peak of 54.8% hit rate and 0.090 MSE. We found this far lower from the original studies numbers from executing a much smaller number of epochs on account of time and power constraints.

## Basics of poker and preprocessing

The card features include a suit and a rank, which range from 1-4 and 1-12, and a single target number, which range from 0-9 and indicate one of ten possible poker hands. The ten hands in poker from weakest to strongest are as follows: “high card”, “one pair”, “two pairs”, “three of a kind”, “straight”, “flush”, “full house”, “four of a kind”, “straight flush”, and “royal flush”. One pair means, there is one pair of equal ranked cards. Two pairs mean we have two pairs of equal ranks. Three of a kind is three equal ranks. Straight means five cards are sequentially ranked without a gap. Flush is the name of a hand consisting five cards with the same suit. Full house is a pair with three of a kind (ranks of the pair and other three are different). Four of a kind is four equal ranks. Straight flush is combination of straight and flush. Royal flush is the always winning hand with Ace, King, Queen, Jack, Ten with flush.

Using Python we read in the training data as a 25010x11 matrix and convert the target class from 0-9 to a bit vector, where the index corresponding to this target class has value 1 and the rest 0. This is done in order to have the same dimensionality between the feature and target vectors. The testing data is also read in, as a 25000x11 matrix, and has this preprocessing applied as well as it will come in handy during the evaluation phase.

## Training Methods

The training methods we chose are two of the three stated in the original study, the reason behind the lacking one being explained above. Gradient descent method with momentum is the first method we examine and it is a back-propagation type neural network. This means the network attempts to find a global minimum along the steepest vector of the error surface by using partial derivatives. The nature of this network also proved to be our downfall in attempting to use another network library, as we had difficulty extending our cost and activation functions due to the intricacy of their derivatives. For this particular method our findings were consistent with the original study in that the learning rate parameter had a large effect on the output of this network. Due to errors in implementation we found the weight values for a network using this method alongside a linear activation function had the problem of reaching infinity very quickly, and this was only remedied by cutting down the learning rate used in the original study, .2 and .02, to, .01 and .001 respectively. However, these smaller learning rates impaired the network as it would require a much larger amount of iterations to reach convergence. We use the momentum parameter to scale the change applied to the weights in each iteration, to increase convergence.

The second and more impressive method used was resilient back-propagation, which functions similarly to a normal back-propagation network with the caveat that it only takes into consideration the sign of the partial derivatives, rather than magnitude. In every iteration, if the weight is found to have changed signs (and thus express instability), that weight is multiplied by 0.5, while weights which remained stable are multiplied by 1.2.

## Simulation Results and Discussion

Simulation data can be viewed in the results/ directory.

We examined both methods in our simulations and used several varied parameters to investigate their effect on the methods. Maybe the most important distinction between this study and the original is the number of epochs used. Due to the lengthy runtime of the network and time constraints imposed upon us by our flip flopping implementations we were forced to settle on cutting down the number of epochs by a magnitude of 10. The effect of this did not wholly prevent us from making conclusions as the data proved to still be statistically significant enough in some cases, but it surely put a damper on results. The original study managed to reach classification rates above 90%, while our best method and parameters could not break 60%. The number of hidden neurons chosen in some simulations varied from 10, to 30, to 50, while the hidden layer activation functions varied between purelin (linear) and tansig (tanh). We fixed the momentum parameter of the gradient descent method to 0.7, as this was the value chosen in the original study. The original study, however, chose learning rates of .2 and .02, and as mentioned above we were forced into using .01 and .001 in order to properly run the gradient descent method. Because of this change, we examined the resilient back-propagation method with all four of these learning rates in order to compare and contrast within our study as well as theirs.

Tables 1-5 were simulated using the gradient descent with momentum method. These tables mostly coincide with the tables in the original study, sans the tables examining the scaled conjugate gradient method which we did not include. Table 1 shows the training results for different neuron sizes. Table 2 shows the same results with the same parameters yet a different learning rate. Table 3-5 shows the effect of different iteration limits for different neuron numbers.

In Table 6 we attempt to compare our two methods using the same parameters for each. The resilient back-propagation method was found to outperform the gradient descent method in the original study, while in ours it showed to be more consistent as gdm had widely variable hit rate, depending on the learning rate given. In Tables 7 and 8 we compare the different hidden layer activation (transfer) functions, purelin and tansig. We observed that tansig consistently outperformed purelin for our data, as tansig suppresses the output value by bounding weights between -1 and 1, while linear performs no operation.

## Future Work

We had many constraints in this project, mostly time and processing power, and also difficulty of finding a proper neural network implementation in Python. We would have also enjoyed to examine the network with a larger set of training methods, which themselves would introduce more learning parameters as well. Our results were consistent with the original study, but with much less statistical accuracy due to our smaller amount of training epochs, so we would wish to reproduce this project with a higher amount of iterations. The problem itself is fairly simple and is useful for proof of neural network accuracy in classification problems, so further work may investigate other neural network training methods other than back-propagation.

## References

Dişken, Gökay. "Predicting Poker Hand’s Strength Wih Artificial Neural Networks." Adana Bilim Ve Teknoloji Üniversitesi (2014).

UCI Machine Learning Repository: Poker Hand Data Set. (2016). Archive.ics.uci.edu. Retrieved 10 March 2016, from https://archive.ics.uci.edu/ml/datasets/Poker+Hand

## Contributors

Hunter Hammond

Jacob Newberry

Chris Watanabe
