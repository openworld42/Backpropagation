<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Oxygen480-actions-office-chart-polar-stacked.svg/128px-Oxygen480-actions-office-chart-polar-stacked.svg.png" 
alt="Backpropagation" align="right" style="right:40px; top:18px; width:128px; border:none;" />

<br />

# Backpropagation

<h3>Neural backpropagation (Java) with examples and training.<h3>

[![Maintenance Status](https://badgen.net/badge/maintenance/active/green)](https://github.com/openworld42/JavaUtil#maintenance-status)
![dependencies](https://img.shields.io/badge/dependencies-none-orange)
[![License](https://badgen.net/badge/issue/active/blue)](https://github.com/openworld42/Backpropagation/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com) 

<br />

You can find java test/example programs in the [test directory][tester_url] on Github.

:construction_worker_man: &nbsp; **TesterSimpleNumbers.java** is the most simple example, training a one-hidden-layer backpropagation network to approximate a result (number) from to input numbers.

:construction_worker_woman: &nbsp; **TesterXOR.java** is a classic problem in artificial neural network research that highlights the major differences between a single-layer perceptron and one that has a few more layers. The XOR function is not linearly separable, meaning that a single-layer perceptron cannot converge on it. However, a perceptron with one hidden layer can accurately classify the XOR inputs. 

:mechanic: &nbsp; **How to use it:**
Download the newest Github release **backpropagation_vx.x.x.jar** file . Write your own neuronal network application or start
with one of the [test program examples][tester_url], referencing the jar like

**java -cp backpropagation_vx.x.x.jar test/TesterSimpleNumbers**

where **x.x.x** is the current version. You need a Java runtime/JDK installed (at least version 17 - check on command line using **java -version**).
To get it: **Linux**: simply use your package manager, **Windows/macOS/others**: download and install JDK from [here](https://openjdk.java.net/).<br /> 
You may also build **Backpropagation** from scratch using **Ant** and the **build.xml** file.<br />

#### Some hints for best practices, using literature:

> [!NOTE]
> Hints can help to solve a problem, but an arbitrary problem may need another (special) treatment

- depending on your problem to be solved, use one or two hidden layers (three are rare, 
	four usually do not have advantages over three).<br />
	
- one hidden layer (or the first of the hidden layers) should have some more neurons
	than the input layer (around 10%, depending on the problem). 
	With many training data sets the number of neurons 
	may be increased - to store more "information".<br />
	
- too many neurons in hidden layers: the model does not "learn", 
	it just "stores" the input as weights and increases
	the amount of computation (time, energy). Therefore an
	unknown new data to the model will not be recognized
	(the model does not "realize" the principle).<br />
	
- not enough neurons: the model is to small to learn "all the data".<br />

- sometimes it makes sense to start with a higher 
	learning rate and reduce it after a while of training.<br />
	
- learning rates are usually in the range of 0.01 to 0.9<br />

- high learning rates may miss an optimum or oscillate over it.<br />

- low learning rates may increase computation time a lot, 
	especially in "flat" areas of optimization and/or get stuck in local minima.<br />
	
The weights (and biases) are initialized in a random way within this model,
to avoid some initialization traps (the model uses "symmetry breaking").<br />

<br />

**Apache 2.0 licensed**, therefore may be used in any other project/program. 

**Credits, Kudos and Attribution:** 
 * David Kriesel
 * Raúl Rojas
 * Logo: link to [commons.wikimedia.org](https://commons.wikimedia.org/wiki/File:Oxygen480-actions-office-chart-polar-stacked.svg), [The Oxygen Team](https://github.com/KDE/oxygen-icons5/blob/master/AUTHORS) (License: LGPL) 

:book: &nbsp; **References, algorithms and literature (see also &nbsp; :coffee: &nbsp; [Javadoc][javadoc_url] overview for more clues):** 

 * David Kriesel, 2007, A Brief Introduction to Neural Networks, [www.dkriesel.com](https://www.dkriesel.com/en/science/neural_networks)
 * R. Rojas, Neural Networks, Springer-Verlag, Berlin

Contributions, examples (or a request :slightly_smiling_face:) from any interested party are welcome - please open an issue with a short description.

<!-- Repository -->

[javadoc_url]: https://raw.githack.com/openworld42/Backpropagation/master/javadoc/index.html
[tester_url]: https://github.com/openworld42/Backpropagation/tree/master/src/test
