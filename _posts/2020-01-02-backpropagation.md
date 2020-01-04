---
title: "Build a Neural Network from Scratch Using Numpy"
excerpt: "Derivation of backpropagation algorithm"

categories: 
    - Machine Learning
tags: [Machine Learning, Deep Learning]
mathjax: true
toc: true
toc_label: Tabel of Contents
classes: wide
---

## Introduction
Although this is the first post that I'm writing on my newly built page, I decided to directly start off with a technical one. I just wanted to leave a record of what I understand about some subjects. It feels really frustrating to find out that you remember nothing about something that you tried really hard to understand.. and sometimes sheets of paper on which you worked all the details are sufficient to retrieve the memory. Well,, so this is the digital version of the notes.

My page is mainly for leaving records of subjects related to details of some algorithms, maths, or papers. However, I do also hope that anyone who happen to come by my blog can benefit from the content, so I'll try to write as understandable as possible.

## Building neural network module from scratch
There are quite a few excellent deep learning libraries out there, e.g. Tensorflow, PyTorch, etc. And we don't really have to work out all the details of backpropagation algorithm unless we are forced (?) to, like when you are taking a course and it's your assignment! But, believe me, I found this exercise of working out all the notations and derivations of backpropagation algorithm helped me better figure out what's going on at the back of the fancy libraries. ~~But it's something that you don't want to do over and over again, which is the reason why I have to leave this as a post..~~ 

I'd say it's almost impossible to actually build a neural network module that is as general as the libraries mentioned above in this short exercise. Instead, the following is my expectation of what are the takeaways when you are done reading this article and following through all the derivation and maths:
* Understand how derivatives are passed from the output layer of a neural network to the inner most layer in Backpropagation algorithm.
* Build an OOP-style neural network module from scratch using the maths developed.
* Play with different activation functions, architectures, and dataset using the built neural network module.

(Note that I'm assuming a basic familiarity with neural networks.)<br>
You'll see that I've put quite some efforts in putting down nitty-gritty of deriving simple vectorized notations of gradients necessary for updating the weights and biases of a neural network. If you want to skip all the details and only see the implementation, go to [this section](https://jihwan-jeong.github.io/machine%20learning/backpropagation/#implementation). Also, the code is available from [my github repo](https://github.com/jihwan-jeong/neuralnet-using-numpy), so you'll want to check this out as well!

## Work with a Single Input
To begin with, we need to define a bunch of notations that we'll use throughout this posting. We'll first look at how to backpropagate an error **when there's a single sample** (like what you'll do with SGD). Then, we'll build upon this a mini-batch notation which we are going to actually implement. 

![A schematic representation of simple neural net](/assets/images/backpropagation/nn_diagram.png)

Look at the diagram above. We denote
* input sample: $$\mathbf{x}\in\mathbb{R}^{1\times m_0}$$
* weights of $$l$$th layer that will be multiplied by the output from $$l-1$$th layer: $$W^l\in\mathbb{R}^{m_{l-1}\times m_l}$$
* bias of $$l$$th layer: $$b^l\in\mathbb{R}^{1\times m_l}$$
* non-linear activation function (element-wise): $$\phi(\cdot)$$
* the input to the $$l$$th layer: $$u^l=h^{l-1}W^l+b^l\in\mathbb{R}^{1\times m_l}$$
* the output of $$l$$th layer: $$h^l=\phi(u^l)\in\mathbb{R}^{1\times m_{l}}$$
* $$L$$ is the total number of layers
* $$t$$ is the target vector
* $$E$$ is the loss function: $$E: \mathbb{R}^{m_L\times m_L}\rightarrow \mathbb{R}$$
* $$\mathbf{y}\in\mathbb{R}^{m_L}$$ is the output of the forward propagation
* $$\eta$$ is the learning rate in the gradient descent

Note that $$h^0 = \mathbf{x}$$ and $$h^L = \mathbf{y}$$ by construction.

When we implement the network using Numpy in Python, it is way faster and simple to work with the matrix (or vectorized) notation. However, I find it always helps my understanding to build a summation based notation first and then come up with the corresponding vectorized version.

### Forward propagation
Transforming an input to an output by feeding the input to the network is pretty straightforward. $$\mathbf{x}$$ will become $$h^1$$ which will become $$h^2$$, ..., and finally $$h^L=\mathbf{y}$$ will be the output of the network. Then, the loss function will be computed by comparing $$h^L$$ with the target $$t$$ (we are doing supervised learning!). I will introduce two popular loss functions (the MSE loss and the cross entropy loss) later, but for now, it suffices to just denote it as $$E$$.

### Backward propagation
Maybe the hardest part starts from here. Although the back-propagation algorithm is nothing but a repetitive application of the chain rule, the fact that we have multiple layers seems to complicate derivation. 

Remember that we are back-propagating the derivatives so as to update our weights $$W^l$$ of the neural network. As for the actual update, we use the gradient descent method, which can be stated as

 $$W^l\leftarrow W^l + \eta \frac{\partial E}{\partial W^l} $$

The last term $$\frac{\partial E}{\partial W^l}$$ is the part where our powerful back-propagation algorithm gives us the solution. Now, let's look at equation by equation to understand how we should go about deriving the derivative w.r.t. each weight!
(Note that I consulted chapter 4 of [Tom Mitchell's book: 'Machine Learning' (1996)](http://profsite.um.ac.ir/~monsefi/machine-learning/pdf/Machine-Learning-Tom-Mitchell.pdf).)

As done in the reference above, it helps to look separately at the output nodes and the hidden nodes.

__Output nodes__

In the output nodes, we get the error signal directly from the loss function, $$E$$.

$$\begin{align}
    \frac{\partial E}{\partial W_{ij}^L} &= \frac{\partial E}{\partial u_j^L} \cdot \frac{\partial u_j^L}{\partial W_{ij}^L}\qquad(\text{where}\quad \frac{\partial E}{\partial u_j^L} = \frac{\partial E}{\partial h^L_j} \cdot \frac{\partial h^L_j}{\partial u_j^L})\\
\Rightarrow \frac{\partial E}{\partial W_{ij}^L} &= \frac{\partial E}{\partial h^L_j}\cdot \frac{\partial h^L_j}{\partial u_j^L}\cdot \frac{\partial u_j^L}{\partial W_{ij}^L}\\
\text{Similarly,}\quad \frac{\partial E}{\partial b_j^L} &= \frac{\partial E}{\partial h_j^L}\cdot \frac{\partial h_j^L}{\partial u_j^L}\cdot \frac{\partial u_j^L}{\partial b_j^L}
\end{align}$$

The first equality holds when the $$j$$th output node affects the loss only through $$u_j^L$$. Also, the relation in the parentheses holds when $$h_j^L$$ is determined by $$u_j^L$$ alone, which is usually the case except for the softmax activation. Two things you need to see here is
1. The $${ij}$$th weight $$W_{ij}^L$$ affects the loss through $$u_j^L$$
2. $$\frac{\partial u_j^L}{\partial W_{ij}^L}$$ and $$\frac{\partial u_j^L}{\partial b_j^L}$$ reduce to:

$$\begin{align}
	\frac{\partial u_j^L}{\partial W_{ij}^L} &= \frac{\partial (\sum_k h_k^{L-1}\cdot W_{kj}^L+b_j^L)}{\partial W_{ij}^L} = h_i^{L-1}\\
\frac{\partial u_j^L }{\partial b_j^L} &= \frac{\partial (\sum_k h_k^{L-1}\cdot W_{kj}^L+b_j^L)}{\partial b_j^L}=1
\end{align}$$

Hence, we have $$\frac{\partial E}{\partial W_{ij}^L}=\frac{\partial E}{\partial h_j^L}\frac{\partial h_j^L}{\partial u_j^L}h_i^{L-1}=\delta_j^L h_i^{L-1}$$ and $$\frac{\partial E}{\partial b_j^L}=\frac{\partial E}{\partial h_j^L}\frac{\partial h_j^L}{\partial u_j^L}=\delta_j^L$$ where we've defined a crucial term $$\delta$$ as follows:

$$\delta_j^l = \frac{\partial E}{\partial u_j^l}$$

That is, the $$j$$th term of $$\delta^l$$ in the $$l$$th layer is the derivative of $$E$$ w.r.t. the $$j$$th input to the layer ($$u_j^l$$). We will see that this plays a key role as it is the one that's going to be passed down to the hidden layers.  

__Hidden units__

In the hidden units, we still want to compute the gradient of the loss w.r.t. each weight $$W_{ij}^l$$ and bias $$b_j^l$$. However, unlike in the output nodes, now $$W_{ij}^l$$ can affect $$E$$ through multiple paths. It's helpful to look at the diagram below.

![Links from hidden units](/assets/images/backpropagation/hidden_units.jpeg)

As can be seen, we need to consider the downstream nodes of $$j$$th node in the $$l$$th layer to correctly compute $$\frac{\partial E}{\partial W_{ij}^l}$$. However, it is still the case that $$E$$ depends on $$W_{ij}^l$$ only through $$u_j^l$$ (except for the softmax activation at the output layer); hence, $$\frac{\partial E}{\partial W_{ij}^l}=\frac{\partial E}{\partial u_j^l}\cdot \frac{\partial u_j^l}{\partial W_{ij}^l}$$ holds. 

Now, let's look at the $$\delta$$:

$$\begin{align}
	\delta_j^l = \frac{\partial E}{\partial u_j^l}&= \sum_{d\in\mathcal{D}_j} \frac{\partial E}{\partial u_d^{l+1}}\frac{\partial u_d^{l+1}}{\partial u_j^l}=\sum_{d\in\mathcal{D}_j} \delta_d^{l+1} \cdot \frac{\partial u_d^{l+1}}{\partial u_j^l}\\
	&= \sum_{d\in\mathcal{D}_j} \delta_d^{l+1}\frac{\partial u_d^{l+1}}{\partial h_j^l}\cdot \frac{\partial h_j^l}{\partial u_j^l}=\sum_{d\in\mathcal{D}_j} \delta_d^{l+1} W_{jd}^{l+1}\cdot \frac{\partial h_j^l}{\partial u_j^l}
\end{align}$$

where the relation $$\frac{\partial u_d^{l+1}}{\partial h_j^l}=\frac{\partial }{\partial h_j^l}\big( \sum_k h_k^l W_{kd}^{l+1}+b_d^{l+1} \big) = W_{jd}^{l+1}$$ is used, and $$\mathcal{D}_j$$ is the set of downstream nodes of the $$j$$th node.

As a result,

$$\begin{align}
	\frac{\partial E}{\partial W_{ij}^l}&=\delta_j^l\cdot\frac{\partial u_j^l}{\partial W_{ij}^l}=\bigg[ \sum_d \delta_d^{l+1} W_{jd}^{l+1}\bigg]\frac{\partial h_j^l}{\partial u_j^l}\cdot \frac{\partial u_j^l}{\partial W_{ij}^l}=\bigg[ \sum_d \delta_d^{l+1} W_{jd}^{l+1}\bigg]\frac{\partial h_j^l}{\partial u_j^l} h_i^{l-1}\\
\text{Similarly,}\quad \frac{\partial E}{\partial b_j^l}&=\frac{\partial E}{\partial u_j^l}\frac{\partial u_j^l}{\partial b_j^l}=\bigg[ \sum_d \delta_d^{l+1}W_{jd}^{l+1}\bigg]\frac{\partial h_j^l}{\partial u_j^l}\frac{\partial u_j^l}{\partial b_j^l}=\bigg[ \sum_d \delta_d^{l+1}W_{jd}^{l+1}\bigg]\frac{\partial h_j^l}{\partial u_j^l}
\end{align}$$

where $$\frac{\partial u_j^l}{\partial W_{ij}^l}=h_i^{l-1}$$ and $$\frac{\partial u_j}{\partial b_j^l}=1$$.

To sum up this part, we can see that we have derived expressions for the gradient of the loss with respect to every weight and bias beginning from the output layer all the way down to the first layer. Clearly, $$\delta^l$$ receives the gradient information from the layer $$l+1$$ through $$\delta^{l+1}$$, $$W_{jd}^{l+1}$$, and the activation. 

__Vectorized Notation__

As mentioned above, it is crucial to be able to write all the expressions derived so far in a vectorized form if we are to implement an efficient neural network module using NumPy. So, let's convert the summations into matrix multiplications and element-wise multiplications. 

- Output nodes

$$
\frac{\partial E}{\partial W^L} = \begin{pmatrix} \frac{\partial E}{\partial W^L_{11}} & \frac{\partial E}{\partial W^L_{12}} &\cdots \\
\frac{\partial E}{\partial W^L_{21}} & \frac{\partial E}{\partial W^L_{22}} & \cdots \\
\vdots & \vdots & \cdots \end{pmatrix} = \begin{pmatrix}
h_1^{L-1}\frac{\partial E}{\partial h^L_1} \frac{\partial h^L_1}{\partial u_1^L} & h_1^{L-1}\frac{\partial E}{\partial h^L_2}\frac{\partial h^L_2}{\partial u_2^L} & \cdots \\
h_2^{L-1}\frac{\partial E}{\partial h^L_1} \frac{\partial h^L_1}{\partial u_1^L} & h_2^{L-1}\frac{\partial E}{\partial h^L_2} \frac{\partial h^L_2}{\partial u_2^L} & \cdots \\
\vdots & \vdots & \cdots
\end{pmatrix}
\in\mathbb{R}^{m_{L-1}\times m_L}
$$

$$\begin{align}
&=(h^{L-1})^\top \bigg[\frac{\partial E}{\partial h^L}\odot \phi'(u^L)\bigg]=(h^{L-1})^\top \delta^L\\
\frac{\partial E}{\partial b^L} &= \frac{\partial E}{\partial h^L}\odot \phi'(u^L)=\delta^L
\end{align}$$

where $$\delta^L = \frac{\partial E}{\partial u^L}=\frac{\partial E}{\partial h^L}\odot \phi'(u^L)$$ and $$\phi’(u^L)= \frac{\partial \phi(u^L)}{\partial u^L}$$, with $$\odot$$ denoting element-wise multiplication.

- Hidden nodes

$$
\begin{align}
\frac{\partial E}{\partial W^l} &= (h^{l-1})^\top \bigg[ \delta^{l+1}(W^{l+1})^\top \odot \phi'(u^l) \bigg]\\
\frac{\partial E}{\partial b^l}&=\delta^{l+1}(W^{l+1})^\top \odot \phi'(u^l)
\end{align}
$$

- Delta

$$\delta^l = [\delta^{l+1}W^{l+1}]\odot \phi'(u^l)$$

***
So far, we've only worked with a single input sample, which may be used in the pure SGD implementation. However, we use mini-batch SGD most of the times! And it is a headache when you try to convert single-sample based notations to multiple-sample ones for the first time. I hope the below walks you through the details :)

## Work with Multiple Inputs
Now, we need to introduce uppercase notations for multiple inputs. 
- The number of input samples: $$n$$
- Input data matrix: $$X\in\mathbb{R}^{n\times m_0}$$
- The input to the $$l$$th layer: $$U^l = H^{l-1}W^l + \boldsymbol{1}b^l\in\mathbb{R}^{n\times m_l}$$ where $$\boldsymbol{1}\in\mathbb{R}^{n\times 1}$$ is a column vector of ones
- The output of the $$l$$th layer: $$H^l = \phi(U^l)\in\mathbb{R}^{n\times m_l}$$

Crucially, we normally have the loss function that can be factorized over each sample, that is, $$E = \sum_{q=1}^{n} E^q$$ where $$E^q$$ is the loss generated from the $$q$$th sample. So, we just need to modify previously developed notations while paying careful attention to the dimension of each matrix or vector. Let's begin with the output layer.

__Output layer__


$$
\begin{align}
\frac{\partial E}{\partial W^L}=\sum_q \frac{\partial E^q}{\partial W^L} &= 
\begin{bmatrix}
H_{11}^{L-1}\frac{\partial E^1}{\partial H_{11}^L}\frac{\partial H_{11}^L}{\partial U_{11}^L} & H_{11}^{L-1}\frac{\partial E^1}{\partial H_{12}^L}\frac{\partial H^L_{12}}{\partial U^L_{12}} & \cdots \\
H_{12}^{L-1}\frac{\partial E^1}{\partial H_{11}^L}\frac{\partial H^L_{11}}{\partial U_{11}^L} & H_{12}^{L-1} \frac{\partial E^1}{\partial H_{12}^L}\frac{\partial H_{12}^L}{\partial U_{12}^L} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix} + 

\begin{bmatrix}
H_{21}^{L-1}\frac{\partial E^2}{\partial H_{21}^L}\frac{\partial H_{21}^L}{\partial U_{21}^L} & H_{21}^{L-1}\frac{\partial E^2}{\partial H_{22}^L}\frac{\partial H^L_{22}}{\partial U^L_{22}} & \cdots \\
H_{22}^{L-1}\frac{\partial E^2}{\partial H_{21}^L}\frac{\partial H^L_{21}}{\partial U_{21}^L} & H_{22}^{L-1} \frac{\partial E^2}{\partial H_{22}^L}\frac{\partial H_{22}^L}{\partial U_{22}^L} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix} + \cdots \\

&= 
\begin{bmatrix}
\sum_q H_{q1}^{L-1}\frac{\partial E^q}{\partial H_{q1}^L}\frac{\partial H_{q1}^L}{\partial U_{q1}^L} & \sum_q H_{q1}^{L-1}\frac{\partial E^q}{\partial H_{q2}^L}\frac{\partial H^L_{q2}}{\partial U^L_{q2}} & \cdots \\
\sum_q H_{q2}^{L-1}\frac{\partial E^q}{\partial H_{q1}^L}\frac{\partial H^L_{q1}}{\partial U_{q1}^L} & \sum_q H_{q2}^{L-1} \frac{\partial E^q}{\partial H_{q2}^L}\frac{\partial H_{q2}^L}{\partial U_{q2}^L} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix} 
\end{align}
$$

Now, if we look at the $$ij$$ component of the above gradient:

$$
\begin{pmatrix}
\frac{\partial E}{\partial W^L}
\end{pmatrix}_{ij} =
\sum_q H_{qi}^{L-1} \frac{\partial E^q}{\partial H_{qj}^L}\frac{\partial H_{qj}^L}{\partial U_{qj}^L}
=\sum_q H_{qi}^{L-1}\delta_{qj}^L
$$

where we can see that $$\delta_{qj}^L=\frac{\partial E^q}{\partial U_{qj}^L}=\frac{\partial E^q}{\partial H_{qj}^L}\frac{\partial H_{qj}^L}{\partial U_{qj}^L}=\frac{\partial E^q}{\partial H_{qj}^L}\phi'(U_{qj}^L)$$, or in vectorized form, $$\delta_{q\cdot}^L=\frac{\partial E^q}{\partial H_{q\cdot}^L}\odot \phi'(U_{q\cdot}^L)$$, which is a $$(1\times m_L)$$ vector for a single sample $$q$$. Hence, in the matrix form, we get below:

$$\frac{\partial E}{\partial W^L}=(H^{L-1})^\top \delta^L$$

where $$\delta^L\in\mathbb{R}^{n\times m_L}$$. 

Similarly, the $$j$$th term of the gradient w.r.t. the bias is:

$$
\begin{align}
\pmatrix{\frac{\partial E}{\partial b^L}}_{j} &=\sum_q \frac{\partial E^q}{\partial U_{qj}^L}\frac{\partial U_{qj}^L}{\partial b_j^L} =\sum_q \delta_{qj}^L \\
&= (\boldsymbol{1}^\top \delta^L)_j
\end{align}
$$

Hence, $$\frac{\partial E}{\partial b^L} = \boldsymbol{1}^\top \delta^L$$.

__Hidden layer__

Finally, for the hidden layer, we have the following:

$$
\begin{align}
\pmatrix{\frac{\partial E}{\partial W^l}}_{ij}&=\pmatrix{\sum_q \frac{\partial E^q}{\partial W^l}}_{ij} = \sum_q \frac{\partial E^q}{\partial W_{ij}^l}=\sum_q \big[ \sum_d \delta_{qd}^{l+1}W_{jd}^{l+1} \big]\frac{\partial H_{qj}^l}{\partial U_{qj}^l}H_{qi}^{l-1}\\
&=\sum_q H_{qi}^{l-1}\delta_{q\cdot}^{l+1}(W_{j\cdot}^{l+1})^\top \frac{\partial H_{qj}^l}{\partial U_{qj}^l} =\sum_q H_{qi}^{l-1}\big[ \delta^{l+1}(W^{l+1})^\top\big]_{qj} \phi'(U_{qj}^l)\\
&= \bigg[ (H^{l-1})^\top \bigg\{ \phi'(U^l)\odot \big[ \delta^{l+1}(W^{l+1})^\top\big] \bigg\}\bigg]_{ij}\\
\therefore \:\frac{\partial E}{\partial W^l} &= (H^{l-1})^\top\bigg\{\ \phi'(U^l)\odot \big[\delta^{l+1}(W^{l+1})^\top \big]\bigg\} \\
\text{and for the bias, } \frac{\partial E}{\partial b^l} &= \boldsymbol{1}^\top \delta^l
\end{align}
$$

As in the single sample case, we need to express $$\delta^l$$ with $$\delta^{l+1}$$ so that the gradient from the output layer can propagate back to hidden layers. 

$$
\begin{align}
\text{In the output layer, }
\delta_{q\cdot}^L &= \frac{\partial E^q}{\partial H_{q\cdot}^L}\odot\phi'(U_{q\cdot}^L)\\
\text{in the hidden layers, } \delta^l_{qj}&=\frac{\partial E^q}{\partial U_{qj}^l}=\sum_{d\in\mathcal{D}_j}\frac{\partial E^q}{\partial U_{qd}^{l+1}}\frac{\partial U^{l+1}_{qd}}{\partial U_{qj}^l}=\sum_{d\in\mathcal{D}_{j}}\frac{\partial E^q}{\partial U_{qd}^{l+1}}W_{jd}^{l+1}\phi'(U_{qj}^l)\\
&=\big[\big\{ \delta^{l+1}(W^{l+1})^\top\big\}\odot \phi'(U^l)\big]_{qj}\\
\text{or, }\:\delta^l &= \big\{\delta^{l+1}(W^{l+1})^\top\big\}\odot\phi'(U^l)
\end{align}
$$

------
Now that we have the vectorized equations of gradients, we are almost there to actually implement a working neural network module. The remaining part is to derive the derivative of a loss function w.r.t. the output as well as the derivative of the activation function ($$\phi(U^l)$$) w.r.t. its input ($$U^l$$).

## Loss functions
Although you may work with a number of different loss functions, we will restrict our derivation to the mean-squared error (MSE) loss and the cross entropy loss.

### MSE loss
This is one of the most commonly used losses (e.g. in the least-squares method), which is defined as follows:
$$
\begin{align}
E = \frac{1}{2n}\sum_{q=1}^n \lVert t_q - y_q \rVert^2 = \frac{1}{2n}\sum_{q=1}^n \sum_{j=1}^{m_L}(t_{qj} - y_{qj})^2
\end{align}\\
$$
where $$t_q\in\mathbb{R}^{1\times m_L}$$ is the $$q$$th target value and $$y^L_q$$ is the corresponding output from the neural net. 

Now, we can simply compute the derivative of $$E$$ w.r.t. $$y$$, and get $$\delta$$:

$$
\begin{align}
\frac{\partial E^q}{\partial y_{qj}}&=\frac{1}{n}(y_{qj}-t_{qj})\\
\delta_{qj}^L &=\frac{\partial E}{\partial y_{qj}}\frac{\partial y_{qj}}{\partial U_{qj}^L}=\frac{1}{n}(y_{qj}-t_{qj})\cdot \phi'(U_{qj}^L)\\
\therefore\quad \delta^L &= \frac{1}{n}(y - t)\odot \phi'(U^L)
\end{align}
$$

Normally in a regression problem, we want the outputs to have unbounded range, so we can simply use the linear output without an activation (i.e., $$y = U^L$$). In a classification problem, however, we want the outputs to be within 0 and 1, so the softmax activation function is applied to the linear output (i.e., $$y = \phi(U^L)$$).

### Cross-entropy loss
While the MSE loss is normally used in the regression problem, the cross-entropy loss is preferred when working with a classification problem. Since the cross-entropy loss takes probabilities as its inputs, we need to squash output values from the output layer such that $$\sum_j H_{qj}^L =1$$ as well as $$H_{qj}^L\in [0, 1]$$ for all $$q$$ and $$j$$. Hence, the activation function in the output layer is set to the softmax function.

Using the cross-entropy loss along with the softmax output changes one of the fundamental assumptions that we made when deriving the gradients, so I'll handle the case in the following section. For now, I'll just present the definition of the cross-entropy loss below:

$$
\begin{align}
E &= \sum_{q=1}^n E^q\\
\text{where}\quad E^q &= -\sum_{j=1}^{m_L}t_{qj}\log(y_{qj})
\end{align}
$$

## Activation functions ($$\phi$$)
An activation function plays an indispensable role in a neural network since it is the only part where nonlinearity is introduced. I'll list some commonly used activation functions along with their derivatives w.r.t. inputs.

### ReLU
__Definition__:   $$
\phi(U) = \max (0, U)
$$

__Derivative__:   $$\phi'(U) = \begin{cases} 
1,\qquad\text{if }U>0,\\
0, \qquad\text{otherwise} \end{cases}
$$

where $$\phi$$ is applied element-wisely.
### Sigmoid function
__Definition__:   $$\phi(U) = \frac{1}{1+\exp(-U)}$$

__Derivative__:   $$\phi'(U) = \phi(U)\big( 1-\phi(U) \big)$$

where $$\phi$$ is applied element-wisely.

### Softmax function
In this part, let's work out the derivative of the cross-entropy loss function with the softmax output. Firstly, the softmax function is defined as follows:

$$\begin{align}
y_{qj} = \phi_{qj} &= \frac{\exp (U_{qj}^L)}{\sum_k \exp (U^L_{qk})}
\end{align}
$$

What changes when using the softmax output is the $$\delta$$ because now we can't write $$\frac{\partial E}{\partial U^L_{qj}}=\frac{\partial E}{\partial y_{qj}}\frac{\partial y_{qj}}{\partial U_{qj}^L}$$. Instead, the input of the $$j$$th output node $$U_{qj}^L$$ affects $$E$$ through all the other nodes, which should be taken into account in the chain rule.

$$\begin{align}
\frac{\partial y_{qc}}{\partial U_{qj}^L}&=
\begin{cases} 
y_{qj} (1-y_{qj})\quad\text{if}\quad c=j,\\
-y_{qc}\cdot y_{qj}\quad\quad\text{otherwise}
 \end{cases}\\
&= y_{qc}(1_{c=j}-y_{qj})
\end{align}
$$

where $$1_{c=j}$$ is the indicator function (or direc-delta function) that gives 1 when $$c=j$$, and $$0$$ otherwise.

The derivative of the cross-entropy loss w.r.t. the $qj$th output is $$\frac{\partial E}{\partial y_{qj}}=-\frac{t_{qj}}{y_{qj}}$$; hence, the delta becomes

$$\begin{align}
\delta_{qj}^L = \frac{\partial E}{\partial U_{qj}^L}&=\sum_c \frac{\partial E}{\partial y_{qc}}\frac{\partial y_{qc}}{\partial U_{qj}^L}\\
&= \sum_c \frac{\partial E}{\partial y_{qc}}y_{qc}(1_{c=j}-y_{qj})\\
&= \frac{\partial E^q}{\partial y_{qj}}y_{qj} - \sum_c \frac{\partial E^q}{\partial y_{qc}}y_{qc} y_{qj}\\
&= -\frac{t_{qj}}{y_{qj}}y_{qj} - \sum_c \bigg( -\frac{t_{qc}}{y_{qc}}\bigg)y_{qc}y_{qj}\\
&= y_{qj}\sum_c t_{qc} - t_{qj} = y_{qj} - t_{qj}\\
\text{or, }\quad \delta^L &= y - t
\end{align}
$$

where I used the fact that $$t_{q\cdot}$$ is one-hot encoded (which is the case in a classification problem). 

## Implementation
Finally! Let's make things actually work. In this post, I'll introduce a way to build a neural network as a module with which you can easily construct a customized network. I consulted [this blog](https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65) for the OOP styles I used in the implementation. 

### Layers
```python
import numpy as np
class Layer:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, delta, lr):
        """
        :param delta: error from the next layer
        :param lr: learning rate
        """
        raise NotImplementedError
```
The first class that we need to build is Layer which is an abstract class. All the hidden layers and the output layers of the network we are going to build can be instantiated by classes that inherit this abstract class. Additionally, we will create an activation layer cause the layer also has forward and backward propagation.

The first argument to the backward method is delta ($$\delta^{l+1}$$), the error from the next layer. The learning rate is to be used for the gradient descent step. 

In this post, we will only look at fully-connected layers which can be defined as follows:

```python
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        """
        :param input_size: the number of nodes in the previous layer
        :param output_size: the number of nodes in 'this' layer
        """
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def forward(self, input_data):
        """
        :param input_data: H^{l-1}
        :return: self.output = U^l = H^{l-1} W^l + 1 b^l
        """
        n = input_data.shape[0]             # size of mini-batch
        self.input_data = input_data        # store the input as attribute
        self.output = self.input_data @ self.weights + np.ones((n, 1)) @ self.bias
        return self.output

    def backward(self, delta_n, lr):
        """
        'delta_n' is the delta from the next layer
        'delta' is the delta to be passed to the previous layer

        Equations:
        * delta = delta_n W.T
        * dEdW = (H^{l-1}).T @ delta_n
        :return delta
        """
        delta = delta_n @ self.weights.T
        dEdW = self.input_data.T @ delta_n
        dEdb = np.sum(delta_n, axis=0, keepdims=True)

        # gradient descent step
        self.weights -= lr * dEdW
        self.bias -= lr * dEdb
        return delta
```

Note how the input to a layer is saved when feed forwarding so that it can be used in the backward step. Also, an activation is intentionally omitted because we will have a separate layer for that. In an activation layer, we can simply multiply $$\phi'(U^l)$$ to delta in an element-wise manner and pass it to the previous layer. 

### Activation
```python
class Activation(Layer):
    def __init__(self, activation, der_activation):
        """
        :param activation:      activation function
        :param der_activation:  derivative of the activation function
        """
        self.activation = activation
        self.der_activation = der_activation

    def forward(self, input_data):
        """
        :param input_data: U^l
        :return: self.output = H^l = \phi(U^l)
        """
        self.input_data = input_data
        self.output = self.activation(self.input_data)
        return self.output

    def backward(self, delta_n, lr):
        return self.der_activation(self.input_data) * delta_n
```

Prior to instantiating an activation layer, an activation function and its derivative should be defined as Python functions. Below are a few examples of activations. 

__ReLU__
```python
def relu(U):
    return np.maximum(U, 0)

def relu_prime(U):
    return (U > 0) * 1.0
```

__tanh__
```python
def tanh(U):
    return np.tanh(U)

def tanh_prime(U):
    return 1 - np.power(np.tanh(U), 2)
```
where $$\tanh(U)=\frac{\exp(U)-\exp(-U)}{\exp(U)+\exp(-U)}$$.

__Sigmoid__
```python
def sigmoid(U):
    # Implemented in this way for numerical stability
    return np.where(U >= 0,
                    1 / (1 + np.exp(-U)),
                    np.exp(U) / (1 + np.exp(U)))

def sigmoid_prime(U):
    Phi = sigmoid(U)
    return Phi * (1-Phi)
```

__Softmax__
```python
def softmax(U):
    # Below is a numerically-stable implementation of the softmax function
    exp = np.exp(U - np.max(U, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)
```
Note that the softmax function defined here will not be used in backpropagating errors, but it will be used in testing.

### Loss functions
```python
class Loss:
    def loss(self, pred, target):
        """
        Computes the loss function values by comparing pred and target
        :param pred:    the output of the output layer
        :param target:  the label
        :return:
        """
        raise NotImplementedError

    def diff_loss(self, pred, target):
        """
        Computes the derivative of the loss function (delta) 
        """
        raise NotImplementedError
```

__MSE loss__
```python
class MSELoss(Loss):
    def loss(self, pred, target):
        return np.sum(np.power(pred - target, 2)) / 2

    def diff_loss(self, pred, target):
        return (pred - target) / pred.shape[0]
```

__Cross-entropy loss__

As mentioned, the cross-entropy loss is **always** conjoined with the softmax output, so both are implemented together in one class as follows:
```python
class CrossEntropyLoss(Loss):
    """
    This class combines the cross entropy loss with the softmax outputs as done in the PyTorch implementation.
    The return value of self.diff_loss will then be directly handed over to the output FCLayer
    """
    def loss(self, pred, target):
        pred = self._softmax(pred)
        return - np.trace(target @ np.log(pred+1e-15).T)

    def diff_loss(self, pred, target):
        pred = self._softmax(pred)
        return pred - target

    @staticmethod
    def _softmax(pred):
        # Below is a numerically-stable implementation of the softmax function
        exp = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
```

### Neural network module
Putting it all together, here is the neural network class that you can instantiate and use for an actual use. Toy examples showing how to use the module will be presented in the next section.

```python
class NeuralNetwork:
    def __init__(self):
        self.layers = []        # FCLayer and activation layers will be appended
        self.loss = None        # the loss function
        self.diff_loss = None   # and its derivative
        self.loss_hist = []

    def add(self, layer):
        """
        Either a FCLayer object or an Activation object will be put in as an argument,
        which will be appended to self.layers.
        """
        self.layers.append(layer)

    def set_loss(self, loss, diff_loss):
        # Setting the loss function and its derivative
        self.loss = loss
        self.diff_loss = diff_loss

    def predict(self, input_data):
        """
        :param input_data: (n x m_0) array of input samples
        :return: the output of the output layer
        """
        out = input_data

        # Feed forward
        for l in self.layers:
            out = l.forward(out)
        return out

    def fit(self, x_train, y_train, epochs, lr, batch_size, x_test=None, y_test=None, verbose=False):
        """
        :param x_train:     input data of training set
        :param y_train:     corresponding target values of training set
        :param epochs:      number of training epochs
        :param lr:          learning rate
        :param batch_size:  mini-batch size
        :param x_test:      test images (if given)
        :param y_test:      test labels (if given)
        """

        inds = list(range(x_train.shape[0]))
        N = len(inds)  # number of training samples

        for i in range(epochs):
            # randomly shuffle the training data at the beginning of each epoch
            inds = np.random.permutation(inds)
            x_train = x_train[inds]
            y_train = y_train[inds]

            loss = 0

            for b in range(0, N, batch_size):
                # get the mini-batch
                x_batch = x_train[b: b + batch_size]
                y_batch = y_train[b: b + batch_size]

                # feed forward
                pred = self.predict(x_batch)

                # Error
                loss += self.loss(pred, y_batch) / N

                # Back propagation
                delta = self.diff_loss(pred, y_batch)
                for l in reversed(self.layers):
                    delta = l.backward(delta, lr)
            
            # record loss per epoch
            self.loss_hist.append(loss)
            
            if verbose:
                print()
                print("Epoch %d/%d\terror=%.5f" % (i + 1, epochs, loss), end='\t', flush=True)

            if x_test is not None:
                accuracy = self.test(x_test, y_test)
                print("Test accuracy: {:2}".format(accuracy), end='')

    def test(self, x_test, y_test):
        # test on the test set in classification
        pred = softmax(self.predict(x_test))
        pred_label, y_test_label = np.argmax(pred, axis=1), np.argmax(y_test, axis=1)
        accuracy = np.mean(pred_label == y_test_label)
        return accuracy
```
In the case of classification problem, we can monitor the accuracy on the test set to see how our model is generalizing over unseen data, which is defined in NeuralNetwork.test() method. 

## Toy Examples
### MNIST handwritten digits
Let's use the famous MNIST dataset to see if our neural network model is correctly implemented. The MNIST dataset contains handwritten digits of 0 to 9, and training a model to predict the correct label for an image is a typical classification problem. The dataset can be downloaded [here](https://github.com/jihwan-jeong/jihwan-jeong.github.io/raw/master/assets/data/mnist.pkl.gz) (downloaded via Keras, then pickled by me). The filename is 'mnist.pkl.gz', and you should place the file in the directory where your Data.py file is. The mnist file is a tuple of train and test set which are composed of numpy arrays of images and labels. 

In a separate file named 'Data.py' write below lines.

```python
import pickle, gzip
import numpy as np

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    (train_image, train_label), (test_image, test_label) = pickle.load(f)

    # Flatten images to (num_images, pixels), normalize pixel values to (-0.5, 0.5) range
    train_image = np.vstack([image.reshape(-1, ) for image in train_image])/255 - 0.5
    test_image = np.vstack([image.reshape(-1, ) for image in test_image])/255 - 0.5

    # One-hot encoding
    train_label, test_label = np.eye(10)[train_label], np.eye(10)[test_label]
    return (train_image, train_label), (test_image, test_label)
```

Now, let's define a neural network with the following specs.
- 1 hidden layer with 64 nodes and an output layer
- ReLU activation in the hidden layer
- Softmax output with cross-entropy error loss
- Mini-batch size is set to 16
- Learning rate is 0.005 
- 10 epochs

```python
# Set a random seed for reproducibility
np.random.seed(1)

# load data
(train_image, train_label), (test_image, test_label) = load_mnist()

# Set hyperparameters
lr = 0.005
batch_size = 16
epochs = 10

# Define a model
nn = NeuralNetwork()
nn.add(FCLayer(train_image.shape[1], 64))       # 784 x 64 fully connected layer
nn.add(Activation(relu, relu_prime))            # relu activation
nn.add(FCLayer(64, train_label.shape[1]))       # 64 x 10 fully connected layer
loss = CrossEntropyLoss()                       
nn.set_loss(loss.loss, loss.diff_loss)          # softmax output with cross-entropy loss

# Train the model
nn.fit(train_image, train_label, epochs=epochs, lr=lr, batch_size=batch_size,
       x_test=test_image, y_test=test_label)

"""Result"""
Epoch 1/10	error=1.20417	Test accuracy: 0.807
Epoch 2/10	error=0.59282	Test accuracy: 0.8504
Epoch 3/10	error=0.48660	Test accuracy: 0.8795
Epoch 4/10	error=0.43013	Test accuracy: 0.885
Epoch 5/10	error=0.38977	Test accuracy: 0.9009
Epoch 6/10	error=0.36007	Test accuracy: 0.901
Epoch 7/10	error=0.33777	Test accuracy: 0.9059
Epoch 8/10	error=0.32317	Test accuracy: 0.9078
Epoch 9/10	error=0.30777	Test accuracy: 0.9169
Epoch 10/10	error=0.29622	Test accuracy: 0.9194
```

There you go! We can see that the neural network module is able to achieve 91.9% accuracy on the test set after being trained for 10 epochs. This is not too bad, but we could've done better if we used a better optimizer (other than mere gradient descent) and tuned hyper-parameters. 

### Regression problem
We've seen how to tackle a multi-class classification problem, so now let's look at a regression problem. We are going to use a synthetic regression problem with the dataset created from [here](http://www.cse.chalmers.se/~richajo/dit866/lectures/l9/Nonlinear%20regression%20toy%20example.html). 

```python
def toy_regression():
    np.random.seed(1)
    N = 2000
    X = 0.5 * np.random.normal(size=N) + 0.35

    Xt = 0.75 * X - 0.35
    X = X.reshape((N, 1))

    Y = -(8 * Xt ** 2 + 0.1 * Xt + 0.1) + 0.05 * np.random.normal(size=N)
    Y = np.exp(Y) + 0.05 * np.random.normal(size=N)
    Y /= max(np.abs(Y))
    Y = Y.reshape(-1, 1)
    return X, Y
```

The dataset looks like below.

<div style="text-align: center"><img src="/assets/images/backpropagation/toy_regression.png" width="600"/></div>

This time, let's use the following specs of a network:
- 2 hidden layers with 16 nodes and an output layer
- ReLU activation on the first hidden layer and sigmoid on the second, while no activation on the output layer
- MSE loss
- Mini-batch size is set to 16
- Learning rate is 0.001
- Number of epochs: 300

As a simple experiment, we can see the role of nonlinear activation functions by first trying with no activations. You'll see that the network is then nothing but a linear regression model.

```python
# Set a random seed for reproducibility
np.random.seed(1)

# load data
Xtrain, Ytrain = regression()

# Define a model (without nonlinear activation)
nn = NeuralNetwork()
nn.add(FCLayer(Xtrain.shape[1], 16))            # 1 x 16 fully connected layer
nn.add(FCLayer(16, 16))                         # 16 x 16 fully connected layer
nn.add(FCLayer(16, 1))                          # 16 x 1 fully connected output layer
loss = MSELoss()
nn.set_loss(loss.loss, loss.diff_loss)

# set hyperparameters
lr = 0.001
batch_size = 16
epochs = 300

# Train the model
nn.fit(Xtrain, Ytrain, epochs, lr, batch_size, verbose=False)

# Plot the result
plt.figure(figsize=(10, 8))
plt.plot(Xtrain[:, 0], Ytrain, '.', label='Target', markersize=2.5)
plt.plot(Xtrain[:, 0], nn.predict(Xtrain), 'r.', label='prediction', markersize=2.5)
plt.legend(fontsize=15)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title("Prediction (Linear)", fontsize=20)
plt.show()
```
Below is the result. It is clear that the network cannot capture the nonlinearity of the train data.

<div style="text-align: center"><img src="/assets/images/backpropagation/prediction_linear.png" width="600"/></div>

So, we need to add nonlinear activations!
```python
# Define a model (with activation functions)
nn = NeuralNetwork()
nn.add(FCLayer(Xtrain.shape[1], 16))            # 1 x 16 fully connected layer
nn.add(Activation(relu, relu_prime))            # relu activation
nn.add(FCLayer(16, 16))                         # 16 x 16 fully connected layer
nn.add(Activation(sigmoid, sigmoid_prime))      # sigmoid activation
nn.add(FCLayer(16, 1))                          # 16 x 1 output layer (no activation to the output!)
loss = MSELoss()
nn.set_loss(loss.loss, loss.diff_loss)

# Train the model
nn.fit(Xtrain, Ytrain, epochs, lr, batch_size, verbose=False)

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(Xtrain[:, 0], Ytrain, '.', label='Target', markersize=2.5)
plt.plot(Xtrain[:, 0], nn.predict(Xtrain), 'r.', markersize=2.5, label='prediction')
plt.legend(fontsize=15)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.title("Prediction (Nonlinear)", fontsize=20)
plt.show()
```

We can see that now the model is able to learn the nonlinearity pretty well.

<div style="text-align: center"><img src="/assets/images/backpropagation/prediction_nonlinear.png" width="600"/></div>

Also, we can plot how the loss value is dropping as we train the model for more epochs. 

```python
plt.figure(figsize=(10, 8))
plt.plot(nn.loss_hist)
plt.ylim(0, 0.05)
plt.xlabel('Number of epochs', fontsize=15)
plt.ylabel('MSE loss per epoch', fontsize=15)
plt.title('MSE loss v.s. Epoch', fontsize=15)
plt.show()
```

<div style="text-align: center"><img src="/assets/images/backpropagation/loss.png" width="600"/></div>

The loss drops quickly in the first 5 epochs, then slowly reduces until reaching a plateau. Although training for too many epochs is a bad idea since it will result in overfitting, we are not particularly interested in the overfitting issue in this post. Rather, we were able to check that indeed our neural network module is learning the nonlinear function pretty nicely!

## Conclusion
Now we are at the end of this long journey of understanding and implementing the backpropagation algorithm. I do hope that this will help you (and me) get the idea of how the fancy neural network libraries work behind the scenes and learn the way to build your own module with the derived equations. Here is the link to [my github repo](https://github.com/jihwan-jeong/neuralnet-using-numpy) containing all the above code, so you can clone and play with the code. Please feel free to leave comments if you know how to improve this post or if you have questions!
