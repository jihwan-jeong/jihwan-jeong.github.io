---
title: "Build a Neural Network from Scratch Using Numpy"
excerpt: "Derivation of backpropagation algorithm"

categories: 
    - Machine Learning
tags: [Machine Learning, Deep Learning]
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
You'll see that I've put quite some efforts in putting down nitty-gritty of deriving simple vectorized notations of gradients necessary for updating the weights and biases of a neural network. 

## Work with a Single Input
To begin with, we need to define a bunch of notations that we'll use throughout this posting. We'll first look at how to backpropagate an error **when there's a single sample** (like what you'll do with SGD). Then, we'll build upon this a mini-batch notation which we are going to actually implement. 

![A schematic representation of simple neural net](/assets/images/nn_diagram.png)

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

![Links from hidden units](/assets/images/hidden_units.jpeg)

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
This is one of the most commonly used loss (e.g. in the least-squares method), which is defined as follows:
$$
\begin{align}
E = \frac{1}{2n}\sum_{q=1}^n \lVert t_q - H_{q\cdot}^L \rVert^2 = \frac{1}{2n}\sum_{q=1}^n \sum_{j=1}^{m_L}(t_{qj} - H_{qj}^L)^2
\end{align}\\
$$
where $$t_q\in\mathbb{R}^{1\times m_L}$$ is the $$q$$th target value and $$H^L_{q\cdot}$$ is the corresponding output from the neural net. 

Now, we can simply compute the derivative of $$E$$ w.r.t. $$H^L$$, and get $$\delta$$:

$$
\begin{align}
\frac{\partial E^q}{\partial H_{qj}^L}&=\frac{1}{n}(H_{qj}^L-t_{qj})\\
\delta_{qj}^L &=\frac{\partial E}{\partial H_{qj}^L}\frac{\partial H_{qj}^L}{\partial U_{qj}^L}=\frac{1}{n}(H_{qj}^L-t_{qj})\cdot \phi'(U_{qj}^L)\\
\therefore\quad \delta^L &= \frac{1}{n}(H^L - t)\odot \phi'(U^L)
\end{align}
$$

### Cross-entropy loss
While the MSE loss is normally used in the regression problem, the cross-entropy loss is preferred when working with a classification problem. Since the cross-entropy loss takes probabilities as its inputs, we need to squash output values from the output layer such that $$\sum_j H_{qj}^L =1$$ as well as $$H_{qj}^L\in [0, 1]$$ for all $$q$$ and $$j$$. Hence, the activation function in the output layer is usually set to the softmax function.

Using the cross-entropy loss along with the softmax output changes one of the fundamental assumptions that we made when deriving the gradients, so I'll handle the case in the following section. For now, I'll just present the definition of the cross-entropy loss below:

$$
\begin{align}
E &= \sum_{q=1}^n E^q\\
\text{where}\quad E^q &= -\sum_{j=1}^{m_L}t_{qj}\log(H_{qj}^L)
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
As mentioned, let's work out the derivative of the cross-entropy loss function with the softmax output. Firstly, the softmax function is defined as follows:

$$\begin{align}
\phi_{qj} &= \frac{\exp (U_{qj}^L)}{\sum_k \exp (U^L_{qk})}
\end{align}
$$

What changes when using the softmax output is the $$\delta$$ because now we can't write $$\frac{\partial E}{\partial U^L_{qj}}=\frac{\partial E}{\partial H^L_{qj}}\frac{\partial H^L_{qj}}{\partial U_{qj}^L}$$. Instead, the input of the $$j$$th output node $$U_{qj}^L$$ affects $$E$$ through all the other nodes, which should be taken into account in the chain rule.

$$\begin{align}
\frac{\partial H_{qc}^L}{\partial U_{qj}^L}&=
\begin{cases} 
H_{qj}^L (1-H_{qj}^L)\quad\text{if}\quad c=j,\\
-H_{qc}^L\cdot H_{qj}^L\quad\quad\text{otherwise}
 \end{cases}\\
&= H_{qc}^L(1_{c=j}-H_{qj}^L)
\end{align}
$$

where $$1_{c=j}$$ is the indicator function (or direc-delta function) that gives 1 when $$c=j$$, and $$0$$ otherwise.

The derivative of the cross-entropy loss w.r.t. the $qj$th output is $$\frac{\partial E}{\partial H_{qj}^L}=-\frac{t_{qj}}{H_{qj}^L}$$; hence, the delta becomes

$$\begin{align}
\delta_{qj}^L = \frac{\partial E}{\partial U_{qj}^L}&=\sum_c \frac{\partial E}{\partial H_{qc}^L}\frac{\partial H_{qc}^L}{\partial U_{qj}^L}\\
&= \sum_c \frac{\partial E}{\partial H_{qc}^L}H_{qc}^L(1_{c=j}-H_{qj}^L)\\
&= \frac{\partial E^q}{\partial H_{qj}^L}H_{qj}^L - \sum_c \frac{\partial E^q}{\partial H_{qc}^L}H_{qc}^L H_{qj}^L\\
&= -\frac{t_{qj}}{H_{qj}^L}H_{qj}^L - \sum_c \bigg( -\frac{t_{qc}}{H_{qc}^L}\bigg)H_{qc}^LH_{qj}^L\\
&= H_{qj}^L\sum_c t_{qc} - t_{qj} = H_{qj}^L - t_{qj}\\
\text{or, }\quad \delta^L &= H^L - t
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
The first class that we need to build is Layer which is an abstract class. All the hidden layers and the output layers of the network we are going to build can be instantiated by classes that inherit this abstract class. Additionally, we will create an activation layer as well cause the layer also has forward and backward propagation.

The first argument to the backward method is delta ($$\delta^{l+1}$$), the error from the next layer. The learning rate is to be used for the gradient descent step. 

In this post, we will only look at fully-connected layers which can be defined as follows:

```python

```