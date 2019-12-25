---
title: "Build a Neural Network from Scratch Using Numpy"
excerpt: "Derivation of backpropagation algorithm"

categories: 
    - Machine Learning
tags: [Machine Learning, Deep Learning]
---

## Introduction
Although this post is the first one that I'm writing on my newly-built page, I decided to directly start off with a technical post. I just wanted to leave a record of what I understand about some subjects. Like you may know, it feels really bad to find out that you have forgot all the things that you tried really hard to understand and learn a few years ago.. and sometimes it's just because you lost sheets of paper on which you worked all the details.

So, this post (and any coming posts as well) is mainly for myself to be able to remember the details of some algorithm, maths, or papers. However, I do also hope that anyone that happen to come by my blog can benefit from the content, so I'll try to write as understandable as possible.

There are quite a few excellent libraries for deep learning out there, e.g. Tensorflow, PyTorch, etc. And we don't really have to work out all the details of backpropagation algorithm unless we are forced (?) to, like when you are taking a course and it's your assignment! But, believe me, I found this exercise of working out all the notations and derivations of backpropagation algorithm helped me better figure out what's going on at the back of the fancy libraries. 

It is impossible to actually build a neural network module that is as general as those available libraries as a mere exercise that can be written in one blog post. So, the following is my expectation of what you can takeaway when you are done reading this article and following through all the derivation and maths:
* Understand how derivatives are passed from the output layer of a neural network to the inner most layer in Backpropagation algorithm.
* Build a OOP-style neural network module from scratch using the maths developed.
* Play with different activation functions, architectures, and dataset using the built neural network module.
Note that I'm assuming a basic familiarity with neural networks.

That's a long introduction! But, what would you expect for a man who's writing a blog for the first time!

## Some notations
To begin with, we need to define a bunch of notations that we'll use throughout this posting. We'll first look at how to backpropagate an error when there's a single sample (like what you'll do with SGD). Then, we'll build upon this a mini-batch notation which we are going to actually implement. 

![A schematic representation of simple neural net](https://github.com/jihwan-jeong/jihwan-jeong.github.io/blob/master/assets/images/nn_diagram.png)














