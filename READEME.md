# DialectMoE

This is the official repository of our paper: DialectMoE: An End-to-End Multi-Dialect Speech Recognition Model with  Mixture-of-Experts

## Abstract

Dialect speech recognition has always been one of the challenges in Automatic Speech Recognition (ASR) systems. While lots of ASR systems perform well in Mandarin, their performance
significantly drops when handling dialect speech. This is mainly due to the obvious differences
between dialects and Mandarin in pronunciation and the data scarcity of dialect speech. In this
paper, we propose DialectMoE, a Chinese multi-dialects speech recognition model based on
Mixture-of-Experts (MoE) in a low-resource conditions. Specifically, DialectMoE assigns input
sequences to a set of experts using a dynamic routing algorithm, with each expert potentially
trained for a specific dialect. Subsequently, the outputs of these experts are combined to derive
the final output. Due to the similarities among dialects, distinct experts may offer assistance in
recognizing other dialects as well. Experimental results on the Datatang dialect public dataset
show that, compared with the baseline model, DialectMoE reduces Character Error Rate (CER)
for Sichuan, Yunnan, Hubei and Henan dialects by 23.6%, 32.6%, 39.2% and 35.09% respectively. The proposed DialectMoE model demonstrates outstanding performance in multi-dialects
speech recognition.
