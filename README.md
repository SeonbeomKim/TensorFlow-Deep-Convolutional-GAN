# TensorFlow-Deep-Convolutional-GAN

(DCGAN)Deep Convolutional Generative Adversarial Network
(c-DCGAN)Conditional Deep Convolutional Generative Adversarial Network

DCGAN = vanilla GAN + CNN Network + Batch Normalization

DCGAN paper : https://arxiv.org/abs/1511.06434  
Batch Normalization paper : https://arxiv.org/abs/1502.03167  
Conditional-GAN paper : https://arxiv.org/pdf/1411.1784.pdf

## Deep Convolutional GAN
    * DCGAN-mnist.py
        * dataset : MNIST
      
    * Conditional-DCGAN-mnist.py
        * DCGAN 모델을 이용하여 원하는 target의 이미지 생성
        * dataset : MNIST
        

## DCGAN-mnist.py result (after 12 epoch)
![DCGAN-mnist.py](./generate/12.png)

## Conditional-DCGAN-mnist.py result (after 2 epoch)
![Conditional-DCGAN-mnist.py](./Conditional-generate/2.png)
