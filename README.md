# TensorFlow-Deep-Convolutional-GAN

(DCGAN)Deep Convolutional Generative Adversarial Network  
(c-DCGAN)Conditional Deep Convolutional Generative Adversarial Network

DCGAN = vanilla GAN + CNN Network + Batch Normalization

DCGAN paper : https://arxiv.org/abs/1511.06434  
Batch Normalization paper : https://arxiv.org/abs/1502.03167  
Conditional-GAN paper : https://arxiv.org/pdf/1411.1784.pdf

## 1.DCGAN.py
    * 원하는 target이 아닌 랜덤한 이미지 생성
    * dataset : MNIST
      
## 2.Conditional-DCGAN.py
    * DCGAN 모델을 이용하여 원하는 target의 이미지 생성
    * dataset : MNIST
        

## DCGAN MNIST result (after 12 epochs of training)
![DCGAN.py](./generate/12.png)

## Conditional-DCGAN MNIST result (after 2 epochs of training)
![Conditional-DCGAN.py](./Conditional-generate/2.png)
