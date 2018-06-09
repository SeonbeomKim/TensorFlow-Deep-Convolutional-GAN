#https://arxiv.org/abs/1511.06434

import tensorflow as tf #version 1.4
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt


tensorboard_path = './Conditional-tensorboard/'
saver_path = './Conditional-saver/'
make_image_path = './Conditional-generate/'

batch_size = 128


class GAN:

	def __init__(self, sess):
		self.channel = 1 #mnist는 흑백
		self.height = 64 #mnist 28*28인데 resize 해줘서 64*64임.
		self.width = 64
		self.noise_dim = 100 # 노이즈 차원
		self.train_rate = 0.0002
		self.target_dim = 10

		with tf.name_scope("placeholder"):
			#class 밖에서 모델 실행시킬때 학습데이터 넣어주는곳.
			self.X = tf.placeholder(tf.float32, [None, self.height, self.width, self.channel])
			#class 밖에서 모델 실행시킬때 class의 Generate_noise 실행한 결과를 넣어주는 곳.
			self.noise_source = tf.placeholder(tf.float32, [None, 1, 1, self.noise_dim])
			#batch_norm
			self.is_train = tf.placeholder(tf.bool)
			#batch_target
			self.Y = tf.placeholder(tf.float32, [None, 1, 1, self.target_dim]) # batch, 1, 1, 10
			self.reshaped_Y = tf.placeholder(tf.float32, [None, self.height, self.width, self.target_dim]) # batch, 64, 64, 10


		

		#노이즈로 데이터 생성. 
		with tf.name_scope("generate_image_from_noise"):
			self.Gen = self.Generator(self.noise_source, self.Y) #batch_size, self.height, self.width, self.channel



		#Discriminator가 진짜라고 생각하는 확률		
		with tf.name_scope("result_from_Discriminator"):
			#학습데이터가 진짜일 확률
			self.D_X, self.D_X_logits = self.Discriminator(self.X, self.reshaped_Y)  #batch_size, 1, 1, 1
			#노이즈로부터 생성된 데이터가 진짜일 확률 
			self.D_Gen, self.D_Gen_logits = self.Discriminator(self.Gen, self.reshaped_Y, True)  #batch_size, 1, 1, 1



		with tf.name_scope("loss"):
			#Discriminator 입장에서 최소화 해야 하는 값
			self.D_loss = self.Discriminator_loss_function(self.D_X_logits, self.D_Gen_logits)
			#Generator 입장에서 최소화 해야 하는 값.
			self.G_loss = self.Generator_loss_function(self.D_Gen_logits)



		#학습 코드
		with tf.name_scope("train"):
			#Batch norm 학습 방법 : https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/layers/batch_normalization
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.train_rate, beta1=0.5) #논문에서 0.5로 해야 안정적이라고 나옴.
					
					#Discriminator와 Generator에서 사용된 variable 분리.
				self.D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')
				self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
					
					#loss 함수 바꿨으므로 maximize가 아닌 minimize 해주는게 의미상 맞음. 즉, -1 안곱해도됨.
				self.D_minimize = self.optimizer.minimize(self.D_loss, var_list=self.D_variables) #G 변수는 고정하고 D로만 학습.
				self.G_minimize = self.optimizer.minimize(self.G_loss, var_list=self.G_variables) #D 변수는 고정하고 G로만 학습.



		#tensorboard
		with tf.name_scope("tensorboard"):
			self.D_loss_tensorboard = tf.placeholder(tf.float32) #Discriminator 입장에서 최소화 해야 하는 값
			self.G_loss_tensorboard = tf.placeholder(tf.float32) #Generator 입장에서 최소화 해야 하는 값.

			self.D_loss_summary = tf.summary.scalar("D_loss", self.D_loss_tensorboard) 
			self.G_loss_summary = tf.summary.scalar("G_loss", self.G_loss_tensorboard) 
			
			self.merged = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter(tensorboard_path, sess.graph)



		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)



		sess.run(tf.global_variables_initializer())



	#노이즈 생성
	def Generate_noise(self, batch_size): #batch_size, 1, 1, self.noise_dim
		return np.random.uniform(-1.0, 1.0, size=[batch_size, 1, 1, self.noise_dim]) #-1~1의 uniform distribution 논문에서 나온대로.



	#데이터의 진짜일 확률
	def Discriminator(self, data, reshaped_Y, reuse=False): #batch_size, 1, 1, 1
		with tf.variable_scope('Discriminator') as scope:
			if reuse == True: #Descriminator 함수 두번 부르는데 두번째 부르는 때에 같은 weight를 사용하려고 함.
				scope.reuse_variables()

			#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/concat
			concat = tf.concat(values=[data, reshaped_Y], axis=3)

			#input layer는 BN 안함.
			conv0 = tf.layers.conv2d(inputs=concat, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 32, 32, 128
			conv0 = tf.nn.leaky_relu(conv0) # default leak is 0.2

			#hidden layer BN 함.
				#1
			conv1 = tf.layers.conv2d(inputs=conv0, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch, 16, 16, 256
			bn1 = tf.layers.batch_normalization(conv1, training=self.is_train)
			conv1 = tf.nn.leaky_relu(bn1)
				#2
			conv2 = tf.layers.conv2d(inputs=conv1, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch 8, 8, 512
			bn2 = tf.layers.batch_normalization(conv2, training=self.is_train)
			conv2 = tf.nn.leaky_relu(bn2)
				#3
			conv3 = tf.layers.conv2d(inputs=conv2, filters=1024, kernel_size=[4, 4], strides=(2, 2), padding='same') #batch 4, 4, 1024
			bn3 = tf.layers.batch_normalization(conv3, training=self.is_train)
			conv3 = tf.nn.leaky_relu(bn3)
				#4 #BN 하면 안됨. 분포를 바꿔버리면 sigmoid의 선형 부분으로만 집중(표현 범위가 매우 좁아짐)될수도 있어서 학습이 진행이 안됨.
			conv4_logits = tf.layers.conv2d(inputs=conv3, filters=self.channel, kernel_size=[4, 4], strides=(1, 1), padding='valid') #batch, 1, 1, 1
			#conv4_logits = tf.layers.batch_normalization(conv4, training=self.is_train)
			conv4_P = tf.nn.sigmoid(conv4_logits)


			return conv4_P, conv4_logits



	#노이즈로 진짜같은 데이터 생성
	def Generator(self, noise, Y): #batch_size, self.height, self.width, self.channel
#if padding == same
	# shape = [batch, input_height*stride_left, input_width*stride_right, filter] 
#if padding == valid
	# if kerner_size > stride_size 인 경우만. 반대의 경우에는 padding == same 인 경우와 같음. 
	# shape = [batch, (input_height*stride_left) + (kerner_size_left-stride_left) , (input_width*stride_right) + (kerner_right-stride_right), filter] 

		#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/concat
		concat = tf.concat(values=[noise, Y], axis=3)

		with tf.variable_scope('Generator'):
			#project and reshape 논문 부분.
			conv0 = tf.layers.conv2d_transpose(inputs=concat, filters=1024, kernel_size=[4, 4], strides=(1, 1), padding='valid') #batch, 4, 4, 1024
			bn0 = tf.layers.batch_normalization(conv0, training=self.is_train)
			conv0 = tf.nn.relu(bn0)

			#4 transpose_convolution
				#1
			conv1 = tf.layers.conv2d_transpose(inputs=conv0, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='same') # batch, 8, 8, 512
			bn1 = tf.layers.batch_normalization(conv1, training=self.is_train)
			conv1 = tf.nn.relu(bn1)
				#2
			conv2 = tf.layers.conv2d_transpose(inputs=conv1, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='same') # batch, 16, 16, 256
			bn2 = tf.layers.batch_normalization(conv2, training=self.is_train) 
			conv2 = tf.nn.relu(bn2)
				#3
			conv3 = tf.layers.conv2d_transpose(inputs=conv2, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='same') # batch, 32, 32, 128
			bn3 = tf.layers.batch_normalization(conv3, training=self.is_train) 
			conv3 = tf.nn.relu(bn3)
				#4 #BN 적용 X
			conv4 = tf.layers.conv2d_transpose(inputs=conv3, filters=self.channel, kernel_size=[4, 4], strides=(2, 2), padding='same') # batch, 64, 64, 1)
			conv4 = tf.nn.tanh(conv4)
			
			return conv4 #생성된 이미지


	
	#Discriminator 학습.
	def Discriminator_loss_function(self, D_X_logits, D_Gen_logits):
		#return tf.reduce_mean(tf.log(D_X) + tf.log(1-D_Gen)) 기존 코드.		
		#위 식이 최대화가 되려면 D_X가 1이 되어야 하며, D_Gen이 0이 되어야 한다.
		#tf.ones_like(X) X와 같은 shape의 1로 이루어진 tensor를 리턴. D_X_logits을 sigmoid 한 결과와 1의 오차.
		D_X_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(D_X_logits), 
					logits=D_X_logits
				)

		D_Gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.zeros_like(D_Gen_logits),
					logits=D_Gen_logits
				)

		#이 두 오차의 합을 최소화 하도록 학습.
		D_loss = tf.reduce_mean(D_X_loss) + tf.reduce_mean(D_Gen_loss)

		return D_loss



	#Generator 입장에서 최소화 해야 하는 값.
	def Generator_loss_function(self, D_Gen_logits):
		#return tf.reduce_mean(tf.log(D_Gen))
		#위 식이 최대화가 되려면 D_Gen이 1이 되어야 함. == 1과의 차이를 최소화 하도록 학습하면 됨.
		G_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(D_Gen_logits), 
					logits=D_Gen_logits
				)

		G_loss = tf.reduce_mean(G_loss)

		return G_loss



def train(model, data, target):
	total_D_loss = 0
	total_G_loss = 0

	iteration = int(np.ceil(len(data)/batch_size))


	for i in range( iteration ):
		#train set. mini-batch
		input_ = data[batch_size * i: batch_size * (i + 1)] # batch, 64, 64, 1
		target_ = target[batch_size * i: batch_size * (i + 1)] #batch, 10 

		#target을 input(batch, 64, 64, 1)과 noise(batch, 1, 1, 10)에 붙여야 하므로 모양 변환이 필요함.
		# 아래 target_ : Generator에서 noise와 concat해서 사용.
		target_ = np.reshape(target_, [-1, 1, 1, model.target_dim]) # => batch, 1, 1, 10
		mult = np.ones([len(target_), model.height, model.width, model.target_dim]) # => batch, 64, 64, 10
		# 아래 reshaped_target_ : Discriminator 에서 input_과 concat해서 사용
		reshaped_target_ = target_ * mult # batch, 64, 64, 10 ==> 이제 input(batch, 64, 64, 1)에 concat 할 수 있게 됨. axis=3


		#노이즈 생성.
		noise = model.Generate_noise(len(input_)) # len(input_) == batch_size, noise = (batch_size, 1, 1, model.noise_dim)
			
		#Discriminator 학습.
		_, D_loss = sess.run([model.D_minimize, model.D_loss], 
					{
						model.X:input_, 
						model.noise_source:noise, 
						model.Y:target_, 
						model.reshaped_Y:reshaped_target_, 
						model.is_train:True
					}
				)

		#Generator 학습. 		#batch_normalization을 하기 때문에 X data도 넣어줘야함.
		_, G_loss = sess.run([model.G_minimize, model.G_loss], 
					{
						model.X:input_, 
						model.noise_source:noise, 
						model.Y:target_, 
						model.reshaped_Y:reshaped_target_, 
						model.is_train:True
					}
				)
		

		#parameter sum
		total_D_loss += D_loss
		total_G_loss += G_loss
	

	return total_D_loss/iteration, total_G_loss/iteration



def write_tensorboard(model, D_loss, G_loss, epoch):
	summary = sess.run(model.merged, 
					{
						model.D_loss_tensorboard:D_loss, 
						model.G_loss_tensorboard:G_loss,
					}
				)

	model.writer.add_summary(summary, epoch)



def gen_image(model, epoch):
	num_generate = 10
	noise = model.Generate_noise(num_generate) # noise = (num_generate, 1, 1, model.noise_dim)
	target_ = np.identity(num_generate) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.identity.html
	target_ = np.reshape(target_, [num_generate, 1, 1, model.target_dim]) # num_generate, 1, 1, 10

	generated = sess.run(model.Gen, 
			{model.noise_source:noise, model.Y:target_, model.is_train:False}) #num_generate, 64*64, 1
	generated = np.reshape(generated, (-1, 64, 64)) #이미지 형태로. #num_generate, 64, 64
		
	fig, axes = plt.subplots(1, num_generate, figsize=(num_generate, 1))

	for i in range(num_generate):
		axes[i].set_axis_off()
		axes[i].imshow(generated[i])
		axes[i].set_title(str(i))

	plt.savefig(make_image_path+str(epoch))
	plt.close(fig)	



def run(model, train_set, target, restore = 0):
	#weight 저장할 폴더 생성
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)
	
	#생성된 이미지 저장할 폴더 생성
	if not os.path.exists(make_image_path):
		os.makedirs(make_image_path)

	#restore인지 체크.
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	
	print('training start')

	#학습 진행
	for epoch in range(restore + 1, 10001):
		#Discriminator 입장에서 최소화 해야 하는 값, #Generator 입장에서 최소화 해야 하는 값
		D_loss, G_loss = train(model, train_set, target)

		print("epoch : ", epoch, " D_loss : ", D_loss, " G_loss : ", G_loss)

		
		if epoch % 1 == 0:
			#tensorboard
			write_tensorboard(model, D_loss, G_loss, epoch)

			#weight save
			#save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")
		
			#image 생성
			gen_image(model, epoch)




sess = tf.Session()

#model
model = GAN(sess) #noise_dim, input_dim

#get mnist data #이미지의 값은 0~1 사이임.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#mnist.train.images = 55000, 784 => train_set = 55000, 28, 28, 1 (batch, height, width, channel)
train_set =  np.reshape(mnist.train.images, ([-1, 28, 28, 1]))
target = mnist.train.labels #55000, 10


#resizing
train_set = sess.run( tf.image.resize_images(train_set, [64, 64]) ) #64,64로 resize


#MNIST는 0~1 값을 갖는데, Generator가 tanh로 생성하니까 -1~1로 정규화 해줘야 함.
train_set -= 0.5 # -0.5 ~ 0.5
train_set /= 0.5 # -1 ~ 1


#run
run(model, train_set, target)

