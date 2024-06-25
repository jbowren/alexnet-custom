import tensorflow as tf

class DataLoader:
	def __init__(self, dataset, imgWidth, imgHeight, inWidth, inHeight, imgChannels) -> None:
		self.numClasses = 1000
		self.numChannels = imgChannels
		self.autotune = tf.data.experimental.AUTOTUNE
		self.imgDims = tf.constant((imgWidth, imgHeight), dtype = tf.int32)
		self.inDims = tf.constant((inWidth, inHeight), dtype = tf.int32)

		self.maxStart = tf.constant((imgWidth - inWidth, imgHeight - inHeight), dtype = tf.float32)

		self.dataset = dataset

	@tf.function
	def preprocess(self, X, y):
		X = X - tf.reduce_mean(X, axis = [1, 2, 3], keepdims = True)
		X = X / tf.math.reduce_std(X, axis = [1, 2, 3], keepdims = True)
		label = tf.one_hot(indices = y, depth = self.numClasses)

		return X, label

	@tf.function
	def augment(self, X, y) -> tuple:
		X = tf.image.random_crop(X, size = [tf.shape(X)[0], 227, 227, 3])
		X = tf.image.random_flip_left_right(X)

		X = X - tf.reduce_mean(X, axis = [1, 2, 3], keepdims = True)
		X = X / tf.math.reduce_std(X, axis = [1, 2, 3], keepdims = True)

		return X, y

	def generator(self, augment = False):
		dataset = self.dataset

		if augment:
			dataset = dataset.map(self.augment, num_parallel_calls = self.autotune)

		dataset = dataset.map(self.preprocess, num_parallel_calls = self.autotune)

		dataset = dataset.prefetch(buffer_size = self.autotune)

		return dataset
