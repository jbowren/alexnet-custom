import tensorflow as tf

class AlexNet(tf.keras.Model):
	def __init__(self, imgDim, inShape, batchSize):
		super().__init__()

		self.inShape = inShape

		self.flips = [1, -1]
		edge = imgDim - inShape[1]
		center = int(tf.math.ceil(edge / 2))
		self.valCoords = [[0, 0], [0, edge], [edge, 0], [edge, edge], [center, center]]
		self.valLocCount = len(self.flips) * len(self.valCoords)

		self.conv1 = tf.keras.layers.Conv2D(filters = 96, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (11, 11), strides = (4, 4), activation = 'relu', padding = 'valid', input_shape = self.inShape)
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv2 = tf.keras.layers.Conv2D(filters = 256, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = 'same')
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv3 = tf.keras.layers.Conv2D(filters = 384, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')

		self.conv4 = tf.keras.layers.Conv2D(filters = 384, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')

		self.conv5 = tf.keras.layers.Conv2D(filters = 256, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')
		self.pool5 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.flatten = tf.keras.layers.Flatten()
		
		self.fc1 = tf.keras.layers.Dense(4096, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		
		self.fc2 = tf.keras.layers.Dense(4096, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout2 = tf.keras.layers.Dropout(0.5)
		
		self.fc3 = tf.keras.layers.Dense(1000, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'softmax')

	def call(self, inputs, training = False):
		res = self.conv1(inputs)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = self.pool1(res)

		res = self.conv2(res)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = self.pool2(res)

		res = self.conv3(res)

		res = self.conv4(res)

		res = self.conv5(res)
		res = self.pool5(res)

		res = self.flatten(res)
		
		if training:
			res = self.fc1(res)
			res = self.dropout1(res)
			res = self.fc2(res)
			res = self.dropout2(res)
		else:
			res = self.fc1(res) * 0.5
			res = self.fc2(res) * 0.5

		res = self.fc3(res)

		return res

	def test_step(self, data):
		X, y = data

		yPred = 0

		for flip in self.flips:
			for coords in self.valCoords:
				coordsRowEnd = coords[0] + self.inShape[0]
				coordsColEnd = coords[1] + self.inShape[1]

				region = X[:, coords[0]:coordsRowEnd, coords[1]:coordsColEnd][:, :, ::flip, :]

				region = region - tf.reduce_mean(region, axis = [1, 2, 3], keepdims = True)
				region = region / tf.math.reduce_std(region, axis = [1, 2, 3], keepdims = True)

				yPred += self(region, training = False)

		yPred /= 10

		self.compiled_loss(y, yPred)

		self.compiled_metrics.update_state(y, yPred)

		return {m.name: m.result() for m in self.metrics}

class AlexNetLarge(tf.keras.Model):
	def __init__(self, imgDim, inShape, batchSize):
		super().__init__()

		self.inShape = inShape

		self.flips = [1, -1]
		edge = imgDim - inShape[1]
		center = int(tf.math.ceil(edge / 2))
		self.valCoords = [[0, 0], [0, edge], [edge, 0], [edge, edge], [center, center]]
		self.valLocCount = len(self.flips) * len(self.valCoords)

		self.conv1 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (11, 11), strides = (4, 4), activation = 'linear', padding = 'valid', input_shape = self.inShape)
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv2 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (5, 5), strides = (1, 1), activation = 'linear', padding = 'same')
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')

		self.conv4 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')

		self.conv5 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')
		self.pool5 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.flatten = tf.keras.layers.Flatten()
		
		self.fc1 = tf.keras.layers.Dense(8192, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		
		self.fc2 = tf.keras.layers.Dense(8192, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout2 = tf.keras.layers.Dropout(0.5)
		
		self.fc3 = tf.keras.layers.Dense(1000, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'softmax')

	def call(self, inputs, training = False):
		res = self.conv1(inputs)
		res = tf.nn.relu(res)
		res = self.pool1(res)

		res = self.conv2(res)
		res = tf.nn.relu(res)
		res = self.pool2(res)

		res = self.conv3(res)
		res = tf.nn.relu(res)

		res = self.conv4(res)
		res = tf.nn.relu(res)

		res = self.conv5(res)
		res = tf.nn.relu(res)
		res = self.pool5(res)

		res = self.flatten(res)
		
		if training:
			res = self.fc1(res)
			res = self.dropout1(res)
			res = self.fc2(res)
			res = self.dropout2(res)
		else:
			res = self.fc1(res) * 0.5
			res = self.fc2(res) * 0.5

		res = self.fc3(res)

		return res

	def test_step(self, data):
		X, y = data

		yPred = 0

		for flip in self.flips:
			for coords in self.valCoords:
				coordsRowEnd = coords[0] + self.inShape[0]
				coordsColEnd = coords[1] + self.inShape[1]

				region = X[:, coords[0]:coordsRowEnd, coords[1]:coordsColEnd][:, :, ::flip, :]

				region = region - tf.reduce_mean(region, axis = [1, 2, 3], keepdims = True)
				region = region / tf.math.reduce_std(region, axis = [1, 2, 3], keepdims = True)

				yPred += self(region, training = False)

		yPred /= 10

		self.compiled_loss(y, yPred)

		self.compiled_metrics.update_state(y, yPred)

		return {m.name: m.result() for m in self.metrics}

class AlexNetLarge2(tf.keras.Model):
	def __init__(self, imgDim, inShape, batchSize):
		super().__init__()

		self.inShape = inShape

		self.flips = [1, -1]
		edge = imgDim - inShape[1]
		center = int(tf.math.ceil(edge / 2))
		self.valCoords = [[0, 0], [0, edge], [edge, 0], [edge, edge], [center, center]]
		self.valLocCount = len(self.flips) * len(self.valCoords)

		self.conv1 = tf.keras.layers.Conv2D(filters = 726, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (11, 11), strides = (4, 4), activation = 'linear', padding = 'valid', input_shape = self.inShape)
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv2 = tf.keras.layers.Conv2D(filters = 1024, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (5, 5), strides = (1, 1), activation = 'linear', padding = 'same')
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')

		self.conv4 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')

		self.conv5 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')
		self.pool5 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.flatten = tf.keras.layers.Flatten()
		
		self.fc1 = tf.keras.layers.Dense(8192, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		
		self.fc2 = tf.keras.layers.Dense(8192, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout2 = tf.keras.layers.Dropout(0.5)
		
		self.fc3 = tf.keras.layers.Dense(1000, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'softmax')

	def call(self, inputs, training = False):
		res = self.conv1(inputs)
		res = tf.nn.relu(res)
		res = self.pool1(res)

		res = self.conv2(res)
		res = tf.nn.relu(res)
		res = self.pool2(res)

		res = self.conv3(res)
		res = tf.nn.relu(res)

		res = self.conv4(res)
		res = tf.nn.relu(res)

		res = self.conv5(res)
		res = tf.nn.relu(res)
		res = self.pool5(res)

		res = self.flatten(res)
		
		if training:
			res = self.fc1(res)
			res = self.dropout1(res)
			res = self.fc2(res)
			res = self.dropout2(res)
		else:
			res = self.fc1(res) * 0.5
			res = self.fc2(res) * 0.5

		res = self.fc3(res)

		return res

	def test_step(self, data):
		X, y = data

		yPred = 0

		for flip in self.flips:
			for coords in self.valCoords:
				coordsRowEnd = coords[0] + self.inShape[0]
				coordsColEnd = coords[1] + self.inShape[1]

				region = X[:, coords[0]:coordsRowEnd, coords[1]:coordsColEnd][:, :, ::flip, :]

				region = region - tf.reduce_mean(region, axis = [1, 2, 3], keepdims = True)
				region = region / tf.math.reduce_std(region, axis = [1, 2, 3], keepdims = True)

				yPred += self(region, training = False)

		yPred /= 10

		self.compiled_loss(y, yPred)

		self.compiled_metrics.update_state(y, yPred)

		return {m.name: m.result() for m in self.metrics}

class AlexNetLarge2Norm(tf.keras.Model):
	def __init__(self, imgDim, inShape, batchSize):
		super().__init__()

		self.inShape = inShape

		self.flips = [1, -1]
		edge = imgDim - inShape[1]
		center = int(tf.math.ceil(edge / 2))
		self.valCoords = [[0, 0], [0, edge], [edge, 0], [edge, edge], [center, center]]
		self.valLocCount = len(self.flips) * len(self.valCoords)

		self.conv1 = tf.keras.layers.Conv2D(filters = 726, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (11, 11), strides = (4, 4), activation = 'linear', padding = 'valid', input_shape = self.inShape)
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv2 = tf.keras.layers.Conv2D(filters = 1024, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (5, 5), strides = (1, 1), activation = 'linear', padding = 'same')
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')

		self.conv4 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')

		self.conv5 = tf.keras.layers.Conv2D(filters = 512, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), kernel_size = (3, 3), strides = (1, 1), activation = 'linear', padding = 'same')
		self.pool5 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2))

		self.flatten = tf.keras.layers.Flatten()
		
		self.fc1 = tf.keras.layers.Dense(8192, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		
		self.fc2 = tf.keras.layers.Dense(8192, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'relu')
		self.dropout2 = tf.keras.layers.Dropout(0.5)
		
		self.fc3 = tf.keras.layers.Dense(1000, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.01), activation = 'softmax')

	def call(self, inputs, training = False):
		res = self.conv1(inputs)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = tf.nn.relu(res)
		res = self.pool1(res)

		res = self.conv2(res)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = tf.nn.relu(res)
		res = self.pool2(res)

		res = self.conv3(res)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = tf.nn.relu(res)

		res = self.conv4(res)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = tf.nn.relu(res)

		res = self.conv5(res)
		res = tf.nn.local_response_normalization(res, depth_radius = 5, bias = 2, alpha = 0.0001, beta = 0.75)
		res = tf.nn.relu(res)
		res = self.pool5(res)

		res = self.flatten(res)
		
		if training:
			res = self.fc1(res)
			res = self.dropout1(res)
			res = self.fc2(res)
			res = self.dropout2(res)
		else:
			res = self.fc1(res) * 0.5
			res = self.fc2(res) * 0.5

		res = self.fc3(res)

		return res

	def test_step(self, data):
		X, y = data

		yPred = 0

		for flip in self.flips:
			for coords in self.valCoords:
				coordsRowEnd = coords[0] + self.inShape[0]
				coordsColEnd = coords[1] + self.inShape[1]

				region = X[:, coords[0]:coordsRowEnd, coords[1]:coordsColEnd][:, :, ::flip, :]

				region = region - tf.reduce_mean(region, axis = [1, 2, 3], keepdims = True)
				region = region / tf.math.reduce_std(region, axis = [1, 2, 3], keepdims = True)

				yPred += self(region, training = False)

		yPred /= 10

		self.compiled_loss(y, yPred)

		self.compiled_metrics.update_state(y, yPred)

		return {m.name: m.result() for m in self.metrics}
