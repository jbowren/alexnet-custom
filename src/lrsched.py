import tensorflow as tf

class ReduceLROnPlateau(tf.keras.callbacks.Callback):
	def __init__(self, factor = 0.1, verbose = 1, min_lr = 0.000002):
		super(ReduceLROnPlateau, self).__init__()

		self.factor = factor
		self.verbose = verbose
		self.min_lr = min_lr
		self.valLossPrev = None

	def on_epoch_end(self, epoch, logs = None):
		if logs is None:
			print('No logs')
			return
	
		valLoss = logs.get('val_loss')

		if valLoss is None:
			return

		if not self.valLossPrev is None and valLoss > self.valLossPrev:
			oldLR = float(self.model.optimizer.lr)
			newLR = oldLR * self.factor

			if newLR >= self.min_lr:
				if self.verbose > 0:
					print(f'\nEpoch {epoch + 1}: ReduceLRonPlateau reducing learning rate to {newLR}.')

				self.model.optimizer.lr = newLR

		self.valLossPrev = valLoss
