import argparse
import torch
from infrastructure.train_runner import TrainRunner



if __name__ == "__main__":

	trainingPath = './train_options/lite_dvd_train.yml'
	trainRunner = TrainRunner(trainingPath)
	options = trainRunner.get_options()

	# check if CUDA available on GPU test run
	if options['use_cuda'] and not torch.cuda.is_available():
		raise OSError("CUDA is not available")

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for key, value in options.items():
		print(f'\t{key}: {value}')
	print('\n')

	trainRunner.train()



