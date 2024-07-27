"""
Different common functions for training the models.
"""
import os
import torch


def extract_model_from_ckpt_and_save(argdict, model, ckpt_number):
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir'], f"ckpt_e{ckpt_number}.pth")
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			model.load_state_dict(checkpoint['state_dict'])
			model_name = argdict['model_description']
			torch.save(model.state_dict(), os.path.join(argdict['log_dir'], f'{model_name}_{ckpt_number}.pth'))


def	resume_training(argdict, model, optimizer):
	""" Resumes previous training or starts anew
	"""
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = argdict['epochs']
			current_lr = argdict['lr']
			argdict = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			argdict['epochs'] = new_epoch
			argdict['lr'] = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = checkpoint['args']
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			argdict['resume_training'] = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 1
		training_params = {}
		training_params['step'] = 1
		training_params['current_lr'] = 0
		training_params['orthog_enabled'] = argdict['orthog_epochs'] > 0

	return start_epoch, training_params

def need_ortog(epoch, argdict) -> bool:

	if argdict['orthog_epochs'] > 0 and epoch <= argdict['orthog_epochs']:
		return True

	return False

def lr_scheduler(epoch, argdict):
	"""Returns the learning rate value depending on the actual epoch number"""
	milestones = argdict['milestones'][::-1]
	learning_rates = argdict['learning_rates'][::-1]

	for idx, milestone in enumerate(milestones):
		if epoch >= milestone:
			return learning_rates[idx]

	return learning_rates[-1]

def binary_lr_scheduler(epoch, argdict):
	learning_rate = argdict['lr']
	k = epoch // 10
	return learning_rate * (0.5 ** k)

def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
	"""Stores the model parameters under 'argdict['log_dir'] + '/model_name.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
	model_name = argdict['model_description']
	torch.save(model.state_dict(), os.path.join(argdict['log_dir'], f'{model_name}.pth'))

	save_dict = { \
		'state_dict': model.state_dict(), \
		'optimizer' : optimizer.state_dict(), \
		'training_params': train_pars, \
		'args': argdict\
		}

	torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

	if epoch % argdict['save_ckpt_every_epochs'] == 0:
		torch.save(save_dict, os.path.join(argdict['log_dir'], f'ckpt_e{epoch}.pth'))

	del save_dict

