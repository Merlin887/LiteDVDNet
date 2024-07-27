import importlib
import os
import pprint
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn

from infrastructure.test_results import TestCaseResult, organize_results, get_test_case_table
from utils import batch_psnr, init_logger_test, \
    variable_to_cv2_image, open_sequence, close_logger, get_current_time, create_folder, \
    save_as_json, load_options, apply_padding, remove_padding, create_video_from_images, calculate_strred, \
    calculate_ssim

NUM_IN_FR_EXT = 5 # temporal size of patch

class TestRunner:
    def __init__(self, options_path: str):
        self.options_path = options_path
        self.options = load_options(options_path)
        self.test_settings =  self.options['test_settings']
        self.suite_path = self.get_path()
        create_folder(self.suite_path)
        self.logger = init_logger_test(self.suite_path)

    def get_path(self) -> str:
        now = datetime.now().strftime("%m%d%Y_%H%M%S")
        save_path = self.options['test_settings']['save_path']
        return str(os.path.join(save_path, self.options['id'] + '_' + now))

    def log(self, message: str):
        self.logger.info(message)

    def close_logger(self):
        close_logger(self.logger)

    def get_device(self) -> torch.device:

        # Sets data type according to CPU or GPU modes
        if self.options['test_settings']['use_cuda']:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    def create_model(self, model) -> nn.Module:

        module = importlib.import_module(model['module'])
        model_class = getattr(module, model['model_name'])
        instance = model_class(**model['model_params'])

        return instance


    def get_models_to_compare(self):
        # Load models to compare
        all_models_dir = self.options['models_to_compare']

        models_to_compare = []
        for model_dir in os.listdir(all_models_dir):
            models_metadata = {}
            model_opts = load_options(os.path.join(all_models_dir, model_dir, f'{model_dir}.yaml'))
            models_metadata['model_name'] = model_opts['model_name']
            models_metadata['description'] = model_dir
            models_metadata['module'] = model_opts['module']
            models_metadata['model_params'] = model_opts['model_params']
            models_metadata['model_params']['inference_mode'] = model_opts['model_params']['inference_mode']
            pretrain_ckpt = os.path.join(all_models_dir, model_dir, f'{model_dir}.pth')
            models_metadata['model_params']['pretrain_ckpt'] = pretrain_ckpt
            models_to_compare.append(models_metadata)

        return models_to_compare

    def run(self):

        test_suite_id = self.options['id']
        options_copy_path = os.path.join(self.suite_path, f'{test_suite_id}.yml')
        shutil.copy(self.options_path, options_copy_path)

        self.log(f"### Starting test runner at {get_current_time()} ###")
        self.log(pprint.pformat(self.options))

        test_case_results = []

        models_to_compare = self.get_models_to_compare()
        noise_sigmas = self.options['test_settings']['noise_sigma']

        for noise_sigma in noise_sigmas:

            self.log(f"\nRunning test case for noise level: {noise_sigma}\n")

            for model_metadata in models_to_compare:
                model_name = model_metadata['model_name']
                model_description = model_metadata['description']
                self.log(f"Running model: {model_name} ({model_description})")

                # create model folder for experiments
                model_folder = os.path.join(self.suite_path, model_description)
                create_folder(model_folder)

                for test_case in self.options['test_cases']:
                    # create model
                    model = self.create_model(model_metadata)
                    test_case['noise_sigma'] = noise_sigma
                    self.log(f"Running test case: {test_case}, buffers empty: {model.are_buffers_empty()}")
                    result = self.run_model(test_case, model, model_description, model_folder)
                    test_case_results.append(result)
                    del model

        self.log(f"### Test completed at {get_current_time()} ###")

        save_path = os.path.join(self.suite_path, f'{test_suite_id}_Results.json')
        test_suite_results = organize_results(test_case_results)
        self.log('\n' + get_test_case_table(test_case_results))
        self.log('\n' + test_suite_results.get_results())
        save_as_json(test_suite_results, save_path)

        # close logger
        self.close_logger()


    def run_model(self, test_case, loaded_model, model_description: str, model_folder: str) -> TestCaseResult:

        experiment_folder = os.path.join(model_folder, f"{test_case['id']}_{test_case['noise_sigma']}")
        create_folder(experiment_folder)

        device = self.get_device()
        loaded_model.to(self.get_device())

        net_params = sum(map(lambda x: x.numel(), loaded_model.parameters()))
        self.log(f'Network: {model_description}, with parameters: {net_params:,d}')

        # Sets the model in evaluation mode (e.g. it removes BN)
        loaded_model.eval()

        with torch.no_grad():
            # process data
            with torch.cuda.amp.autocast(True):
                original_seq, loadtime = self.load_sequence(test_case, device)
                noisy_seq, denoised_seq, runtime = self.denoise_sequence(original_seq, loaded_model, test_case, device)

        strred_score = 0
        ssim_score = 0
        psnr = batch_psnr(denoised_seq, original_seq, 1., False)
        psnr_noisy = batch_psnr(noisy_seq.squeeze(), original_seq, 1., False)
        seq_length = original_seq.size()[0]
        average_frame_time = runtime / seq_length

        self.log(f"Finished denoising {test_case['test_data_path']}")
        self.log(f"\tDenoised {seq_length} frames in {runtime:.3f}s, loaded seq in {loadtime:.3f}s")
        self.log(f"\tSingle frame denoising time {round(average_frame_time, 3) * 1000} msec")
        self.log(f"\tPSNR noisy {psnr_noisy:.4f}dB, PSNR result {psnr:.4f}dB")

        # Save outputs
        if self.options['test_settings']['save_results']:

            original_data_folder = test_case['test_data_path']
            filename = os.path.basename(test_case['test_data_path'])
            # Save sequence
            self.save_out_seq(noisy_seq = noisy_seq,
                              denoised_seq = denoised_seq,
                              save_dir=experiment_folder,
                              filename=filename,
                              fext=self.test_settings['suffix'],
                              save_noisy=self.test_settings['save_noisy'])


            frames = self.test_settings['max_num_fr_per_seq']
            orig_video_path = create_video_from_images(original_data_folder,f'{filename}_original.mp4', experiment_folder,frames, 30)
            denoised_video_path = create_video_from_images(experiment_folder,f'{filename}_denoised.mp4',experiment_folder,frames, 30)
            if self.options['test_settings']['calculate_strred']:
                strred_score = calculate_strred(orig_video_path, denoised_video_path, frames)
                self.log(f'ST-RRED score: {strred_score}')

            if self.options['test_settings']['calculate_ssim']:
                ssim_score_array = calculate_ssim(orig_video_path, denoised_video_path, frames)
                ssim_score = np.mean(np.array(ssim_score_array))
                self.log(f'SSIM score: {ssim_score}')



        tc_result = TestCaseResult(model_name=model_description,
                              test_case_name=test_case['id'],
                              noise_level=test_case['noise_sigma'],
                              strred_score=ssim_score.item(),
                              ssim_score=strred_score.item(),
                              psnr_noisy=round(psnr_noisy, 3),
                              psnr_clean=round(psnr, 3),
                              total_denoising_time=round(runtime, 3) * 1000,
                              single_frame_time=round(average_frame_time, 3) * 1000)

        return tc_result

    def load_sequence(self, test_case, device):

        start_time = time.time()

        # process data
        original_seq, _, _ = open_sequence(test_case['test_data_path'],
                                           self.test_settings['gray'],
                                           expand_if_needed=False,
                                           max_num_fr=self.test_settings['max_num_fr_per_seq'])

        original_seq = torch.from_numpy(original_seq).to(device)
        loading_time = time.time() - start_time

        return  original_seq, loading_time

    def denoise_sequence(self, original_seq, model_temp, test_case, device):

        noise_level = test_case['noise_sigma'] / 255

        # Add noise
        noise = torch.empty_like(original_seq).normal_(mean=0, std=noise_level).to(device)
        noisy_seq = original_seq + noise
        noisestd = torch.FloatTensor([noise_level]).to(device)

        numframes, C, H, W = noisy_seq.shape
        noise_map = noisestd.expand((1, 1, H, W))
        padded_noisyseq, padded_noisemap = apply_padding(noisy_seq, noise_map)

        start_time = time.time()

        denoised_seq = self.denoise_seq_fastdvd(model=model_temp,
                                                seq=padded_noisyseq,
                                                noise_map=padded_noisemap,
                                                temp_psz=NUM_IN_FR_EXT)

        denoising_time = time.time() - start_time

        denoised_seq = remove_padding(original_seq, denoised_seq)
        return noisy_seq, denoised_seq, denoising_time

    def save_out_seq(self, noisy_seq, denoised_seq, save_dir, filename, fext, save_noisy):
        """Saves the denoised and noisy sequences under save_dir
        """
        seq_len = noisy_seq.size()[0]
        for idx in range(seq_len):
            # Build Outname
            noisy_name = os.path.join(save_dir, f'{filename}_noisy_{idx:04d}{fext}')
            denoised_name = os.path.join(save_dir, f'{filename}_denoised_{idx:04d}{fext}')

            # Save result
            if save_noisy:
                noisyimg = variable_to_cv2_image(noisy_seq[idx].clamp(0., 1.))
                cv2.imwrite(noisy_name, noisyimg)

            outimg = variable_to_cv2_image(denoised_seq[idx].unsqueeze(dim=0))
            cv2.imwrite(denoised_name, outimg)

    def denoise_seq_fastdvd(self, model, seq, noise_map, temp_psz):
        r"""Denoises a sequence of frames with FastDVDnet.

        Args:
            seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
            noise_map: Tensor. Noise map
            temp_psz: size of the temporal patch
            model_temp: instance of the PyTorch model of the temporal denoiser
        Returns:
            denframes: Tensor, [numframes, C, H, W]
        """
        # init arrays to handle contiguous frames and related patches
        numframes, C, H, W = seq.shape
        ctrlfr_idx = int((temp_psz - 1) // 2)
        inframes = list()
        denframes = torch.empty((numframes, C, H, W)).to(seq.device)

        for fridx in range(numframes):
            # load input frames
            if not inframes:
                # if list not yet created, fill it with temp_patchsz frames
                for idx in range(temp_psz):
                    relidx = abs(idx - ctrlfr_idx)  # handle border conditions, reflect
                    inframes.append(seq[relidx])
            else:
                del inframes[0]
                relidx = min(fridx + ctrlfr_idx, -fridx + 2 * (numframes - 1) - ctrlfr_idx)  # handle border conditions
                inframes.append(seq[relidx])

            inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz * C, H, W)).to(seq.device)

            # append result to output list
            denframes[fridx] = torch.clamp(model(inframes_t, noise_map), 0., 1.)

        # free memory up
        del inframes
        del inframes_t
        torch.cuda.empty_cache()

        # convert to appropiate type and return
        return denframes









