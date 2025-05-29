# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import cv2
from contextlib import contextmanager
from functools import partial
import torch.nn.functional as F
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import numpy as np
from typing import Optional, Literal
import itertools

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        
        # condition
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)
        
        
        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt


    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond, _ = self.model(
                    latent_model_input, t=timestep, **arg_c)
                noise_pred_uncond, _ = self.model(
                    latent_model_input, t=timestep, **arg_null)
                
                noise_pred_cond, noise_pred_uncond = noise_pred_cond[0], noise_pred_uncond[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    

    def load_video_frames(self, video_path, size=(832, 480)):
        r"""
        Load video frames from the given path and preprocess them.

        Args:
            video_path (str): Path to the video file.
            size (tuple[`int`], *optional*, defaults to (1280,720)): Target resolution for resizing frames.

        Returns:
            torch.Tensor: Tensor of video frames with shape (frame_num, C, H, W).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to target size
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            # Convert to tensor and normalize to [-1, 1]
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
            # Convert to tensor and normailize to [0, 1]
            # frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255
            frames.append(frame)

        cap.release()
        if not frames:
            raise ValueError(f"No frames found in video: {video_path}")

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3).to(self.device)
        latents = self.vae.video_encode(frames_tensor)
        return latents # [C, F, H, W]


    def find_subtokens_range(self, source_tokens, target_tokens):
        """
        查找 target_tokens 在 source_tokens 中的位置，不包括 '
        返回起始索引和结束索引（类似 slice），找不到返回 None
        """
        valid_len = len(source_tokens)
        # 在 source 的有效范围内滑动窗口匹配 target
        for i in range(valid_len - len(target_tokens) + 1):
            if source_tokens[i:i + len(target_tokens)] == target_tokens:
                # return (i, i + len(target_tokens)-1)
                return list(range(*(i, i + len(target_tokens))))
            

        return None


    def create_binary_mask( # Renamed slightly for clarity
        self,
        attn_map: torch.Tensor,
        n: int, 
        pooling_mode: Literal['max', 'avg'] = 'max',
        threshold: Optional[float] = 0.5,
        threshold_method: Literal['fixed', 'otsu'] = 'fixed',
        normalize_per_slice=False
    ) -> torch.Tensor:

        # --- Preparation ---
        original_shape = attn_map.shape
        C, T, H, W = original_shape
        device = attn_map.device

        map_batched = attn_map.reshape(C * T, 1, H, W)

        padding = (n - 1) // 2

        if pooling_mode == 'max':
            smoothed_map_batched = torch.nn.functional.max_pool2d(map_batched, kernel_size=n, stride=1, padding=padding)
        elif pooling_mode == 'avg':
            smoothed_map_batched = torch.nn.functional.avg_pool2d(map_batched, kernel_size=n, stride=1, padding=padding)

        smoothed_map = smoothed_map_batched.squeeze(1).view(C, T, H, W)
        # smoothed_map has shape (C, T, H, W)

        map_to_binarize = smoothed_map_batched # Start with the smoothed map
        if normalize_per_slice:

            flat_map = map_to_binarize.view(C * T, -1)

            # Calculate min and max per slice (image in the batch C*T)
            min_vals = torch.min(flat_map, dim=1, keepdim=True)[0]
            max_vals = torch.max(flat_map, dim=1, keepdim=True)[0]

            # Calculate range, handle the case where min == max (flat slice)
            range_vals = max_vals - min_vals

            range_vals = torch.where(range_vals == 0,
                                    torch.tensor(1.0, device=device, dtype=map_to_binarize.dtype),
                                    range_vals)

            min_vals_b = min_vals.view(C * T, 1, 1, 1)
            range_vals_b = range_vals.view(C * T, 1, 1, 1)

            normalized_map_batched = (map_to_binarize - min_vals_b) / range_vals_b

            map_to_binarize = torch.clamp(normalized_map_batched, 0.0, 1.0)

        map_to_binarize = map_to_binarize.squeeze(1).view(C, T, H, W)
        binary_mask = torch.zeros_like(map_to_binarize, dtype=torch.bool, device=device)
        
        if threshold_method == 'fixed':
            if normalize_per_slice: 
                binary_mask = map_to_binarize > torch.mean(map_to_binarize).item()
            else:
                binary_mask = map_to_binarize > threshold

        return binary_mask.to(device=device, dtype=torch.float32)          
   

    def soften_mask_edges(
        self,
        binary_mask: torch.Tensor,
        decay_factor: float = 0.1
    ) -> torch.Tensor:
        """
        Softens the edges of a binary mask by assigning values to background pixels (0)
        based on their distance to the nearest foreground pixel (1).

        Pixels originally equal to 1 remain 1.
        Pixels originally equal to 0 get a value exp(-decay_factor * distance),
        where distance is the Euclidean distance to the nearest 1.
        This means pixels closer to the original mask edge get values closer to 1,
        and pixels farther away get values closer to 0.

        Args:
            binary_mask (torch.Tensor): The input binary mask. Expected to be a 4D
                                        tensor (C, T, H, W) with values 0.0 or 1.0.
            decay_factor (float): Controls how quickly the softened value decays
                                with distance. A larger value means a faster decay
                                (sharper transition near the edge), a smaller value
                                means a slower decay (softer, more spread-out transition).
                                Defaults to 0.1.

        Returns:
            torch.Tensor: A mask tensor with the same dimensions as the input,
                        where original mask areas are 1.0 and background areas
                        have softened values based on distance. Output is float32.

        Raises:
            TypeError: If binary_mask is not a PyTorch Tensor.
            ValueError: If binary_mask is not 4D.
            ImportError: If scipy is not installed.
        """
        from scipy.ndimage import distance_transform_edt
        # --- Input Validation ---
        if not isinstance(binary_mask, torch.Tensor):
            raise TypeError("binary_mask must be a PyTorch Tensor.")
        if binary_mask.ndim != 4:
            raise ValueError(f"Input binary_mask must be 4D (C, T, H, W), got {binary_mask.ndim}D")
        if not decay_factor > 0:
            raise ValueError("decay_factor must be positive.")

        # --- Preparation ---
        original_shape = binary_mask.shape
        C, T, H, W = original_shape
        device = binary_mask.device
        dtype = torch.float32 # Ensure output is float

        # Create an output tensor initialized with zeros
        softened_mask = torch.zeros_like(binary_mask, dtype=dtype, device=device)

        # --- Process each slice (C, T) independently ---
        for c in range(C):
            for t in range(T):
                # Extract the 2D slice
                mask_slice = binary_mask[c, t, :, :]

                # Move to CPU and convert to NumPy boolean array for SciPy
                # distance_transform_edt expects background (0) to be True
                # and foreground (1) to be False.
                inverted_mask_slice_np = (mask_slice == 0).cpu().numpy()

                # Compute Euclidean Distance Transform
                # distance_map_np contains the distance from each True pixel (background)
                # to the nearest False pixel (foreground/mask).
                # Pixels that were originally part of the mask (False in inverted) will have distance 0.
                distance_map_np = distance_transform_edt(inverted_mask_slice_np)

                # Convert distances to softened values (0 to 1) using exponential decay
                # exp(-k * distance). Larger distance -> smaller value.
                # Distance 0 (original mask) -> exp(0) = 1.
                # Add a small epsilon to distance before applying exp if you want to strictly avoid 1.0 in non-mask areas
                # but exp(-k*dist) naturally handles this.
                softened_values_np = np.exp(-decay_factor * distance_map_np)

                # Convert back to PyTorch tensor and move to the original device
                softened_values_slice = torch.from_numpy(softened_values_np).to(device=device, dtype=dtype)

                # --- Combine original mask and softened background ---
                # Use torch.where for clarity and efficiency:
                # Where the original mask was 1, keep 1.0.
                # Where the original mask was 0, use the calculated softened value.
                final_slice = torch.where(
                    mask_slice.bool(), # Condition: True where original mask was 1
                    torch.tensor(1.0, device=device, dtype=dtype), # Value if True
                    softened_values_slice # Value if False (use calculated softened value)
                )

                # Place the processed slice into the output tensor
                softened_mask[c, t, :, :] = final_slice

        return softened_mask                

    
    def generate_conflict_map( # Renamed slightly for clarity
        self,
        vsrc: torch.Tensor,
        vtar: torch.Tensor,
        normalize: bool = True, # Setting default to True as it simplifies tuning 'k' later
        norm_type: int = 2,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Generates a conflict map indicating the magnitude of difference between
        two velocity fields (vsrc and vtar) and returns it as a 4D tensor.

        The conflict at each spatial-temporal location is defined as the L-p norm
        (default L2, Euclidean distance) of the difference vector between vsrc and
        vtar along the channel dimension. The resulting scalar map (F, H, W) is
        then expanded to match the input shape (C, F, H, W) by repeating the
        scalar value across the channel dimension.

        Args:
            vsrc (torch.Tensor): The source velocity field tensor. Expected shape:
                                (Channels, Frames, Height, Width).
            vtar (torch.Tensor): The target velocity field tensor. Must have the
                                same shape and device as vsrc.
            normalize (bool, optional): If True, normalize the underlying 3D conflict
                                        map to the range [0, 1] using min-max scaling
                                        before expanding it to 4D. Recommended for
                                        easier tuning of 'k' in downstream functions.
                                        Defaults to True.
            norm_type (int, optional): The order of the norm (p-norm). Default is 2 (L2 norm).
                                    Use 1 for L1 norm (Manhattan distance), etc.
            epsilon (float, optional): A small value added to the denominator during
                                    normalization to prevent division by zero if
                                    all conflict values are identical. Defaults to 1e-6.

        Returns:
            torch.Tensor: The conflict map tensor, expanded to 4D.
                        Shape: (Channels, Frames, Height, Width). The value is
                        uniform across the channel dimension for each (F,H,W).

        Raises:
            TypeError: If inputs are not PyTorch tensors.
            ValueError: If input tensors do not have the same shape or are not 4D.
        """
        # --- Input Validation ---
        if not isinstance(vsrc, torch.Tensor) or not isinstance(vtar, torch.Tensor):
            raise TypeError("Inputs vsrc and vtar must be PyTorch tensors.")

        if vsrc.ndim != 4 or vtar.ndim != 4:
            raise ValueError(f"Input tensors must be 4D (C, F, H, W), "
                            f"got shapes {vsrc.shape} and {vtar.shape}")

        if vsrc.shape != vtar.shape:
            raise ValueError(f"Input tensors vsrc and vtar must have the same shape, "
                            f"got {vsrc.shape} and {vtar.shape}")

        if vsrc.device != vtar.device:
            logging.warning(f"Input tensors are on different devices ({vsrc.device}, {vtar.device}). "
                            f"Proceeding with calculations, but ensure this is intended.")

        # --- Conflict Calculation ---
        vsrc_float = vsrc.float()
        vtar_float = vtar.float()
        num_channels = vsrc.shape[0]

        difference = vtar_float - vsrc_float

        # Calculate the norm along the channel dimension -> shape (F, H, W)
        conflict_map_3d = torch.norm(difference, p=norm_type, dim=0)

        # --- Optional Normalization (applied to the 3D map) ---
        if normalize:
            # Avoid normalization issues on empty tensors
            if conflict_map_3d.numel() > 0:
                min_val = torch.min(conflict_map_3d)
                max_val = torch.max(conflict_map_3d)
                denominator = max_val - min_val

                if denominator < epsilon:
                    logging.warning(f"Conflict map values are nearly constant ({min_val.item()} to {max_val.item()}). "
                                    f"Normalization results in a map of all zeros.")
                    conflict_map_3d = torch.zeros_like(conflict_map_3d)
                else:
                    conflict_map_3d = (conflict_map_3d - min_val) / denominator
            else:
                logging.warning("Conflict map has zero elements, skipping normalization.")


        # --- Expand to 4D ---
        # Add channel dim and expand
        conflict_map_4d = conflict_map_3d.unsqueeze(0).expand(num_channels, -1, -1, -1)

        return conflict_map_4d


    def compute_dynamic_source_mask( # Renamed slightly for clarity
        self,
        conflict_map_4d: torch.Tensor,
        k: float,
        function_type: str = 'exponential_squared',
        clamp_output: bool = True
        # warn_threshold removed as normalization is best done in the generating function
    ) -> torch.Tensor:
        """
        Computes the Dynamic Source Mask M(p, t) based on a 4D conflict map input.

        Applies a chosen function element-wise to the conflict map values to generate
        the mask. High conflict should result in a mask value near 0, while low
        conflict should result in a value near 1. Assumes the input conflict map
        may have redundant values across the channel dimension.

        Args:
            conflict_map_4d (torch.Tensor): A tensor representing the conflict C(p, t)
                                            between Vsrc and Vtar. Expected shape:
                                            (Channels, Frames, Height, Width).
                                            Normalizing this input (e.g., via
                                            generate_conflict_map_4d with normalize=True)
                                            is recommended for easier tuning of 'k'.
            k (float): A positive sensitivity parameter. Controls how aggressively
                    the mask value decreases as conflict increases.
            function_type (str, optional): Specifies the function f(C) used to map
                                        conflict C to the mask value M element-wise.
                                        Options:
                                        - 'exponential_squared': M = exp(-k * C^2)
                                        - 'exponential': M = exp(-k * C)
                                        - 'inverse': M = 1 / (1 + k * C)
                                        - 'inverse_squared': M = 1 / (1 + k * C^2)
                                        Defaults to 'exponential_squared'.
            clamp_output (bool, optional): If True, clamps the output mask values
                                        strictly to the range [0.0, 1.0].
                                        Defaults to True.

        Returns:
            torch.Tensor: The dynamic source mask tensor M(p, t).
                        Shape: (Channels, Frames, Height, Width).

        Raises:
            TypeError: If conflict_map_4d is not a PyTorch tensor.
            ValueError: If conflict_map_4d is not 4D, k is not positive, or an
                        invalid function_type is provided.
        """
        # --- Input Validation ---
        if not isinstance(conflict_map_4d, torch.Tensor):
            raise TypeError("Input conflict_map_4d must be a PyTorch tensor.")

        if conflict_map_4d.ndim != 4:
            raise ValueError(f"Input conflict_map_4d must be 4D (C, F, H, W), "
                            f"got {conflict_map_4d.ndim}D shape {conflict_map_4d.shape}")

        # num_channels = conflict_map_4d.shape[0] # Get C if needed elsewhere

        if not isinstance(k, (int, float)) or k <= 0:
            raise ValueError(f"Parameter k must be a positive number, got {k}")

        supported_functions = ['exponential_squared', 'exponential', 'inverse', 'inverse_squared']
        if function_type not in supported_functions:
            raise ValueError(f"Invalid function_type '{function_type}'. "
                            f"Supported types are: {supported_functions}")

        # --- Mask Calculation (Applied element-wise on 4D tensor) ---
        # Note: conflict_map_4d already includes the (potentially redundant) channel dim
        conflict_map_float = conflict_map_4d.float()
        k_float = float(k)

        if function_type == 'exponential_squared':
            # Applies exp(-k * C^2) element-wise
            mask_4d = torch.exp(-k_float * torch.pow(conflict_map_float, 2))
        elif function_type == 'exponential':
            # Applies exp(-k * C) element-wise
            mask_4d = torch.exp(-k_float * conflict_map_float)
        elif function_type == 'inverse':
            # Applies 1 / (1 + k * C) element-wise
            mask_4d = 1.0 / (1.0 + k_float * conflict_map_float)
        elif function_type == 'inverse_squared':
            # Applies 1 / (1 + k * C^2) element-wise
            mask_4d = 1.0 / (1.0 + k_float * torch.pow(conflict_map_float, 2))
        else:
            raise NotImplementedError(f"Function type {function_type} calculation not implemented.")

        # --- Optional Clamping ---
        if clamp_output:
            mask_4d = torch.clamp(mask_4d, min=0.0, max=1.0)

        # --- Return the 4D Mask ---
        # No expansion needed, it's already 4D
        return mask_4d
    
                    
    def edit(self,
             target_prompt,
             size=(832, 480),
             frame_num=81,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=50,
             guide_scale=5.0,
             tar_guide_scale=10.0,
             n_prompt="",
             seed=-1,
             offload_model=True,
             source_video_path=None,
             source_prompt=None,
             nmax_step=50,
             nmin_step=0,     
             n_avg=5,
             worse_avg=3,
             omega=3,
             source_words=None,
             target_words=None,
             window_size=11,
             decay_factor=0.1,
             tmd_window_size=11,
             tmd_stride=8
             ):
        # preprocess
        F = frame_num
        W, H = size
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1], size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) * 
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        # seed_g = torch.Generator(device=self.device)
        torch.manual_seed(seed)

        # 加载源视频潜在表示和参考图像
        x_src = self.load_video_frames(source_video_path)
        C_latent, F_latent, H_latent, W_latent = x_src.shape

        # Validate TMD parameters
        if tmd_window_size > F_latent:
            logging.warning(f"tmd_window_size ({tmd_window_size}) > latent frames ({F_latent}). Using full sequence as one window.")
            tmd_window_size = F_latent
            tmd_stride = F_latent
        elif tmd_stride <= 0:
             logging.warning(f"Invalid tmd_stride ({tmd_stride}). Setting stride to window size / 2.")
             tmd_stride = max(1, tmd_window_size // 2)
             
             
        # 计算提示词相对位置
        source_words_idx = None
        target_words_idx = None 
        if source_words:
            tk1 = self.text_encoder.tokenizer.tokenizer.tokenize(source_prompt, add_special_tokens=True)
            tk2 = self.text_encoder.tokenizer.tokenizer.tokenize(source_words, add_special_tokens=True)
            source_words_idx = self.find_subtokens_range(tk1[:-1], tk2[:-1])
            
        if target_words:
            tk1 = self.text_encoder.tokenizer.tokenizer.tokenize(target_prompt, add_special_tokens=True)
            tk2 = self.text_encoder.tokenizer.tokenizer.tokenize(target_words, add_special_tokens=True)
            target_words_idx = self.find_subtokens_range(tk1[:-1], tk2[:-1])

        # 编码源和目标文本提示
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context_src = self.text_encoder([source_prompt], self.device)
            context_tar = self.text_encoder([target_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context_tar = self.text_encoder([target_prompt], torch.device('cpu'))
            context_src = self.text_encoder([source_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context_src = [t.to(self.device) for t in context_src]
            context_tar = [t.to(self.device) for t in context_tar]
            context_null = [t.to(self.device) for t in context_null]

        arg_src_c = {'context': context_src, 'seq_len': seq_len, 'words_indices': source_words_idx, 'block_id': 18, 'type': 'src'}
        arg_tar_c = {'context': context_tar, 'seq_len': seq_len, 'words_indices': target_words_idx, 'block_id': 18, 'type': 'tar'}
        arg_unc = {'context': context_null, 'seq_len': seq_len}

        
        # 初始化编辑路径
        zt_edit = x_src.clone() # [16, 21, 60, 104]: C x Frames x H x W 
        conflict_mask = torch.ones_like(x_src)

        # 设置采样调度器
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False, solver_order=2)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False, solver_order=1)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        self.model.to(self.device)
        # 编辑过程
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            for index, t in enumerate(tqdm(timesteps)):
                t_next = timesteps[timesteps.tolist().index(t) + 1] if t > timesteps[-1] else 0
                arg_src_c["timestep"] = t
                arg_tar_c["timestep"] = t
                arg_unc["timestep"] = t
                timestep = torch.tensor([t], device=self.device)
                relative_index = nmax_step - (sampling_steps - index)
                v_list = []

                if sampling_steps - (index + 1) >= nmax_step:
                    continue
                if sampling_steps - (index + 1) >= nmin_step:
                    t_i = t / 1000.0
                    t_im1 = t_next / 1000.0
                    v_delta_sum = torch.zeros_like(zt_edit)
                    v_worse = torch.zeros_like(zt_edit)
                    v_mask = torch.zeros_like(zt_edit)
                    for time in range(n_avg):
                        # --- Temporal MultiDiffusion Logic ---
                        V_delta_accumulator = torch.zeros_like(zt_edit)
                        window_counts = torch.zeros_like(zt_edit) # Use float for division later
                        window_mask_sum = torch.zeros_like(zt_edit)    
                        fwd_noise = torch.randn_like(zt_edit[:, 0:tmd_window_size, :, :], device=self.device)
                        # for f_start in range(0, F_latent, tmd_stride):
                        if tmd_window_size >= F_latent:
                            window_starts = [0]
                        else:
                            window_starts = list(range(0, F_latent - tmd_window_size, tmd_stride))
                            last_possible_start = F_latent - tmd_window_size
                            if not window_starts or window_starts[-1] < last_possible_start:
                                if last_possible_start >= 0:
                                    window_starts.append(last_possible_start)

                        for f_start in window_starts:
                            f_end = F_latent if tmd_window_size >= F_latent else f_start + tmd_window_size

                            # --- Calculate V_delta within the window ---
                            # Extract window slices
                            zt_edit_w = zt_edit[:, f_start:f_end, :, :]
                            x_src_w = x_src[:, f_start:f_end, :, :]  

                            # 计算 zt_src
                            zt_src = (1 - t_i) * x_src_w + t_i * fwd_noise
                            zt_tar = zt_edit_w + zt_src - x_src_w
                                                
                        
                            # 计算源和目标噪声预测
                            noise_pred_src, src_attn_map = self.model([zt_src], t=timestep, **arg_src_c)
                            noise_pred_tar, tar_attn_map = self.model([zt_tar], t=timestep, **arg_tar_c)
                            noise_pred_src, noise_pred_tar = noise_pred_src[0], noise_pred_tar[0]
                            # uncond
                            noise_pred_uncond_src, _ = self.model([zt_src], t=timestep, **arg_unc)
                            noise_pred_uncond_tar, _ = self.model([zt_tar], t=timestep, **arg_unc)
                            noise_pred_uncond_src, noise_pred_uncond_tar = noise_pred_uncond_src[0], noise_pred_uncond_tar[0]

                            # 计算引导后的噪声预测
                            noise_pred_src_guided = noise_pred_uncond_src + guide_scale * (noise_pred_src - noise_pred_uncond_src)
                            noise_pred_tar_guided = noise_pred_uncond_tar + tar_guide_scale * (noise_pred_tar - noise_pred_uncond_tar)
                            
                            sum_attn_mask = torch.zeros_like(zt_edit_w)
                            
                            conflict_map = self.generate_conflict_map(noise_pred_src_guided, noise_pred_tar_guided)
                            raw_mask = self.compute_dynamic_source_mask(conflict_map, 0.5)
                            clamped_mask = torch.clamp(raw_mask, min=0.0, max=1.0)
                            conflict_mask = 5.0 - 4.0 * clamped_mask
                                                        
                            if src_attn_map is not None:
                                src_attn_mask = self.create_binary_mask(src_attn_map,
                                                                        n=window_size,
                                                                        pooling_mode='avg',
                                                                        threshold=torch.mean(src_attn_map).item(),
                                                                        threshold_method='fixed')
                                
                                sum_attn_mask += src_attn_mask
                                
                            if tar_attn_map is not None:
                                tar_attn_mask = self.create_binary_mask(tar_attn_map,
                                                                        n=window_size,
                                                                        pooling_mode='avg',
                                                                        threshold=torch.mean(tar_attn_map).item(),
                                                                        threshold_method='fixed')
                                
                                sum_attn_mask += tar_attn_mask
                                
                            sum_attn_mask = torch.clamp(sum_attn_mask, min=0.0, max=1.0)
                        
                            V_delta = noise_pred_tar_guided - noise_pred_src_guided
                            
                            # Accumulate results
                            V_delta_accumulator[:, f_start:f_end, :, :] += V_delta
                            window_counts[:, f_start:f_end, :, :] += 1.0
                            window_mask_sum[:, f_start:f_end, :, :] += sum_attn_mask

                        V_delta_final = V_delta_accumulator / torch.clamp(window_counts, min=1.0) # Avoid division by zero
                        v_delta_sum += V_delta_final
                        v_list.append(V_delta_final)
                        v_mask += window_mask_sum
                        if time < worse_avg:
                            v_worse += V_delta_final

                    v_list = torch.stack(v_list, dim=0) 
                    V_delta_better = v_list.mean(dim=0)
                    
                    v_trend = []
                    for worse_set in itertools.combinations(v_list, worse_avg):
                        v_worse = torch.zeros_like(V_delta_better)
                        for worse in worse_set:
                            v_worse += worse
                        v_worse = v_worse / worse_avg
                        v_trend.append(V_delta_better - v_worse)
                    v_trend = torch.stack(v_trend, dim=0)
                    v_trend = v_trend.mean(dim=0)
                    
                    v_mask = torch.clamp(v_mask, min=0.0, max=1.0)
                    v_mask = self.soften_mask_edges(v_mask, decay_factor=decay_factor)
                    
                    V_delta_final = V_delta_better + (omega - 1) * v_trend
                    V_delta_final = V_delta_final * v_mask

                    zt_edit = zt_edit + (t_im1 - t_i) * V_delta_final                
                    
                else:
                    # 使用tar进行采样
                    noise_pred_uncond, _ = self.model([zt_edit], t=timestep, context=context_null, seq_len=seq_len)
                    noise_pred_cond, _ = self.model([zt_edit], t=timestep, context=context_tar, seq_len=seq_len)
                    noise_pred_cond, noise_pred_uncond = noise_pred_cond[0], noise_pred_uncond[0]
                    noise_pred = noise_pred_uncond + 6 * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        zt_edit.unsqueeze(0),
                        return_dict=False)[0]
                    
                    zt_edit = temp_x0.squeeze(0)
             
        # 解码编辑结果
        if offload_model:
            self.model.cpu()
        if self.rank == 0:
            videos = self.vae.decode([zt_edit])
            return videos[0]
        return None
    

