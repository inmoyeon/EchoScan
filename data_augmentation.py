import torch
import torchaudio
import random


def data_augmentation(X, augmentation_type, fs):
    selected_type = random.choice(augmentation_type)

    if selected_type == "none":
        pass

    elif selected_type == "filter_augmentation":
        filter_types = [
            "lowpass",
            "highpass",
            "bandpass",
            "bandreject",
        ]  # 0: None, 1: lowpass, 2: highpass
        filter_type = random.choice(filter_types)

        if filter_type == "lowpass":
            target_freq = random.randint(500, 2000)
            X = torchaudio.functional.lowpass_biquad(
                X, sample_rate=fs, cutoff_freq=target_freq
            )
        elif filter_type == "highpass":
            target_freq = random.randint(100, 1000)
            X = torchaudio.functional.highpass_biquad(
                X, sample_rate=fs, cutoff_freq=target_freq
            )
        elif filter_type == "bandpass":
            target_freq = random.randint(100, 2000)
            X = torchaudio.functional.bandpass_biquad(
                X, sample_rate=fs, central_freq=target_freq
            )
        elif filter_type == "bandreject":
            target_freq = random.randint(100, 2000)
            X = torchaudio.functional.bandreject_biquad(
                X, sample_rate=fs, central_freq=target_freq
            )

    elif selected_type == "sliding":
        sliding_types = ["roll_sliding", "zero_sliding"]
        sliding_type = random.choice(sliding_types)

        if sliding_type == "roll_sliding":
            sliding_value = random.randint(0, X.shape[-1])
            X = torch.roll(X, sliding_value, dims=-1)
        elif sliding_type == "zero_sliding":
            sliding_range = [-500, 500]
            sliding_value = random.randint(sliding_range[0], sliding_range[1])
            if not sliding_value == 0:
                X = torch.roll(X, sliding_value, dims=-1)
                if sliding_value > 0:
                    X[:, :sliding_value] = 0.0
                elif sliding_value < 0:
                    X[:, -sliding_value:] = 0.0

    elif selected_type == "time_masking":
        time_masking_types = ["long", "medium", "short"]
        time_masking_type = random.choice(time_masking_types)

        if time_masking_type == "long":
            X = time_masking_1d(X, mask_length=128, nb_masks=1)
        elif time_masking_type == "medium":
            X = time_masking_1d(X, mask_length=64, nb_masks=2)
        elif time_masking_type == "short":
            X = time_masking_1d(X, mask_length=32, nb_masks=4)

    elif selected_type == "channel_masking":
        mask_channels = random.sample(range(X.shape[0]), 1)
        X[mask_channels] = 0

    return X


def time_masking_1d(X, mask_length, nb_masks):
    masks = []
    for i in range(nb_masks):
        mask_start = random.randint(0, X.shape[-1] - mask_length)
        mask_end = mask_start + mask_length
        mask = [mask_start, mask_end]
        masks.append(mask)

    for mask in masks:
        X[:, mask[0] : mask[1]] = 0

    return X
