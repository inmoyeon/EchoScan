import os
import random
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from data_augmentation import data_augmentation


def get_mean_std(train_path, sample_num=1000):
    rirs = []
    files = all_files_in_directory(train_path)
    files = random.sample(files, sample_num)
    for i in tqdm(files):
        rir = torch.load(i)["rir"]
        rirs.append(rir)

    rirs = torch.stack(rirs, dim=0)
    mean = torch.mean(rirs)
    std = torch.std(rirs)
    return [mean, std]


def all_files_in_directory(dir):
    file_list = []
    for root, dirs, files in os.walk(dir):
        if files:
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    return file_list


class EchoscanDataset(Dataset):
    def __init__(
        self,
        conf=None,
        split="train",
        augmentation=False,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.conf = conf
        self.split = split
        self.augmentation = augmentation

        if split == "train":
            self.datapath = conf["path_dataset_train"]
            self.data = all_files_in_directory(self.datapath)
            self.use_augmentation = True if len(self.augmentation) > 0 else False
            self.snr_range = conf.loader.tr_snr_range

        elif split == "valid":
            self.datapath = conf["path_dataset_valid"]
            self.data = all_files_in_directory(self.datapath)
            self.use_augmentation = False
            self.snr_range = conf.loader.val_snr_range

    def __getitem__(self, index):
        filename = self.text_to_filename(self.data[index])
        datum = torch.load(self.data[index])

        rir = datum["rir"]
        image_label = datum["filled_image_labels"]
        height_label = datum["height_labels"]
        rt60 = datum["rt60"]
        room_type = datum["room_type"]
        if "los" in datum.keys():
            los = datum["los"]
        else:
            los = torch.Tensor([1.0])

        if self.conf.loader.add_g_noise:
            rir = self.add_noise(rir, self.snr_range)

        if self.conf.loader.use_normalize:
            rir = self.normalize_wav(
                rir,
                normalize_type=self.conf.loader.normalize_type,
                mean_std=[torch.mean(rir), torch.std(rir)],
            )

        if self.use_augmentation:
            rir = data_augmentation(rir, self.augmentation, fs=self.conf.model.fs)

        if torch.sum(torch.isnan(rir)) > 0:
            print("NaN in rir: ", filename)
            exit()
        elif torch.sum(torch.isinf(rir)) > 0:
            print("Inf in rir", filename)
            exit()

        data = {
            "filename": filename,
            "rir": rir,
            "image_label": image_label,
            "height_label": height_label,
            "rt60": rt60,
            "room_type": room_type,
            "los": los,
        }
        return data

    def text_to_filename(self, text):
        return text.split("/")[-1].strip(".pt")

    def normalize_wav(self, waveform, normalize_type, mean_std=None):
        """
        normalize_type: minmax, zscore, rms
        """
        if normalize_type == "minmax":
            waveform = waveform - torch.mean(waveform)
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            return waveform
        elif normalize_type == "zscore":
            assert (
                mean_std is not None
            ), "mean_std should be provided for zscore normalization"
            trainset_m = mean_std[0]
            trainset_std = mean_std[1]
            waveform = (waveform - trainset_m) / (trainset_std + 1e-8)
            return waveform
        elif normalize_type == "rms":
            rms = torch.sqrt(torch.mean(torch.square(waveform)))
            waveform = waveform / (rms + 1e-8)
            return waveform
        else:
            raise ValueError(
                "Unknown norm type in [minmax, zscore, rms]: %s" % normalize_type
            )

    def add_noise(self, rir, snr_range):
        snr = ((snr_range[1] - snr_range[0]) * torch.rand(1)) + snr_range[0]
        noise = torch.randn_like(rir)

        pow_rir = torch.sum(torch.square(rir))
        pow_noise = torch.sum(torch.square(noise))

        scaled_pow_noise = pow_rir / (10 ** (snr / 10))
        scaling = torch.sqrt(scaled_pow_noise) / torch.sqrt(pow_noise)
        scaled_noise = scaling * noise

        noisy_rir = rir + scaled_noise
        return noisy_rir

    def __len__(self):
        return len(self.data)
