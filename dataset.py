from torch.utils.data import Dataset


class DSDNDataset(Dataset):
    def __init__(self, noise_im1, noise_pd1, noise_im2, noise_pd2, cd):
        super(DSDNDataset, self).__init__()
        self.noise_im1 = noise_im1
        self.noise_pd1 = noise_pd1
        self.noise_im2 = noise_im2
        self.noise_pd2 = noise_pd2
        self.cd = cd

    def __getitem__(self, index):
        batch_noise_im1 = self.noise_im1[index]
        batch_noise_pd1 = self.noise_pd1[index]
        batch_noise_im2 = self.noise_im2[index]
        batch_noise_pd2 = self.noise_pd2[index]
        batch_cd = self.cd[index]
        return batch_noise_im1.float(), batch_noise_pd1.float(), batch_noise_im2.float(), batch_noise_pd2.float(), batch_cd.float()

    def __len__(self):
        return self.noise_im1.size(0)
