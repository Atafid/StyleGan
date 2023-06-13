import torch
import torchvision
from pathlib import Path
from PIL import Image
from typing import Tuple, Iterator

from labml import tracker, monit, experiment, lab
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.utils import cycle_dataloader

from src.generator import Generator, MappingNetwork, GeneratorLoss
from src.discriminator import Discriminator, DiscriminatorLoss
from src.utils import PathLengthPenalty, GradientPenalty

import matplotlib.pyplot as plt
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size):
        super().__init__()

        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]

        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize([img_size, img_size]), torchvision.transforms.ToTensor()])

    def __len__(self):
        return (len(self.paths))

    def __getitem__(self, index):
        return (self.transform(Image.open(self.paths[index])))


class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()

    d_latent: int = 512
    batch_size: int = 32
    img_size: int = 32

    dataset: Dataset
    dataset_path: str
    loader: Iterator

    discriminator: Discriminator
    generator: Generator
    mapping_network: MappingNetwork

    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss

    discriminator_optimizer: torch.optim.Adam
    generator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam

    num_gen_block: int

    learning_rate: float = 1e-3
    mn_learning_rate: float = 1e-5
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    training_steps: int = 1000

    style_mix_prob: float = 0.9

    log_generated_interval: int = 500
    gradient_accumulate_steps: int = 1

    gradient_penalty: GradientPenalty
    path_length_penalty: PathLengthPenalty

    lazy_gradient_penalty_interval: int = 4
    gradient_penalty: GradientPenalty
    gradient_penalty_coefficient: float = 10.

    lazy_path_penalty_interval: int = 32
    lazy_path_penalty_after: int = 5000

    save_checkpoint_interval: int = 2000

    log_layer_outputs: bool = False
    mode = ModeState()

    def init(self, path):
        self.dataset_path = path

        self.dataset = Dataset(self.dataset_path, self.img_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=8,
                                                 shuffle=True, drop_last=True, pin_memory=True)

        self.loader = cycle_dataloader(dataloader)

        self.discriminator = Discriminator(
            self.img_size, n_features=64, max_features=512).to(self.device)
        self.generator = Generator(
            self.img_size, self.d_latent, n_features=32, max_features=512).to(self.device)

        self.num_gen_block = self.generator.num_blocks

        self.mapping_network = MappingNetwork(
            8, self.d_latent).to(self.device)

        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)
        self.gradient_penalty = GradientPenalty()

        if (self.log_layer_outputs):
            hook_model_outputs(self.mode, self.discriminator, 'discriminator')
            hook_model_outputs(self.mode, self.generator, 'generator')
            hook_model_outputs(
                self.mode, self.mapping_network, 'mapping_network')

        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(), lr=self.mn_learning_rate, betas=self.adam_betas)

        tracker.set_image("generated", True)

    def get_w(self, batch_size):
        if (torch.rand(()).item() < self.style_mix_prob):
            cross_over_point = int(torch.rand(()).item() * self.num_gen_block)

            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)

            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(
                self.num_gen_block-cross_over_point, -1, -1)

            return (torch.cat((w1, w2), dim=0))

        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)

            w = self.mapping_network(z)

            return (w[None, :, :].expand(self.num_gen_block, -1, -1))

    def get_noise(self, batch_size):
        noise = []

        res = 4

        for i in range(self.num_gen_block):
            if (i == 0):
                n1 = None

            else:
                n1 = torch.randn(batch_size, 1, res, res, device=self.device)

            n2 = torch.randn(batch_size, 1, res, res, device=self.device)

            noise.append((n1, n2))
            res *= 2

        return (noise)

    def generate_images(self, batch_size):
        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)

        img = self.generator(w, noise)

        return (img, w)

    def step(self, index):
        with (monit.section("Discriminator")):
            self.discriminator_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):
                with (self.mode.update(is_log_activations=(index+1) % self.log_generated_interval == 0)):
                    gen_img, w = self.generate_images(self.batch_size)

                    fake_output = self.discriminator(gen_img.detach())

                    real_img = next(self.loader).to(self.device)

                    if ((index+1) % self.lazy_gradient_penalty_interval == 0):
                        real_img.requires_grad_()

                    real_output = self.discriminator(real_img)

                    real_loss, fake_loss = self.discriminator_loss(
                        real_output, fake_output)
                    disc_loss = real_loss+fake_loss

                    if ((index+1) % self.lazy_gradient_penalty_interval == 0):
                        gp = self.gradient_penalty(real_img, real_output)
                        tracker.add('loss.gp', gp)

                        disc_loss = disc_loss + 0.5*self.gradient_penalty_coefficient * \
                            gp*self.lazy_gradient_penalty_interval

                    disc_loss.backward()

                    tracker.add('loss_discriminator', disc_loss)

            if ((index+1) % self.log_generated_interval == 0):
                tracker.add('discriminator', self.discriminator)

            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.)
            self.discriminator_optimizer.step()

        with (monit.section("Generator")):
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):
                gen_img, w = self.generate_images(self.batch_size)

                fake_output = self.discriminator(gen_img)
                gen_loss = self.generator_loss(fake_output)

                if (index > self.lazy_path_penalty_after and (index+1) % self.lazy_path_penalty_interval == 0):
                    plp = self.path_length_penalty(w, gen_img)

                    if (not (torch.isnan(plp))):
                        tracker.add('loss.plp', plp)
                        gen_loss = gen_loss + plp

                gen_loss.backward()
                tracker.add('generator_loss', gen_loss)

            if ((index+1) % self.log_generated_interval == 0):
                tracker.add('generator', self.generator)
                tracker.add('mapping_network', self.mapping_network)

            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), max_norm=1.)
            torch.nn.utils.clip_grad_norm_(
                self.mapping_network.parameters(), max_norm=1.)

            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()

        if ((index+1) % self.log_generated_interval == 0):
            tracker.add('generated', torch.cat(
                [gen_img[:6], real_img[:3]], dim=0))

        if ((index+1) % self.save_checkpoint_interval == 0):
            experiment.save_checkpoint()

        if ((index+1) % 50 == 0):
            img, w = self.generate_images(self.batch_size)

            #f, axarr = plt.subplots(4, 8)

            # for i in range(32):
            #     real_img = self.dataset.__getitem__(i)
            #     real_img = np.transpose(real_img.detach().cpu().numpy(), (1, 2, 0))

            #     axarr[int(i/8)][i % 8].imshow(real_img)
            #     axarr[int(i/8)][i % 8].axis("off")

            f, axarr = plt.subplots(4, 8)

            for i in range(img.shape[0]):
                gen_img = np.transpose(
                    img[i].detach().cpu().numpy(), (1, 2, 0))

                axarr[int(i/8)][i % 8].imshow(gen_img)
                axarr[int(i/8)][i % 8].axis("off")

            plt.show()

    tracker.save()

    def train(self):
        for i in monit.loop(self.training_steps):
            self.step(i)

            if ((i+1) % self.log_generated_interval == 0):
                tracker.new_line()


def main():
    experiment.create(name="waifu-gan")

    configs = Configs()

    experiment.configs(configs, {
        'device.cuda_device': 0,
        'img_size': 64,
        'log_generated_interval': 200
    })

    configs.init(str(lab.get_data_path() / 'stylegan'))

    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    with (experiment.start()):
        configs.train()


def generate_new_image():
    configs = Configs()

    configs.init('D:\Documents\Code\Python\Waifu-GAN\data\img_align_celeba')

    configs.mapping_network.load_state_dict(
        torch.load("mapping_network.pth"))
    configs.generator.load_state_dict(torch.load("generator.pth"))

    img, w = configs.generate_images(configs.batch_size)

    f, axarr = plt.subplots(4, 8)

    for i in range(32):
        real_img = configs.dataset.__getitem__(i)
        real_img = np.transpose(real_img.detach().cpu().numpy(), (1, 2, 0))

        if (i == 0):
            print(real_img)

        axarr[int(i/8)][i % 8].imshow(real_img)
        axarr[int(i/8)][i % 8].axis("off")

    for i in range(img.shape[0]):
        gen_img = np.transpose(img[i].detach().cpu().numpy(), (1, 2, 0))

        if (i == 0):
            print(gen_img)

        axarr[int(i/8)][i % 8].imshow(gen_img)
        axarr[int(i/8)][i % 8].axis("off")

    plt.show()


if (__name__ == '__main__'):
    train = input("Train new model (y/n) : ")

    if (train == "y"):
        main()
    else:
        generate_new_image()
