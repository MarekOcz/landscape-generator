import torch
# noinspection PyProtectedMember
from torch.autograd import grad
from torch.nn import Module
from torch.linalg import vector_norm
from torch import device, Tensor, mean, rand, ones_like, save, load, randn, inference_mode
from tqdm.auto import tqdm
from pathlib import Path
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn
from time import sleep


def gradient_penalty(critic: Module, real: Tensor, fake: Tensor, device: str | device) -> Tensor:
    batch_size, c_channels, height, width = real.shape
    epsilon = rand((batch_size, 1, 1, 1), device=device).repeat(1, c_channels, height,
                                                                width)  # Creates the proportion of fake images in the interpolated images
    mixed_images = real * epsilon + fake * (
            1 - epsilon)  # Creates the interpolated images that are a mixture of fake and real images
    mixed_scores = critic(mixed_images)
    gradient = grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view([gradient.shape[0], -1])
    return mean((vector_norm(gradient, dim=1) - 1) ** 2)


def save_gan_to_train(generator: Module, gen_optim: Optimizer, critic: Module, critic_optim: Optimizer,
                      epoch: int, batch_idx: int or None, save_path: Path):
    save({
        'generator': generator.state_dict(),
        'critic': critic.state_dict(),
        'generator_optimizer': gen_optim.state_dict(),
        'critic_optimizer': critic_optim.state_dict(),
        'epoch': epoch,
        'batch index': batch_idx
    }, save_path)


def load_gan_to_train(generator: Module, gen_optim: Optimizer, critic: Module, critic_optim: Optimizer,
                      load_path: Path):
    checkpoint = load(load_path)
    generator.load_state_dict(checkpoint['generator'])
    critic.load_state_dict(checkpoint['critic'])
    gen_optim.load_state_dict(checkpoint['generator_optimizer'])
    critic_optim.load_state_dict(checkpoint['critic_optimizer'])
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch index']
    return epoch, batch_idx


def wgan_gp_train_loop(generator: Module, gen_optim: Optimizer, critic: Module, critic_optim: Optimizer, epochs: int,
                       train_data: DataLoader, critic_iterations: int, device: str | device, z_dim: int = 100,
                       lda: int = 10, model_save_path: Path = None, load_path: Path = None, epoch_start: int = 0,
                       start_batch_idx: int = None, save_interval: int = 10, writer: SummaryWriter = None, start_from_load=True, step_interval: int  =10):
    if load_path is not None:
        if start_from_load:
            epoch_start, start_batch_idx = load_gan_to_train(generator, gen_optim, critic, critic_optim, load_path)
        else:
            load_gan_to_train(generator, gen_optim, critic, critic_optim, load_path)

    fixed_noise = randn(32, z_dim, 1, 1, device=device)
    step = 0
    num_iterations = len(train_data)
    non_blocking = torch.cuda.is_available()
    for epoch in range(epoch_start, epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        sleep(0.1)
        train_loop = tqdm(enumerate(train_data), total=num_iterations)
        # noinspection PyTypeChecker
        for batch_idx, (real, _) in train_loop:
            if start_batch_idx is not None:
                if batch_idx == start_batch_idx:
                    start_batch_idx = None
                    continue
                else:
                    continue
            real = real.to(device, non_blocking=non_blocking)
            generator.train()
            critic.train()
            batch_size = real.shape[0]
            for _ in range(critic_iterations):  # Critic training loop
                noise = randn(batch_size, z_dim, 1, 1, device=device)
                fake = generator(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)  # Calculating and flattening the critic scores
                grad_pen = gradient_penalty(critic, real, fake, device)
                critic_loss = (-(mean(critic_real) - mean(critic_fake)) + lda * grad_pen)
                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_optim.step()

            output = critic(fake).reshape(-1)
            generator_loss = -(mean(output))
            generator.zero_grad()
            generator_loss.backward()
            gen_optim.step()
            train_loop.set_description(f"Gen: {generator_loss:.3f} Critic: {critic_loss:.3f}")
            if batch_idx % save_interval == 0:
                generator.eval()
                with inference_mode():
                    fake = generator(fixed_noise)
                    img_grid = make_grid(fake, normalize=True)
                    writer.add_image("Generated images", img_grid, global_step=step)
                    save_gan_to_train(generator, gen_optim, critic, critic_optim, epoch, batch_idx, model_save_path)
                    step += 1


def init_weights(model: Module, mean: float = 0.0, standard_deviation: float = 0.02):
    for layer in model.modules():  # Iterates through the layers
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight.data, mean=mean, std=standard_deviation)


class Critic(Module):
    def __init__(self, img_channels: int, features: int):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(  # input 128*128
            nn.Conv2d(img_channels, features, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),  # 64*64
            self.block(features, features * 2, 4, 2, 1),  # 32*32
            self.block(features * 2, features * 4, 4, 2, 1),  # 16*16
            self.block(features * 4, features * 8, 4, 2, 1),  # 8 * 8
            self.block(features * 8, features * 16, 4, 2, 1),  # 4 * 4
            nn.Conv2d(features * 16, 1, 4),  # 1 * 1 * 1
            nn.Sigmoid()  # Outputs between 0 and 1
        )

    def block(self, in_features, out_features, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=True),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.critic(x)


class Generator(Module):
    def __init__(self, z_dim: int, img_channels: int, features: int):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(  # input 128*128
            self.block(z_dim, features * 32, 4, 1, 0),
            self.block(features * 32, features * 16, 4, 2, 1),
            self.block(features * 16, features * 8, 4, 2, 1),
            self.block(features * 8, features * 4, 4, 2, 1),
            self.block(features * 4, features * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features * 2, img_channels, kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.Tanh()  # Outputs between -1 and 1
        )

    def block(self, in_features, out_features, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)
