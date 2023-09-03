
if __name__ == '__main__':
    from code import loading_data
    from code.training import *
    from code.utils import diagnostics
    import torch
    from torch.optim import Adam
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    # HYPERPARAMETERS
    num_features = 48
    lambda_gp = 10
    lr_critic = 5e-5
    lr_generator = 5e-5
    critic_it = 5
    z_dim = 100

    # FOLDERS
    dataset_folder = Path("")
    load_model_path = Path("")
    save_model_path = Path("")

    torch.manual_seed(21)
    torch.cuda.manual_seed(21)


    device = str(diagnostics())
    train_data = loading_data.load_dataset(dataset_folder, batch_size=32,
                                           num_workers=2, device=device, pin_memory=True)
    generator = Generator(z_dim=z_dim, features=num_features, img_channels=3).to(device)
    critic = Critic(img_channels=3, features=num_features).to(device)
    init_weights(critic)
    init_weights(generator)

    optim_gen = Adam(params=generator.parameters(), betas=(0.5, 0.9), lr=lr_generator)
    optim_critic = Adam(params=critic.parameters(), betas=(0.5, 0.9), lr=lr_critic)
    writer = SummaryWriter("")

    wgan_gp_train_loop(generator, optim_gen, critic, optim_critic, 20, train_data, critic_it, device, z_dim, lambda_gp,
                       save_model_path, writer=writer, load_path=load_model_path, start_from_load=True)
