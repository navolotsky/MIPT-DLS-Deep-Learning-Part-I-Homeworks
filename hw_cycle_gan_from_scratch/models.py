from torch import nn


class CycleGANGenerator(nn.Module):
    def __init__(self, res_num=9):
        super().__init__()
        # c7s1-64
        # Using bias before normalization is meaningless,
        # but authors of the reference implementation do:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L551
        # Thanks god, I am not the only one who noticed this:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/981
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=1, padding=3, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # d128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # d256
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        # R256 x res_num
        self.residuals = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(num_features=256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(num_features=256)
            )
            for _ in range(res_num)
        )
        # u128
        self.conv4 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4_norm_act = nn.Sequential(
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        # u64
        self.conv5 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv5_norm_act = nn.Sequential(
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        # c7s1-3
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7,
                      stride=1, padding=3, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.conv1(input)
        conv1_output_size = output.size()
        output = self.conv2(output)
        conv2_output_size = output.size()
        output = self.conv3(output)
        for residual in self.residuals:
            output = residual(output) + output
        output = self.conv4(output, output_size=conv2_output_size)
        output = self.conv4_norm_act(output)
        output = self.conv5(output, output_size=conv1_output_size)
        output = self.conv5_norm_act(output)
        output = self.conv6(output)
        return output


class CycleGANDiscriminator(nn.Module):  # PatchGAN 70x70
    def __init__(self):
        super().__init__()
        # C64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # C128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # C256
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # C512
        # Stride for this layer and the last one is 1:
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/162
        # This is missed in the paper and they haven't still fixed this.
        # I was damn confused by this missing when trying to implement from scratch
        # basing just on the original paper with no any look in the reference implementation.
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.output = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.output(output)
        return output


class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_generator = CycleGANGenerator()
        self.backward_generator = CycleGANGenerator()
        self.forward_discriminator = CycleGANDiscriminator()
        self.backward_discriminator = CycleGANDiscriminator()

    def forward(self, input, target):
        generated_from_input = self.forward_generator(input)
        reconstructed_input = self.backward_generator(generated_from_input)
        generated_from_target = self.backward_generator(target)
        reconstructed_target = self.forward_generator(generated_from_target)
        forward_discriminator_output_on_generated_from_input = self.forward_discriminator(
            generated_from_input)
        forward_discriminator_output_on_target = self.forward_discriminator(
            target)
        backward_discriminator_output_on_generated_from_target = self.backward_discriminator(
            generated_from_target)
        backward_discriminator_output_on_input = self.backward_discriminator(
            input)
        return (generated_from_input, reconstructed_input,
                generated_from_target, reconstructed_target,
                forward_discriminator_output_on_generated_from_input,
                forward_discriminator_output_on_target,
                backward_discriminator_output_on_generated_from_target,
                backward_discriminator_output_on_input)
