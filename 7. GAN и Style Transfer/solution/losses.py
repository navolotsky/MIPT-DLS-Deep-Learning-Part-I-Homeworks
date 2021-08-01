import torch
from torch import nn


class AdversarialLoss(nn.Module):
    def forward(
            self,
            discriminator_output_on_fake,
            discriminator_output_on_real):
        probs_on_fake = torch.sigmoid(
            discriminator_output_on_fake).mean((-2, -1))
        probs_on_real = torch.sigmoid(
            discriminator_output_on_real).mean((-2, -1))
        gen_term = torch.mean((probs_on_fake - 1) ** 2)
        dis_term = torch.mean(probs_on_fake ** 2) + \
            torch.mean((probs_on_real - 1) ** 2)
        return gen_term, dis_term


class OneDirectionCycleConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.L1Loss(reduction='mean')

    def forward(self, input, reconstructed_input):
        loss = self._loss(input, reconstructed_input)
        return loss


class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one_direction_loss = OneDirectionCycleConsistencyLoss()

    def forward(self, input, reconstructed_input, target, reconstructed_target):
        return (
            self.one_direction_loss(input, reconstructed_input) +
            self.one_direction_loss(target, reconstructed_target))


class IdentityMappingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.L1Loss(reduction='mean')

    def forward(self, input, generated_from_input, target, generated_from_target):
        return (self._loss(input, generated_from_input) + self._loss(target, generated_from_target))


class CycleGANLoss(nn.Module):
    # I wanted to implement this in more intelligent way,
    # but this feature (detach for model weights) is not implemented so far:
    # https://discuss.pytorch.org/t/how-to-use-a-layer-with-gradient-but-without-weight-adjustment/86190

    """
    To use it correctly, you must manually freeze
    discriminators parameters when backpropagating generators losses
    and freeze generators parameters when backpropagating discriminators losses.
    """

    def __init__(
        self,
        cycle_consistency_weight=10,  # aka lambda
        include_identity_loss=False,  # used for some specific applications
        # results in given value * lambda:
        identity_loss_fraction_from_cycle_consistency_weight=0.5
    ):
        super().__init__()
        self.adversarial_loss = AdversarialLoss()
        self.cycle_consistency_loss = CycleConsistencyLoss()
        self.cyc_weight = cycle_consistency_weight
        self.include_identity_loss = include_identity_loss
        if include_identity_loss:
            self.identity_mapping_loss = IdentityMappingLoss()
        self.id_fraction_from_cyc_weight = identity_loss_fraction_from_cycle_consistency_weight

    def forward(
        self,
        input, target,
        generated_from_input, reconstructed_input,
        generated_from_target, reconstructed_target,
        forward_discriminator_output_on_generated_from_input,
        forward_discriminator_output_on_target,
        backward_discriminator_output_on_generated_from_target,
        backward_discriminator_output_on_input
    ):
        fgen_loss, fdis_loss = self.adversarial_loss(
            forward_discriminator_output_on_generated_from_input,
            forward_discriminator_output_on_target
        )
        bgen_loss, bdis_loss = self.adversarial_loss(
            backward_discriminator_output_on_generated_from_target,
            backward_discriminator_output_on_input
        )
        cyc_loss = self.cycle_consistency_loss(
            input, reconstructed_input,
            target, reconstructed_target
        )
        weighted_cyc_loss = self.cyc_weight * cyc_loss
        losses = [fgen_loss, fdis_loss, bgen_loss,
                  bdis_loss, weighted_cyc_loss]
        if self.include_identity_loss:
            id_loss = self.identity_mapping_loss(
                input, generated_from_input,
                target, generated_from_target
            )
            weighted_id_loss = self.id_fraction_from_cyc_weight * self.cyc_weight * id_loss
            losses.append(weighted_id_loss)
        return losses
