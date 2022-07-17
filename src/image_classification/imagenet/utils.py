"""
Utilities script containing all helper functions
"""


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# Define Smooth Loss
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Define KL divergence loss
class KL_div_Loss(nn.Module):
    """
    We use formulation of Hinton et. for KD loss.

    $T^2$ scaling is implemented to avoid gradient rescaling when using T!=1
    """

    def __init__(self, temperature):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(KL_div_Loss, self).__init__()
        self.temperature = temperature
        #print( "Setting temperature = {} for KD (Only Teacher)".format(self.temperature) )
        print( "Setting temperature = {} for KD".format(self.temperature) )


    def forward(self, y, teacher_scores):
        p = F.log_softmax(y / self.temperature, dim=1) # Hinton formulation
        
        #p = F.log_softmax(y, dim=1) # Muller et. al used this. 

        q = F.softmax(teacher_scores / self.temperature, dim=1)
        l_kl = F.kl_div(p, q, reduction='batchmean')
        return l_kl*(self.temperature**2) # $T^2$ scaling is important



def get_imagenet_trained_teacher_for_imagenet_student(args):
    """
    Get teacher trained on ImageNet.
    """
    weights_path, teacher_alpha = args.teacher_weights_path, args.teacher_alpha
    ckpt = torch.load(weights_path)

    teacher = models.__dict__[ckpt['arch']](pretrained=False)

    state_dict = ckpt['state_dict']
    remapped_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module')}
    teacher.load_state_dict(remapped_state_dict)
    
    print( "Loading {} ImageNet teacher with best top1/ top5 accuracy = {:.3f} / {:.3f}".format(
        ckpt['arch'], ckpt['best_acc1'], ckpt['best_acc5']
    ) )

    return teacher, ckpt['arch']