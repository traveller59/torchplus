import torch
from torch.nn import functional as F


def group_norm(input,
               running_mean,
               running_var,
               weight=None,
               bias=None,
               use_input_stats=True,
               momentum=0.1,
               eps=1e-5,
               group=1):
    r"""Applies Group Normalization for each channel in each data sample in a
    batch.

    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
    :class:`~torch.nn.InstanceNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError(
            'Expected running_mean and running_var to be not None when use_input_stats=False'
        )
    N, C, *Ds = input.shape
    # N, C = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(N)
    if bias is not None:
        bias = bias.repeat(N)

    def _group_norm(input,
                    running_mean=None,
                    running_var=None,
                    weight=None,
                    bias=None,
                    use_input_stats=None,
                    momentum=None,
                    eps=None,
                    group=1):
        # Repeat stored stats and affine transform params if necessary
        G = group
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(N)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(N)

        # Apply group norm
        input_reshaped = input.contiguous().view(N, G, C // G, *Ds)
        input_reshaped = input_reshaped.view(1, N * G,
                                             *input_reshaped.size()[2:])

        out = F.batch_norm(
            input_reshaped,
            running_mean,
            running_var,
            weight=None,
            bias=None,
            training=use_input_stats,
            momentum=momentum,
            eps=eps)

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(
                running_mean.view(N, G).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(
                running_var.view(N, G).mean(0, keepdim=False))
        # out = out.view()
        param_shape = [1] * len(input.shape)
        param_shape[1] = C
        param_shape[0] = N

        ret = out.view(N, C, *input.size()[2:])
        return ret * weight.view(*param_shape) + bias.view(*param_shape)

    return _group_norm(
        input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
        group=group)

def one_hot(tensor, depth, dim=-1, on_value=1.0):
    tensor_onehot = torch.FloatTensor(*list(tensor.shape),
                                      depth).zero_().cuda()
    if isinstance(tensor, torch.autograd.Variable):
        tensor_onehot.scatter_(dim,
                               tensor.unsqueeze(dim).data, on_value).type(
                                   torch.cuda.FloatTensor)
        return torch.autograd.Variable(tensor_onehot)
    tensor_onehot.scatter_(dim,
                           tensor.unsqueeze(dim).data, on_value).type(
                               torch.cuda.FloatTensor)
    return tensor_onehot
