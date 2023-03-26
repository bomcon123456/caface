import torch
from torch import nn
from torch.autograd import Function


class GradientReversal(Function):
    """_summary_

    Args:
        Function (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.save_for_backward(x, coeff)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, coeff = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = coeff * grad_output.neg()
        return grad_input, None


revgrad = GradientReversal.apply


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, coeff):
        coeff = torch.tensor(coeff, requires_grad=False)
        
        return revgrad(x, coeff)
