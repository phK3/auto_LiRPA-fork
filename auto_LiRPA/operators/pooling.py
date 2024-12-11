#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Pooling operators."""
from collections import OrderedDict
from .base import *
from .activation_base import BoundOptimizableActivation
import numpy as np
from .solver_utils import grb
from .gurobi_maxpool_lp import compute_maxpool_bias


class BoundMaxPool(BoundOptimizableActivation):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])

        self.requires_input_bounds = [0]
        self.kernel_size = attr['kernel_shape']
        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.use_default_ibp = True
        self.alpha = {}
        self.init = {}
        # use this attribute to select the relaxation mode
        # 'original_abcrown' - the original code in this file
        # 'deeppoly' - my refactorization of the original code
        # 'xiao2024' - xiao2024 relaxation (own implementation using formula in the paper)
        # 'xiao2024_original' - xiao2024 relaxation (using their implementation for auto_LiRPA -> https://github.com/xiaoyuanpigo/maxlin/blob/15909427c2ac643010124604e78f703b915e0c72/auto_lirpa_maxlin.py#L230)
        # 'gurobi_lp' - optimizable upper bound via differentiable LP
        self.relax_mode = options['maxpool_relaxation']

    def forward(self, x):
        output, _ = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                                 return_indices=True, ceil_mode=self.ceil_mode)
        return output
    
    def deeppoly_lower(self, ls, us, shape):
        max_lower, max_lower_index = F.max_pool2d(ls, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)
        
        lower_b = torch.zeros(ls.shape[0], *self.output_shape[1:], device=ls.device)
        
        lower_d = torch.zeros(shape, device=ls.device)
        # set \hat{z} >= z_i where i is the input element with largest lower bound.
        lower_d = torch.scatter(lower_d.flatten(-2), -1,
                                max_lower_index.flatten(-2), 1.0)
        lower_d = lower_d.view(shape)
        
        return lower_d, lower_b
    
    def deeppoly_upper(self, ls, us, shape):
        max_lower, max_lower_index = F.max_pool2d(ls, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)
        max_upper, max_upper_index = F.max_pool2d(us, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)

        # find largest upper bound that does not belong to the same entry as max_lower (respecting stride and padding)
        padding = tuple((self.padding[0], self.padding[0], self.padding[1], self.padding[1]))
        remaining_upper = torch.scatter(F.pad(us, padding).flatten(-2), -1, max_lower_index.flatten(-2), -torch.inf).view(us.shape)
        max_upper_remaining, max_upper_index_remaining = F.max_pool2d(remaining_upper, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)

        values = torch.zeros_like(max_lower, device=us.device)
        values[max_lower >= max_upper_remaining] = 1.

        upper_d = torch.zeros(shape, device=us.device)
        upper_d = torch.scatter(upper_d.flatten(-2), -1, max_lower_index.flatten(-2), values.flatten(-2)).view(shape)
        
        # when we are fixed (i.e. l_i >= second largest upper bound, then bias = 0, else bias = u_i)
        upper_b = torch.where(max_lower >= max_upper_remaining, 0., max_upper)
        upper_b = upper_b.view(shape[0], *self.output_shape[1:])
        
        return upper_d, upper_b

    def xiao2024_lower(self, ls, us, shape):
        ms = 0.5 * (ls + us)

        mval, mind = F.max_pool2d(ms, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)
        
        # lower coefficients
        lower_d = torch.zeros(shape, device=ls.device)
        lower_d = torch.scatter(lower_d.flatten(-2), -1, mind.flatten(-2), 1)
        lower_d = lower_d.view(shape)
        
        lower_b = torch.zeros(ls.shape[0], *self.output_shape[1:], device=ls.device)
        
        return lower_d, lower_b
    
    def xiao2024_upper(self, ls, us, shape):       
        lval, lind = F.max_pool2d(ls, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)

        # find largest, second largest and third largest concrete upper bound (while respecting stride and padding)
        padding = tuple((self.padding[0], self.padding[0], self.padding[1], self.padding[1]))
        # find largest entry
        u1, u1ind = F.max_pool2d(us, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode, return_indices=True)
        # add padding to us and set largest entry to -inf so it is not picked again
        us2 = torch.scatter(F.pad(us, padding).flatten(-2), -1, u1ind.flatten(-2), -torch.inf).view(us.shape)
        # find 2nd-largest entry (but without padding, since we explicitly added the padding above)
        u2, u2ind = F.max_pool2d(us2, self.kernel_size, self.stride, 0, ceil_mode=self.ceil_mode, return_indices=True)

        l_i = ls.flatten(-2).gather(-1, u1ind.flatten(-2)).view(u1.shape)

        ub_slope = (u1 - u2) / (u1 - l_i)

        # bias terms
        upper_b = -l_i * ub_slope + u2
        upper_b = torch.where((lval >= u2) & (l_i >= lval), 0., upper_b)
        upper_b = upper_b.view(us.shape[0], *self.output_shape[1:])

        # coefficients
        # need .clone(), otherwise values[..] = 1. would be an inplace operation which is bad for autodiff (see https://discuss.pytorch.org/t/what-is-in-place-operation/16244/15)
        values = ub_slope.clone()  
        # if fixed, then slope is 1
        values[lval >= u2] = 1.

        upper_d = torch.zeros(shape, device=us.device)
        upper_d = torch.scatter(upper_d.flatten(-2), -1, u1ind.flatten(-2), values.flatten(-2)).view(shape)       

        return upper_d, upper_b
    
    def xiao2024_lower_original(self, ls, us, shape):
        middle=(ls + us)/2
        max_m, max_m_index = F.max_pool2d(
            middle, self.kernel_size, self.stride, self.padding,
            return_indices=True, ceil_mode=self.ceil_mode)
        
        #lower_bound
        lower_d = torch.zeros((shape), device=ls.device)
        lower_d = torch.scatter(torch.flatten(lower_d, -2), -1,
                                torch.flatten(max_m_index, -2),
                                1.0).view(shape)
        if self.padding[0] > 0 or self.padding[1] > 0:
            lower_d = lower_d[...,self.padding[0]:-self.padding[0],
                            self.padding[1]:-self.padding[1]]
            
        lower_b = torch.zeros(shape[0], *self.output_shape[1:], device=ls.device)

        return lower_d, lower_b

    def xiao2024_upper_original(self, ls, us, shape):
        upper_d = torch.zeros(shape, device=us.device)
        upper_b = torch.zeros(shape[0], *self.output_shape[1:], device=us.device)

        max_lower, max_lower_index = F.max_pool2d(
            ls, self.kernel_size, self.stride, self.padding,
            return_indices=True, ceil_mode=self.ceil_mode)

        max_upper, max_upper_index = F.max_pool2d(
            us, self.kernel_size, self.stride, self.padding,
            return_indices=True, ceil_mode=self.ceil_mode)
        
        paddings = tuple((self.padding[0], self.padding[0], self.padding[1], self.padding[1]))
        if paddings == (0,0,0,0):
            delete_upper = torch.scatter(
                torch.flatten(us, -2), -1,
                torch.flatten(max_upper_index, -2), -torch.inf).view(upper_d.shape)
        else:
            delete_upper = torch.scatter(
                torch.flatten(F.pad(us, paddings), -2), -1,
                torch.flatten(max_upper_index, -2),
                -torch.inf).view(upper_d.shape)

        # Find the the second max upper bound 
        max_upper2, max_upper2_index = F.max_pool2d(
            delete_upper, self.kernel_size, self.stride, 0,
            return_indices=True, ceil_mode=self.ceil_mode)
        

        if paddings == (0,0,0,0):
            scatter_ = torch.scatter(
                torch.flatten(torch.zeros(shape, device=us.device), -2), -1,
                torch.flatten(max_upper_index, -2),
                1.0).view(upper_d.shape)
            max_upper_lower= torch.where(scatter_!=0,ls, -torch.inf)
            scatter_deviation = torch.where(scatter_!=0,us-ls,0)
            scatter_deviation2 = torch.where(scatter_!=0,1.0/(us - ls),0)
        else:
            scatter_ = torch.scatter(
                torch.flatten(F.pad(torch.zeros(shape, device=us.device), paddings), -2), -1,
                torch.flatten(max_upper_index, -2),
                1.0).view(upper_d.shape)
            max_upper_lower=torch.where(scatter_!=0,ls, -torch.inf)
            scatter_deviation = torch.where(scatter_!=0,us - ls,0)
            scatter_deviation2 = torch.where(scatter_!=0,1.0/(us - ls),0)
        deviation,_=F.max_pool2d(
            scatter_deviation, self.kernel_size, self.stride, 0,
            return_indices=True, ceil_mode=self.ceil_mode)
        max_upper_lower,_=F.max_pool2d(
            max_upper_lower, self.kernel_size, self.stride, 0,
            return_indices=True, ceil_mode=self.ceil_mode)

        deviation2,_=F.max_pool2d(
            scatter_deviation2, self.kernel_size, self.stride, 0,
            return_indices=True, ceil_mode=self.ceil_mode)

        values = torch.zeros_like(max_lower)
        values[max_lower >= max_upper2] = 1.0
        temp=(max_upper-max_upper2)/deviation
        values[max_lower < max_upper2]=temp[max_lower < max_upper2]
        values[max_upper_lower < max_lower]=temp[max_upper_lower < max_lower]
        
        upper_d = torch.scatter(
            torch.flatten(upper_d, -2), -1,
            torch.flatten(max_upper_index, -2),
            torch.flatten(values, -2)).view(upper_d.shape)

        b=-(max_upper-max_upper2)*deviation2*max_upper_lower+max_upper2#-max_lower
        upper_b[max_upper2 > max_lower]=b[max_upper2 > max_lower]  
        upper_b[max_upper_lower < max_lower]=b[max_upper_lower < max_lower]

        return upper_d, upper_b


    def project_simplex(self, patches):
        sorted = torch.flatten(patches, -2)
        sorted, _ = torch.sort(sorted, -1, descending=True)
        rho_sum = torch.cumsum(sorted, -1)
        rho_value = 1 - rho_sum
        rho_value = (sorted + rho_value/torch.tensor(
            range(1, sorted.size(-1)+1), dtype=torch.float,
            device=sorted.device)) > 0
        _, rho_index = torch.max(torch.cumsum(rho_value, -1), -1)
        rho_sum = torch.gather(rho_sum, -1, rho_index.unsqueeze(-1)).squeeze(-1)
        lbd = 1/(rho_index+1)* (1-rho_sum)

        return torch.clamp(patches + lbd.unsqueeze(-1).unsqueeze(-1), min=0)

    def _init_opt_parameters_impl(self, size_spec, name_start):
        if name_start == '_forward':
            warnings.warn("MaxPool's optimization is not supported for forward mode")
            return None

        ref = self.inputs[0].lower # a reference variable for getting the shape

        if self.relax_mode == 'gurobi_lp':
            # lower and upper relaxation are optimizable
            n_params = 2
        else:
            # originally, only the lower relaxation is optimized
            n_params = 1

        alpha = torch.empty(
            [n_params, size_spec, self.input_shape[0], self.input_shape[1],
            self.output_shape[-2], self.output_shape[-1],
            self.kernel_size[0], self.kernel_size[1]],
            dtype=torch.float, device=ref.device, requires_grad=True)
        self.init[name_start] = False
        return alpha

    @staticmethod
    @torch.jit.script
    def jit_mutiply(Apos, Aneg, pos, neg):
        return pos.contiguous() * Apos + neg.contiguous() * Aneg

    def bound_backward(self, last_lA, last_uA, x, start_node=None,
                       unstable_idx=None, **kwargs):
        # self.padding is a tuple of two elements: (height dimension padding, width dimension padding).
        paddings = tuple((self.padding[0], self.padding[0], self.padding[1], self.padding[1]))

        if self.stride[0] != self.kernel_size[0]:
            raise ValueError("self.stride ({}) != self.kernel_size ({})".format(self.stride, self.kernel_size))

        shape = self.input_shape
        batch_size = x.lower.shape[0]
        shape = list(shape[:-2]) + [a + 2*b for a, b in zip(self.input_shape[-2:], self.padding)]
        shape[0] = batch_size

        if self.relax_mode == 'original_abcrown':
            # Lower and upper D matrices. They have size (batch_size, input_c, x, y) which will be multiplied on enlarges the A matrices via F.interpolate.
            upper_d = torch.zeros(shape, device=x.device)
            lower_d = None

            # Size of upper_b and lower_b: (batch_size, output_c, h, w).
            upper_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)
            lower_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)

            # Find the maxpool neuron whose input bounds satisfy l_i > max_j u_j for all j != i. In this case, the maxpool neuron is linear, and we can set upper_d = lower_d = 1.
            # We first find which indices has the largest lower bound.
            max_lower, max_lower_index = F.max_pool2d(
                x.lower, self.kernel_size, self.stride, self.padding,
                return_indices=True, ceil_mode=self.ceil_mode)
            # Set the upper bound of the i-th input to -inf so it will not be selected as the max.

            if paddings == (0,0,0,0):
                delete_upper = torch.scatter(
                    torch.flatten(x.upper, -2), -1,
                    torch.flatten(max_lower_index, -2), -torch.inf).view(upper_d.shape)
            else:
                delete_upper = torch.scatter(
                    torch.flatten(F.pad(x.upper, paddings), -2), -1,
                    torch.flatten(max_lower_index, -2),
                    -torch.inf).view(upper_d.shape)
            # Find the the max upper bound over the remaining ones.
            max_upper, _ = F.max_pool2d(
                delete_upper, self.kernel_size, self.stride, 0,
                return_indices=True, ceil_mode=self.ceil_mode)

            # The upper bound slope for maxpool is either 1 on input satisfies l_i > max_j u_j (linear), or 0 everywhere. Upper bound is not optimized.
            values = torch.zeros_like(max_lower)
            values[max_lower >= max_upper] = 1.0
            upper_d = torch.scatter(
                torch.flatten(upper_d, -2), -1,
                torch.flatten(max_lower_index, -2),
                torch.flatten(values, -2)).view(upper_d.shape)
            

            # can we just put that here?
            # For the upper bound, we set the bias term to concrete upper bounds for maxpool neurons that are not linear.
            max_upper_, _ = F.max_pool2d(x.upper, self.kernel_size, self.stride,
                                        self.padding, return_indices=True,
                                        ceil_mode=self.ceil_mode)
            upper_b[max_upper > max_lower] = max_upper_[max_upper > max_lower]

        elif self.relax_mode == 'deeppoly':
            # need to define lower_b here becaues this branch is executed every time, while below, we only execute self.deeppoly_lower
            # at initialization and use the values stored in alpha in the following iterations (so no lower_b will be available)
            # TODO: just use torch.zeros() instead of calling self.deeppoly_lower to get the lower bias (it is always zero anyways)?
            #_, lower_b = self.deeppoly_lower(x.lower, x.upper, self.output_shape)
            lower_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)
            upper_d, upper_b = self.deeppoly_upper(x.lower, x.upper, shape)
        elif self.relax_mode == 'xiao2024':
            # see above in self.relax_mode == 'deeppoly' why we need to define lower_b here
            lower_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)
            upper_d, upper_b = self.xiao2024_upper(x.lower, x.upper, shape)
        elif self.relax_mode == 'xiao2024_original':
            lower_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)
            upper_d, upper_b = self.xiao2024_upper_original(x.lower, x.upper, shape)
        elif self.relax_mode == 'gurobi_lp':
            # still need lower bound of zero, just as in deeppoly or original_abcrown case, since we use their relaxation for the lower bound
            lower_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)
        else:
            raise ValueError(f"self.relax_mode = {self.relax_mode} not supported! Choose from original_abcrown, deeppoly, xiao2024, xiao2024_original, gurobi_lp")


        if self.opt_stage == 'opt':
            if unstable_idx is not None and self.alpha[start_node.name].size(1) != 1:
                if isinstance(unstable_idx, tuple):
                    raise NotImplementedError('Please use --conv_mode matrix')
                elif unstable_idx.ndim == 1:
                    # Only unstable neurons of the start_node neurons are used.
                    if self.relax_mode == 'gurobi_lp':
                        # use self.alpha[][:1] instead of self.alpha[][0] to retain ndims in shape
                        alpha = self.non_deter_index_select(
                            self.alpha[start_node.name][:1], index=unstable_idx, dim=1)
                        alpha_u = self.non_deter_index_select(
                            self.alpha[start_node.name][1:], index=unstable_idx, dim=1)
                    else:
                        alpha = self.non_deter_index_select(
                            self.alpha[start_node.name], index=unstable_idx, dim=1)
                elif unstable_idx.ndim == 2:
                    # Each element in the batch selects different neurons.
                    if self.relax_mode == 'gurobi_lp':
                        alpha = batched_index_select(
                            self.alpha[start_node.name][:1], index=unstable_idx, dim=1)
                        alpha_u = batched_index_select(
                            self.alpha[start_node.name][1:], index=unstable_idx, dim=1)
                    else:
                        alpha = batched_index_select(
                            self.alpha[start_node.name], index=unstable_idx, dim=1)
                else:
                    raise ValueError
            else:
                if self.relax_mode == 'gurobi_lp':
                    # TODO: rename all alpha to alpha_l
                    alpha = self.alpha[start_node.name][:1]
                    alpha_u = self.alpha[start_node.name][1:]
                else:
                    alpha = self.alpha[start_node.name]

            if not self.init[start_node.name]:
                if self.relax_mode == 'original_abcrown':
                    lower_d = torch.zeros((shape), device=x.device)
                    # [batch, C, H, W]
                    lower_d = torch.scatter(
                        torch.flatten(lower_d, -2), -1,
                        torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
                elif self.relax_mode == 'deeppoly':
                    lower_d, lower_b = self.deeppoly_lower(x.lower, x.upper, shape)
                elif self.relax_mode == 'xiao2024':
                    lower_d, lower_b = self.xiao2024_lower(x.lower, x.upper, shape)
                elif self.relax_mode == 'xiao2024_original':
                    lower_d, lower_b = self.xiao2024_lower_original(x.lower, x.upper, shape)
                elif self.relax_mode == 'gurobi_lp':
                    # use some method for initialization
                    lower_d, lower_b = self.xiao2024_lower_original(x.lower, x.upper, shape)
                    upper_d, upper_b = self.xiao2024_upper_original(x.lower, x.upper, shape)
                else:
                    raise ValueError(f"self.relax_mode = {self.relax_mode} not supported! Choose from original_abcrown, deeppoly, xiao2024, xiao2024_original, gurobi_lp")
                
                # shape [batch, C*k*k, L]
                lower_d_unfold = F.unfold(
                    lower_d, self.kernel_size, 1, stride=self.stride)

                # [batch, C, k, k, out_H, out_W]
                alpha_data = lower_d_unfold.view(
                    lower_d.shape[0], lower_d.shape[1], self.kernel_size[0],
                    self.kernel_size[1], self.output_shape[-2], self.output_shape[-1])

                # [batch, C, out_H, out_W, k, k]
                alpha.data.copy_(alpha_data.permute((0,1,4,5,2,3)).clone().detach())
                # In optimization mode, we use the same lower_d once builded.
                if self.padding[0] > 0 or self.padding[1] > 0:
                    lower_d = lower_d[...,self.padding[0]:-self.padding[0],
                                      self.padding[1]:-self.padding[1]]
                    
                if self.relax_mode == 'gurobi_lp':
                    # now also need to init upper relaxation
                    # just copy what abCROWN people did for lower relaxation

                    # shape [batch, C*k*k, L]
                    upper_d_unfold = F.unfold(
                        upper_d, self.kernel_size, 1, stride=self.stride)

                    # [batch, C, k, k, out_H, out_W]
                    alpha_u_data = upper_d_unfold.view(
                        upper_d.shape[0], upper_d.shape[1], self.kernel_size[0],
                        self.kernel_size[1], self.output_shape[-2], self.output_shape[-1])

                    # [batch, C, out_H, out_W, k, k]
                    alpha_u.data.copy_(alpha_u_data.permute((0,1,4,5,2,3)).clone().detach())
                    # In optimization mode, we use the same upper_d once builded.
                    if self.padding[0] > 0 or self.padding[1] > 0:
                        upper_d = upper_d[...,self.padding[0]:-self.padding[0],
                                        self.padding[1]:-self.padding[1]]
                        
                self.init[start_node.name] = True

            # The lower bound coefficients must be positive and projected to an unit simplex.
            alpha.data = self.project_simplex(alpha.data).clone().detach()  # TODO: don't do this, never re-assign the .data property. Use copy_ instead.
            # permute the last 6 dimensions of alpha to [batch, C, k, k, out_H, out_W], which prepares for the unfold operation.
            alpha = alpha.permute((0,1,2,3,6,7,4,5))
            alpha_shape = alpha.shape
            alpha = alpha.reshape((alpha_shape[0]*alpha_shape[1]*alpha_shape[2],
                                   -1, alpha_shape[-2]*alpha_shape[-1]))
            lower_d = F.fold(alpha, self.input_shape[-2:], self.kernel_size, 1,
                             self.padding, self.stride)
            lower_d = lower_d.view(alpha_shape[0], alpha_shape[1],
                                   alpha_shape[2], *lower_d.shape[1:])
            lower_d = lower_d.squeeze(0)

            if self.relax_mode == 'gurobi_lp':
                # we don't need to change alpha_u, as all we do is compute the bias in GurobiLP
                # permute the last 6 dimensions of alpha to [batch, C, k, k, out_H, out_W], which prepares for the unfold operation.
                alpha_u = alpha_u.permute((0,1,2,3,6,7,4,5))
                alpha_u_shape = alpha_u.shape
                alpha_u = alpha_u.reshape((alpha_u_shape[0]*alpha_u_shape[1]*alpha_u_shape[2],
                                    -1, alpha_u_shape[-2]*alpha_u_shape[-1]))
                upper_d = F.fold(alpha_u, self.input_shape[-2:], self.kernel_size, 1,
                                self.padding, self.stride)
                upper_d = upper_d.view(alpha_u_shape[0], alpha_u_shape[1],
                                    alpha_u_shape[2], *upper_d.shape[1:])
                upper_d = upper_d.squeeze(0)

                # TODO: check if correct - do we only need to look at unfolded lower and upper bounds?
                # we only look at unfolded bounds: 
                # - if we have 3x3 input and kernel size 2x2 and no padding, we only look at the first 2x2 inputs
                #   the other inputs are irrelevant
                # - just execute a = torch.arange(18, dtype=torch.float32), F.unfold(a, (2, 2), 1, stride=2) to check
                #upper_b = compute_maxpool_bias(x.lower, x.upper, upper_d)
                # shape [batch, C*k*k, L]
                lower_unfold = F.unfold(x.lower, self.kernel_size, 1, stride=self.stride)
                upper_unfold = F.unfold(x.upper, self.kernel_size, 1, stride=self.stride)
                # upper_d also has spec_size dimension 
                # reshape to (spec*batch, out_channels, in[0], in[1])
                upper_d_reshape = upper_d.view(upper_d.shape[0]*upper_d.shape[1], -1, upper_d.shape[-2], upper_d.shape[-1])
                upper_d_unfold = F.unfold(upper_d_reshape, self.kernel_size, 1, stride=self.stride)
                # if we want to use compute_maxpool_bias, we need one set of lower and upper bounds for each set of coefficients
                # but if the coefficients tensor has an additional spec-dim (and the lower and upper bounds have not),
                # then we need to repeat the bounds tensors to match the number of elements
                lower_unfold = lower_unfold.repeat(upper_d_unfold.shape[0], *[1] * (lower_unfold.dim() - 1))
                upper_unfold = upper_unfold.repeat(upper_d_unfold.shape[0], *[1] * (upper_unfold.dim() - 1))
                lower_unfold = lower_unfold.view(batch_size, -1, self.kernel_size[0], self.kernel_size[1]) 
                upper_unfold = upper_unfold.view(batch_size, -1, self.kernel_size[0], self.kernel_size[1]) 
                upper_d_unfold = upper_d_unfold.view(batch_size, -1, self.kernel_size[0], self.kernel_size[1])               
                upper_b = compute_maxpool_bias(lower_unfold, upper_unfold, upper_d_unfold)
                # reshape to (spec_size, batch_size, ...)
                upper_b = upper_b.view(upper_d.shape[0], batch_size, *self.output_shape[1:])

        else:
            if self.relax_mode == 'original_abcrown':
                lower_d = torch.zeros((shape), device=x.device)
                # [batch, C, H, W]
                lower_d = torch.scatter(
                    torch.flatten(lower_d, -2), -1,
                    torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
            elif self.relax_mode == 'deeppoly':
                lower_d, lower_b = self.deeppoly_lower(x.lower, x.upper, shape)
            elif self.relax_mode == 'xiao2024':
                lower_d, lower_b = self.xiao2024_lower(x.lower, x.upper, shape)
            elif self.relax_mode == 'xiao2024_original':
                lower_d, lower_b = self.xiao2024_lower_original(x.lower, x.upper, shape)
            elif self.relax_mode == 'gurobi_lp':
                # why is this executed even if we set alpha-crown and not only crown as method?
                # use some method for initialization
                lower_d, lower_b = self.xiao2024_lower_original(x.lower, x.upper, shape)
                upper_d, upper_b = self.xiao2024_upper_original(x.lower, x.upper, shape)
            else:
                raise ValueError(f"self.relax_mode = {self.relax_mode} not supported! Choose from original_abcrown, deeppoly, xiao2024, xiao2024_original")

            if self.padding[0] > 0 or self.padding[1] > 0:
                lower_d = lower_d[...,self.padding[0]:-self.padding[0],
                                  self.padding[1]:-self.padding[1]]

            # why only look for padding[0] in upper_d???    
            if self.padding[0] > 0:
                upper_d = upper_d[...,self.padding[0]:-self.padding[0],
                                  self.padding[0]:-self.padding[0]]


        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0

            bias = 0

            if isinstance(last_A, torch.Tensor):
                pos_A = last_A.clamp(min=0)
                neg_A = last_A.clamp(max=0)

                # TODO: introduces bugs in the original abCROWN, if patches are used
                #if b_pos is not None:
                #    # check if both the slopes and the bias have a spec dimension
                #    # if we use the original abCROWN overapproximation, they use the same biases
                #    # for each spec output, so the biases don't have a spec dimension
                #    if len(d_pos.shape) == 5 and len(b_pos.shape) == 5:
                #        # with our formulation, we can have different biases for the maxpool neuron for each spec
                #        # but still need spec x batch shape in the output bias
                #        bias = bias + torch.einsum('sb...,sb...->sb', pos_A, b_pos)
                #    else:
                #        # just use the bias computation that was there before otherwise
                #        bias = bias + self.get_bias(pos_A, b_pos)
                #if b_neg is not None:
                #    if len(d_neg.shape) == 5 and len(b_neg.shape) == 5:
                #        bias = bias + torch.einsum('sb...,sb...->sb', neg_A, b_neg)
                #    else:
                #        bias = bias + self.get_bias(neg_A, b_neg)

                if b_pos is not None:
                    # This is matrix mode, and padding is considered in the previous layers
                    bias = bias + self.get_bias(pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + self.get_bias(neg_A, b_neg)

                # Here we should comfirm that the maxpool patches are not overlapped.
                shape = last_A.size()

                padding = [self.padding[0], self.padding[0], self.padding[1], self.padding[1]]
                d_pos = F.pad(d_pos, padding)
                d_neg = F.pad(d_neg, padding)

                pos_A = F.interpolate(
                    pos_A.view(shape[0] * shape[1], *shape[2:]),
                    scale_factor=self.kernel_size)
                if d_pos.shape[-2] > pos_A.shape[-2] or d_pos.shape[-1] > pos_A.shape[-1]:
                    if not (d_pos.shape[-2] > pos_A.shape[-2] and d_pos.shape[-1] > pos_A.shape[-1]):
                        raise NotImplementedError(
                            "Asymmetric padding of maxpool not implemented.")
                    pos_A = F.pad(pos_A, (0, d_pos.shape[-2] - pos_A.shape[-2],
                                          0, d_pos.shape[-1] - pos_A.shape[-1]))
                else:
                    d_pos = F.pad(d_pos, (0, pos_A.shape[-2] - d_pos.shape[-2],
                                          0, pos_A.shape[-1] - d_pos.shape[-1]))
                pos_A = pos_A.view(shape[0], shape[1], *pos_A.shape[1:])

                neg_A = F.interpolate(neg_A.view(shape[0] * shape[1], *shape[2:]),
                                      scale_factor=self.kernel_size)
                if d_neg.shape[-2] > neg_A.shape[-2] or d_neg.shape[-1] > neg_A.shape[-1]:
                    if not (d_neg.shape[-2] > neg_A.shape[-2] and d_neg.shape[-1] > neg_A.shape[-1]):
                        raise NotImplementedError("Asymmetric padding of maxpool not implemented.")
                    neg_A = F.pad(neg_A, (0, d_neg.shape[-2] - neg_A.shape[-2],
                                          0, d_neg.shape[-1] - neg_A.shape[-1]))
                else:
                    d_neg = F.pad(d_neg, (0, neg_A.shape[-2] - d_neg.shape[-2],
                                          0, neg_A.shape[-1] - d_neg.shape[-1]))
                neg_A = neg_A.view(shape[0], shape[1], *neg_A.shape[1:])

                next_A = self.jit_mutiply(pos_A, neg_A, d_pos, d_neg)
                if self.padding[0] > 0 or self.padding[1] > 0:
                    next_A = next_A[...,self.padding[0]:-self.padding[0],
                                    self.padding[1]:-self.padding[1]]
            elif isinstance(last_A, Patches):
                # The last_A.patches was not padded, so we need to pad them here.
                # If this Conv layer is followed by a ReLU layer, then the padding was already handled there and there is no need to pad again.
                one_d = torch.ones(tuple(1 for i in self.output_shape[1:]),
                                   device=last_A.patches.device, dtype=last_A.patches.dtype).expand(self.output_shape[1:])
                # Add batch dimension.
                one_d = one_d.unsqueeze(0)
                # After unfolding, the shape is (1, out_h, out_w, in_c, h, w)
                one_d_unfolded = inplace_unfold(
                    one_d, kernel_size=last_A.patches.shape[-2:],
                    stride=last_A.stride, padding=last_A.padding,
                    inserted_zeros=last_A.inserted_zeros,
                    output_padding=last_A.output_padding)
                if last_A.unstable_idx is not None:
                    # Move out_h, out_w dimension to the front for easier selection.
                    one_d_unfolded_r = one_d_unfolded.permute(1, 2, 0, 3, 4, 5)
                    # for sparse patches the shape is (unstable_size, batch, in_c, h, w). Batch size is 1 so no need to select here.
                    one_d_unfolded_r = one_d_unfolded_r[
                        last_A.unstable_idx[1], last_A.unstable_idx[2]]
                else:
                    # Append the spec dimension.
                    one_d_unfolded_r = one_d_unfolded.unsqueeze(0)
                patches = last_A.patches * one_d_unfolded_r

                if b_pos is not None:
                    patch_pos = Patches(
                        patches.clamp(min=0), last_A.stride, last_A.padding,
                        last_A.shape, unstable_idx=last_A.unstable_idx,
                        output_shape=last_A.output_shape)
                    bias = bias + self.get_bias(patch_pos, b_pos)
                if b_neg is not None:
                    patch_neg = Patches(
                        patches.clamp(max=0), last_A.stride, last_A.padding,
                        last_A.shape, unstable_idx=last_A.unstable_idx,
                        output_shape=last_A.output_shape)
                    bias = bias + self.get_bias(patch_neg, b_neg)

                # bias = bias.transpose(0,1)
                shape = last_A.shape
                pos_A = last_A.patches.clamp(min=0)
                neg_A = last_A.patches.clamp(max=0)

                def upsample(last_patches, last_A):
                    if last_A.unstable_idx is None:
                        patches = F.interpolate(
                            last_patches.view(shape[0] * shape[1] * shape[2], *shape[3:]),
                            scale_factor=[1,]+self.kernel_size)
                        patches = patches.view(shape[0], shape[1], shape[2], *patches.shape[1:])
                    else:
                        patches = F.interpolate(
                            last_patches, scale_factor=[1,] + self.kernel_size)
                    return Patches(
                        patches, stride=last_A.stride, padding=last_A.padding,
                        shape=patches.shape, unstable_idx=last_A.unstable_idx,
                        output_shape=last_A.output_shape)

                pos_A = upsample(pos_A, last_A)
                neg_A = upsample(neg_A, last_A)

                padding, stride, output_padding = compute_patches_stride_padding(
                    self.input_shape, last_A.padding, last_A.stride, self.padding,
                    self.stride, last_A.inserted_zeros, last_A.output_padding)

                pos_A.padding, pos_A.stride, pos_A.output_padding = padding, stride, output_padding
                neg_A.padding, neg_A.stride, neg_A.output_padding = padding, stride, output_padding

                # unsqueeze for the spec dimension
                d_pos = maybe_unfold_patches(d_pos.unsqueeze(0), pos_A)
                d_neg = maybe_unfold_patches(d_neg.unsqueeze(0), neg_A)

                next_A_patches = self.jit_mutiply(
                    pos_A.patches, neg_A.patches, d_pos, d_neg)

                if start_node is not None:
                    self.patch_size[start_node.name] = next_A_patches.size()

                next_A = Patches(
                    next_A_patches, stride, padding, next_A_patches.shape,
                    unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape,
                    inserted_zeros=last_A.inserted_zeros, output_padding=output_padding)

            return next_A, bias

        uA, ubias = _bound_oneside(last_uA, upper_d, lower_d, upper_b, lower_b)
        lA, lbias = _bound_oneside(last_lA, lower_d, upper_d, lower_b, upper_b)

        return [(lA, uA)], lbias, ubias

    def bound_forward(self, dim_in, x):
        lower_d, lower_b, upper_d, upper_b = self.bound_relax(x, init=False)

        def _bound_oneside(w_pos, b_pos, w_neg, b_neg, d, b):
            d_pos, d_neg = d.clamp(min=0), d.clamp(max=0)
            w_new = d_pos.unsqueeze(1) * w_pos + d_neg.unsqueeze(1) * w_neg
            b_new = d_pos * b_pos + d_neg * b_neg
            if isinstance(self.kernel_size, list) and len(self.kernel_size) == 2:
                tot_kernel_size = prod(self.kernel_size)
            elif isinstance(self.kernel_size, int):
                tot_kernel_size = self.kernel_size ** 2
            else:
                raise ValueError(f'Unsupported kernel size {self.kernel_size}')
            w_pooled = (F.avg_pool2d(w_new.view(-1, *w_new.shape[2:]),
                self.kernel_size, self.stride, self.padding,
                ceil_mode=self.ceil_mode) * tot_kernel_size)
            w_pooled = w_pooled.reshape(w_new.shape[0], -1, *w_pooled.shape[1:])
            b_pooled = F.avg_pool2d(b_new, self.kernel_size, self.stride, self.padding,
                ceil_mode=self.ceil_mode) * tot_kernel_size + b
            return w_pooled, b_pooled

        lw, lb = _bound_oneside(x.lw, x.lb, x.uw, x.ub, lower_d, lower_b)
        uw, ub = _bound_oneside(x.uw, x.ub, x.lw, x.lb, upper_d, upper_b)

        return LinearBound(lw, lb, uw, ub)

    def bound_relax(self, x, init=False, dim_opt=None):
        if init:
            self.init_linear_relaxation(x, dim_opt)

        # Only used by forward mode
        paddings = tuple(self.padding + self.padding)
        self.upper, self.lower = x.upper, x.lower

        # A_shape = last_lA.shape if last_lA is not None else last_uA.shape
        # batch_size, input_c, x, y
        upper_d = torch.zeros_like(x.lower)
        lower_d = torch.zeros_like(x.lower)

        upper_d = F.pad(upper_d, paddings)
        lower_d = F.pad(lower_d, paddings)

        # batch_size, output_c, x, y
        upper_b = torch.zeros((list(self.output_shape))).to(x.lower)
        lower_b = torch.zeros((list(self.output_shape))).to(x.lower)

        # 1. find the index i where li > uj for all j, then set upper_d = lower_d = 1
        max_lower, max_lower_index = F.max_pool2d(x.lower, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        delete_upper = torch.scatter(torch.flatten(F.pad(x.upper, paddings), -2), -1, torch.flatten(max_lower_index, -2), -torch.inf).view(upper_d.shape)
        max_upper, _ = F.max_pool2d(delete_upper, self.kernel_size, self.stride, 0, return_indices=True, ceil_mode=self.ceil_mode)

        values = torch.zeros_like(max_lower)
        values[max_lower >= max_upper] = 1.0
        upper_d = torch.scatter(torch.flatten(upper_d, -2), -1, torch.flatten(max_lower_index, -2), torch.flatten(values, -2)).view(upper_d.shape)

        if self.opt_stage == 'opt':
            raise NotImplementedError
        else:
            lower_d = torch.scatter(torch.flatten(lower_d, -2), -1,
                                    torch.flatten(max_lower_index, -2),
                                    1.0).view(upper_d.shape)
            if self.padding[0] > 0:
                lower_d = lower_d[...,self.padding[0]:-self.padding[0],
                                  self.padding[0]:-self.padding[0]]

        values[:] = 0.0
        max_upper_, _ = F.max_pool2d(x.upper, self.kernel_size, self.stride,
                                     self.padding, return_indices=True,
                                     ceil_mode=self.ceil_mode)
        values[max_upper > max_lower] = max_upper_[max_upper > max_lower]
        upper_b = values

        if self.padding[0] > 0:
            upper_d = upper_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        return lower_d, lower_b, upper_d, upper_b

    def dump_optimized_params(self):
        ret = {'alpha': self.alpha}
        ret['init'] = self.init
        return ret

    def restore_optimized_params(self, alpha):
        self.alpha = alpha['alpha']
        self.init = alpha['init']

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,32,27,27)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,32,6,6)
        this_layer_shape = self.output_shape
        assert this_layer_shape[2] ==  ((2 * self.padding[0] + pre_layer_shape[2] - (self.stride[0] - 1))//self.stride[0])

        new_layer_gurobi_vars = []
        neuron_idx = 0
        pre_ubs = self.forward(self.inputs[0].upper).detach().cpu().numpy()

        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    a_sum = 0.0
                    v = model.addVar(lb=-float('inf'), ub=float('inf'),
                                            obj=0, vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{self.name}_{neuron_idx}')
                    for ker_row_idx in range(self.kernel_size[0]):
                        in_row_idx = -self.padding[0] + self.stride[0] * out_row_idx + ker_row_idx
                        if (in_row_idx < 0) or (in_row_idx == len(gvars_array[out_chan_idx][ker_row_idx])):
                            # This is padding -> value of 0
                            continue
                        for ker_col_idx in range(self.kernel_size[1]):
                            in_col_idx = -self.padding[1] + self.stride[1] * out_col_idx + ker_col_idx
                            if (in_col_idx < 0) or (in_col_idx == pre_layer_shape[3]):
                                # This is padding -> value of 0
                                continue
                            var = gvars_array[out_chan_idx][in_row_idx][in_col_idx]
                            a = model.addVar(vtype=grb.GRB.BINARY)
                            a_sum += a
                            model.addConstr(v >= var)
                            model.addConstr(v <= var + (1 - a) * pre_ubs[
                                0, out_chan_idx, out_row_idx, out_col_idx])
                    model.addConstr(a_sum == 1, name=f'lay{self.name}_{neuron_idx}_eq')
                    out_row_vars.append(v)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        model.update()



class BoundGlobalAveragePool(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        output = nn.AdaptiveAvgPool2d((1, 1)).forward(x)  # adaptiveAveragePool with output size (1, 1)
        return output

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        H, W = self.input_shape[-2], self.input_shape[-1]

        lA = (last_lA.expand(list(last_lA.shape[:-2]) + [H, W]) / (H * W)) if last_lA is not None else None
        uA = (last_uA.expand(list(last_uA.shape[:-2]) + [H, W]) / (H * W)) if last_uA is not None else None

        return [(lA, uA)], 0, 0

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        h_L = F.adaptive_avg_pool2d(h_L, (1, 1))
        h_U = F.adaptive_avg_pool2d(h_U, (1, 1))
        return h_L, h_U


class BoundAveragePool(Bound):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        # assumptions: ceil_mode=False, count_include_pad=True
        super().__init__(attr, inputs, output_index, options)

        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])

        self.kernel_size = attr['kernel_shape']
        assert len(self.kernel_size) == 2
        self.stride = attr['strides']
        assert len(self.stride) == 2
        # FIXME (22/07/02): padding is inconsistently handled. Should use 4-tuple.

        if 'pads' not in attr:
            self.padding = [0, 0]
        else:
            self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.count_include_pad = True
        self.use_default_ibp = True

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if isinstance(last_A, torch.Tensor):
                shape = last_A.size()
                # propagate A to the next layer, with batch concatenated together
                next_A = F.interpolate(
                    last_A.reshape(shape[0] * shape[1], *shape[2:]),
                    scale_factor=self.kernel_size) / (prod(self.kernel_size))
                next_A = F.pad(next_A, (0, self.input_shape[-2] - next_A.shape[-2], 0, self.input_shape[-1] - next_A.shape[-1]))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            elif isinstance(last_A, Patches):
                patches = last_A.patches
                shape = patches.size()
                # When the number of inserted zeros can cancel out the stride, we use a shortcut that can reduce computation.
                simplify_patch = ((last_A.inserted_zeros + 1 == self.kernel_size[0])
                                  and (self.kernel_size[0] == self.kernel_size[1]))
                padding, stride, output_padding = compute_patches_stride_padding(
                        self.input_shape, last_A.padding, last_A.stride,
                        self.padding, self.stride,
                        inserted_zeros=last_A.inserted_zeros,
                        output_padding=last_A.output_padding,
                        simplify=not simplify_patch)
                inserted_zeros = last_A.inserted_zeros
                if last_A.inserted_zeros == 0:
                    # No inserted zeros, can be handled using interpolate.
                    if last_A.unstable_idx is None:
                        # shape is: [out_C, batch, out_H, out_W, in_c, patch_H, patch_W]
                        up_sampled_patches = F.interpolate(
                            patches.view(shape[0] * shape[1],
                                         shape[2] * shape[3], *shape[4:]),
                            scale_factor=[1,] + self.kernel_size)
                        # The dimension of patch-H and patch_W has changed.
                        up_sampled_patches = up_sampled_patches.view(
                            *shape[:-2], up_sampled_patches.size(-2),
                            up_sampled_patches.size(-1))
                    else:
                        # shape is: [spec, batch, in_c, patch_H, patch_W]
                        up_sampled_patches = F.interpolate(
                            patches, scale_factor=[1,] + self.kernel_size)
                    # Divided by the averaging factor.
                    up_sampled_patches = up_sampled_patches / prod(self.kernel_size)
                elif simplify_patch:
                    padding = tuple(p // s - o for p, s, o in zip(padding, stride, output_padding))
                    output_padding = (0, 0, 0, 0)
                    stride = 1  # Stride and inserted zero canceled out. No need to insert zeros and add output_padding.
                    inserted_zeros = 0
                    value = 1. / prod(self.kernel_size)
                    # In the case where the stride and adding_zeros cancel out, we do not need to insert zeros.
                    weight = torch.full(
                        size=(self.input_shape[1], 1, *self.kernel_size),
                        fill_value=value, dtype=patches.dtype,
                        device=patches.device)
                    if last_A.unstable_idx is None:
                        # shape is: [out_C, batch, out_H, out_W, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(
                            patches.reshape(
                                shape[0] * shape[1] * shape[2] * shape[3],
                                *shape[4:]
                            ), weight, stride=1, groups=self.input_shape[1])
                    else:
                        # shape is: [spec, batch, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(
                            patches.reshape(shape[0] * shape[1], *shape[2:]),
                            weight, stride=1, groups=self.input_shape[1])
                    up_sampled_patches = up_sampled_patches.view(
                        *shape[:-2], up_sampled_patches.size(-2), up_sampled_patches.size(-1))
                else:
                    # With inserted zeros, must be handled by treating pooling as general convolution.
                    value = 1. / prod(self.kernel_size)
                    weight = torch.full(size=(self.input_shape[1], 1, *self.kernel_size),
                                        fill_value=value, dtype=patches.dtype,
                                        device=patches.device)
                    weight = insert_zeros(weight, last_A.inserted_zeros)
                    if last_A.unstable_idx is None:
                        # shape is: [out_C, batch, out_H, out_W, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(
                            patches.reshape(shape[0] * shape[1] * shape[2] * shape[3], *shape[4:]),
                            weight, stride=self.kernel_size,
                            groups=self.input_shape[1])
                    else:
                        # shape is: [spec, batch, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(
                            patches.reshape(shape[0] * shape[1], *shape[2:]),
                            weight, stride=self.kernel_size,
                            groups=self.input_shape[1])
                    up_sampled_patches = up_sampled_patches.view(
                        *shape[:-2], up_sampled_patches.size(-2),
                        up_sampled_patches.size(-1))
                next_A = last_A.create_similar(
                    up_sampled_patches, stride=stride, padding=padding,
                    output_padding=output_padding,
                    inserted_zeros=inserted_zeros)
            else:
                raise ValueError(f'last_A has unexpected shape {type(last_A)}')
            return next_A, 0.

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,32,27,27)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,32,6,6)
        this_layer_shape = self.output_shape
        assert this_layer_shape[2] ==  (
            (2 * self.padding[0] + pre_layer_shape[2] - (self.stride[0] - 1)
        ) // self.stride[0])

        value = 1.0/(self.kernel_size[0] * self.kernel_size[1])
        new_layer_gurobi_vars = []
        neuron_idx = 0
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    # print(self.bias.shape, out_chan_idx, out_lbs.size(1))
                    lin_expr = 0.0
                    for ker_row_idx in range(self.kernel_size[0]):
                        in_row_idx = -self.padding[0] + self.stride[0] * out_row_idx + ker_row_idx
                        if (in_row_idx < 0) or (in_row_idx == len(gvars_array[out_chan_idx][ker_row_idx])):
                            # This is padding -> value of 0
                            continue
                        for ker_col_idx in range(self.kernel_size[1]):
                            in_col_idx = -self.padding[1] + self.stride[1] * out_col_idx + ker_col_idx
                            if (in_col_idx < 0) or (in_col_idx == pre_layer_shape[3]):
                                # This is padding -> value of 0
                                continue
                            coeff = value
                            lin_expr += coeff * gvars_array[out_chan_idx][in_row_idx][in_col_idx]
                    v = model.addVar(lb=-float('inf'), ub=float('inf'),
                                            obj=0, vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{self.name}_{neuron_idx}')
                    model.addConstr(lin_expr == v, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1

                    out_row_vars.append(v)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        model.update()
