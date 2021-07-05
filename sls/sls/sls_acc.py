import torch
import copy
import time
import math

from . import utils as ut

class SlsAcc(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        n_batches_per_epoch=500,
        init_step_size=1,
        c=0.1,
        beta_b=0.9,
        gamma=2.0,
        momentum=0.6,
        reset_option=1,
        acceleration_method="polyak"
    ):
        params = list(params)
        super().__init__(params, {})
        self.params = params
        self.momentum = momentum
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.acceleration_method = acceleration_method
        self.n_batches_per_epoch = n_batches_per_epoch
        self.state['step'] = 0
        self.state['step_size'] = init_step_size
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.reset_option = reset_option
        self.state['params_last'] = copy.deepcopy(self.params)
        self.state['params_current'] = copy.deepcopy(self.params)
        # if acceleration_method == "nesterov":
        #     self.state['lambda_old'] = 0
        #     self.state['lambda_current'] = 1
        #     self.state['tau'] = 1

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()
        batch_step_size = self.state['step_size']
        step_size = ut.reset_step(step_size=batch_step_size,
            n_batches_per_epoch=self.n_batches_per_epoch,
            gamma=self.gamma,
            reset_option=self.reset_option,
            init_step_size=self.init_step_size)
        loss = closure_deterministic()
        self.state['n_forwards'] += 1
        # get loss and compute gradients
        if self.acceleration_method == "polyak":
            loss.backward()
            # increment # forward-backward calls
            self.state['n_backwards'] += 1
            grad_current = ut.get_grad_list(self.params)
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        params_last = copy.deepcopy(self.state['params_last'])
        # if self.acceleration_method == 'polyak':
        try_momentum_update(
            self.params,
            params_current,
            params_last,
            gamma=self.momentum
        )
        # elif self.acceleration_method == 'nesterov':
        #     try_momentum_update(
        #         self.params,
        #         params_current,
        #         params_last,
        #         gamma=self.state['tau']
        #     )
        half_loss = closure_deterministic()
        self.state['n_forwards'] += 1
        half_loss.backward()
        self.state['n_backwards'] += 1
        half_grad = ut.get_grad_list(self.params)
        self.state['params_current'] = copy.deepcopy(params_current)
        # only do the check if the gradient norm is big enough
        with torch.no_grad():
            if ut.compute_grad_norm_squared(half_grad) >= 1e-8:
                # check if condition is satisfied
                found = 0
                for e in range(100):
                    # print(e, step_size)
                    if self.acceleration_method == "polyak":
                        # try a prospective step
                        try_polyak_update(
                            self.params,
                            step_size=step_size,
                            params_current=params_current,
                            params_last=params_last,
                            grad_current=grad_current,
                            momentum=self.momentum
                        )
                        # compute the loss at the next step; no need to compute gradients.
                        loss_next = closure_deterministic()
                        self.state['n_forwards'] += 1
                        armijo_results = ut.check_polyak_armijo_conditions(
                            step_size=step_size,
                            half_loss=half_loss,
                            grad=grad_current,
                            half_grad=half_grad,
                            loss_next=loss_next,
                            c=self.c,
                            momentum=self.momentum,
                            beta_b=self.beta_b
                        )
                    elif self.acceleration_method == "nesterov":
                        try_nesterov_update(
                            self.params,
                            step_size=step_size,
                            half_grad=half_grad,
                            params_current=params_current,
                            params_last=self.state["params_last"],
                            gamma=self.momentum
                        )
                        # compute the loss at the next step; no need to compute gradients.
                        loss_next = closure_deterministic()
                        self.state['n_forwards'] += 1
                        armijo_results = ut.check_nesterov_armijo_conditions(
                            step_size=step_size,
                            half_loss=half_loss,
                            loss_next=loss_next,
                            half_grad=half_grad,
                            c=self.c,
                            beta_b=self.beta_b
                        )
                    found, step_size = armijo_results
                    if found == 1:
                        break
                # if line search exceeds max_epochs
                if found == 0:
                    print('rip')
                    # all else fails, sgd will do :)
                    ut.try_sgd_update(self.params, 1e-6, params_current, half_grad)
        self.state["params_last"] = copy.deepcopy(params_current)
        # if self.acceleration_method == "nesterov":
        #     lambda_tmp = self.state['lambda_current']
        #     self.state['lambda_current'] = (1 + math.sqrt(1 + 4 * self.state['lambda_old']  *
        #                                                   self.state['lambda_old'] )) / 2
        #     self.state['lambda_old'] = lambda_tmp
        #     self.state['tau'] = (1 - self.state['lambda_old']) / self.state['lambda_current']
        # save the new step-size
        self.state['step_size'] = step_size
        self.state['step'] += 1
        return loss


def try_momentum_update(params_model, params_current, params_last, gamma):
    zipped = zip(params_model, params_current, params_last)
    for p_model, p_current, p_old in zipped:
        p_model.data = (1+gamma)*p_current - gamma*p_old.cuda()


def try_polyak_update(params_model, step_size, params_current, grad_current, params_last, momentum):
    zipped = zip(params_model, params_current, grad_current, params_last)
    for p_model, p_current, g_current, p_old in zipped:
        if g_current is None:
            continue
        p_model.data = p_current - step_size*g_current + momentum*(p_current-p_old.cuda())


def try_nesterov_update(params_model, step_size, half_grad, params_current, params_last, gamma):
    zipped = zip(params_model, half_grad, params_current, params_last)
    for p_model, half_g, p_current, p_old in zipped:
        if half_g is None:
            continue
        p_model.data = (1+gamma)*p_current - gamma*p_old.cuda() - step_size*half_g
