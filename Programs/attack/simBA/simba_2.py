"""
This module implements the black-box attack `SimBA`.
"""

import numpy as np
import concurrent.futures
import proj_lp
import torch
from tqdm import tqdm
import model_arch as ma

class SimBA():
    """
    Implementation of the `SimBA` attack. Paper link: https://arxiv.org/abs/1905.07121
    """

    def __init__(self,  classifier, device):
        """
        Create a class: `SimBA` instance.
        - classifier: model to attack
        """
        self.classifier = classifier
        self.device = device
    
    def init(self, x):
        """
        Initialize the attack.
        """
        # y_pred = self.classifier.predict(np.array([x]))[0]
        y_pred = self.classifier(x)[0]
        y_pred = y_pred.cpu().detach().numpy()
        x_adv = x
        y_list = []
        y_list.append(y_pred[np.argmax(y_pred)])

        perm = np.random.permutation(x.cpu().detach().numpy().reshape(-1).shape[0])

        return x_adv, y_pred, y_list, perm

    def step(self, x_adv, y_pred, y_list, perm, index, epsilon=0.1):
        """
        Single step for non-distributed attack.
        """
        y_adv = None
        x_adv_new = x_adv

        diff = np.zeros(x_adv.reshape(-1).shape[0])
        diff[perm[index]] = epsilon * 2
        diff = torch.from_numpy(diff).float()
        diff = diff.reshape(x_adv.shape)
        diff = ma.to_device(diff, self.device)

        # print(y_list)

        # plus = self.classifier.predict(np.array([np.clip(x_adv + diff, 0, 1)]))[0]
        plus = self.classifier(torch.clamp(diff + x_adv  , 0, 1))[0]
        plus = plus.cpu().detach().numpy()
        if plus[np.argmax(y_pred)] < y_list[-1]:
            x_adv_new = torch.clamp(x_adv + diff, 0, 1)
            y_adv = plus
        else:
            # minus = self.classifier.predict(np.array([np.clip(x_adv - diff, 0, 1)]))[0]
            minus = self.classifier(torch.clamp(x_adv - diff, 0, 1))[0]
            minus = minus.cpu().detach().numpy()
            if minus[np.argmax(y_pred)] < y_list[-1]:
                x_adv_new = torch.clamp(x_adv - diff, 0, 1)
                y_adv = minus
        
        if y_adv is not None:
            y_list.append(y_adv[np.argmax(y_pred)])

        return x_adv_new, y_adv, y_list

    def batch(self, x_adv, y_pred, y_list, perm, index, epsilon=0.1, max_workers=10, batch=50):
        """
        Single step for distributed attack.
        """
        noise = np.zeros(x_adv.shape)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.step, x_adv, y_pred, y_list, perm, index+j, epsilon): j for j in range(0, batch)}
            for future in concurrent.futures.as_completed(future_to_url):
                j = future_to_url[future]
                try:
                    x_adv_new, _, _ = future.result()
                    noise = noise + (x_adv_new - x_adv)
                except Exception as exc:
                    print('Task %r generated an exception: %s' % (j, exc))
                else:
                    pass

        if(np.sum(noise) != 0):
            noise = proj_lp(noise, xi = 1)

        x_adv = np.clip(x_adv + noise, 0, 1)

        y_adv = self.classifier.predict(np.array([x_adv]))[0] 
        y_list.append(y_adv[np.argmax(y_pred)])

        return x_adv, y_adv, y_list

    def attack(self, x, epsilon=0.1, max_it=1000, distributed=False, batch=50, max_workers=10):
        """
        Initiate the attack.

        - x: input data
        - epsilon: perturbation on each pixel
        - max_it: number of iterations
        - distributed: if True, use distributed attack
        - batch: number of queries per worker
        - max_workers: number of workers
        """

        x_adv, y_pred, y_list, perm = self.init(x)

        if distributed:
            pbar = tqdm(range(0, max_it, batch), desc="Distributed SimBA")
        else:
            pbar = tqdm(range(0, max_it), desc="SimBA")

        for i in pbar:
            if distributed:
                x_adv, y_adv, y_list = self.batch(x_adv, y_pred, y_list, perm, i, epsilon, max_workers, batch)
            else:
                x_adv, y_adv, y_list = self.step(x_adv, y_pred, y_list, perm, i, epsilon)

            pbar.set_postfix({'origin prob': y_list[-1], 'l2 norm': np.sqrt(np.power(x_adv.cpu().detach().numpy() - x.cpu().detach().numpy(), 2).sum())})

            # Early break
            if y_adv is not None:
                if(np.argmax(y_adv) != np.argmax(y_pred)):
                    break
     
        return x_adv