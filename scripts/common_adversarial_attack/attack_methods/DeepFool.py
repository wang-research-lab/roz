import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

class DeepFoolAttack():
    def __init__(self, model, tmp=None, search='orig', num_classes=1000, max_iters=50, overshoot=0.02):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.max_iters = max_iters
        self.overshoot = overshoot


    def attack(self, test_loader, epsilon=8/255):
        correct = 0
        total_num = 0
        for image, target in tqdm(test_loader):
            total_num += 1
            image = image[0].to(self.device)
            f_image = self.model.forward(
                Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()

            I = (np.array(f_image)).flatten().argsort()[::-1]

            I = I[0:self.num_classes]
            label = I[0]
            if label != target:
                continue

            input_shape = image.cpu().numpy().shape
            pert_image = copy.deepcopy(image)
            w = np.zeros(input_shape)
            r_tot = np.zeros(input_shape)

            loop_i = 0



            x = Variable(pert_image[None, :], requires_grad=True)
            fs = self.model.forward(x)

            k_i = label

            while k_i == label and loop_i < self.max_iters:

                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)
                grad_orig = x.grad.data.cpu().numpy().copy()

                for k in range(1, self.num_classes):
                    zero_gradients(x)

                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = x.grad.data.cpu().numpy().copy()

                    # set new w_k and new f_k
                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                    pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(), ord=1)

                    # determine which w_k to use
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k

                # compute r_i and r_tot
                # Added 1e-4 for numerical stability
                r_i = (pert + 1e-4) * np.sign(w)
                r_tot = np.float32(r_tot + r_i)
                pert_image = image + (1 + self.overshoot) * torch.from_numpy(r_tot).to(self.device)
                pert_image = torch.clamp(pert_image, -1.8, 2.15)
                x = Variable(pert_image, requires_grad=True)
                fs = self.model.forward(x)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())

                loop_i += 1
            if k_i == label:
                print('Attack failure')
                raise RuntimeError
            else:
                # successful attack, need to check the perturbation
                r_tot = (1 + self.overshoot) * r_tot
                r_tot = r_tot.flatten()
                perterb = np.linalg.norm(r_tot, ord=np.inf)
                if perterb > epsilon:
                    correct += 1

        print('Test accuracy under perturbation: {}'.format(correct/total_num))






    def attack_one_sample(self, images, target):
        images = images[0].to(self.device)

        f_image = self.model.forward(Variable(images[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()

        I = (np.array(f_image)).flatten().argsort()[::-1]
        I = I[0:self.num_classes]
        label = I[0]
        if label != target:
            return 0
        input_shape = images.cpu().numpy().shape
        pert_image = copy.deepcopy(images)

        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = self.model.forward(x)

        k_i = label


        while k_i == label and loop_i < self.max_iters:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(), ord=1)

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * np.sign(w)
            r_tot = np.float32(r_tot + r_i)
            pert_image = images + (1 + self.overshoot) * torch.from_numpy(r_tot).to(self.device)
            pert_image = torch.clamp(pert_image, -1.8, 2.15)
            x = Variable(pert_image, requires_grad=True)
            fs = self.model.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        if k_i == label:
            print('Attack failure')
            raise RuntimeError
        else:
            # successful attack
            r_tot = (1 + self.overshoot) * r_tot
            r_tot = r_tot.flatten()
            perterb = np.linalg.norm(r_tot, ord=np.inf)
            return perterb


