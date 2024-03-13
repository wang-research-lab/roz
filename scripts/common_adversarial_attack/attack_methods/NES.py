import torch
import torch.nn as nn
from torchvision import models



class NESAttack():
    def __init__(self, model, attack_setting='score', search='orig'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.binary = True if search == 'binary' else False
        self.q = 100




    def attack_one_sample(self, image, target):
        image, target = image.to(self.device), target.to(self.device)
        search_space = range(1, 11)
        if self.binary:
            attack_flag = self.iterate_attack(image,target,0, 0)
            if attack_flag:
                return 0
            search_space = list(search_space)
            min_epsilon = self.search_attack(search_space, 0, len(search_space), image,target)
            if min_epsilon == -1:
                return 1e10
            else:
                return min_epsilon
        else:
            search_space = range(1,101)
            for epislon in search_space:
                input_epsilon = epislon / 1000
                self.alpha = input_epsilon * 0.15
                steps = 20
                flag = self.iterate_attack(image, target, steps, input_epsilon)
                # print('here', epislon)
                if flag:
                    return input_epsilon

            return 10000000



    def search_attack(self, search_space, left, right, image, target):
        if left == right:
            return -1
        target_idx = (right-left) // 2 + left
        input_epsilon = search_space[target_idx] / 1000
        self.alpha = input_epsilon * 0.15
        steps = 20
        attack_flag = self.iterate_attack(image, target, steps, input_epsilon)
        if attack_flag:
            # succeesful attack, reduce epsilon
            min_epsilon = self.search_attack(search_space, left, target_idx, image=image, target=target)
            if min_epsilon == -1:
                return input_epsilon
            else:
                return min_epsilon
        else:
            # failuer attack, need to search for larger epsilon
            min_epsilon = self.search_attack(search_space, target_idx+1, right,  image=image, target=target)
            if min_epsilon == -1:
                return -1
            else:
                return min_epsilon





    def attack(self, test_loader, epsilon=8 / 255):
        steps = int(min(epsilon * 255 + 4, 1.25 * epsilon * 255))
        success = 0
        total = 0
        self.alpha = epsilon * 0.15
        from tqdm import tqdm
        for images, labels in tqdm(test_loader):
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            flag = self.iterate_attack(images, labels, steps, epsilon)
            if flag:
                success += 1
            total += 1
            # print('ACC:  {}'.format((total-success)/total))
        print('ACC:  {}'.format((total - success) / total))




    def clip_epsilon_ball(self, ori_image, adv_image, epsilon):
        a = torch.clamp(ori_image - epsilon, min=-1.8)

        b = (adv_image >= a).float() * adv_image \
            + (a > adv_image).float() * a

        c = (b > ori_image + epsilon).float() * (ori_image + epsilon) \
            + ((ori_image + epsilon) >= b).float() * b
        images = torch.clamp(c, max=2.15).detach()
        return images


    def compute_grad(self, images, target_label):
        with torch.no_grad():
            sigma = 0.001
            g = 0
            size = images.shape[-1]
            u = torch.randn(self.q, 3, size, size).to(self.device)
            logits1 = self.model(images+sigma*u)  # self.q, 1000
            # print(logits1.shape) # 100, 1000
            # print(torch.max(torch.cat((logits1[:, :target_label], logits1[:, target_label+1:]), dim=1), dim=1)[0].shape)
            tmp1 = logits1[:, target_label] - torch.max(torch.cat((logits1[:, :target_label], logits1[:, target_label+1:]), dim=1), dim=1)[0]
            # tmp shape : self.q
            logits2 = self.model(images-sigma*u)
            tmp2 = logits2[:, target_label] - torch.max(torch.cat((logits2[:, :target_label], logits2[:, target_label+1:]), dim=1), dim=1)[0]
            # 100, 1, 1, 1  *  100,3,224,224
            g = (tmp1-tmp2).view(self.q, 1, 1, 1) * u  # 100,3,224,224
            g = torch.sum(g, dim=0)
            # assert g.shape == (3,224,224)

            g = g / (2*self.q*sigma)


            return g.unsqueeze(0)



            # for i in range(self.q):
            #     print(i, 'inside compute grad')
            #     u = torch.randn(3, size, size).to(self.device)
            #     logits = self.model(images + (sigma * u).unsqueeze(0)).squeeze()
            #     g = g + (logits[target_label] - torch.max(torch.cat((logits[:target_label], logits[target_label+1:])))) * u
            #     logits = self.model(images - (sigma * u).unsqueeze(0)).squeeze()
            #     g = g - (logits[target_label] - torch.max(torch.cat((logits[:target_label], logits[target_label+1:])))) * u
            #
            # g = g / (2 * self.q * sigma)
            # print(g, torch.sum(g))
            # return g.unsqueeze(0)



    def iterate_attack(self, images, labels, steps, epsilon):
        with torch.no_grad():
            ori_images = images.clone().detach()
            images.requires_grad = False
            if torch.max(self.model(ori_images.clone().detach()), dim=1)[1].item() != labels.item():
                return True
            for i in range(steps):
                # print(i, 'inside iterate attack')
                imgs_grad = self.compute_grad(images, labels.squeeze())
                adv_images = images - self.alpha * imgs_grad.sign()
                adv_images = self.clip_epsilon_ball(ori_images, adv_images, epsilon)
                images = adv_images.clone().detach()
                if torch.max(self.model(images.clone().detach()), dim=1)[1].item() != labels.item():
                    return True
            return False