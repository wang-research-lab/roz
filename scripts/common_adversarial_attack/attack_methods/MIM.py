import torch
import torch.nn as nn
from torchvision import models

class MIMAttack():
    def __init__(self, model, attack_setting='white', search='orig'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.imgNet = self.model.img_Net
        attack_setting_dict = {'white': 0,
                               'transfer': 1,
                               'decision': 2}
        self.attack_setting = attack_setting_dict[attack_setting]
        if self.attack_setting == 1:
            self.substitue_model = models.resnet152(pretrained=True)
            if self.imgNet:
                pass
            else:
                self.substitue_model.fc = nn.Linear(2048, 10, bias=True)
                state_dict = torch.load('substitute.pkl')
                self.substitue_model.load_state_dict(state_dict=state_dict)
            self.substitue_model = self.substitue_model.to(self.device)
            self.substitue_model.eval()


        self.binary = True if search == 'binary' else 'orig'


    def search_attack(self, search_space, left, right, image, target):
        if left == right:
            return -1
        target_idx = (right-left) // 2 + left
        input_epsilon = search_space[target_idx] / 1000
        self.alpha = input_epsilon * 0.15
        attack_flag = self.iterate_attack(image, target, input_epsilon)
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


    def attack_one_sample(self, image, target):
        image, target = image.to(self.device), target.to(self.device)
        search_space = range(1,601) if self.attack_setting == 1 else range(1, 10)
        if self.binary:
            min_epsilon = self.search_attack(list(search_space), 0, len(list(search_space)), image, target)
            if min_epsilon == -1:
                return 1e10
            else:
                return min_epsilon
        else:
            for epislon in search_space:
                input_epsilon = epislon / 1000
                flag = self.iterate_attack(image, target, input_epsilon)
                if flag:
                    return input_epsilon

            return 10000000



    def attack(self, test_loader, epsilon=8 / 255):


        success = 0
        total = 0
        from tqdm import tqdm
        for images, labels in tqdm(test_loader) :
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            flag = self.iterate_attack(images, labels, epsilon)
            if flag:
                success += 1
            total += 1

        print('ACC:  {}'.format((total - success) / total))

    def clip_epsilon_ball(self, ori_image, adv_image, epsilon):
        a = torch.clamp(ori_image - epsilon, min=-1.8)

        b = (adv_image >= a).float() * adv_image \
            + (a > adv_image).float() * a

        c = (b > ori_image + epsilon).float() * (ori_image + epsilon) \
            + ((ori_image + epsilon) >= b).float() * b
        images = torch.clamp(c, max=2.15).detach()
        return images


    def iterate_attack(self, images, labels, epsilon):


        iter = 10 if self.attack_setting == 1 else 20
        decay_factor = 1.0
        pert_out = images.clone().detach()
        alpha = 0.15 * epsilon

        if torch.max(self.model(pert_out.clone().detach()), dim=1)[1].item() != labels.item():
            return True
        g = 0
        for i in range(iter-1):
            pert_out.requires_grad = True

            if self.attack_setting == 1:
                output = self.substitue_model(pert_out)
            else:
                output = self.model(pert_out)

            loss = self.criterion(output, labels)
            if self.attack_setting == 1:
                self.substitue_model.zero_grad()
            else:
                self.model.zero_grad()
            loss.backward()
            data_grad = pert_out.grad
            g = decay_factor * g + data_grad / torch.norm(data_grad, p=1)

            pert_out = pert_out + alpha * torch.sign(g)
            pert_out = self.clip_epsilon_ball(images, pert_out, epsilon)

            if self.attack_setting == 1:
                pass
            else:
                if torch.max(self.model(pert_out.clone().detach()), dim=1)[1].item() != labels.item():
                    return True
            pert_out = pert_out.clone().detach()

        if self.attack_setting == 1:
            if torch.max(self.model(pert_out), dim=1)[1].item() != labels.item():
                return True

        return False