import torch
from tqdm import tqdm
import time
from torchvision import models
import torch.nn as nn


class FGSMAttack():
    def __init__(self, model, attack_setting='score', search='orig'):
        self.model = model
        self.imgNet = self.model.img_Net

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.CrossEntropyLoss()
        attack_setting_dict = {'white': 0,
                             'transfer':1,
                             'decision':2}
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
        self.binary = search


    def attack(self, test_loader,epsilon=0.1):
        # Accuracy counter
        correct = 0
        total_nums = 0


        # Loop over all examples in test set

        for data, target in tqdm(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            if self.attack_setting == 1:
                pred = torch.max(self.model(data.clone().detach()), dim=1)[1]

                if pred.item() != target.item():
                    continue
                else:
                    output = self.substitue_model(data)

            else:
                output = self.model(data)
                init_pred = torch.max(output, dim=1)[1]  # get the index of the max log-probability
                if init_pred.item() != target.item():
                    continue
            total_nums += 1
            loss = self.criterion(output, target)
            self.model.zero_grad()
            if self.attack_setting == 1:
                self.substitue_model.zero_grad()

            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_update(data, epsilon, data_grad)
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1


        final_acc = correct / len(test_loader)
        print("Epsilon: {}\tTest Accuracy： {}".format(epsilon, final_acc))
        print(correct, total_nums, len(test_loader))
        return final_acc



    def attack_one_sample(self, image, target):
        image, target = image.to(self.device), target.to(self.device)
        image.requires_grad = True

        if self.attack_setting == 1:
            pred = torch.max(self.model(image.clone().detach()), dim=1)[1]
            if pred.item() != target.item():
                return 0
            output = self.substitue_model(image)
        else:
            output = self.model(image)
            init_pred = torch.max(output, dim=1)[1]
            if init_pred.item() != target.item():
                return 0


        loss = self.criterion(output, target)
        self.model.zero_grad()
        if self.attack_setting == 1:
            self.substitue_model.zero_grad()
        loss.backward()
        self.data_grad = image.grad.data



        with torch.no_grad():
            search_space = range(1, 101) if self.attack_setting == 1 else range(1, 10)
            if self.binary:
                min_epsilon = self.search_attack(list(search_space), 0, len(list(search_space)), image, target)
                if min_epsilon == -1:
                    return 1e10
                else:
                    return min_epsilon
            else:
                for epsilon in range(1, 601):
                    input_epsilon = epsilon / 1000
                    # print(data_grad)
                    perturbed_data = self.fgsm_update(image, input_epsilon, self.data_grad)
                    new_output = self.model(perturbed_data)
                    final_pred = new_output.max(1, keepdim=True)[1]
                    if final_pred.item() != target.item():
                        return input_epsilon
                return 1e10

    def search_attack(self, search_space, left, right, image, target):
        if left == right:
            return -1
        target_idx = (right - left) // 2 + left
        input_epsilon = search_space[target_idx] / 1000
        # attack_flag = self.iterate_attack(image, target, input_epsilon)
        perturbed_data = self.fgsm_update(image, input_epsilon, self.data_grad)
        new_output = self.model(perturbed_data)
        final_pred = new_output.max(1, keepdim=True)[1]

        if final_pred.item() != target.item():
            attack_flag = True
        else:
            attack_flag = False


        if attack_flag:
            # succeesful attack, reduce epsilon
            min_epsilon = self.search_attack(search_space, left, target_idx, image=image, target=target)
            if min_epsilon == -1:
                return input_epsilon
            else:
                return min_epsilon
        else:
            # failuer attack, need to search for larger epsilon
            min_epsilon = self.search_attack(search_space, target_idx + 1, right, image=image, target=target)
            if min_epsilon == -1:
                return -1
            else:
                return min_epsilon





    def fgsm_update(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # print(perturbed_image)
        # Adding clipping to maintain [0,1] range

        # mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        # (-1.79226,  2.14589699)
        perturbed_image = torch.clamp(perturbed_image, -1.8, 2.15)
        # Return the perturbed image
        return perturbed_image