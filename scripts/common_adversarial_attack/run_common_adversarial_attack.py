import argparse
from attack_methods.FGSM import FGSMAttack
from attack_methods.DeepFool import DeepFoolAttack
from attack_methods.BIM import BIMAttack
from attack_methods.MIM import MIMAttack
from attack_methods.DIM import DIMAttack
from attack_methods.NES import NESAttack
from attack_methods.SPSA import SPSAAttack
from load_utils import get_imageNet_loader, load_pretrained_vision_models, get_CIFAR10_loader
from wrapper import VictimModel
import clip


parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', default=8/255, type=float)
parser.add_argument('--attack_type', default='V1')
parser.add_argument('--attack_setting', default='white')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--prompts_id_file',default='./Prompts/RN50file.log')
parser.add_argument('--sample_p', default=5, type=int)

params = parser.parse_args()


eval_types = ['CLIP_ensemble','CLIP_AUTO']
clip_types = ['RN50', 'RN101', 'ViT-B/32','ViT-B/16',  'RN50x4', 'RN50x16']
attack_method = ['FGSM', 'DeepFool', 'MIM', 'BIM', 'DIM', 'NES', 'SPSA']

for clip_type in clip_types:
    for method in attack_method:
        for eval_type in eval_types:
            attack_mode = method
            imgNet = True
            test_loader = get_imageNet_loader(batch_size=params.batch_size,sample=False, sample_type=params.sample_type)

            victim_model = VictimModel(eval_type=eval_type, clip_type=clip_type,
                                       vision_model=None, model=None, imgNet=imgNet,
                                       prompts_id_file=params.prompts_id_file,sample_p=params.sample_p)
            victim_model.eval()
            Attacker = eval(params.attack_method + 'Attack')(victim_model, params.attack_setting, 'binary')

            Attacker.attack(test_loader, 8 / 255)

            from tqdm import tqdm
            import numpy as np
            epsilon_li = []
            for image, label in tqdm(test_loader):
                min_epsilon = Attacker.attack_one_sample(image, label)
                epsilon_li.append(min_epsilon)
                print('medium perturbation: {}'.format(np.median(epsilon_li)))

