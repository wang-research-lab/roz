import argparse
from attack_methods.FGSM import FGSMAttack
from attack_methods.DeepFool import DeepFoolAttack
from attack_methods.BIM import BIMAttack
from attack_methods.MIM import MIMAttack
from attack_methods.DIM import DIMAttack
from attack_methods.NES import NESAttack
from attack_methods.SPSA import SPSAAttack
from load_utils import get_imageNet_loader, load_pretrained_vision_models, get_CIFAR10_loader
from Wrapper import VictimModel
import clip


parser = argparse.ArgumentParser()
parser.add_argument('--attack_method', default='FGSM')
parser.add_argument('--traditional_model', default='resnet50')
parser.add_argument('--eval_type', default='traditional', choices=['CLIP',
                                                                   'traditional',
                                                                   'CLIP_ensemble',
                                                                   'CLIP_AUTO'])
parser.add_argument('--clip_type', default='ViT-B/32')
parser.add_argument('--epsilon', default=8/255, type=float)
parser.add_argument('--attack_type', default='V1')
parser.add_argument('--attack_setting', default='white')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--search', default='orig', choices=['orig', 'binary'])
parser.add_argument('--sample', default='True')
parser.add_argument('--imgNet', default='True')
parser.add_argument('--prompts_id_file',default='')
parser.add_argument('--sample_type', default=224, type=int)
parser.add_argument('--sample_p', default=1, type=int)

params = parser.parse_args()

attack_mode = params.attack_method
print(params.sample)
imgNet = eval(params.imgNet)
if imgNet:
    test_loader = get_imageNet_loader(batch_size=params.batch_size,sample=eval(params.sample), sample_type=params.sample_type)
else:
    test_loader = get_CIFAR10_loader(batch_size=params.batch_size, sample=eval(params.sample), sample_type=params.sample_type)


victim_model = VictimModel(eval_type=params.eval_type, clip_type=params.clip_type,
                           vision_model=params.traditional_model, model=None, imgNet=imgNet,
                           prompts_id_file=params.prompts_id_file,sample_p=params.sample_p)





victim_model.eval()

Attacker = eval(params.attack_method+'Attack')(victim_model, params.attack_setting, params.search)




if params.attack_type == 'V1':
    Attacker.attack(test_loader, 8 / 255)
else:
    from tqdm import tqdm
    import numpy as np
    epsilon_li = []
    for image, label in tqdm(test_loader):
        min_epsilon = Attacker.attack_one_sample(image, label)
        epsilon_li.append(min_epsilon)
        print(np.median(epsilon_li))


    print('medium perturbation: {}'.format(np.median(epsilon_li)))

