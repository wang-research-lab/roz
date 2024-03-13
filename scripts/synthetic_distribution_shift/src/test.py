from torchvision.models import resnet50
from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization

registry.add_model(
    Model(
        name = 'test',
        transform = StandardTransform(img_resize_size=256, img_crop_size=224),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = lambda: resnet50(pretrained=True),
        eval_batch_size = 256,

    )
)
