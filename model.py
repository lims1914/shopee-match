from torch import nn
import torch.nn.functional as F
import timm
import transformers


class Ori_ImageModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(Ori_ImageModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if model_name == 'tf_efficientnet_b5_ns':
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        elif model_name == 'eca_nfnet_l0' or 'eca_nfnet_l1':
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()
        else:
            print('wrong input')
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, image, label):
        batch_size = image.shape[0]
        x = self.backbone(image)
        x = self.pooling(x).view(batch_size, -1)

        return x


class Textnet(nn.Module):

    def __init__(self,model_name='bert-base-uncased',):
        """
        param model_name: name of model from pretrainedmodels
        """
        super(Textnet, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids,attention_mask=attention_mask)

        features = x[0]
        features = features[:,0,:]
        return F.normalize(features)