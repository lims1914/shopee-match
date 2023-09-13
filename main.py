import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms


from dataset import ImageDataset, TextDataset
from get_embedding import get_text_embedding, get_image_embedding, get_clip_embeddings
from utils import get_voted_predictions, get_image_predictions, pca
import matplotlib.pyplot as plt



img_size = 512
batch_size = 16
device = 'cuda'


data = pd.read_csv('./train.csv')
#data = cudf.DataFrame(data)
image_path = "./train_images/" + data['image']
tmp = data.groupby('label_group').posting_id.agg('unique').to_dict()
data['target'] = data.label_group.map(tmp)
data['target'] = data['target'].apply(lambda x: ' '.join(x))


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
])

image_dataset = ImageDataset(image_path=image_path, transform=transform)
train_loader = torch.utils.data.DataLoader(
    image_dataset,
    batch_size = batch_size,
    pin_memory=True,
    drop_last=False,
    num_workers=2
)


text_dataset = TextDataset(data)
text_loader = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=2
    )

clip_embeddings = get_clip_embeddings(data, '../input/shopee-product-matching/train_images')

pretrained_model_path = "../input/best-multilingual-model/sentence_transfomer_xlm_best_loss_num_epochs_25_arcface.bin"
text_embedding = get_text_embedding(pretrained_model_path, text_loader)


models = ['eca_nfnet_l0', 'eca_nfnet_l1', 'tf_efficientnet_b5_ns']
image_embedding1 = get_image_embedding(models[0], train_loader)
image_embedding2 = get_image_embedding(models[1], train_loader)
image_embedding3 = get_image_embedding(models[2], train_loader)


score = get_voted_predictions(data, image_embedding1, image_embedding2, image_embedding3, threshold = 0.3)

embedding1 = pca(text_embedding)
embedding2 = pca(image_embedding1)
merge_embedding = np.concatenate((embedding1, embedding2), axis=1)


temp = []
thresholds = np.linspace(0, 1, num=20)
for threshold in thresholds:
    predict = get_image_predictions(data, image_embedding1, threshold=threshold)
    temp.append(predict)
plt.plot(thresholds, temp, color='b', label=r'f1_score', lw=2, alpha=.8)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('threshold')
plt.ylabel('f1_score')
plt.legend(loc='lower right')
plt.show()


metrics = ["euclidean", "manhattan", "minkowski", "cosine"]
temp1 = []
for metric in metrics:
    predict1 = get_image_predictions(data, image_embedding1, threshold = 0.3, metric=metric)
    temp1.append(predict1)


predict = get_image_predictions(data, clip_embeddings, threshold = 0.3)