from model import Textnet, Ori_ImageModel
import torch
import numpy as np
import tqdm
import clip
from dataset import CLIPDataset


text_model = "../input/sentence-transformer-models/paraphrase-xlm-r-multilingual-v1/0_Transformer"
def get_text_embedding(pretrained_model_path, text_loader):
    embeds = []
    #model = ShopeeNet(**model_params)
    model = Textnet(model_name=text_model)
    model.eval()
    model.load_state_dict(dict(list(torch.load(pretrained_model_path).items())[:-1]))
    model.cuda()
    with torch.no_grad():
        for input_id, attention_mask in tqdm(text_loader):
            input_id = input_id.cuda()
            attention_mask = attention_mask.cuda()
            features = model(input_ids=input_id, attention_mask=attention_mask).detach().cpu().numpy()
            embeds.append(features)
        text_embedding = np.concatenate(embeds)
    return text_embedding


def get_image_embedding(train_loader):
    #model = ImageModel(model_name = 'tf_efficientnet_b5_ns')
    model = Ori_ImageModel(model_name = 'tf_efficientnet_b5_ns')
    #model = replace_activations(model, torch.nn.SiLU, Mish())
    model = model.cuda()
    embed = []
    with torch.no_grad():
        for img,label in tqdm(train_loader):
            img = img.cuda()
            label = label.cuda()
            feat = model(img,label)
            image_embeddings = feat.detach().cpu().numpy()
            embed.append(image_embeddings)
        image_embedding = np.concatenate(embed)
    return image_embedding


def get_clip_embeddings(df, images_path):
    embed_dim = 512
    ds = CLIPDataset(df, images_path)
    dl = torch.utils.data.DataLoader(ds, batch_size=2 * 16, shuffle=False, num_workers=4)

    model, _ = clip.load("../input/openai-clip/RN50x4.pt", device='cuda', jit=False)

    # Allocate memory for features
    features = np.empty((len(df), 2 * embed_dim), dtype=np.float32)

    # Begin predict
    i = 0
    for images, texts, _ in tqdm(dl):
        n = len(images)
        with torch.no_grad():
            # Generate image and text features
            images_features = model.encode_image(images.cuda())
            texts_features = model.encode_text(texts.cuda())

        # Concat features (first images then texts)
        features[i:i + n, :embed_dim] = images_features.cpu()
        features[i:i + n, embed_dim:] = texts_features.cpu()

        i += n

    # Option to save these features (may be usefull to tune cut value
    print(f'Our clip embeddings shape is {features.shape}')
    #     np.save("clip_embeddings.npy", features)

    # l2-normalize
    features /= np.linalg.norm(features, 2, axis=1, keepdims=True)

    return features
