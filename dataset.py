from PIL import Image
import torch
from torch.utils.data import Dataset
import transformers
from utils import tokenize, strip_emoji
import clip


class ImageDataset(Dataset):
    def __init__(self, image_path, transform):
        self.image_path = image_path
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(1)

    def __len__(self):
        return self.image_path.shape[0]


tokenize_model = "../input/sentence-transformer-models/paraphrase-xlm-r-multilingual-v1/0_Transformer"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(tokenize_model)


class TextDataset(Dataset):
    def __init__(self, csvfile):
        self.csv_file = csvfile.reset_index()

    def __getitem__(self, index):
        text = self.csv_file.iloc[index]
        text = text.title
        text = TOKENIZER(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        return input_ids, attention_mask

    def __len__(self):
        return self.csv_file.shape[0]


_, preprocess = clip.load("../input/openai-clip/RN50x4.pt", device='cuda', jit=False)
class CLIPDataset(Dataset):
    def __init__(self, df, images_path):
        super().__init__()
        self.df = df
        self.images_path = images_path
        self.has_target = ('label_group' in df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = preprocess(Image.open(self.images_path + '/' + row['image']))
        text = tokenize([strip_emoji(row['title'])])[0]

        if self.has_target:
            return image, text, row['label_group']
        else:
            return image, text, 0