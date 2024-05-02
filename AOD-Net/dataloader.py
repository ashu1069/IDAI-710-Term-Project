import os
from PIL import Image
from torch.utils.data import Dataset

class DehazingDataset(Dataset):
    def __init__(self, root_dir, subset='test', transform=None):
        '''
        Arguments:
        root_dir: it contains two folders- foggy and original
        The default value of subset is set to 'test' therefore in train and val loader, 
        we have explicitly changes that to 'train' and 'val'

        Return:
        foggy_image
        original_image
        filename: this is same for both foggy and original
        '''
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.foggy_path = os.path.join(root_dir, 'foggyX_images', subset)
        self.original_path = os.path.join(root_dir, 'original_images', subset)
        self.image_files = [f for f in os.listdir(self.foggy_path) if os.path.isfile(os.path.join(self.foggy_path, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        foggy_image_path = os.path.join(self.foggy_path, filename)
        original_image_path = os.path.join(self.original_path, filename)

        #converting the image to RGB
        foggy_image = Image.open(foggy_image_path).convert('RGB')
        original_image = Image.open(original_image_path).convert('RGB')

        #applying transform if mentioned
        if self.transform:
            foggy_image = self.transform(foggy_image)
            original_image = self.transform(original_image)

        return {'foggy_image': foggy_image, 'original_image': original_image, 'filename': filename}
