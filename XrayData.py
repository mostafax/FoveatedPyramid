import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import json


class HeadXrays(Dataset):

    def __init__(self, directory,junior=True):
        self.json_file_name = "/content/drive/MyDrive/data/all_data.json"
        self.root_dir = "/content/images/"
        '''
        self.anno_dir = os.path.join('/', "home/mostafa/work/Ceph-Model/AnnotationsByMD")
        img_dir = os.path.join('/', "home/mostafa/work/Ceph-Model/RawImage/TrainingData")
        images = filter(lambda f: not f.startswith("."),os.listdir(img_dir))

        parse_id = lambda img: int(img.split(".bmp")[0])

        images = [(parse_id(img), img) for img in images]

        images.sort(key = lambda x: x[0])
        self.files = np.array([(os.path.join(img_dir,img[1]),) + self.loadAnnotations(img[0]) for img in images])
        '''
        with open('{}'.format(self.json_file_name)) as json_file:
            self.landmarks = json.load(json_file)
        #self.root_dir = root_dir
        self.images_names = []
        for idx in range(len(self.landmarks)):
            self.images_names.append(os.path.join(self.root_dir,self.landmarks[idx]["imagePath"]))
        all_points = []
        list_of_points = []
        for j in range(len(self.landmarks)):
            landmarks = self.landmarks[j]["points"]
            landmarks = sorted(landmarks, key=lambda k: k['pointName'])
            list_of_points = []
            for i in range(len(landmarks)):
                list_of_points.append([landmarks[i]["X"], landmarks[i]["Y"]])
            list_of_points = np.array([list_of_points])
            list_of_points = list_of_points.astype('float').reshape(-1, 2)
            all_points.append(list_of_points)
        #print(len(all_points[0]))
        #for i in range(len(l)):
        all_data = []
        for idx in range(len(self.landmarks)):
            all_data.append([self.images_names[idx],all_points[idx],all_points[idx]])
        self.files = np.array(all_data)    
        #print(self.files[0])
        #print(self.files.shape)
        #return 0
        #print(sorted(self.images_names))


    def loadAnnotations(self,id):
        anno = ()
        for i, doctor in enumerate(["junior","senior"]):
            path = os.path.join(self.anno_dir,f"400_{doctor}",f"{id:03d}.txt")

            with open(path,"r") as f:
                annotations = f.readlines()[:19]

            anno+= (np.array(list(map(lambda c: (int(c[0]), int(c[1])), map(lambda a: a.split(","), annotations)))),)
        return anno


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path, junior, senior = self.files[idx]

        image = Image.open(path)
        return image, junior, senior


    def __len__(self):
        return len(self.files)


class HeadXrayAnnos(HeadXrays):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.files[idx]



class Transform(Dataset):
    def __init__(self, dataset, indices=None, tx=lambda x:x, ty=lambda x:x):
        self.dataset = dataset
        self.tx = tx
        self.ty = ty
        if indices is None:
            indices = np.arange(len(self.dataset))
        self.indices = indices

    def __getitem__(self, idx):
        x, junior, senior = self.dataset[self.indices[idx]]

        return self.tx(x), self.ty(junior), self.ty(senior)

    def __len__(self):
        return len(self.indices)

class TransformedHeadXrayAnnos(Transform):
    def __init__(self, indices, landmarks):
        tx = lambda x: x

        middle = np.array([1198, 1438]) / 2

        ty = lambda x: (x[landmarks] - middle) / 1198. * 2
        path = "images/RawImage"
        if 'SLURM_TMPDIR' in os.environ:
            path = os.path.join(os.environ['SLURM_TMPDIR'],'RawImage')
        super().__init__(HeadXrays(path),indices = indices,tx=tx,ty=ty)

class TransformedXrays(Transform):
    def __init__(self, indices, landmarks):
        tx = transforms.Compose([
            transforms.Pad((0, 0, 0, 32)),
            transforms.ToTensor(),
            lambda x: x[:, :, :1198].sum(dim=0, keepdim=True),
            transforms.Normalize([1.4255656], [0.8835338])])

        middle = np.array([1198, 1438]) / 2

        ty = lambda x: (x[landmarks] - middle) / 1198. * 2
        path = "/content/RawImage"
        if 'SLURM_TMPDIR' in os.environ:
            path = os.path.join(os.environ['SLURM_TMPDIR'],'RawImage')
        super().__init__(HeadXrays(path),indices = indices,tx=tx,ty=ty)

def get_train_val(landmarks, trainset, valset):
    splits = ['train', 'val']

    ranges = {'train': trainset, 'val': valset}

    datasets = {x: TransformedXrays(indices=ranges[x], landmarks=landmarks) for x in splits}

    return splits, datasets

def get_folded(landmarks, fold, num_folds, fold_size, batchsize):
    folds = np.arange(num_folds * fold_size).reshape(num_folds, fold_size)

    val_fold = fold == np.arange(num_folds)
    val_set = folds[val_fold].flatten()
    train_set = folds[~val_fold].flatten()
    splits, datasets = get_train_val(landmarks, train_set, val_set)
    annos = TransformedHeadXrayAnnos(indices=train_set, landmarks=landmarks)
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                  batch_size=batchsize, shuffle=(x == 'train'), num_workers=2,
                                                  pin_memory=True)
                   for x in splits}

    return splits, datasets, dataloaders, annos

def get_shuffled(landmarks, seed):

    splits = ['train', 'val', 'test']

    train_val = np.arange(360)
    np.random.seed(seed)
    np.random.shuffle(train_val)
    train = train_val[:324]
    val = train_val[324:]

    ranges = {'train': train, 'val': val, 'test': np.arange(360, 400)}

    datasets = {x: TransformedXrays(indices=ranges[x], landmarks = landmarks) for x in splits}

    return splits, datasets
