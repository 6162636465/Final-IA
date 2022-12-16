import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader,Subset
import random




def load_data(batch_size, valid_ratio):
    transform = Compose([Resize((224, 224)), ToTensor()])
    path = r'E:\IA\FinalFinal\VIT_DogCat\With lightning\data\PetImages'
    data_train = ImageFolder(path, transform)
    n = len(data_train)  # número total de conjuntos de datos
    n_test = random.sample(range(1, n), int(valid_ratio * n))  # Tome una lista de números aleatorios proporcionalmente
    valid_set = torch.utils.data.Subset(data_train, n_test)  # Tome el conjunto de prueba de acuerdo con la lista de números aleatorios
    train_set = torch.utils.data.Subset(data_train, list(set(range(1, n)).difference(set(n_test))))  # El conjunto de prueba se deja como el conjunto de entrenamiento.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader,data_train
