import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Secventa de cod luata din laboratorul 6 :https://fmi-unibuc-ia.github.io/ia/Laboratoare/Laboratorul%206.pdf
class CustomImageDataset(Dataset):
    def __init__(self, fisier_csv, folder_imagini, transform=None):
        #initizam datele
        self.folder_imagini = folder_imagini
        self.eticheta_imagini = pd.read_csv(fisier_csv)
        self.transform = transform
 

    def __len__(self):
        #pentru a testa erorile de compilare
        return 20
        # return len(self.eticheta_imagini)

    def __getitem__(self, idx):
        #citim imaginile
        nume_imagine = self.eticheta_imagini.iloc[idx, 0]
        cale_imagine = os.path.join(self.folder_imagini, f"{nume_imagine}.png")
        #normalizam imagini pentru a fi in intervalul [0,1]
        imagine = read_image(cale_imagine).float() / 255.0

        #daca avem imagini alb-negru le expandam pe 3 canale
        if imagine.shape[0] == 1:
            imagine = imagine.expand(3, -1, -1)

        #aplicam o normalizare daca exista
        if self.transform:
            imagine = self.transform(imagine)
            
        eticheta = self.eticheta_imagini.iloc[idx, 1]

        return nume_imagine,imagine, eticheta

class CustomTrainClass(Dataset):
    def __init__(self, fisier_csv, folder_imagini, transform=None):
        self.etichete_imagini = pd.read_csv(fisier_csv)
        self.transform = transform
        self.folder_imagini = folder_imagini

    def __len__(self):
        return len(self.etichete_imagini)

    def __getitem__(self, idx):
        #citim imaginile de test
        nume_imagini = self.etichete_imagini.iloc[idx, 0]
        cale_imagini = os.path.join(self.folder_imagini, f"{nume_imagini}.png")
        imagine = read_image(cale_imagini)
        #convertim la float si impartim la 255 pentru a obtine pixeli cu valori in [0,1]
        imagine = imagine.float() / 255.0
        #imaginile alb negru le expandam
        if imagine.shape[0] == 1:
            imagine = imagine.expand(3, -1, -1)

        #aplicam daca exista o normalizare
        if self.transform:
            imagine = self.transform(imagine)
        label = self.etichete_imagini.iloc[idx, 0]
        return imagine, label

class ReteaConvolutionala(nn.Module):
    def __init__(self):
        super().__init__()
        #toate straturile le  punem intr-o lista pentru usurinta operatiei de forward
        self.convolutii_normalizare_relu = []
        #avem o convolutie, normalizare (re-centering si re-scaling ) si Relu ca functie de activare
        #deoarce avem batchNorm2d, bias-ul nu mai este relevant deci va fi setat pe False
        self.convolutii_normalizare_relu.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.convolutii_normalizare_relu.append(nn.BatchNorm2d(64))
        self.convolutii_normalizare_relu.append(nn.ReLU(inplace=False))

        self.convolutii_normalizare_relu.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.convolutii_normalizare_relu.append(nn.BatchNorm2d(64))
        self.convolutii_normalizare_relu.append(nn.ReLU(inplace=False))

        self.convolutii_normalizare_relu.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.convolutii_normalizare_relu.append(nn.BatchNorm2d(128))
        self.convolutii_normalizare_relu.append(nn.ReLU(inplace=False))

        self.convolutii_normalizare_relu.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.convolutii_normalizare_relu.append(nn.BatchNorm2d(128))
        self.convolutii_normalizare_relu.append(nn.ReLU(inplace=False))

        self.straturi=nn.Sequential(*self.convolutii_normalizare_relu)
        #spargem ce a gasit reteau in 3 clase diferite, conform criteriului de clasificare
        self.linear = nn.Linear(128, 3)

        #reducem dimensiunile la 1x1 prin pastrarea valorii medii
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    #definim cum vor trece imaginile prin retea
    #prima data prin straturile  (Conv, batchnorm si relu ), apoin prin avg_pooling,
    # mai apoi sunt aplatizate la un tensor de dim 1, apoi impartite in cele 3 categorii
    def forward(self, input):
        out = self.straturi(input)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out

def train(retea, opt, functie_pierdere, loader_training, dispozitiv):
    retea.train(True)
    #pentru a putea rula pe GPU
    retea.to(dispozitiv)
    predictii_corecte,predictii_totale,pierdere_train = 0,0,0

    #luam toate imaginile din load_train
    for _, imagini, etichete in loader_training:
        #sunt mutatate toate operatiile pe GPU pentru rapiditate
        imagini, etichete = imagini.to(dispozitiv), etichete.long().to(dispozitiv)
        #aici reteau face predictii
        predictii = retea(imagini)
        #se calculeaza pierderea (adica ce a clasificat gresit reteaua)
        pierdere = functie_pierdere(predictii, etichete)
        #imbunatatim reteau
        pierdere.backward()
        opt.step()
        opt.zero_grad()
        #aici calculam acuratetea si loss-ul la training
        pierdere_train += pierdere.item()
        pred = torch.argmax(predictii, dim=1)
        predictii_totale += etichete.size(0)
        predictii_corecte += pred.eq( etichete).type(torch.int32).sum().item()
    return pierdere_train / len(train_loader), predictii_corecte / predictii_totale



def validate(functie_pierdere,retea, validation_loader, dispozitiv):
    retea.eval()
    #mutam operatiile pe GPU
    retea.to(dispozitiv)
    lista_predictii,lista_id,lista_corect_y = [],[],[]
    predictii_corecte,pierdere_validare,predictii_totale = 0,0,0

    with torch.no_grad():
        #pentru validare nu mai vrem sa imbunatatim reteau, vrem doar sa vedem ce
        # acuratete da pentru un set nou de date, diferit de cel de train, dar pentru care avem etichetele corecte,
        #deci putem afla acuratetea
        for id,imagini, clasificare_target in validation_loader:
            lista_id += id
            lista_corect_y+=clasificare_target
            #se muta iar procesul pe gpu
            imagini, clasificare_target = imagini.to(dispozitiv), clasificare_target.long().to(dispozitiv)
            predictii_totale += clasificare_target.size(0)
            #reteau a facut preziceri
            dupa_clasificare = retea(imagini)
            pierderi = functie_pierdere(dupa_clasificare, clasificare_target)

            #aici aflam acuratea si loss-ul
            predictii = torch.argmax(dupa_clasificare, dim=1)
            lista_predictii += predictii.cpu().tolist()
            pierdere_validare += pierderi.item()
            predictii_corecte += predictii.eq (clasificare_target).type(torch.int32).sum().item()

    #rezultatul va fi un dictionar de forma id, eticheta si eticheta corecta
    rezultat = {"id": lista_id, "label": lista_predictii, "corect":lista_corect_y}
    return pierdere_validare / len(validation_loader), predictii_corecte / predictii_totale, rezultat

def predictii_pt_imgTest(retea, dispozitiv, test_loader):
    retea.eval()
    retea.to(dispozitiv)
    lista_id,lista_predictii = [],[]

    with torch.no_grad():
        #nu mai vrem ca reteau sa se imbunatateasca deci se apeleaza iar torch.no_grad()
        for imagini, ids in test_loader:
            lista_id += ids
            imagini = imagini.to(dispozitiv)
            dupa_test = retea(imagini)
            predictions = torch.argmax(dupa_test, dim=1)
            lista_predictii += predictions.cpu().tolist()

    #punem rezultatele intr-un dictionar cu formatul cerut
    rezultat = {'image_id': lista_id, 'label': lista_predictii}
    return rezultat

def incarcarea_retelei(path):
    checkpoint = torch.load(path)
    retea = ReteaConvolutionala()
    retea.load_state_dict(checkpoint)
    return retea

def salvarea_retelei(network, path):
    state = network.state_dict()
    torch.save(state, path)

def confusion_matrix(y_true, y_pred):
    #functie ce calculeaza matricea de confuzie (invatat in laborator)
    num_classes = len(np.unique(y_true))
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        conf_mat[true_label, pred_label] += 1

    return conf_mat

def plot_confusion_matrix(conf_mat, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("./matrix_confusion.jpg")
    plt.show()

if __name__ == '__main__':
    imagini_training = CustomImageDataset("realistic-image-classification/train.csv", "realistic-image-classification/train", transforms.Normalize((0.4912, 0.4820, 0.4464), (0.2472, 0.2436, 0.2617)))
    train_loader = DataLoader(imagini_training, batch_size=128, shuffle=True, drop_last=False)
    imagini_validare = CustomImageDataset("realistic-image-classification/validation.csv", "realistic-image-classification/validation", transforms.Normalize((0.4912, 0.4820, 0.4464), (0.2472, 0.2436, 0.2617)))
    validation_loader = DataLoader(imagini_validare, batch_size=128, shuffle=False, drop_last=False)
    acc_val_best = 0
    locatie_retea = './retea_0.pt'

    #initializam datele ce sunt folosite pentru antrenare, validarea si testarea retelei
    retea = ReteaConvolutionala()
    functie_pierdere = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(retea.parameters(), lr=0.005, momentum=0.9)
    dispozitiv = "cuda" if torch.cuda.is_available() else "cpu"

    valori_cc =[]
    cele_bune_rez ={}

    #am un sistem bazat pe epoci
    for epoch in range(5):
        pierdere_train, acc_antrenare= train(retea, optimizer, functie_pierdere, train_loader, dispozitiv)
        validare_pierdere, acc_validare, rezultat = validate(functie_pierdere, retea, validation_loader, dispozitiv)
        valori_cc.append(acc_validare)
        if acc_validare > acc_val_best:
            cele_bune_rez=rezultat
            acc_val_best = acc_validare
            salvarea_retelei(retea, locatie_retea)


    #plotam un grafic cu evolutia acuratetii pentru validare
    plt.figure(figsize=(10, 6))
    plt.plot(valori_cc, marker='o', linestyle='-', color='#660187', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig("./img_val.jpg")
    plt.show()
    #matrice de confuzie
    conf_mat = confusion_matrix(cele_bune_rez["corect"], cele_bune_rez["label"])
    class_names = ['Class 0', 'Class 1', 'Class 2']
    plot_confusion_matrix(conf_mat, class_names)


    imagini_test = CustomTrainClass("realistic-image-classification/test.csv", "realistic-image-classification/test", transforms.Normalize((0.4912, 0.4820, 0.4464), (0.2472, 0.2436, 0.2617)))
    retea = incarcarea_retelei(locatie_retea)

    test_loader = DataLoader(imagini_test, batch_size=128, shuffle=False, drop_last=False)
    rez = predictii_pt_imgTest(retea, dispozitiv, test_loader)
    date = pd.DataFrame(rez)
    date.to_csv('predictii_test.csv', index=False)






