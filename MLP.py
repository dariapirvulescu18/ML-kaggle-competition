import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from torchvision.io import read_image
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms


#cum am scris si in documnetatie, folosesc iar clasa CustomImageDataset invatat in laboratorul 6
#pentru a citi si prelucra datele
class CustomImageDataset(Dataset):
    def __init__(self, fisier_csv, folder_imagini, transform=None):
        self.folder_imagini = folder_imagini
        self.eticheta_imagini = pd.read_csv(fisier_csv)
        self.transform = transform


    def __len__(self):
        # return 20
        return len(self.eticheta_imagini)

    def __getitem__(self, idx):
        #citim si normalizam imaginile
        nume_imagine = self.eticheta_imagini.iloc[idx, 0]
        cale_imagine = os.path.join(self.folder_imagini, f"{nume_imagine}.png")
        imagine = read_image(cale_imagine).float() / 255.0

        if imagine.shape[0] == 1:
            imagine = imagine.expand(3, -1, -1)

        if self.transform:
            imagine = self.transform(imagine)
        imagine = imagine.reshape(-1)
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
        #citim si normalizam imaginile
        nume_imagini = self.etichete_imagini.iloc[idx, 0]
        cale_imagini = os.path.join(self.folder_imagini, f"{nume_imagini}.png")
        imagine = read_image(cale_imagini)
        imagine = imagine.float() / 255.0
        if imagine.shape[0] == 1:
            imagine = imagine.expand(3, -1, -1)
        if self.transform:
            imagine = self.transform(imagine)
        imagine = imagine.reshape(-1)

        label = self.etichete_imagini.iloc[idx, 0]
        return imagine, label


imagini_training = CustomImageDataset("realistic-image-classification/train.csv", "realistic-image-classification/train", transforms.Normalize((0.4912, 0.4820, 0.4464), (0.2472, 0.2436, 0.2617)))
imagini_validare = CustomImageDataset("realistic-image-classification/validation.csv", "realistic-image-classification/validation", transforms.Normalize((0.4912, 0.4820, 0.4464), (0.2472, 0.2436, 0.2617)))
test_image_set = CustomTrainClass("realistic-image-classification/test.csv", "realistic-image-classification/test", transforms.Normalize((0.4912, 0.4820, 0.4464), (0.2472, 0.2436, 0.2617)))

#luam ce ne trebuie(imaginea sau eticheta returnata din datele initiale citite cu CustomImageDataset

X_train = []
y_train =[]
for nume,img,eticheta  in imagini_training:
    X_train.append(img)
    y_train.append(eticheta)
X_train= np.array(X_train)
y_train= np.array(y_train)

X_validation =[]
y_validation =[]
for nume,img,eticheta in imagini_validare:
    X_validation.append(img)
    y_validation.append(eticheta)

X_test =[]
nume_test =[]
for img,nume in test_image_set:
    X_test.append(img)
    nume_test.append(nume)


#definim modelul
print(X_train.shape)
print(y_train.shape)
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(150), activation='relu', solver='adam',
                                     alpha=0.01, batch_size=64, learning_rate='adaptive',
                                     learning_rate_init=0.05, power_t=0.5, max_iter=200, shuffle=True,
                                     random_state=None, tol=0.0001, momentum=0.9, early_stopping=False,
                                     validation_fraction=0.1, n_iter_no_change=10)

best_accuracy = 0
best_epoch = 0
y_validation_pred=0
#pentru a vedea progresul am folosit tqdm
for epoch in range(1, mlp_classifier_model.max_iter + 1):
    with tqdm(total=len(X_train), desc=f"Training Epoch {epoch}") as pbar:
        #training-ul
        mlp_classifier_model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        pbar.update(len(X_train))

    # Validarea
    y_validation_pred = mlp_classifier_model.predict(X_validation)
    accuracy = np.mean(y_validation_pred == y_validation)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch

    print(f"Epoch {epoch}: Validation Accuracy = {accuracy:.4f}")

print(f"Best Validation Accuracy: {best_accuracy:.4f} at Epoch {best_epoch}")


# Predictii pentru datele de validare
# with tqdm(total=len(X_validation), desc="Validating") as pbar:
#     y_validation_pred = mlp_classifier_model.predict(X_validation)
#     pbar.update(len(X_validation))
#     print("Validation Results")
#     print(confusion_matrix(y_validation, y_validation_pred))
#     print(classification_report(y_validation, y_validation_pred))

# Predctii pentru test
with tqdm(total=len(X_test), desc="Testing") as pbar:
    test_predictions = mlp_classifier_model.predict(X_test)
    pbar.update(len(X_test))


print("Validation Results")
print(confusion_matrix(y_validation, y_validation_pred))
print(classification_report(y_validation, y_validation_pred))


# Cream fisierul csv
submission = pd.DataFrame({
    'image_id': nume_test,
    'label': test_predictions
})

submission['image_id'] = submission['image_id'].str.replace('.png', '')

submission.to_csv('submission.csv', index=False)

cm = confusion_matrix(y_validation, y_validation_pred)

#matricea de confuzie plotata
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mlp_classifier_model.classes_, yticklabels=mlp_classifier_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('./matrice_neuro.jpg')
plt.show()

history = mlp_classifier_model.fit(X_train, y_train)


#plotoam evolutia loss-ului
plt.plot(history.loss_curve_)
plt.title('Evolution of Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('./evolution_neuro.jpg')
plt.show()
