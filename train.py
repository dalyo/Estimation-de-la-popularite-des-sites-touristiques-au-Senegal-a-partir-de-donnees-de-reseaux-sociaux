# USAGE
# python train.py --dataset donnee_tourist --epochs 25

# importation des paquets nécessaires
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construction de l'argument parser et du parse des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


# Récuperation de la liste des images dans notre répertoire, 
# puis initialisation de la liste des données et des images de classe

print("[INFO] Chargement des données...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#  Boucler sur les paths d'images 
for imagePath in imagePaths:
	# extraire le label de classe du nom du fichier
	label = imagePath.split(os.path.sep)[-2]

	# charger l'image et la redimensionner pour qu'elle ait une taille fixe 
	# de 128x128 pixels,
	# ignorer le rapport d'aspect
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (128, 128))

	# mettre à jour les listes de données et d'étiquettes, respectivement
	data.append(image)
	labels.append(label)

# convertir les données et les étiquettes en tableaux NumPy
data = np.array(data)
labels = np.array(labels)

# effectuer un one hot-encoding sur les étiquettes
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

# initialisation de l'objet d'augmentation des données de formation
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialiser l'objet d'augmentation des données de validation/test 
#(auquel nous ajouterons une soustraction moyenne)
valAug = ImageDataGenerator()

# définir la soustraction moyenne d'ImageNet 
#(dans l'ordre RGB) et fixer la valeur de la soustraction moyenne 
# pour chacun des objets d'augmentation des données
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# charger le VGG16, en veillant à ne pas utiliser les jeux 
# de couches FC de tête, tout en adaptant la taille du tenseur 
# d'image d'entrée au réseau
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(128, 128, 3)))

# montrer un résumé du modèle de base
print("[INFO] Description du modèle de base...")
print(baseModel.summary())

# construire la tête du modèle qui sera placée 
# sur le modèle de base
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# boucler sur toutes les couches du modèle de base et les geler
# afin qu'elles ne soient pas mises à jour lors du premier processus de formation
for layer in baseModel.layers:
	layer.trainable = False

# compiler notre modèle (cela doit être fait après avoir 
# réglé nos couches de manière à ce qu'elles ne puissent pas être formées)
print("[INFO] compilation du model...")
opt = Adam(lr=1e-4)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# former la tête du réseau pendant quelques époques 
#(toutes les autres couches sont gelées) 
# - cela permettra aux nouvelles couches FC de commencer 
# à s'initialiser avec des valeurs réelles "apprises" par opposition à l'aléatoire pur
print("[INFO] Entrainement de la tête ...")
H = model.fit(
	x=trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])

# Evaluation du réseau
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# tracer la perte de formation et la précision
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])