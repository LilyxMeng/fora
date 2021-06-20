from flask import Flask, render_template, Response, request
import cv2 
import requests
from PIL import Image
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__, static_folder='static')

# loading machine learning model code
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # generate predictions
        loss = F.cross_entropy(out, labels) # calculate loss
        return loss
      
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # generate predictions
        loss = F.cross_entropy(out, labels)   # calculate loss
        acc = accuracy(out, labels)           # calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class Fruits360CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 50 x 50

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 25 x 25

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),#output :256*25*25
            nn.MaxPool2d(5, 5), # output: 256 x 5 x 5

            nn.Flatten(), 
            nn.Linear(256*5*5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 131))
               
    def forward(self, xb):
        return self.network(xb)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

model = to_device(Fruits360CnnModel(), device)
model.load_state_dict(torch.load('fora-model.pth'))

CLS = ['Tomato 4', 'Apple Red Delicious', 'Tomato 3', 'Huckleberry', 'Blueberry', 'Pear Red', 'Banana Lady Finger', 'Melon Piel de Sapo', 'Pear', 'Cherry 1', 'Strawberry', 'Nut Forest', 'Avocado', 'Tomato 2', 'Pomegranate', 'Dates', 'Carambula', 'Potato Red Washed', 'Granadilla', 'Kohlrabi', 'Tamarillo', 'Pepper Red', 'Fig', 'Ginger Root', 'Kiwi', 'Cherry Wax Yellow', 'Lemon', 'Guava', 'Apple Golden 2', 'Pear Stone', 'Apple Red 1', 'Cauliflower', 'Mandarine', 'Quince', 'Strawberry Wedge', 'Pear Monster', 'Raspberry', 'Pitahaya Red', 'Nut Pecan', 'Apple Golden 3', 'Redcurrant', 'Apple Red Yellow 1', 'Pepper Yellow', 'Grape Pink', 'Banana Red', 'Cucumber Ripe 2', 'Physalis', 'Cherry Rainier', 'Maracuja', 'Chestnut', 'Plum', 'Potato Sweet', 'Cucumber Ripe', 'Hazelnut', 'Nectarine', 'Cherry Wax Black', 'Cantaloupe 2', 'Lychee', 'Pepper Orange', 'Clementine', 'Watermelon', 'Pear Kaiser', 'Mangostan', 'Cherry 2', 'Pineapple Mini', 'Rambutan', 'Grape White', 'Tomato Yellow', 'Apple Braeburn', 'Tomato Maroon', 'Onion White', 'Onion Red Peeled', 'Mango', 'Potato White', 'Apple Crimson Snow', 'Potato Red', 'Corn Husk', 'Cocos', 'Mulberry', 'Avocado ripe', 'Tomato 1', 'Passion Fruit', 'Apple Granny Smith', 'Beetroot', 'Kumquats', 'Grape White 2', 'Apricot', 'Eggplant', 'Limes', 'Corn', 'Grape White 4', 'Grape White 3', 'Tomato Heart', 'Apple Pink Lady', 'Plum 3', 'Pear Williams', 'Tomato not Ripened', 'Peach 2', 'Pomelo Sweetie', 'Salak', 'Grapefruit Pink', 'Apple Golden 1', 'Banana', 'Apple Red 2', 'Onion Red', 'Physalis with Husk', 'Apple Red Yellow 2', 'Grape Blue', 'Lemon Meyer', 'Plum 2', 'Pepino', 'Tangelo', 'Cactus fruit', 'Papaya', 'Apple Red 3', 'Walnut', 'Pear Abate', 'Pear 2', 'Pear Forelle', 'Pineapple', 'Tomato Cherry Red', 'Cherry Wax Red', 'Mango Red', 'Orange', 'Nectarine Flat', 'Kaki', 'Pepper Green', 'Grapefruit White', 'Peach', 'Cantaloupe 1', 'Peach Flat']

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    pred = torch.argmax(preds)
    return CLS[pred.item()]
    # Retrieve the class label
    #return model.classes[preds[0].item()]


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/scanning", methods=["POST", "GET"])
def scanning():
    prediction = "The image is invalid or unidentifiable"
    return render_template("scanning.html", prediction=prediction)


@app.route("/info", methods=["POST", "GET"])
def info():
    return render_template("info.html")


@app.route("/img", methods=["POST", "GET"])
def img():
    prediction = "The image is invalid or unidentifiable"

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("result.html", prediction=prediction)

        file = request.files["file"]

        if file.filename == "":
            return render_template("scanning.html", prediction=prediction)

        if file:
            img = Image.open(file).convert('RGB')

            transform = transforms.Compose([transforms.Resize((100, 100)),transforms.ToTensor()])

            image = transform(img)

            prediction = predict_image(image, model)
            
            print(prediction)

            if "Apple" in prediction:
                prediction = "Apple"
                nutrition = food_search("apple")

            elif "Tomato" in prediction:
                prediction = "Tomato"
                nutrition = food_search("tomato")

            elif "Huckleberry" in prediction:
                prediction = "Huckleberry"
                nutrition = food_search("huckleberry")

            elif "Blueberry" in prediction:
                prediction = "Blueberry"
                nutrition = food_search("blueberry")

            elif "Pear" in prediction:
                prediction = "Pear"
                nutrition = food_search("pear")

            elif "Banana" in prediction:
                prediction = "Banana"
                nutrition = food_search("banana")
            
            elif "Melon" in prediction:
                prediction = "Melon"
                nutrition = food_search("melon")

            elif "Cherry" in prediction:
                prediction = "Cherry"
                nutrition = food_search("cherry")

            elif "Strawberry" in prediction:
                prediction = "Strawberry"
                nutrition = food_search("strawberry")

            elif "Nut" in prediction:
                prediction = "Nut"
                nutrition = food_search("nut")

            elif "Avocado" in prediction:
                prediction = "Avocado"
                nutrition = food_search("avocado")
            
            elif "Pomegranate" in prediction:
                prediction = "Pomegranate"
                nutrition = food_search("pomegranate")

            elif "Dates" in prediction:
                prediction = "Dates"
                nutrition = food_search("dates")

            elif "Carambula" in prediction:
                prediction = "Carambula"
                nutrition = food_search("carambula")

            elif "Potato" in prediction:
                prediction = "Potato"
                nutrition = food_search("potato")

            elif "Granadilla" in prediction:
                prediction = "Granadilla"
                nutrition = food_search("granadilla")

            elif "Kohlrabi" in prediction:
                prediction = "Kohlrabi"
                nutrition = food_search("kohlrabi")

            elif "Tamarillo" in prediction:
                prediction = "Tamarillo"
                nutrition = food_search("tamarillo")

            elif "Pepper" in prediction:
                prediction = "Pepper"
                nutrition = food_search("pepper")

            elif "Fig" in prediction:
                prediction = "Fig"
                nutrition = food_search("fig")

            elif "Ginger" in prediction:
                prediction = "Ginger"
                nutrition = food_search("ginger")

            elif "Kiwi" in prediction:
                prediction = "Kiwi"
                nutrition = food_search("kiwi")

            elif "Lemon" in prediction:
                prediction = "Lemon"
                nutrition = food_search("lemon")

            elif "Guava" in prediction:
                prediction = "Guava"
                nutrition = food_search("guava")

            elif "Cauliflower" in prediction:
                prediction = "Cauliflower"
                nutrition = food_search("cauliflower")

            elif "Mandarine" in prediction:
                prediction = "Mandarine"
                nutrition = food_search("mandarine")

            elif "Quince" in prediction:
                prediction = "Quince"
                nutrition = food_search("quince")

            elif "Raspberry" in prediction:
                prediction = "Raspberry"
                nutrition = food_search("raspberry")

            elif "Pitahaya" in prediction:
                prediction = "Pitahaya"
                nutrition = food_search("pitahaya")

            elif "Redcurrant" in prediction:
                prediction = "Redcurrant"
                nutrition = food_search("redcurrant")

            elif "Chestnut" in prediction:
                prediction = "Chestnut"
                nutrition = food_search("chestnut")

            elif "Plum" in prediction:
                prediction = "Plum"
                nutrition = food_search("plum")

            elif "Cucumber" in prediction:
                prediction = "Cucumber"
                nutrition = food_search("cucumber")

            elif "Hazelnut" in prediction:
                prediction = "Hazelnut"
                nutrition = food_search("hazelnut")

            elif "Nectarine" in prediction:
                prediction = "Nectarine"
                nutrition = food_search("nectarine")

            elif "Cantaloupe" in prediction:
                prediction = "Cantaloupe"
                nutrition = food_search("cantaloupe")

            elif "Lychee" in prediction:
                prediction = "Lychee"
                nutrition = food_search("lychee")

            elif "Clementine" in prediction:
                prediction = "Clementine"
                nutrition = food_search("clementine")

            elif "Watermelon" in prediction:
                prediction = "Watermelon"
                nutrition = food_search("watermelon")

            elif "Mangostan" in prediction:
                prediction = "Mangostan"
                nutrition = food_search("mangostan")

            elif "Pineapple" in prediction:
                prediction = "Pineapple"
                nutrition = food_search("pineapple")

            elif "Rambutan" in prediction:
                prediction = "Rambutan"
                nutrition = food_search("rambutan")

            elif "Grape" in prediction:
                prediction = "Grape"
                nutrition = food_search("grape")

            elif "Onion" in prediction:
                prediction = "Onion"
                nutrition = food_search("onion")

            elif "Mango" in prediction:
                prediction = "Mango"
                nutrition = food_search("mango")

            elif "Corn" in prediction:
                prediction = "Corn"
                nutrition = food_search("corn")

            elif "Cocos" in prediction:
                prediction = "Cocos"
                nutrition = food_search("cocos")

            elif "Mulberry" in prediction:
                prediction = "Mulberry"
                nutrition = food_search("mulberry")

            elif "Passion Fruit" in prediction:
                prediction = "Passion Fruit"
                nutrition = food_search("passion fruit")

            elif "Beetroot" in prediction:
                prediction = "Beetroot"
                nutrition = food_search("beetroot")

            elif "Kumquats" in prediction:
                prediction = "Kumquats"
                nutrition = food_search("kumquats")

            elif "Apricot" in prediction:
                prediction = "Apricot"
                nutrition = food_search("apricot")

            elif "Eggplant" in prediction:
                prediction = "Eggplant"
                nutrition = food_search("eggplant")

            elif "Limes" in prediction:
                prediction = "Limes"
                nutrition = food_search("limes")

            elif "Physalis" in prediction:
                prediction = "Physalis"
                nutrition = food_search("physalis")
            
            elif "Pepino" in prediction:
                prediction = "Pepino"
                nutrition = food_search("pepino")

            elif "Tangelo" in prediction:
                prediction = "Tangelo"
                nutrition = food_search("Tangelo")

            elif "Cactus fruit" in prediction:
                prediction = "Cactus fruit"
                nutrition = food_search("cactus fruit")

            elif "Papaya" in prediction:
                prediction = "Papaya"
                nutrition = food_search("papaya")

            elif "Walnut" in prediction:
                prediction = "Walnut"
                nutrition = food_search("walnut")

            elif "Kaki" in prediction:
                prediction = "Kaki"
                nutrition = food_search("kaki")

            elif "Grapefruit" in prediction:
                prediction = "Grapefruit"
                nutrition = food_search("grapefruit")

            elif "Peach" in prediction:
                prediction = "Peach"
                nutrition = food_search("peach")

            elif "Cantaloupe" in prediction:
                prediction = "Cantaloupe"
                nutrition = food_search("cantaloupe")


    return render_template("scanning.html", nutrition=nutrition, prediction=prediction)

    return render_template("scanning.html", nutrition=nutrition, prediction=prediction)


def food_search(query):
#https://api.edamam.com/api/nutrition-data?app_id=997f7f0c&app_key=07144bd90ea108d780ade51ac08aa0f0&nutrition-type=logging&ingr=banana


    curl = f"https://api.edamam.com/api/nutrition-data?app_id=997f7f0c&app_key=07144bd90ea108d780ade51ac08aa0f0&nutrition-type=logging&ingr=={query}" #\
           #f"&pageSize=2&api_key={'9KUGdc5Li2UFBwwgyBmRK0hUU1LZi4kBmC7UmCcw'}"
    
    response = requests.get(curl)
    nutrition = response.json()['totalNutrients']
    nutritional_list = []
    for k in nutrition:
        nutritionl = nutrition[k]['label']
        
        nutritionq = nutrition[k]['quantity']
        nutritionq = "{:.2f}".format(nutritionq)
        nutritionu = nutrition[k]['unit']

        nutritional = nutritionl  + " " + nutritionq + nutritionu
        nutritional_list.append(nutritional)

    #print(nutritional_list)

    #print(type(nutritionq))


    #print(nutrition['FASAT']['label'])

    #print(nutrition.keys())

    return nutritional_list

if __name__ == "__main__":
    app.run(debug=True)
