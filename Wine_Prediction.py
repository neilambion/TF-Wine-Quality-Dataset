import tensorflow as tf
import os
from keras.models import load_model

def clear():
    os.system('cls')

clear()
sample = {}
for feature in list(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']):
    while True:
        try:
            print("Wine Type and Quality Predictor")
            sample["%s" % feature] = float(input("Please input data for {}: ".format(feature.replace("_"," "))))
        except ValueError:
            print("Please enter an number!")
            continue
        else:
            break
    clear()

model = load_model('wine_model.h5')
input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

clear()
print(
    f"This wine is {'white' if predictions[1]==1 else 'red'}"
    f"\nQuality: {int(predictions[0])}"
)