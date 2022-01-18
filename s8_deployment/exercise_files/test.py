# from fileinput import filename
# from google.cloud import storage
import pickle

filename="model.pkl"
my_model = pickle.load(open(filename, 'rb'))

""" will to stuff to your request """
data="5.1,3.5,1.4,0.2"

input_data = list(map(float, data.split(',')))

print(input_data)
prediction = my_model.predict([input_data]) 
print(f'Belongs to class: {prediction}')

