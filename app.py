#!/usr/bin/env python
# encoding: utf-8
import json
from transformers import BertModel, BertTokenizer
from flask import Flask, jsonify, request
# from flask_restful import Resource, Api 
import pandas as pd
import numpy as np
import pickle
import torch
app = Flask(__name__)
# api = Api(app)

def extract_features(text):
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize the text
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Get the hidden states for each token
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs[2]
    # Concatenate the last 4 hidden states
    token_vecs = []
    for layer in range(-4, 0):
        token_vecs.append(hidden_states[layer][0])
    # Calculate the mean of the last 4 hidden states
    features = []
    for token in token_vecs:
        features.append(torch.mean(token, dim=0))
    # Return the features as a tensor
    return torch.stack(features)

    


    # Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))

@app.route('/url-checker', methods=['POST'])
def get():
	
    data = request.get_json()
    url = data['url']
    pred = extract_features(url).numpy()
    pred = np.reshape(pred, (1, 3072))
    model = pickle.load(open('model.pkl','rb'))      
    print(model.predict(pred))
    prediction_array = model.predict(pred)
    prediction_value = prediction_array[0]
    response = {"prediction_value": prediction_value}
    # print(prediction_value)
    return jsonify(response)
			# return "jshsa" 	

# class UrlPrediction(Resource): 



     
    
# api.add_resource(UrlPrediction, '/')     
if __name__ == "__main__":
    app.run(debug=True)


# # using flask_restful 
# from flask import Flask, jsonify, request 
# from flask_restful import Resource, Api 

# # creating the flask app 
# app = Flask(__name__) 
# # creating an API object 
# api = Api(app) 

# # making a class for a particular resource 
# # the get, post methods correspond to get and post requests 
# # they are automatically mapped by flask_restful. 
# # other methods include put, delete, etc. 
# class Hello(Resource): 

# 	# corresponds to the GET request. 
# 	# this function is called whenever there 
# 	# is a GET request for this resource 
# 	def get(self): 

# 		return jsonify({'message': 'hello world'}) 

# 	# Corresponds to POST request 
# 	def post(self): 
		
# 		data = request.get_json()	 # status code 
# 		return jsonify({'data': data}), 201


# # another resource to calculate the square of a number 
# class Square(Resource): 

# 	def get(self, num): 

# 		return jsonify({'square': num**2}) 


# # adding the defined resources along with their corresponding urls 
# api.add_resource(Hello, '/') 
# api.add_resource(Square, '/square/<int:num>') 


# # driver function 
# if __name__ == '__main__': 

# 	app.run(debug = True) 






# from flask import Flask, jsonify, request

# app = Flask(__name__)

# incomes = [
#     { 'description': 'salary', 'amount': 5000 }
# ]


# @app.route('/incomes')
# def get_incomes():
#     return jsonify(incomes)


# @app.route('/incomes', methods=['POST'])
# def add_income():
#     data = request.get_json()
#     print(data["amount"])
#     incomes.append(request.get_json())
#     return jsonify(data["amount"])