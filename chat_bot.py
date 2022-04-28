import json

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model

with open('database.json') as database:
    data1 = json.load(database)

tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"input patterns": inputs, 'tags': tags})

import string

data['input patterns'] = data['input patterns'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['input patterns'] = data['input patterns'].apply(lambda wrd: ''.join(wrd))

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['input patterns'])
train = tokenizer.texts_to_sequences(data['input patterns'])
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(train)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]

vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
train = model.fit(x_train, y_train, epochs=300)

import random

while True:
    texts_p = []
    prediction_input = input("You : ")
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)
    output = model.predict(prediction_input)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    print("Trippy : ", random.choice(responses[response_tag]))
    hotelid = ""

    if response_tag == 'goodbye':
        break
    if response_tag == 'location':
        import requests

        print("Tripy: In which city do you want me to search a hotel? ")
        city = input("You: ")
        print("Tripy: First tell me your check in date yyyy-mm-dd format !")
        checkin = input("You: ")
        print("Tripy: Let me know your check out date yyyy-mm-dd format !")
        checkout = input("You: ")
        print("Tripy: How many rooms do you require?")
        nr = input("You: ")
        print("Tripy: Number of adults?")
        adult = input("You: ")
        print("Tripy: Number of children? (Ages should be less than or equal to 5)")
        children = input("You: ")
        url = "https://booking-com.p.rapidapi.com/v1/hotels/search"

        querystring = {
            "checkout_date": checkout, "room_number": nr, "filter_by_currency": "INR", "dest_type": "city",
            "locale": "en-us", "checkin_date": checkin,
            "adults_number": adult, "order_by": "popularity", "units": "metric", "dest_id": "",
            "children_number": children,
            "categories_filter_ids": "class::2,class::4,free_cancellation::1",
            "children_ages": "5,0", "include_adjacency": "true", "page_number": "0"
        }

        if (city.lower()) == 'pune' or (city.lower()) == 'pnq':
            querystring.update({"dest_id": "-2108361"})
        elif (city.lower()) == 'mumbai' or (city.lower()) == 'bom' or (city.lower()) == 'bombay':
            querystring.update({"dest_id": "-2092174"})
        elif (city.lower()) == 'delhi' or (city.lower()) == 'del':
            querystring.update({"dest_id": "-2106102"})
        elif (city.lower()) == 'blr' or (city.lower()) == 'banglore' or (city.lower()) == 'bengaluru':
            querystring.update({"dest_id": "-2090174"})

        headers = {
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com",
            "X-RapidAPI-Key": "85926a460emsh91319b0ae6a051ap188c86jsn74155e259fc6"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)

        data = response.json()

        print(f"---------- Top 5 hotels in {city} ----------")
        for i in range(5):
            hotel = data["result"][i]["hotel_name"]
            print(i + 1, "", hotel)


        def getHotelDetails(id):
            print("Rs ", round(data["result"][id]["min_total_price"]), " /-")
            print("Address: ", data["result"][id]["address"])
            print("Distance from city center: ", data["result"][id]["distances"][0]["text"])
            print("Review: ", data["result"][id]["review_score"], " ", data["result"][id]["review_score_word"])

            url1 = "https://booking-com.p.rapidapi.com/v1/hotels/facilities"
            querystring = {"hotel_id": data["result"][id]["hotel_id"], "locale": "en-us"}
            headers = {
                "X-RapidAPI-Host": "booking-com.p.rapidapi.com",
                "X-RapidAPI-Key": "85926a460emsh91319b0ae6a051ap188c86jsn74155e259fc6"
            }
            response = requests.request("GET", url1, headers=headers, params=querystring)
            print("Facilities: ")
            data1 = response.json()
            for i in range(42):
                print(data1[i]["facility_name"])


        choice = int(input("Select Hotel: "))
        if choice == 1:
            getHotelDetails(0)
        elif choice == 2:
            getHotelDetails(1)
        elif choice == 3:
            getHotelDetails(2)
        elif choice == 4:
            getHotelDetails(3)
        elif choice == 5:
            getHotelDetails(4)
