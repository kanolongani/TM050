import requests
import json
import pickle
import pandas as pd
from json.decoder import JSONDecodeError
from flask import Flask, render_template, request

app = Flask(__name__)

def nums_length_ratio(fullname):
    num_count = sum(c.isdigit() for c in fullname)
    
    total_length = len(fullname)
    
    ratio = num_count / total_length if total_length != 0 else 0
    
    return ratio

def nums_length_ratio_(username):
    num_count = sum(c.isdigit() for c in username)
    
    total_length = len(username)
    
    ratio = num_count / total_length if total_length != 0 else 0
    
    return ratio

def take_prediction(username):

    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}&hl=en"
    print(url)
    payload = {}
    headers = {
    'authority': 'www.instagram.com',
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'cookie': 'csrftoken=QvuYxq5LH9kpnwABcR8cifVlFF1geyDc; mid=Y4iqcQALAAFIbXWp8jsQkNBZoDYn; ig_nrcb=1; ig_did=63AC029B-4F79-4342-984A-7608041E7273; datr=cKqIYwvHcPYCzQYyTY8jA7x4; dpr=1.25; csrftoken=QvuYxq5LH9kpnwABcR8cifVlFF1geyDc',
    'dpr': '1.25',
    'referer': 'https://www.instagram.com/_karan__longani_/?hl=en',
    'sec-ch-prefers-color-scheme': 'dark',
    'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
    'sec-ch-ua-full-version-list': '"Chromium";v="116.0.5845.188", "Not)A;Brand";v="24.0.0.0", "Google Chrome";v="116.0.5845.188"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': '"Windows"',
    'sec-ch-ua-platform-version': '"15.0.0"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'viewport-width': '983',
    'x-ig-app-id': '936619743392459',
    'x-ig-www-claim': '0',
    'x-requested-with': 'XMLHttpRequest'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    response = response.text
    response = json.loads(response)
    print(response)


    followers = response['data']['user']['edge_followed_by']['count']
    following = response['data']['user']['edge_follow']['count']
    fullname = response['data']['user']['full_name']
    fname = fullname.split()
    acc_private = response['data']['user']['is_private']
    external_url = response['data']['user']['external_url']
    profile_pic = response['data']['user']['profile_pic_url']
    username = response['data']['user']['username']
    post = response['data']['user']['edge_owner_to_timeline_media']['count']
    description  = response['data']['user']['biography']



    print(f"followers : {followers}")
    print(f"following : {following}")
    print(f"name : {fullname}")
    print(f"username : {username}")
    print(f"account private : {acc_private}")
    print(f"no. of post :{post}")

    if fullname == username:
        namequeal = 1
    else:
        namequeal = 0


    if external_url:
        ex_url = 1
    else:
        ex_url = 0

    print(f"external url : {ex_url}")
    print(f"name_username_match: {namequeal}")
    print(f"length of description : {len(description)}")


    if profile_pic:
        print("profile_pic is found")
    else:
        print("profile_pic is not found")



    ratio_for_fullname = nums_length_ratio(fullname)
    print("Numerical characters ratio in the fullname:", ratio_for_fullname)



    ratio_for_username = nums_length_ratio_(username)
    print("Numerical characters ratio in the username:", ratio_for_username)

    try:
        activity_ratio = post / followers
    except:
        activity_ratio = 0

    if followers > following:
        real = 1
    else:
        real = 0


    with open('catboost_model.pkl', 'rb') as f:
        catboost_pipeline = pickle.load(f)

    data = pd.DataFrame({'profile pic': [1],                     
                        'nums/length username': [ratio_for_username],                
                        'fullname words': [len(fname)],                 
                        'nums/length fullname': [ratio_for_fullname],               
                        'name==username': [namequeal],           
                        'description length': [len(description)],           
                        'external URL': [ex_url],           
                        'private': [acc_private],                
                        '#posts': [post],                       
                        '#followers': [followers],                  
                        '#follows': [following],  
                        'activity ratio':[activity_ratio],
                        '#followers > #follows?':[real]
                        }) 
    # predictions = catboost_pipeline.predict(data.drop(['fullname', 'external_url', 'profile_pic', 'username', 'description'], axis=1))
    predictions = catboost_pipeline.predict(data)
    print("Predictions:", predictions)

    return predictions



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/store_username', methods=['POST'])
def store_username():
    username = request.form['username']
    print("Received username:", username)
    try:
        predictions = take_prediction(username)
        print("Predictions:", predictions)
        return render_template('prediction_result.html', predictions=predictions)
    
    except JSONDecodeError:
        return render_template('invalid.html')
    

if __name__ == '__main__':
    app.run()
