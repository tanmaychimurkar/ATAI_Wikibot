from question_answer import *
from assertion_statements import *
from multimedia import *
from recommendation_questions import *
import requests
import json
import time
from transformers import pipeline

link_list = ['page', 'webpage', 'link', 'url', 'website', 'IMDb', 'wikidata', 'wiki']

context_json = {}
context_json_name = {}

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
candidate_labels = ['question', 'recommendation', 'image']


def full_function(query_input, room_id):
    try:
        if room_id not in context_json.keys():
            context_json[room_id] = []
            context_json_name[room_id] = []
        query_input_check = remove_imdb_name(query_input)
        if room_id in context_json:
            if len(movie_entity_code(query_input_check)) == 0:
                try:
                    query_entity = [context_json[room_id][-1]]
                    query_entity_name = [context_json_name[room_id][-1]]
                except:
                    pass
            else:
                query_entity = movie_entity_code(query_input_check)
                query_entity_name = movie_entity_name(query_input_check)
                context_json[room_id].append(query_entity[0])
                context_json_name[room_id].append(query_entity_name[0])
        else:
            query_entity = movie_entity_code(query_input_check)
            query_entity_name = movie_entity_name(query_input_check)
            context_json[room_id].append(query_entity[0])
            context_json_name[room_id].append(query_entity_name[0])

        text = query_input.translate(str.maketrans('', '', string.punctuation))
        text_clean = text.split(' ')
        if len(set(link_list) & set(text_clean)) > 0:
            out_query = get_url(query_input, query_entity, query_entity_name, text_clean)
        elif len({'frame', 'still', 'stillframe'} & set(text_clean)) > 0:
            out_query = fetch_image2(query_input, query_entity, query_entity_name)
        else:
            text = remove_imdb_name(text)
            zero_shot = classifier(text, candidate_labels)
            if zero_shot['labels'][0] == 'question':
                print('inside_question')
                out_query = assertion_function(query_input, query_entity)
            elif zero_shot['labels'][0] == 'recommendation':
                print("inside recommendation")
                out_query = generate_recommendations(query_input, query_entity)
            elif zero_shot['labels'][0] == 'image':
                print('inside image')
                out_query = fetch_image2(query_input, query_entity, query_entity_name)
    except:
        out_query = "Hmmm, it seems I cannot answer this mysterious question from the realm of real Humans. Can you please try with a different question kind Human! Thank you very much :D"
    return out_query


# url of the speakeasy server
url = "https://speakeasy.ifi.uzh.ch"

# get the api specification
r = requests.get(url + "/client-specs")
spec = json.loads(r.text)


# user login
def login(username: str, password: str):
    return requests.post(url=url + "/api/login", json={"username": username, "password": password})


# check details of the current user
def current(session_token: str):
    return requests.get(url=url + "/api/user/current", params={"session": session_token})


# user logout
def logout(session_token: str):
    return requests.get(url=url + "/api/logout", params={"session": session_token})


# check available chat rooms
def check_rooms(session_token: str):
    return requests.get(url=url + "/api/rooms", params={"session": session_token})


# check the state of a chat room
def check_room_state(room_id: str, since: int, session_token: str):
    return requests.get(url=url + "/api/room/{}/{}".format(room_id, since),
                        params={"roomId": room_id, "since": since, "session": session_token})


# post a message to a chat room
def post_message(room_id: str, session_token: str, message: str):
    return requests.post(url=url + "/api/room/{}".format(room_id), params={"roomId": room_id, "session": session_token},
                         data=message)


random_greetings = ['Hellllllo', 'Hi there', 'Holaaa', 'Gruezi', 'Hello there kind Human',
                    'Hello. I might be a robot, but I like cats']

agent_details = login(username='removed', password='removed').json()
print("--- agent details:")
print(json.dumps(agent_details, indent=4))

chatroom_messages = {}
while True:
    current_rooms = check_rooms(session_token=agent_details["sessionToken"]).json()["rooms"]
    print("--- {} chatrooms available".format(len(current_rooms)))

    for idx, room in enumerate(current_rooms):
        room_id = room["uid"]
        print("chat room - {}: {}".format(idx, room_id))

        new_room_state = check_room_state(room_id=room_id, since=0, session_token=agent_details["sessionToken"]).json()
        new_messages = new_room_state["messages"]
        print("found {} messages".format(len(new_messages)))

        if room_id not in chatroom_messages.keys():
            chatroom_messages[room_id] = []
            response = 'Hello there, superior Human Being. Thanks for joining my chat room. I am bot_752, ' \
                       'at your service! I can answer questions of these type: 1)Who is the actor in Top Gun? 2)Who ' \
                       'is the director of it? 3)Show me movies similar to Top Gun. 4) Show me action movies by Tom ' \
                       'Cruise. 5)Show me a picture of him. 5) Show me a stillframe from Top Gun. 6) When was it ' \
                       'released? 7)What is Top Gun. Please feel free to ask other questions that you might have, ' \
                       'I will try my best to answer them :D '.encode(
                'utf-8')
            post_message(room_id=room_id, session_token=agent_details["sessionToken"], message=response)

        if len(chatroom_messages[room_id]) != len(new_messages):
            for message in new_messages:
                if message["ordinal"] >= len(chatroom_messages[room_id]) and message["session"] != agent_details[
                    "sessionId"]:
                    if len(message['message']) < 10:
                        response = random.choice(random_greetings)
                    else:
                        response = full_function(message['message'], room_id).encode('utf-8')
                    post_message(room_id=room_id, session_token=agent_details["sessionToken"], message=response)

        chatroom_messages[room_id] = new_messages

    time.sleep(1)
    print("")
