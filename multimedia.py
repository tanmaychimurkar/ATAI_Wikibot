import json
import random
import string
import difflib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from rdflib import Graph

g = Graph()
g.parse('14_graph.nt', format='turtle')

with open('images.json') as f:
    images_json_new = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

movie_human_pics = pd.read_csv('new_movie_df.csv')
movie_human_df = pd.read_csv('movie_human_locs.csv')
movie_human_df = movie_human_df[['label', 'codes']]
movie_human_df.drop_duplicates(inplace=True)
property_df_with_names_final_subset = pd.read_csv('property_df_with_names_final_subset.csv', index_col=0)
movie_human_pics_list = movie_human_pics['lbl'].tolist()

question_category_property = {'film_locations_where': 'filming location',
                              'film_locations_when': 'publication date',
                              'human_born_where': 'place of birth',
                              'human_born_when': 'date of birth',
                              'film_actor': 'cast member',
                              'soundtrack': 'performer',
                              'producer': 'producer',
                              'screenwriter': 'screenwriter',
                              'director': 'director',
                              'film_actor2': 'cast member',
                              'desc': 'desc'}

question_category_who = {
    'film_actor': ['cast', 'actor', 'lead', 'hero', 'protagonist', 'played', 'member', 'acted'],
    'soundtrack': ['soundtrack', 'music', 'score', 'performer', 'composer', 'sound', 'track'],
    'producer': ['producer', 'produced'],
    'screenwriter': ['screen', 'writer', 'screenwriter'],
    'director': ['directed', 'director', 'direct', 'shot', 'filmed', 'made', 'created', 'make', 'work']
}


# generating the entity image from the required query text input from the entity history

# show the image type of the entity from the instance match found or the predicate match found from the query input
def get_entity_image(text):
    ner_results = nlp(text)

    # broken ner, get list for reverse matching
    object_ner = []
    for entry in ner_results:
        object_ner.append(entry['word'])
    matched_ner = [' '.join(object_ner)][0]

    try:
        # fetching an entity from the current query text
        nearest_match = difflib.get_close_matches(matched_ner, movie_human_pics_list)[0]
        movie_entity = movie_human_df[movie_human_df.label == nearest_match]['codes'].tolist()[0]
    except:
        # fetching an entity from the previous entity list
        nearest_match = ''
        movie_entity = []
    return nearest_match, movie_entity


# missing property_df_with_names_final_subset, is rel2id
def question_category_property_fetch(x, text_clean):
    i = 0
    insersection_length = []
    try:
        for value in x.values():
            if len(list(set(value) & set(text_clean))) > 0:
                insersection_length.append(len(list(set(value) & set(text_clean))))
            else:
                insersection_length.append(0)
        if np.array(insersection_length).sum() > 0:
            i = np.argmax(np.array(insersection_length))
            property_name = question_category_property[str(list(x.keys())[i])]
            pcode = property_df_with_names_final_subset['property_codes'][
                property_df_with_names_final_subset.label == property_name].tolist()
        else:
            pcode = []
    except:
        pcode = []
    return pcode


# function for cleaning the query text

# getting the property when an entity match is not found
def get_property_image(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text_clean = text.split(' ')
    property_code = question_category_property_fetch(question_category_who, text_clean)
    return property_code


# getting property code from the matched list of property names
def property_image_condition(entity, pcode):
    query_obj = """
    PREFIX ddis: <http://ddis.ch/atai/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>

    SELECT DISTINCT ?label
    WHERE{
    wd:%s wdt:%s ?x.
    ?x rdfs:label ?label.
    }
    """ % (entity, pcode[0])
    qres = g.query(query_obj)
    qres_all = []
    for row in qres:
        qres_all.append(str(row.asdict()['label'].toPython()))
    if len(qres_all) == 0:
        print(
            "Aaaaaah, that's a tricky one, I'm afraid i won't be able to answer that correctly.\nGive me a moment to "
            "make a guess, good Human :D")
        query_out = ''
    else:
        query_out = random.choice(qres_all)
    return query_out


def fetch_image2(query_input, query_entity, query_entity_name):
    try:
        name, entity = get_entity_image(query_input)
        if len(entity) == 0:
            name, entity = query_entity_name[0], query_entity[0]
        pcode = get_property_image(query_input)
        if len(pcode) > 0:
            query_output = property_image_condition(entity, pcode)
            imdb_code = movie_human_pics[movie_human_pics.lbl == query_output]['imdb'].tolist()[0]
        else:
            imdb_code = movie_human_pics[movie_human_pics.lbl == name]['imdb'].tolist()[0]

        # filtering the case where a movie gives back a still frame
        if imdb_code[:2] == 'tt':
            if 'frame' in query_input:
                img = [obj for obj in images_json_new if imdb_code in obj['movie'] and obj['type'] == 'still_frame'][0][
                    'img']
            else:
                img = [obj for obj in images_json_new if imdb_code in obj['movie'] and obj['type'] == 'poster'][0][
                    'img']

            if len(img) == 0:
                img = [obj for obj in images_json_new if imdb_code in obj['movie']][0]['img']

        # filtering the case where a cast member gives back an event image
        elif imdb_code[:2] == 'nm':
            img = [obj for obj in images_json_new if imdb_code in obj['cast'] and obj['type'] == 'event'][0]['img']
            if len(img) == 0:
                img = [obj for obj in images_json_new if imdb_code in obj['cast']][0]['img']
        img_final = 'Here is a cool image I found' + 'image:' + img[:-4]
    except:
        img_final = "Hmmmm, there seems to be no image of this object. Seems like it likes to keep an air of mystery " \
                    "about itself. Can you try again with a different object name? Thank you, superior Human being :D "
    return img_final
