import re
import string
import difflib
import rdflib
from rdflib import Graph
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from question_answer import query_single, property_code, movie_entity_name

g = Graph()
g.parse('14_graph.nt', format='turtle')
movie_human_pics = pd.read_csv('new_movie_df.csv')

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

movie_human_df = pd.read_csv('movie_human_locs.csv')
movie_human_df = movie_human_df[['label', 'codes']]
movie_human_df.drop_duplicates(inplace=True)
human_movie_full = movie_human_df.label.tolist()


def multiple_entitites(matched_ner_people, matched_ner_object):
    movie_entity = []
    try:
        # populating the case where movie entities are found
        nearest_match_people = difflib.get_close_matches(matched_ner_people, human_movie_full)[0]
        people_entity_code = movie_human_df[movie_human_df.label == nearest_match_people]['codes'].tolist()
        movie_entity.append(people_entity_code[0])

        # populating the case where movie entities are found
        nearest_match_object = difflib.get_close_matches(matched_ner_object, human_movie_full)[0]
        object_entity_code = movie_human_df[movie_human_df.label == nearest_match_object]['codes'].tolist()
        movie_entity.append(object_entity_code[0])

    except:
        movie_entity = []
    return movie_entity


def movie_entity_code(text):
    ner_results = nlp(text)

    # broken ner, get list for reverse matching
    object_ner = []
    for entry in ner_results:
        object_ner.append(entry['word'])
    matched_ner = [' '.join(object_ner)][0]

    try:
        nearest_match = difflib.get_close_matches(matched_ner, human_movie_full)[0]
        movie_entity = [movie_human_df[movie_human_df.label == nearest_match]['codes'].tolist()[0]]
    except:
        movie_entity = []
    return movie_entity


# all the entities found inside the query were put inside a single input query statement
def all_entitites(query_input, query_entity):
    query_input = query_input.translate(str.maketrans('', '', string.punctuation))
    ner_results = nlp(query_input)
    people = []
    object = []

    # for all tags of tokenized ner objects found which were a person
    for entry in ner_results:
        if entry['entity'][-3:] == 'PER':  # catching person
            people.append(entry['word'])
        else:
            # for all tags of tokenized ner objects found which were not a person
            object.append(entry['word'])

    # match the entries for people and other objects from the query input
    matched_ner_people = [' '.join(people)][0]
    matched_ner_object = [' '.join(object)][0]

    # defining the loop for an assertion statement from the total input entities found inside the query
    if (len(matched_ner_people) > 0) & (len(matched_ner_object) > 0):
        movie_entity = multiple_entitites(matched_ner_people, matched_ner_object)
    else:
        # falling into the second assertion when the question doesn't refer to the case of people and movie entity
        # match found in the query
        if len(movie_entity_code(query_input)) == 0:
            movie_entity = [query_entity]
        # falling into the case where the person entity is found inside the query text
        else:
            movie_entity = movie_entity_code(query_input)
    return movie_entity


# function for assertion statement for entity property code
def assertion_relation(movie_entity, prop_code):
    try:
        if (rdflib.term.URIRef(WD + movie_entity[1]), rdflib.term.URIRef(WDT + prop_code[0]),
            rdflib.term.URIRef(WD + movie_entity[0])) in g:
            return 'Yup, it seems that is True'
        else:
            return 'Nope, this seems False'
    except:
        return 'Nope, this seems False'


# function to define the assertion flow that follows from the input entities
def assertion_function(text, query_entity):
    try:
        movie_entity = all_entitites(text, query_entity)
        if len(movie_entity) == 1:
            query_out = query_single(text, query_entity)
        else:
            prop_code = property_code(text)
            query_out = assertion_relation(movie_entity, prop_code)
    except:
        query_out = "Hmmm, this question is too tough for my tiny graph brain. Can you please ask a different " \
                    "question? Thank you kind Human. :D "
    return query_out


def remove_imdb_name(x):
    pattern1 = r'IMDb'
    pattern2 = r'imdb'
    pattern3 = r'Wiki'
    pattern4 = r'WIKI'
    mod_string1 = re.sub(pattern1, '', x)
    mod_string2 = re.sub(pattern2, '', mod_string1)
    mod_string3 = re.sub(pattern3, '', mod_string2)
    mod_string4 = re.sub(pattern4, '', mod_string3)
    return mod_string4


def get_url(query_input, query_entity, query_entity_name, text_clean):
    global out_query
    try:
        if len({'imdb', 'IMDb'} & set(text_clean)) > 0:
            text = remove_imdb_name(query_input)
            entity_name = movie_entity_name(text)[0]
            if len(entity_name) == 0:
                entity_name = query_entity_name
            imdb_code = movie_human_pics[movie_human_pics.lbl == entity_name]['imdb'].tolist()[0]
            out_query = 'imdb:' + str(imdb_code)

            # return imdb:imdb id
        elif len({'WIKI', 'wikidata', 'Wikidata', 'wiki'} & set(text_clean)) > 0:
            text = remove_imdb_name(query_input)
            entity_code = movie_entity_code(text)
            pcode = property_code(text)
            if len(pcode) > 0:
                wiki_code = 'wdt:' + str(pcode[0])
                out_query = wiki_code
            elif len(entity_code) == 0:
                entity_code = query_entity
                wiki_code = 'wd:' + str(entity_code[0])
                out_query = wiki_code
    except:
        out_query = "Hmmm, this entity or property seems to be mysterious. I am unable to find it. Can you please " \
                    "retry with a different entity/property's IMDb or wikidata link? Thank you sooo muchhh :D "
    return out_query
