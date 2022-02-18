import random
import difflib
import numpy as np
import pandas as pd
import string
from sklearn.metrics import pairwise_distances
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from rdflib import Graph

g = Graph()
g.parse('14_graph.nt', format='turtle')

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

property_df_with_names_final_subset = pd.read_csv('property_df_with_names_final_subset.csv', index_col=0)
movie_human_df = pd.read_csv('movie_human_locs.csv')
movie_human_df = movie_human_df[['label', 'codes']]
movie_human_df.drop_duplicates(inplace=True)
human_movie_full = movie_human_df.label.tolist()

ent2id = pd.read_csv('ent2id.csv')
lbl2ent = pd.read_csv('lbl2ent.csv')
rel2id = pd.read_csv('rel2id.csv')

# load the embeddings
entity_emb = np.load('entity_embeds.npy')
relation_emb = np.load('relation_embeds.npy')

question_category_where = {
    'film_locations_where': ['shot', 'located', 'filmed', 'country', 'city', 'film', 'location', ],
    'human_born_where': ['born', 'birthplace', 'grew up', 'lives', 'lived', 'live', 'childhood', 'grow', 'grew', 'up'],
}

question_category_when = {
    'human_born_when': ['when', 'born', 'date of birth', 'birthday'],
    'film_locations_when': ['released', 'release', 'release', 'publication', 'date', 'shot']
}

question_category_who = {
    'film_actor': ['cast', 'actor', 'lead', 'hero', 'protagonist', 'played', 'member', 'acted'],
    'soundtrack': ['soundtrack', 'music', 'score', 'performer', 'composer', 'sound', 'track'],
    'producer': ['producer', 'produced'],
    'screenwriter': ['screen', 'writer', 'screenwriter'],
    'director': ['directed', 'director', 'direct', 'shot', 'filmed', 'made', 'created', 'make', 'work']
}
question_category_what = {
    'desc': ['what', 'tell', 'is', 'who', 'about', 'What', 'Who', 'Tell', 'occupation']
}

question_category_general = {
    'film_actor': ['cast', 'actor', 'lead', 'hero', 'protagonist', 'played', 'member', 'acted'],
    'soundtrack': ['soundtrack', 'music', 'score', 'performer', 'composer', 'sound', 'track'],
    'producer': ['producer', 'produced'],
    'screenwriter': ['screen', 'writer', 'screenwriter'],
    'director': ['directed', 'director', 'direct', 'shot', 'filmed', 'made', 'created', 'make', 'work'],
    'film_actor2': ['in']
}

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


def movie_entity_name(text):
    ner_results = nlp(text)

    # broken ner, get list for reverse matching
    object_ner = []
    for entry in ner_results:
        object_ner.append(entry['word'])
    matched_ner = [' '.join(object_ner)][0]

    try:
        nearest_match = difflib.get_close_matches(matched_ner, human_movie_full)
        # movie_entity = [movie_human_df[movie_human_df.label == nearest_match]['codes'].tolist()[0]]
    except:
        nearest_match = ''
    return nearest_match


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


all_properties = property_df_with_names_final_subset.label.tolist()


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


def property_code(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text_clean = text.split(' ')

    if 'when' in [x.lower() for x in text_clean]:
        property_code = question_category_property_fetch(question_category_when, text_clean)

    elif 'where' in [x.lower() for x in text_clean]:
        property_code = question_category_property_fetch(question_category_where, text_clean)

    elif 'who' in [x.lower() for x in text_clean]:
        property_code = question_category_property_fetch(question_category_who, text_clean)
    else:
        property_code = question_category_property_fetch(question_category_general, text_clean)

    if len(property_code) == 0:
        for x in all_properties:
            if x in text_clean:
                index_val = all_properties.index(x)
                property_code = [property_df_with_names_final_subset.iloc[index_val]['property_codes']]

    if len(property_code) == 0:
        property_code = []

    return property_code


def embedding_res_check(query_input, query_entity):
    try:
        head = entity_emb[ent2id[ent2id.entity == query_entity[0]]['index_num'].tolist()]
        pred = relation_emb[rel2id[rel2id.property == property_code(query_input)[0]]['index_num'].tolist()]
        lhs = head + pred
        dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
        most_likely = dist.argsort()
        ranks = dist.argsort().argsort()
        output_df = pd.DataFrame(columns=['Entity', 'Label', 'Score', 'Rank'])
        for rank, idx in enumerate(most_likely[:50]):
            try:
                row_list = {'Entity': ent2id.iloc[idx]['entity'],
                            'Label': lbl2ent[lbl2ent.code == ent2id.iloc[idx]['entity']]['index_num'].tolist()[0],
                            'Score': dist[idx], 'Rank': rank + 1}
                output_df = output_df.append(row_list, ignore_index=True)
            except:
                row_list = {'Entity': 'unk', 'Label': 'unk', 'Score': 999999, 'Rank': rank + 1}
                output_df = output_df.append(row_list, ignore_index=True)

        output_single = output_df['Label'].tolist()
    except:
        output_single = []
    return output_single


def query_single(text, query_entity):
    """
    Query to return single object, doesn't contain keyword 'all' in user query
    :return: string obj
    """
    global qres_all
    try:
        movie_entity = movie_entity_code(text)
        prop_code = property_code(text)

        if len(movie_entity) == 0:
            movie_entity = query_entity
        if len(prop_code) == 0:
            text = text.translate(str.maketrans('', '', string.punctuation))
            text_clean = text.split(' ')
            for value in question_category_what.values():
                if len(list(set(value) & set(text_clean))) > 0:
                    query_obj = """
                    PREFIX ddis: <http://ddis.ch/atai/>
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX schema: <http://schema.org/>
                    PREFIX p: <http://www.wikidata.org/prop/>
                    PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
                    PREFIX ps: <http://www.wikidata.org/prop/statement/>

                    SELECT DISTINCT ?desc
                    WHERE{
                    wd:%s schema:description ?desc.
                    }
                    """ % (movie_entity[0])
                    qres = g.query(query_obj)
                    qres_all = []
                    for row in qres:
                        qres_all.append(str(row.asdict()['desc'].toPython()))

        elif prop_code[0] == 'P577':
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
            wd:%s wdt:%s ?label.
            }
            """ % (movie_entity[0], prop_code[0])

            qres = g.query(query_obj)
            qres_all = []
            for row in qres:
                qres_all.append(str(row.asdict()['label'].toPython()))

        else:
            query_obj = """
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>

            SELECT DISTINCT ?x ?label
            WHERE{
            wd:%s wdt:%s ?x.
            ?x rdfs:label ?label.
            }
            """ % (movie_entity[0], prop_code[0])

            qres = g.query(query_obj)
            qres_all = []
            for row in qres:
                qres_all.append(str(row.asdict()['label'].toPython()))

        if len(qres_all) == 0:
            print(
                "Aaaaaah, that's a tricky one, let me call my supervisor for more brain power.\nGive me a moment "
                "please, good Human :D")
            query_out = ''
        else:
            query_out = random.choice(qres_all)

        if len(prop_code) > 0:
            embed_outputs = embedding_res_check(text, movie_entity)
            if len(embed_outputs) == 0:
                query_out = 'Here is what I found: ' + query_out
            else:
                if query_out in embed_outputs:
                    query_out = 'Here is what I found: ' + query_out
                else:
                    query_out = 'Here is what I found: ' + embed_outputs[0]
    except:
        query_out = "Dear Superior Human, I am afraid this is beyond my skill. My supervisor didn't know about this " \
                    "either. Can you please ask a different question? I promise I will use all my brain power for the " \
                    "next one :D. "

    return query_out
