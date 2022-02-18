import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import ast
import re
from rdflib import Graph
import string
from question_answer import movie_entity_code

g = Graph()
g.parse('14_graph.nt', format='turtle')

ent2id = pd.read_csv('ent2id.csv')
lbl2ent = pd.read_csv('lbl2ent.csv')
rel2id = pd.read_csv('rel2id.csv')
entity_emb = np.load('entity_embeds.npy')
relation_emb = np.load('relation_embeds.npy')
movie_names_all = pd.read_csv('movie_names_all.csv')
property_df_with_names_final_subset = pd.read_csv('property_df_with_names_final_subset.csv', index_col=0)
genre_df = pd.read_csv('genre_df.csv')

recommendation_json = {
    'director': 'director',
    'actor': 'cast member',
    'composer': 'composer',
}

city_tag = ['city', 'country', 'state', 'town', 'village', 'nation']


def get_property(x):
    return x.split('/')[-1]


def get_movie_recommendation(query_entity):
    try:
        ent = ent2id[ent2id.entity == query_entity]['index_num'].tolist()[0]
        dist = pairwise_distances(entity_emb[ent].reshape(1, -1), entity_emb).reshape(-1)
        most_likely = dist.argsort()
        output_df = pd.DataFrame(columns=['Entity', 'Label', 'Score', 'Rank'])
        for rank, idx in enumerate(most_likely[:10]):
            try:
                row_list = {'Entity': ent2id.iloc[idx]['entity'],
                            'Label': lbl2ent[lbl2ent.code == ent2id.iloc[idx]['entity']]['index_num'].tolist()[0],
                            'Score': dist[idx], 'Rank': rank + 1}
                output_df = output_df.append(row_list, ignore_index=True)
            except:
                row_list = {'Entity': 'unk', 'Label': 'unk', 'Score': 999999, 'Rank': rank + 1}
                output_df = output_df.append(row_list, ignore_index=True)
        # output_single = output_df['Label'].tolist()
        output_df = output_df[1:]
        movies_only = pd.merge(output_df, movie_names_all, on=['Label', 'Entity'], how='inner')
        output_single = movies_only['Label'].tolist()
    except:
        output_single = []
        output_df = pd.DataFrame()
    return output_single


def get_location_recommendations(query_entity):
    try:
        query_obj = """
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>

        SELECT DISTINCT ?Entity ?Label
        WHERE{
          ?Entity wdt:P31 wd:Q11424.
          ?Entity wdt:P915 wd:%s.
          ?Entity rdfs:label ?Label.
        }
        limit 10

        """ % (query_entity)

        qres = g.query(query_obj)
        out_query = pd.DataFrame(qres.bindings)
        out_query.columns = ['Entity', 'Label']
        out_query['Entity'] = out_query['Entity'].apply(lambda x: str(x))
        out_query['Label'] = out_query['Label'].apply(lambda x: str(x))
        out_query['Entity'] = out_query['Entity'].apply(get_property)
        movies_only = pd.merge(out_query, movie_names_all, on=['Label', 'Entity'], how='inner')

        output_single = movies_only['Label'].tolist()
    except:
        output_single = ''
    return output_single


# adding function for human entities and property embedding
def human_property_embedding(query_entity, pcode):
    tail = entity_emb[ent2id[ent2id.entity == query_entity]['index_num'].tolist()[0]]
    pred = relation_emb[rel2id[rel2id.property == pcode]['index_num'].tolist()[0]]
    lhs = tail - pred
    dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
    most_likely = dist.argsort()
    output_df = pd.DataFrame(columns=['Entity', 'Label', 'Score', 'Rank'])
    for rank, idx in enumerate(most_likely[:10]):
        try:
            row_list = {'Entity': ent2id.iloc[idx]['entity'],
                        'Label': lbl2ent[lbl2ent.code == ent2id.iloc[idx]['entity']]['index_num'].tolist()[0],
                        'Score': dist[idx], 'Rank': rank + 1}
            output_df = output_df.append(row_list, ignore_index=True)
        except:
            row_list = {'Entity': 'unk', 'Label': 'unk', 'Score': 999999, 'Rank': rank + 1}
            output_df = output_df.append(row_list, ignore_index=True)
    movies_only = pd.merge(output_df, movie_names_all, on=['Label', 'Entity'], how='inner')
    output_single = movies_only['Label'].tolist()
    return output_single


# generate recommendations from humans
def get_human_recommendations(query_entity, desc_str):
    try:
        for value in recommendation_json.keys():
            if value in desc_str:
                correct_label = recommendation_json[value]
                pcode = property_df_with_names_final_subset['property_codes'][
                    property_df_with_names_final_subset.label == correct_label].tolist()
                output_df = human_property_embedding(query_entity, pcode[0])

    except:
        output_df = []
        pcode = []
    return output_df


def remove_film_movie(x):
    pattern1 = r'film'
    pattern2 = r'movie'
    pattern3 = r'Film'
    pattern4 = r'Movie'
    mod_string1 = re.sub(pattern1, '', x)
    mod_string2 = re.sub(pattern2, '', mod_string1)
    mod_string3 = re.sub(pattern3, '', mod_string2)
    mod_string4 = re.sub(pattern4, '', mod_string3)
    return mod_string4


genrename_list = genre_df['genrename_list'].tolist()
genrename_list = [ast.literal_eval(x) for x in genrename_list]


# query_input = "Show me action film"
# function for generating genre recommendations for previous entities
def generate_genre_recommendation(query_input):
    try:
        # adding the part of the total movie recommendations from the part
        query_input = query_input.translate(str.maketrans('', '', string.punctuation))
        query_input = query_input.split(' ')
        instances_list = []
        for x in genrename_list:
            if len(list(set(query_input) & set(x))) > 0:
                instances_list.append(genrename_list.index(x))
        random_index = instances_list[0]
        genre_entity = genre_df.iloc[random_index]['entity']

        if len(genre_entity) > 0:
            query_obj = """
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>

            SELECT DISTINCT ?instancename
            WHERE{
            ?instance wdt:P31 wd:Q11424.
            ?instance wdt:P136 wd:%s.
            ?instance rdfs:label ?instancename.
            }
            limit 10
            """ % (genre_entity)

            qres = g.query(query_obj)
            out_query = pd.DataFrame(qres.bindings)
            out_query.columns = ['name']
            out_query['name'] = out_query['name'].apply(lambda x: str(x))
            output_single = out_query['name'].tolist()
            output_single = ', '.join(output_single)
            output_single = 'These movies fit the genre: ' + output_single
    except:
        output_single = ''
    return output_single


# main helper function to generate the recommendations of all types for the given query input from the database
def generate_recommendations(query_input, movie_entity):
    try:
        text = query_input.translate(str.maketrans('', '', string.punctuation))
        text = text.split(' ')
        query_entity = movie_entity_code(query_input)
        flag = 0
        if len(query_entity) == 0:
            for x in genrename_list:
                if (len(list(set(text) & set(x))) > 0):
                    flag = 1

        if flag == 1:
            recommendation_result = generate_genre_recommendation(query_input)
        else:
            query_entity = movie_entity
            query_obj = """
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>

            SELECT DISTINCT ?desc ?instanceof
            WHERE{
            wd:%s schema:description ?desc.
            wd:%s wdt:P31 ?category.
            ?category rdfs:label ?instanceof.
            FILTER(LANG(?desc) = "en").
            FILTER(LANG(?instanceof) = "en").
            }
            """ % (query_entity[0], query_entity[0])

            qres = g.query(query_obj)
            out_query = pd.DataFrame(qres.bindings)
            out_query.columns = ['desc', 'instanceof']
            out_query['desc'] = out_query['desc'].apply(lambda x: str(x))
            out_query['instanceof'] = out_query['instanceof'].apply(lambda x: str(x))

            instanceof_str = out_query['instanceof'].tolist()[0]
            instanceof_str = instanceof_str.translate(str.maketrans('', '', string.punctuation))
            instanceof_str = instanceof_str.split(' ')

            desc_str = out_query['desc'].tolist()[0]
            desc_str = desc_str.translate(str.maketrans('', '', string.punctuation))
            desc_str = desc_str.split(' ')

            # generating the case where a film related question is asked
            if ('film' in instanceof_str):
                recommendation_result = get_movie_recommendation(query_entity[0])
            # generating the case where a location related question is asked
            elif len(list(set(instanceof_str) & set(city_tag))) > 0:
                recommendation_result = get_location_recommendations(query_entity[0])
            # generating the case where a human related question is asked
            else:  # human
                recommendation_result = get_human_recommendations(query_entity[0], desc_str)
            recommendation_result = ', '.join(recommendation_result)
            recommendation_result = 'Hellllo, these movies look similar to what you are looking forr ' + recommendation_result
    except:
        # raising an exception when the following cases occur:
        # 1) Matching entity was not found
        # 2) Entity history was not found
        # 3) RDF graph didn't have entity-predicate relationship
        recommendation_result = "I am unable to find similar items related to this query. Can you please ask a " \
                                "different question? Thank you kind Human :D "
        print("Beyond my skill")
    return recommendation_result
