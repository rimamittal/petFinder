import os
import json


def construct_df_from_text_json(json_files, df, path_to_json):
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)

            pet_id = js.partition('.json')[0]

            doc_sent = json_text.get('documentSentiment', 'N/A')
            if doc_sent != 'N/A':
                magnitude = doc_sent['magnitude']
                score = doc_sent['score']

                df.loc[index] = [pet_id, magnitude, score]
    return df
