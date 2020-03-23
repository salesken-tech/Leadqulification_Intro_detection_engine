import numpy as np
import pandas as pd
from src.utilities.objects import VadChunk
from src.utilities import sken_singleton, sken_logger, constants
from src.services import question_detection

logger = sken_logger.get_logger("snippet_service")


# def agent_customer_sequence(input_excel_file):
#     df = pd.read_excel(input_excel_file)
#     df.text_ = df.text.astype(str)
#     df['a_bin'] = 0
#     df['b_bin'] = 0
#     df.a_bin = df.speaker.apply(lambda x: 0 if x == 'Agent' else 1)
#     df.b_bin = df.speaker.apply(lambda x: 0 if x == 'Customer' else 1)
#     df['a_bin_cumsum'] = df.a_bin.cumsum()
#     df['b_bin_cumsum'] = df.b_bin.cumsum()
#     df = df.drop(['a_bin', 'b_bin'], axis=1)
#     df['a_bin'] = df.speaker.apply(lambda x: 1 if x == 'Agent' else 0)
#     df['b_bin'] = df.speaker.apply(lambda x: 1 if x == 'Customer' else 0)
#     df['a_con'] = df.a_bin_cumsum * df.a_bin
#     df['b_con'] = df.b_bin_cumsum * df.b_bin
#     df.drop(['a_bin_cumsum', 'b_bin_cumsum', 'a_bin', 'b_bin'], axis=1, inplace=True)
#     df['identifier'] = df.a_con + df.b_con
#     df['name_idnet'] = df.speaker + "_" + df.identifier.astype(str)
#     df.drop(['a_con', 'b_con'], axis=1, inplace=True)
#     df['text_'] = df['text'] + ". "
#     df1 = df[['name_idnet', 'text_']].groupby(['name_idnet'], as_index=False).sum()
#     df2 = df.drop_duplicates("name_idnet")[['speaker', 'name_idnet']]
#     df2 = df2.merge(df1, on='name_idnet')
#     df2 = df2.drop(["name_idnet"], axis=1)
#     df2['text_'] = df2.text_.apply(lambda x: x.strip("."))
#     return df2


# def agent_customer_sequence(input_excel_file):
#     df = pd.read_excel(input_excel_file)
#     df.text = df.text.astype(str)
#     df['a_bin'] = 0
#     df['b_bin'] = 0
#     df.a_bin = df.speaker.apply(lambda x: 0 if x == 'Agent' else 1)
#     df.b_bin = df.speaker.apply(lambda x: 0 if x == 'Customer' else 1)
#     df['a_bin_cumsum'] = df.a_bin.cumsum()
#     df['b_bin_cumsum'] = df.b_bin.cumsum()
#     df = df.drop(['a_bin', 'b_bin'], axis=1)
#     df['a_bin'] = df.speaker.apply(lambda x: 1 if x == 'Agent' else 0)
#     df['b_bin'] = df.speaker.apply(lambda x: 1 if x == 'Customer' else 0)
#     df['a_con'] = df.a_bin_cumsum * df.a_bin
#     df['b_con'] = df.b_bin_cumsum * df.b_bin
#     df.drop(['a_bin_cumsum', 'b_bin_cumsum', 'a_bin', 'b_bin'], axis=1, inplace=True)
#     df['identifier'] = df.a_con + df.b_con
#     df['name_idnet'] = df.speaker + "_" + df.identifier.astype(str)
#     df.drop(['a_con', 'b_con', 'identifier'], axis=1, inplace=True)
#     df['nindex'] = df.index
#
#     def summing(df):
#         df = df.copy().reset_index(drop=True)
#         n_ind = df.nindex[0].astype(int)
#         if df.shape[0] == 1:
#             df['nindx'] = n_ind
#             return df.reset_index(drop=True)
#         else:
#             df['text'] = df.text.astype(str) + " "
#             df['id'] = df.id.astype(str) + " "
#             text = df.text.sum().strip()
#             name = df.name_idnet[0]
#             from_time = df.from_time[0]
#             to_time = df.to_time[df.shape[0] - 1]
#             ids = df.id.sum().strip().replace(" ", ",")
#             data = {'nindx': [n_ind], 'id': [ids], 'speaker': [df.speaker[0]], 'name_idnet': [df.name_idnet[0]],
#                     'text': [text], 'from_time': [from_time], 'to_time': [to_time]}
#             df1 = pd.DataFrame(data)
#             return df1.reset_index(drop=True)
#
#     df_result = df.groupby(['speaker', 'name_idnet'], as_index=False).apply(summing).drop('nindex', axis=1).sort_values(
#         'nindx').reset_index(drop=True)
#     df_result = df_result.drop(['name_idnet', 'nindx'], axis=1)
#     df_result = df_result[['id', 'speaker', 'text', 'from_time', 'to_time']]
#     return df_result

def agent_customer_sequence(input_excel_file):
    cached_snippets = []
    df = pd.read_excel(input_excel_file)
    if len(df) != 0:
        logger.info("Making new snippets for {} snippets".format(len(df)))
        for i in range(len(df)):
            if len(cached_snippets) == 0:
                cached_snippets.append({"orignal_ids": [df["id"][i]], "speaker": df["speaker"][i], "text": df["text"][i],
                                        "from_time": df["from_time"][i], "to_time": df["to_time"][i],
                                        "task_id": df["task_id"][i]})
            else:
                if df["speaker"][i] == cached_snippets[-1]['speaker']:
                    cached_snippets[-1]["orignal_ids"].append(df["id"][i])
                    cached_snippets[-1]["text"] += ". " + df["text"][i]
                    cached_snippets[-1]["to_time"] = df["to_time"][i]
                    cached_snippets[-1]["task_id"] = df["task_id"][i]
                else:
                    cached_snippets.append(
                        {"orignal_ids": [df["id"][i]], "speaker": df["speaker"][i], "text": df["text"][i],
                         "from_time": df["from_time"][i], "to_time": df["to_time"][i], "task_id": df["task_id"][i]})
        new_snippets_df = pd.DataFrame(cached_snippets)
        logger.info("Made {} new snippets from {} old snippets".format(len(new_snippets_df),len(df)))
        return new_snippets_df
    else:
        return []


def make_snippets(df, snippet_ids, task_id):
    if len(df) != 0:
        sentences = df["text"].to_list()
        sentence_vectors = sken_singleton.Singletons.get_instance().perform_embeddings(sentences)
        vad_chunks = []
        for i in range(len(df)):
            vad_chunks.append(
                VadChunk( snippet_ids[i], df["from_time"][i], df["to_time"][i], df["speaker"][i], df["text"][i], sentence_vectors[i], None, task_id, df["orignal_ids"][i],
                 questions=None,
                 q_encoding=None,
                 encoding_method=constants.fetch_constant("encoding_method")))

        return vad_chunks
    else:
        return []


def check_snippet_speaker(vad_chunk):
    """
    This method checks if the snippet speaker is agent or customer
    :param vad_chunk:
    :return: true if speaker is agent else false
    """
    return True if vad_chunk.speaker == "Agent" else False


def find_snippet_questions(vad_chunk):
    """
    Extractes questions from snippet text and sets the snippet question if any question is detected else sets if to None
    :param vad_chunk:
    :return: None
    """
    questions = question_detection.find_questions(vad_chunk.text)
    if len(questions) != 0:
        logger.info("Found {} question for snippet_id={}".format(len(questions), vad_chunk.sid))
        vad_chunk.set_questions(questions)
    else:
        logger.info("Did not find any question in snippet {}".format(vad_chunk.sid))
        vad_chunk.set_questions(None)


def make_snippet_question_embeddings(vad_chunk):
    """
    Sets the sentence embedding of snippet questions if present else sets it to None
    :param vad_chunk:
    :return: None
    """
    if vad_chunk.questions is not None:
        vad_chunk.set_question_encoding(
            sken_singleton.Singletons.get_instance().perform_embeddings(vad_chunk.questions),
            constants.fetch_constant("encoding_method"))
        logger.info(
            "Calculated  embeddings for {} snippet questions for snippet_id ={}".format(
                len(vad_chunk.questions),
                vad_chunk.sid))
    else:
        logger.info("There were not snippet questions for snippet_id={}".format(vad_chunk.sid))
        vad_chunk.set_question_encoding(None, None)


if __name__ == "__main__":
    df = agent_customer_sequence("/home/andy/Desktop/new_test.xlsx")
    df.to_excel("/home/andy/Desktop/data_for_emotion.xlsx")
    print(df.head())
