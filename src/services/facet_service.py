import os

import numpy as np
import pandas as pd
from google.cloud import translate
from src.utilities import db, constants, sken_logger, sken_singleton
from src.utilities.objects import CaughtFacetSignals


client = translate.TranslationServiceClient()
logger = sken_logger.get_logger("facet_signal")
from zipfile import ZipFile


def make_facet_signal_entries(file_path, org_id, prod_id):
    df = pd.read_excel(file_path)
    for facet_signal in range(len(df)):
        sql = """insert into facet_signal (value, facet_id, org_id, product_id) values(%s, (select 
        id from facet where name_ = %s), %s, %s) """
        db.DBUtils.get_instance().execute_query(sql, (df.text[facet_signal], df.facet[facet_signal], org_id, prod_id),
                                                is_write=True, is_return=False)


def praphrase_sentences(text, depth=int(constants.fetch_constant("language_depth")),
                        project_id=constants.fetch_constant("google_project_id")):
    parent = client.location_path(project_id, "global")
    x = client.get_supported_languages(parent)
    target_laguages = [item.language_code for item in x.languages[:depth]]
    translated_text = []
    for language in target_laguages:
        response = client.translate_text(
            parent=parent,
            contents=[text],
            mime_type='text/plain',  # mime types: text/plain, text/html
            source_language_code='en-IN',
            target_language_code=language)
        for translation in response.translations:
            translated_text.append(translation.translated_text)
    result = []
    for lg, sentence in zip(target_laguages, translated_text):
        response = client.translate_text(
            parent=parent,
            contents=[sentence],
            mime_type='text/plain',  # mime types: text/plain, text/html
            source_language_code=str(lg),
            target_language_code="en")
        for translation in response.translations:
            result.append(translation.translated_text)
    return result


def make_generated_facet_siganls(org_id, product_id):
    sql = "select 	dimension.id as dimid, 	dimension.name_ as dimname, 	facet.id as facet_id, 	facet.name_ as " \
          "facet_name, 	facet_signal.id as fsid, 	facet_signal.value as fsval, 	facet_signal.org_id, 	" \
          "facet_signal.product_id from 	dimension left join facet on 	facet.dim_id = dimension.id left join " \
          "facet_signal on 	facet_signal.facet_id = facet.id left join generated_facet_signals on 	" \
          "generated_facet_signals.facet_signal_id = facet_signal.id where facet_signal.org_id = %s and " \
          "facet_signal.product_id = %s group by 	dimension.id, 	facet.id, 	facet_signal.id "
    rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_id), is_write=False, is_return=True)
    if len(rows) != 0:
        for row in rows:
            logger.info("Making paraphrase sentences for text={} ".format(row[col_names.index("fsval")]))
            sentences = praphrase_sentences(row[col_names.index("fsval")])
            sentences.append(row[col_names.index("fsval")])
            for sentence in list(set(sentences)):
                sql = "INSERT INTO public.generated_facet_signals (value, facet_signal_id) VALUES(%s, %s)"
                db.DBUtils.get_instance().execute_query(sql, (sentence, row[col_names.index("fsid")]), is_write=True,
                                                        is_return=False)
            logger.info("Generated paraphrased sentences for facet_signal_id={}".format(row[col_names.index("fsid")]))
        logger.info("done with making paraphrases for org_id={} and product_id={}".format(org_id, product_id))
        return True
    else:
        logger.info("No facet signals found for org_id={} and product_id={}".format(org_id, product_id))
        return False


def create_lq_maches(vad_chunks, threshold):
    """
    This method returns the caught lead_qualification facets that are caught for each snippet, only one facet signal
    can be caught across all the facets
    """
    caught_lq_facets = []
    logger.info("Making caught facets for lead_qualification")
    for vad_chunk in vad_chunks:
        if vad_chunk.q_encoding is not None:
            for i, question in enumerate(vad_chunk.questions):
                scores = np.zeros(shape=(len(sken_singleton.Singletons.get_instance().get_cached_lq_dims()),
                                         max([len(x.facet_signals) for x in
                                              sken_singleton.Singletons.get_instance().get_cached_lq_dims().values()])))
                for x, facet in enumerate(sken_singleton.Singletons.get_instance().get_cached_lq_dims()):
                    for y, facet_signal in enumerate(
                            sken_singleton.Singletons.get_instance().get_cached_lq_dims()[facet].facet_signals):
                        score = (np.dot([vad_chunk.q_encoding[i]], np.array(facet_signal.embedding).T) / (
                                np.linalg.norm([vad_chunk.q_encoding[i]]) * np.linalg.norm(
                            facet_signal.embedding)))[0][0]
                        scores[x, y] = score
                if np.amax(scores) >= float(threshold):
                    facet_index, facet_signal_index = np.where(scores == np.amax(scores))[0][0], \
                                                      np.where(scores == np.amax(scores))[1][0]
                    facet = sken_singleton.Singletons.get_instance().get_cached_lq_dims()[
                        list(sken_singleton.Singletons.get_instance().get_cached_lq_dims().keys())[
                            facet_index]]
                    facet_signal = facet.facet_signals[facet_signal_index]
                    caught_lq_facets.append(
                        CaughtFacetSignals(vad_chunk, vad_chunk.text, question, facet.name, facet_signal,
                                           facet_signal.text,
                                           np.amax(scores),
                                           constants.fetch_constant("encoding_method"), "Lead-Qualification"))
    return caught_lq_facets


def create_intro_matches(vad_chunks, threshold):
    """
        This method returns the caught Introduction facets that are caught for each snippet, only one facet signal
        can be caught across all the facets
        """
    caught_intro_facets = []
    logger.info("Making caught facets for Introduction")
    for vad_chunk in vad_chunks:
        if vad_chunk.text_encoding is not None:
            scores = np.zeros(shape=(len(sken_singleton.Singletons.get_instance().get_cached_intro_dims()),
                                     max([len(x.facet_signals) for x in
                                          sken_singleton.Singletons.get_instance().get_cached_intro_dims().values()])))
            for x, facet in enumerate(sken_singleton.Singletons.get_instance().get_cached_intro_dims()):
                for y, facet_signal in enumerate(
                        sken_singleton.Singletons.get_instance().get_cached_intro_dims()[facet].facet_signals):
                    score = (np.dot([vad_chunk.text_encoding], np.array(facet_signal.embedding).T) / (
                            np.linalg.norm([vad_chunk.text_encoding]) * np.linalg.norm(
                        facet_signal.embedding)))[0][0]
                    scores[x, y] = score
            if np.amax(scores) >= float(threshold):
                facet_index, facet_signal_index = np.where(scores == np.amax(scores))[0][0], \
                                                  np.where(scores == np.amax(scores))[1][0]
                facet = sken_singleton.Singletons.get_instance().get_cached_intro_dims()[
                    list(sken_singleton.Singletons.get_instance().get_cached_intro_dims().keys())[
                        facet_index]]
                facet_signal = facet.facet_signals[facet_signal_index]
                caught_intro_facets.append(
                    CaughtFacetSignals(vad_chunk, vad_chunk.text, None, facet.name, facet_signal,
                                       facet_signal.text,
                                       np.amax(scores),
                                       constants.fetch_constant("encoding_method"), "Introduction"))
    return caught_intro_facets


def make_result_lq(caught_facets, org_id, product_i):
    if len(caught_facets) != 0:
        result = []
        a_id = []
        b_id = []
        i_id = []
        n_id = []
        for item in caught_facets:
            sql = "select id,value,facet_id,org_id,product_id from facet_signal where facet_signal.id =%s"
            rows, col_names = db.DBUtils.get_instance().execute_query(sql,
                                                                      (str(item.facet_signal.orignal_facet_signal_id),),
                                                                      is_write=False, is_return=True)
            tmp_dict = {"Task_id": item.snippet.task_id, "Snippet_id": item.snippet.sid,
                        "Snippet_text": item.snippet_text,
                        "Snippet_Ques": item.snippet_question, "Dimension": "Lead-Qualification",
                        "Facet": item.facet_name,
                        "Facet_Signal_id": rows[0][col_names.index("id")],
                        "Face_Signal_text": rows[0][col_names.index("value")],
                        "Score": str(item.score)}
            result.append(tmp_dict)
            if item.facet_name.lower() == "authority":
                a_id.append(item.facet_signal.orignal_facet_signal_id)
            elif item.facet_name.lower() == "interest":
                i_id.append(item.facet_signal.orignal_facet_signal_id)
            elif item.facet_name.lower() == "budget":
                b_id.append(item.facet_signal.orignal_facet_signal_id)
            else:
                n_id.append(item.facet_signal.orignal_facet_signal_id)

        sql = "select id from facet_signal where facet_id = (select facet.id from facet where facet.name_ like " \
              "'authority' ) and org_id=%s and product_id=%s "
        rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_i), is_write=False,
                                                                  is_return=True)
        ao_id = [row[col_names.index("id")] for row in rows]

        sql = "select id from facet_signal where facet_id = (select facet.id from facet where facet.name_ like " \
              "'budget' ) and org_id=%s and product_id=%s "
        rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_i), is_write=False,
                                                                  is_return=True)
        bo_id = [row[col_names.index("id")] for row in rows]

        sql = "select id from facet_signal where facet_id = (select facet.id from facet where facet.name_ like " \
              "'interest' ) and org_id=%s and product_id=%s "
        rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_i), is_write=False,
                                                                  is_return=True)
        io_id = [row[col_names.index("id")] for row in rows]

        sql = "select id from facet_signal where facet_id = (select facet.id from facet where facet.name_ like " \
              "'need investigation' ) and org_id=%s and product_id=%s "
        rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_i), is_write=False,
                                                                  is_return=True)
        no_id = [row[col_names.index("id")] for row in rows]

        count_dict = [{"Facet": "authority", "caught_count": len(set(a_id)), "caught_facetsignal_id": set(a_id),
                       "uncaught_count": len(ao_id) - len(set(a_id)),
                       "uncaught_facetsignal_id": set(ao_id) - set(a_id)},
                      {"Facet": "interest", "caught_count": len(set(i_id)), "caught_facetsignal_id": set(i_id),
                       "uncaught_count": len(io_id) - len(set(i_id)),
                       "uncaught_facetsignal_id": set(io_id) - set(i_id)},
                      {"Facet": "budget", "caught_count": len(set(b_id)), "caught_facetsignal_id": set(b_id),
                       "uncaught_count": len(bo_id) - len(set(b_id)),
                       "uncaught_facetsignal_id": set(bo_id) - set(b_id)},
                      {"Facet": "need_investigation", "caught_count": len(set(n_id)),
                       "caught_facetsignal_id": set(n_id), "uncaught_count": len(no_id) - len(set(n_id)),
                       "uncaught_facetsignal_id": set(no_id) - set(n_id)}]

        return result, count_dict
    else:
        return []


def make_result_intro(caught_facets, org_id, product_id):
    if len(caught_facets) != 0:
        result = []
        as_id = []
        kvp_id = []
        for item in caught_facets:
            sql = "select id,value,facet_id,org_id,product_id from facet_signal where facet_signal.id =%s"
            rows, col_names = db.DBUtils.get_instance().execute_query(sql,
                                                                      (str(item.facet_signal.orignal_facet_signal_id),),
                                                                      is_write=False, is_return=True)
            tmp_dict = {"Task_id": item.snippet.task_id, "Snippet_id": item.snippet.sid,
                        "Snippet_text": item.snippet_text,
                        "Snippet_Ques": item.snippet_question, "Dimension": "Introduction",
                        "Facet": item.facet_name,
                        "Facet_Signal_id": rows[0][col_names.index("id")],
                        "Face_Signal_text": rows[0][col_names.index("value")],
                        "Score": str(item.score)}
            result.append(tmp_dict)
            if item.facet_name.lower() == "aspiration setting":
                as_id.append(item.facet_signal.orignal_facet_signal_id)
            else:
                kvp_id.append(item.facet_signal.orignal_facet_signal_id)

        sql = "select id from facet_signal where facet_id = (select facet.id from facet where facet.name_ like " \
              "'aspiration setting' ) and org_id=%s and product_id=%s "
        rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_id), is_write=False,
                                                                  is_return=True)
        aso_id = [row[col_names.index("id")] for row in rows]

        sql = "select id from facet_signal where facet_id = (select facet.id from facet where facet.name_ like " \
              "'key value proposition' ) and org_id=%s and product_id=%s "
        rows, col_names = db.DBUtils.get_instance().execute_query(sql, (org_id, product_id), is_write=False,
                                                                  is_return=True)
        kvpo_id = [row[col_names.index("id")] for row in rows]

        count_dict = [
            {"Facet": "aspiration setting", "caught_count": len(set(as_id)), "caught_facetsignal_id": set(as_id),
             "uncaught_count": len(aso_id) - len(set(as_id)),
             "uncaught_facetsignal_id": set(aso_id) - set(as_id)},
            {"Facet": "key value proposition", "caught_count": len(set(kvp_id)), "caught_facetsignal_id": set(kvp_id),
             "uncaught_count": len(kvpo_id) - len(set(kvp_id)),
             "uncaught_facetsignal_id": set(kvpo_id) - set(kvp_id)}
        ]

        return result, count_dict
    else:
        return []


def make_combined_result(intro_result, intro_count, lq_result, lq_count, task_id, input_path):
    df = pd.DataFrame(intro_result + lq_result)
    df2 = pd.DataFrame(intro_count + lq_count)
    file_path = input_path.split(".")[0] + "_" + str(task_id)
    result_path = file_path + "_" + "caught_facets.xlsx"
    count_path = file_path + "_" + "count.xlsx"
    df.to_excel(result_path)
    df2.to_excel(count_path)
    output_path = file_path + ".zip"
    zipObject = ZipFile(output_path, 'w')
    zipObject.write(result_path)
    zipObject.write(count_path)
    zipObject.close()
    return output_path
