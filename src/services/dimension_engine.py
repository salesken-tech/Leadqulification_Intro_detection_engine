import time

import numpy as np

from src.utilities import sken_singleton, sken_logger, constants, sken_exceptions
import os
import pandas
from src.utilities.db import DBUtils
from src.utilities.objects import FacetSignal, Facet, CaughtFacetSignals
from src.services import snippet_service, facet_service

from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

pool = ThreadPool(2)

logger = sken_logger.get_logger("dimension_engine")


def refresh_cached_dims(org_id, prod_id):
    """
    This method refreshes the cached_dimensions singleton ,the method clears the cached dimensions whenever a new
    product request is made
    :return:
    """

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(sken_singleton.Singletons.get_instance().get_cached_lq_dims().clear())
        executor.submit(sken_singleton.Singletons.get_instance().get_cached_lq_dims().clear())
    logger.info("Cleared cached dimensions for org_id ={} and product_id={}".format(org_id, prod_id))


def make_cached_dimensions(org_id, prod_id):
    """
    This method caches the facet signals for the particular product and organization
    :param org_id:
    :param prod_id:
    :return:
    """
    if len(sken_singleton.Singletons.get_instance().get_cached_lq_dims()) == 0 or len(
            sken_singleton.Singletons.get_instance().get_cached_intro_dims()) == 0:
        logger.info("Creating cached_dimensions for organization={} and product={}".format(org_id, prod_id))
        sql = "select dimension.id as dimid,dimension.name_ as dimname,facet.id as facet_id,facet.name_ as " \
              "facet_name,facet_signal.id as fsid,facet_signal.value as fsval,generated_facet_signals.id as gsid," \
              "generated_facet_signals.value as gs_value,facet_signal.org_id,facet_signal.product_id from dimension " \
              "left join facet on facet.dim_id = dimension.id left join facet_signal on facet_signal.facet_id = " \
              "facet.id left join generated_facet_signals on 	generated_facet_signals.facet_signal_id = " \
              "facet_signal.id where facet_signal.org_id=%s and facet_signal.product_id=%s group by dimension.id," \
              "facet.id,facet_signal.id,generated_facet_signals.id "

        rows, col_names = DBUtils.get_instance().execute_query(sql, (org_id, prod_id), is_write=False, is_return=True)
        kvp_id = as_id = a_id = b_id = i_id = n_id = None
        if len(rows) != 0:
            start = time.time()
            logger.info("Making cache facet signals for organization= {} and product={}".format(org_id, prod_id))
            kvp_facet_signals = []
            as_facet_signals = []
            authority_facet_singals = []
            budget_facte_singals = []
            interest_face_signals = []
            need_facet_singals = []
            for row in rows:
                if str(row[col_names.index("dimname")]).lower() == "introduction":
                    if str(row[col_names.index("facet_name")]).lower() == "key value proposition":
                        kvp_id = row[col_names.index("fsid")]
                        if row[col_names.index("gs_value")] is not None:
                            kvp_facet_signals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=sken_singleton.Singletons.get_instance().perform_embeddings(
                                                row[col_names.index("gs_value")]),
                                            embedding_method=constants.fetch_constant("encoding_method")))
                        else:
                            kvp_facet_signals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=None,
                                            embedding_method=None))
                    else:
                        as_id = row[col_names.index("fsid")]
                        if row[col_names.index("gs_value")] is not None:
                            as_facet_signals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=sken_singleton.Singletons.get_instance().perform_embeddings(
                                                row[col_names.index("gs_value")]),
                                            embedding_method=constants.fetch_constant("encoding_method")))
                        else:
                            as_facet_signals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=None,
                                            embedding_method=None))
                else:
                    if str(row[col_names.index("facet_name")]).lower() == "authority":
                        a_id = row[col_names.index("fsid")]
                        if row[col_names.index("gs_value")] is not None:
                            authority_facet_singals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=sken_singleton.Singletons.get_instance().perform_embeddings(
                                                row[col_names.index("gs_value")]),
                                            embedding_method=constants.fetch_constant("encoding_method")))
                        else:
                            authority_facet_singals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=None,
                                            embedding_method=None))
                    elif str(row[col_names.index("facet_name")]).lower() == "budget":
                        b_id = row[col_names.index("fsid")]
                        if row[col_names.index("gs_value")] is not None:
                            budget_facte_singals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=sken_singleton.Singletons.get_instance().perform_embeddings(
                                                row[col_names.index("gs_value")]),
                                            embedding_method=constants.fetch_constant("encoding_method")))
                        else:
                            budget_facte_singals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=None,
                                            embedding_method=None))
                    elif str(row[col_names.index("facet_name")]).lower() == "interest":
                        i_id = row[col_names.index("fsid")]
                        if row[col_names.index("gs_value")] is not None:
                            interest_face_signals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=sken_singleton.Singletons.get_instance().perform_embeddings(
                                                row[col_names.index("gs_value")]),
                                            embedding_method=constants.fetch_constant("encoding_method")))
                        else:
                            interest_face_signals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=None,
                                            embedding_method=None))
                    else:
                        n_id = row[col_names.index("fsid")]
                        if row[col_names.index("gs_value")] is not None:
                            need_facet_singals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=sken_singleton.Singletons.get_instance().perform_embeddings(
                                                row[col_names.index("gs_value")]),
                                            embedding_method=constants.fetch_constant("encoding_method")))
                        else:
                            need_facet_singals.append(
                                FacetSignal(row[col_names.index("gsid")], row[col_names.index("gs_value")],
                                            row[col_names.index("fsid")],
                                            embedding=None,
                                            embedding_method=None))

            with ThreadPoolExecutor(max_workers=6) as executor:
                executor.submit(sken_singleton.Singletons.get_instance().set_cached_intro_dims, "key_value_proposition",
                                Facet(kvp_id, "key value proposition", kvp_facet_signals))
                executor.submit(sken_singleton.Singletons.get_instance().set_cached_intro_dims, "aspiration_setting",
                                Facet(as_id, "aspiration setting", as_facet_signals))
                executor.submit(sken_singleton.Singletons.get_instance().set_cached_lq_dims, "authority",
                                Facet(a_id, "authority", authority_facet_singals))
                executor.submit(sken_singleton.Singletons.get_instance().set_cached_lq_dims, "budget",
                                Facet(b_id, "budget", budget_facte_singals))
                executor.submit(sken_singleton.Singletons.get_instance().set_cached_lq_dims, "interest",
                                Facet(i_id, "interest", interest_face_signals))
                executor.submit(sken_singleton.Singletons.get_instance().set_cached_lq_dims, "need_investigation",
                                Facet(n_id, "need investigation", need_facet_singals))

            logger.info("Cached {} facet signals for org={} and product={} in {}".format(len(
                kvp_facet_signals + as_facet_signals + authority_facet_singals + interest_face_signals + budget_facte_singals + need_facet_singals),
                org_id, prod_id, (time.time() - start)))

        else:
            logger.info("No facet_signals found for organization={} and product={}".format(org_id, prod_id))
            raise sken_exceptions.NoFacetFound(org_id, prod_id)
    else:
        logger.info(
            "Skipping caching of facet_signals for organization={} and product_id ={}, they already exist in RAM".format(
                org_id, prod_id))


def wraper_method(input_file_path, org_id, product_id, threshold):
    df = snippet_service.agent_customer_sequence(input_file_path)
    sql = "select max(task_id) as task_id from new_snippet"
    rows, col_names = DBUtils.get_instance().execute_query(sql, (), is_write=False, is_return=True)
    task_id = rows[0][col_names.index("task_id")]
    if task_id is None:
        task_id = 1
    else:
        task_id = int(task_id) + 1

    logger.info("Making new_snippet entries in the database")
    data = []
    if len(df) != 0:
        for i in range(len(df)):
            data.append(tuple(
                [df["from_time"][i], df["to_time"][i], 0, df["text"][i], df["speaker"][i], str(df["orignal_ids"][i]),
                 task_id]))
        snippet_ids = DBUtils.get_instance().insert_bulk("new_snippet",
                                                         "from_time, to_time, confidence, text_, speaker, "
                                                         "snippet_list,task_id", data, return_parameter=[
                "id"])
    vad_chunks = snippet_service.make_snippets(df, snippet_ids, task_id)
    if len(vad_chunks) != 0:
        try:
            make_cached_dimensions(org_id, product_id)
            for vad_chunk in vad_chunks:
                snippet_service.find_snippet_questions(vad_chunk)
                snippet_service.make_snippet_question_embeddings(vad_chunk)
            with ThreadPoolExecutor(max_workers=2) as executore:
                caught_intro_facets = executore.submit(facet_service.create_intro_matches, vad_chunks,
                                                       threshold).result()
                caught_lq_facets = executore.submit(facet_service.create_lq_maches, vad_chunks, threshold).result()
            with ThreadPoolExecutor(max_workers=2) as executor:
                lq_result, lq_count = executor.submit(facet_service.make_result_lq, caught_lq_facets, org_id,
                                                      product_id).result()
                intro_result, intro_count = executor.submit(facet_service.make_result_intro, caught_intro_facets,
                                                            org_id, product_id).result()
            for item in lq_result:
                sql = "INSERT INTO public.caught_facets (new_snippet_id, snippet_text, fact_signal_id, " \
                      "facet_signal_text, facet_name, dimension_name,score) VALUES(%s, %s, %s, %s, %s, %s,%s); "
                DBUtils.get_instance().execute_query(sql, (
                    item["Snippet_id"], item["Snippet_text"], item["Facet_Signal_id"], item["Face_Signal_text"],
                    item["Facet"], item["Dimension"], item["Score"]), is_write=True, is_return=False)
            for item in intro_result:
                sql = "INSERT INTO public.caught_facets (new_snippet_id, snippet_text, fact_signal_id, " \
                      "facet_signal_text, facet_name, dimension_name,score) VALUES(%s, %s, %s, %s, %s, %s,%s); "
                DBUtils.get_instance().execute_query(sql, (
                    item["Snippet_id"], item["Snippet_text"], item["Facet_Signal_id"], item["Face_Signal_text"],
                    item["Facet"], item["Dimension"], item["Score"]), is_write=True, is_return=False)
            logger.info("Made entries for caught facets")
            output_path = facet_service.make_combined_result(intro_result,
                                                             intro_count, lq_result,
                                                             lq_count,
                                                             task_id,
                                                             input_file_path)
            return output_path

        except sken_exceptions.NoFacetFound as e:
            print(e.message)


if __name__ == '__main__':
    wraper_method("/home/andy/Downloads/snippets_test.xlsx", 1, 150, 0.7)
