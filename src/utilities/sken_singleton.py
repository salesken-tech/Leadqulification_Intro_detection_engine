from src.utilities import sken_logger, constants
from laserembeddings import Laser
from sentence_transformers import SentenceTransformer

logger = sken_logger.get_logger("sken_singleton")


class Singletons:
    __instance = None
    laser_embedder = cached_lq_dims = cached_intro_dims = None

    # robert_embedder = None

    @staticmethod
    def get_instance():
        """Static access method"""
        if Singletons.__instance is None:
            logger.info("Calling private constructor for embedder initialization ")
            Singletons()
        return Singletons.__instance

    def __init__(self):
        if Singletons.__instance is not None:
            raise Exception("The singleton is already initialized you are attempting to initialize it again get lost")
        else:
            logger.info("Initializing Laser embedder")
            self.laser_embedder = Laser()
            self.cached_lq_dims = {}
            self.cached_intro_dims = {}

            # logger.info("Initializing Roberta embedder")
            # self.robert_embedder = SentenceTransformer(constants.fetch_constant("robeta_path"))
            Singletons.__instance = self

    def perform_embeddings(self, all_sentences):
        """
        This method embeds all the sentences passed using Laser embedder
        :param all_sentences:
        :return: list of sentence embeddings
        """
        if self.laser_embedder is not None:
            sentence_embeddings = self.laser_embedder.embed_sentences(all_sentences, ["en"] * len(all_sentences))
            return sentence_embeddings
        else:
            logger.info("the embedder is not set please restart the service")

    # def perform_embeddings(self, all_sentences):
    #     """
    #     This method embeds all the sentences passed using Laser embedder
    #     :param all_sentences:
    #     :return: list of sentence embeddings
    #     """
    #     if self.robert_embedder is not None:
    #         sentence_embeddings = self.robert_embedder.encode(all_sentences)
    #         return sentence_embeddings
    #     else:
    #         logger.info("the embedder is not set please restart the service")

    def get_cached_lq_dims(self):
        """
        :return: the dictionary of cached facets
        """
        return self.cached_lq_dims

    def set_cached_lq_dims(self, facet_name, facet):
        """
        :return: the dictionary of cached facets
        """
        self.cached_lq_dims[facet_name] = facet

    def get_cached_intro_dims(self):
        """
        :return: the dictionary of cached facets
        """
        return self.cached_intro_dims

    def set_cached_intro_dims(self, facet_name, facet):
        """
        :return: the dictionary of cached facets
        """
        self.cached_intro_dims[facet_name] = facet

