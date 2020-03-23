class VadChunk(object):
    """this class holds the incoming snippet text and stores the encoding of each snippet"""

    def __init__(self, sid, from_time, to_time, speaker, text, text_encoding, confidence, task_id, old_snippet_list,
                 questions=None,
                 q_encoding=None,
                 encoding_method=None):
        self.sid = sid
        self.from_time = from_time
        self.to_time = to_time
        self.speaker = speaker
        self.text = text
        self.task_id = task_id
        self.old_snippet_list = old_snippet_list
        self.questions = questions
        self.confidence = confidence
        self.q_encoding = q_encoding
        self.text_encoding = text_encoding
        self.encoding_method = encoding_method

    def set_sid(self, sid):
        self.sid = sid

    def set_questions(self, questions):
        self.questions = questions

    def set_question_encoding(self, encoding, encoding_method):
        self.q_encoding = encoding
        self.encoding_method = encoding_method

    def set_text_encoding(self, encoding, encoding_method):
        self.text_encoding = encoding
        self.encoding_method = encoding_method


class FacetSignal(object):
    """This class stores the embedding of all the facet_signal """

    def __init__(self, fsid, text, orignal_facet_signal_id, embedding=None,
                 embedding_method=None):
        self.fsid = fsid
        self.text = text
        self.embedding = embedding
        self.embedding_method = embedding_method
        self.orignal_facet_signal_id = orignal_facet_signal_id

    def set_embedding(self, embedding, embedding_method):
        self.embedding = embedding
        self.embedding_method = embedding_method


class Facet(object):
    """ This class represents the facets that are defined for lead qualification and introduction, the facets have names(
        eg:authority,budget,KVP etc) , the list of facet_signal object and the id of facet signals caught for the particular
        facet type
    """

    def __init__(self, fid, name, facet_signals):
        self.fid = fid
        self.name = name
        self.facet_signals = facet_signals

    def set_id(self, fid, name):
        self.fid = fid
        self.name = name

    def set_facet_signals(self, facet_signal):
        self.facet_signals = facet_signal


class Dimension(object):
    """This class holds the dimensions """

    def __init__(self, dimid, name, facets):
        self.dimid = dimid
        self.name = name
        self.facets = facets

    def set_id(self, dimid, name):
        self.dimid = dimid
        self.name = name

    def set_facet_signals(self, facets):
        self.facets = facets


# class CaughtDimensions(object):
#     def __init__(self, cid, snippet, dimension, score, method):
#         self.snippet = snippet
#         self.dimension = dimension
#         self.score = score
#         self.method = method
#
#     def set_id(self, cid):
#         self.cid = cid


class CaughtFacetSignals(object):
    def __init__(self, snippet, snippet_text, snippet_question, facet_name, facet_signal, facet_signal_text, score,
                 method, dimension):
        self.snippet = snippet
        self.snippet_text = snippet_text
        self.snippet_question = snippet_question
        self.facet_name = facet_name
        self.facet_signal = facet_signal
        self.facet_signal_text = facet_signal_text
        self.score = score
        self.method = method
        self.dimension = dimension
