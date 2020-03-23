class SkenExceptions(Exception):
    pass


class NoFacetFound(SkenExceptions):
    """This exception is raised if thee are no facets for the particular organisation """

    def __init__(self, org_id, product_id):
        self.code = 9999
        self.message = "{}: The facet for organization={} and product_id={} ".format(self.code, org_id, product_id)
