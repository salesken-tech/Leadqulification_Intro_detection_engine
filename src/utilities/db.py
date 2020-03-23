import psycopg2
import sys
from src.utilities import constants
from src.utilities import sken_logger
from inspect import getframeinfo, stack
from psycopg2 import pool
from psycopg2.extras import execute_values
import re

logger = sken_logger.get_logger("db")


class DBUtils:
    __instance = None
    sales_pool = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if DBUtils.__instance is None:
            logger.info("Calling private constructor for connection pool initialization")
            DBUtils()
        return DBUtils.__instance

    def __init__(self):
        if DBUtils.__instance is not None:
            raise Exception("This is a singleton class ")
        else:
            logger.info(
                "Initializing connection pool for database connection, should happen only once during startup. with {}".format(
                    constants.fetch_constant("host")))
            self.sales_pool = pool.ThreadedConnectionPool(constants.fetch_constant("min_pool"),
                                                          constants.fetch_constant("max_pool"),
                                                          host=constants.fetch_constant("host"),
                                                          user="postgres",
                                                          password=constants.fetch_constant("password"),
                                                          port="5432",
                                                          database=constants.fetch_constant("db_name"))

            logger.info("Made {} max_connections ".format(self.sales_pool.maxconn))
            DBUtils.__instance = self

    def execute_query(self, sql, data, is_write=False, is_return=True):
        """This function is used to execute sql queries but don't use to insert multiple vales in db as insertion
        operation is slow so use next method """
        caller = getframeinfo(stack()[1][0])
        rows = []
        col_names = None
        con = None
        try:
            con = self.sales_pool.getconn()
            cur = con.cursor()
            try:
                logger.info("[%s]:%d - %s" % (
                    caller.filename, caller.lineno, re.sub('\\s+', ' ', cur.mogrify(sql, data).decode())))
            except (TypeError, IndexError) as e:
                logger.info("[%s]:%d - %s" % (caller.filename, caller.lineno, sql))
            cur.execute(sql, data)
            if cur.description:
                col_names = [desc[0] for desc in cur.description]
            if is_write:
                con.commit()
            if is_return:
                while True:
                    row = cur.fetchone()
                    if row is None:
                        break
                    rows.append(row)
        except psycopg2.DatabaseError as e:
            if con:
                con.rollback()
            logger.error(e)
            sys.exit(1)
        finally:
            if con:
                con.close()
                self.sales_pool.putconn(con)
        return rows, col_names

    def insert_bulk(self, table_name, col_names, data, return_parameter=None):
        """
        Used for making bulk inserts into the database
        @param table_name:
        @param col_names:
        @param data:
        @param return_parameter: The column name of the returning parameter 'id'
        @return:
        """
        rows = []
        try:
            con = self.sales_pool.getconn()
            cur = con.cursor()
            try:
                if return_parameter is None:
                    sql_insert_generic = "INSERT INTO {} ({}) VALUES %s)".format(table_name, col_names)
                    logger.info(sql_insert_generic)
                    logger.info(
                        "Inserting {} entries into {} table with no return parameter".format(len(data), table_name))
                    execute_values(cur, sql_insert_generic, data, template=None, page_size=len(data))
                    con.commit()
                else:
                    return_parameter_string = ""
                    for item in return_parameter:
                        return_parameter_string += item + ","
                    return_parameter_string = return_parameter_string[:-1]
                    sql_insert_return = "INSERT INTO {} ({}) VALUES  %s returning {}".format(table_name, col_names,
                                                                                             return_parameter_string)
                    logger.info(sql_insert_return)
                    logger.info(
                        "Inserting {} entries into {} table with {} return parameter".format(len(data), table_name,
                                                                                             return_parameter))
                    execute_values(cur, sql_insert_return, data, template=None, page_size=len(data))
                    con.commit()
                    for row in cur.fetchall():
                        for item in row:
                            rows.append(item)
            except psycopg2.DataError as e:
                logger.error(e)
                con.rollback()
                pass
        except psycopg2.DatabaseError as e:
            logger.error(e)
            pass
        finally:
            if con:
                con.close()
                self.sales_pool.putconn(con)
        return rows
