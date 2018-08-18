import json
import os.path
from sys import stderr
from traceback import format_exc

import MySQLdb
from MySQLdb.cursors import DictCursor


class DBCtrl(object):
    """Database Controller"""
    config_file = os.path.join(os.path.dirname(__file__), "config.json")
    encoding = "utf8mb4"
    DATABASE_NOT_FOUND = 1049
    DUPLICATE_ENTRY = 1062
    SERVER_HAS_GONE = 2006

    def __init__(self):
        try:
            with open(self.config_file) as f:
                self.config = json.load(f)['db']
        except Exception as ex:
            print("Config file (%s) error: %s\n" % (self.config_file, ex), file=stderr, flush=True)
            exit(1)
        try:
            self.connection = MySQLdb.connect(
                user=self.config['user']['username'],
                passwd=self.config['user']['password'],
                host=self.config['host'],
                cursorclass=DictCursor,
                use_unicode=True
            )
            self.connection.set_character_set(self.encoding)
        except MySQLdb.OperationalError as ex:
            print("DB Connection Error: %s\n" % ex, file=stderr, flush=True)
            exit(1)
        try:
            cursor = self._get_cursor()
            cursor.execute("use %s;" % (self.config['name']))
            self.connection.commit()
        except MySQLdb.OperationalError as ex:
            print("DB Preparation Error: %s\n" % ex, file=stderr, flush=True)
            self.connection.rollback()
            exit(1)

    def _get_cursor(self):
        cursor = None
        while not cursor:
            try:
                self.connection.ping(True)
                cursor = self.connection.cursor()
                cursor.execute("SET NAMES %s;" % self.encoding)
                cursor.execute("SET CHARACTER SET %s;" % self.encoding)
                cursor.execute("SET character_set_connection=%s;" % self.encoding)
            except MySQLdb.Error as ex:
                print("Cursor Error: %s\n\033[31m%s\033[0m\n" % (ex, format_exc()), file=stderr, flush=True)
        return cursor

    def add_row(self, table, values, rerais=False):
        """Add new row to a table of database.

        Not Implemented.
        """
        raise NotImplementedError("INSERT is not implemented in DBCtrl.")

    def get_rows(self, table, columns=[], values={}, rerais=False):
        """Get list of rows from a table of database."""
        res = tuple()
        cursor = self._get_cursor()
        try:
            cursor.execute(
                "select %s from %s%s;" % (
                    ["*", ", ".join(columns)][columns is not None and columns != []],
                    table,
                    ["", " where %s" % " and ".join(
                        ["%s=%%s" % key for key in values.keys()]
                    )][values is not None and values != {}]
                ),
                tuple(value for value in values.values())
            )
            self.connection.commit()
            res = cursor.fetchall()
        except Exception as ex:
            self.connection.rollback()
            if rerais:
                raise
            else:
                print("Select Error: %s\n\033[31m%s\033[0m\n" % (ex, format_exc()), file=stderr, flush=True)
        finally:
            cursor.close()
            return res

    def get_rows_by_query(self, table, columns=[], query="", values=[], rerais=False):
        """Get list of rows from a table of database by query and a list of values."""
        res = tuple()
        cursor = self._get_cursor()
        try:
            cursor.execute(
                "select %s from %s%s;" % (
                    ["*", ", ".join(columns)][columns is not None and columns != []],
                    table,
                    ["", " where %s" % query][query is not None and query != ""]
                ),
                values
            )
            self.connection.commit()
            res = cursor.fetchall()
        except Exception as ex:
            self.connection.rollback()
            if rerais:
                raise
            else:
                print("Select Error: %s\n\033[31m%s\033[0m\n" % (ex, format_exc()), file=stderr, flush=True)
        finally:
            cursor.close()
            return res

    def update_rows(self, table, conditions, values, rerais=False):
        """Update rows matching conditions in a table of database.

        Not Implemented.
        """
        raise NotImplementedError("UPDATE is not implemented in DBCtrl.")
