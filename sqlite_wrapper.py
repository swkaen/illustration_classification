import sqlite3

def create_table(connection, cursor, table_name, columns):
    cursor.execute("create table {table_name}({columns})".format(table_name=table_name, columns=columns))
    connection.commit()

