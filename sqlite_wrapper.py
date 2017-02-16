import sqlite3

def create_table(connection, cursor, table_name, columns):
    cursor.execute("create table IF NOT EXISTS {table_name}({columns})".format(table_name=table_name, columns=columns))
    connection.commit()


def insert_data(connection, cursor, table_name, data):
    cursor.execute("insert into {table_name} values {data}".format(table_name=table_name, data=data))
    connection.commit()

