from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Mysql(object):
    def __init__(self):
        self.engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/recommendation')
        self.db_session = sessionmaker(bind=self.engine)
