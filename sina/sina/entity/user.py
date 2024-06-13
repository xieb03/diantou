from typing import Any

from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

from ..dao.mysql_db import Mysql

Base = declarative_base()


class User(Base):
    __tablename__ = 'User'
    id = Column(Integer(), primary_key=True)
    username = Column(String(20))
    # password 之所以设置了很长的字符，是因为我们要对密码进行加密。根据加密方式的不同长度也会不同，在这里我就设置的相对长一点
    password = Column(String(500))
    nick = Column(String(20))
    gender = Column(String(10))
    age = Column(String(2))
    city = Column(String(10))

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        mysql = Mysql()
        engine = mysql.engine
        Base.metadata.create_all(engine)
