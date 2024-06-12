#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import redis


class Redis(object):
    def __init__(self):
        self.redis = redis.StrictRedis(host='127.0.0.1',
                                       port=6379,
                                       password='',
                                       decode_responses=True)
