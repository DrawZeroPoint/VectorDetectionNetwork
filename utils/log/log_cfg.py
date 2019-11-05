#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import time
import os
import utils.web.util as web_util

path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")


class Log(object):
    """
封装后的logging
    """

    def __init__(self, logger=None, log_cate='lingxi'):
        """
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        """

        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        self.log_time = time.strftime("%Y_%m_%d")
        # file_dir = os.getcwd() + '/../../../../Logs/lingxi'  # test
        file_dir = path + '/../../../../Logs/lingxi'  # server

        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        # self.log_name = self.log_path + "/" + log_cate + "." + self.log_time + '.log'
        self.log_name = self.log_path + "/" + 'lingxi_{}.log'.format(web_util.get_curr_time_str())
        formatter = logging.Formatter(
            '[%(asctime)s] %(filename)s line:%(lineno)d [%(levelname)s]%(message)s')

        # print(self.log_name)

        f_handler = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        self.logger.addHandler(f_handler)

        # s_handler = logging.StreamHandler()
        # s_handler.setLevel(logging.INFO)
        # s_handler.setFormatter(formatter)
        # self.logger.addHandler(s_handler)

        f_handler.close()
        # s_handler.close()

    def get_log(self):
        return self.logger
