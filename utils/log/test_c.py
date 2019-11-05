#!/usr/bin/python
# -*- coding:utf-8 -*-

import perception.utils.log.test_a
import perception.utils.log.test_b
from perception.utils.log.log_cfg import Log

log = Log(__name__).get_log()
log.info("I am c.py")
