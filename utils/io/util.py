#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import pytz
import datetime


def get_current_time_str() -> str:
    """.

    :return datetime
    """
    tz = pytz.timezone('Asia/Shanghai')
    date = datetime.datetime.now(tz)
    return date.strftime('%Y%m%d%H%M%S')
