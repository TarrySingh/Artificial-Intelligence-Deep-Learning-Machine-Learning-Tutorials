from functools import partial
from datetime import datetime

import constants
from parse_utils import group_data


def group_key(row):
    return row.vehicle_make


def filter_key(cutoff_date, gender, row):
    return row.last_updated >= cutoff_date and row.gender == gender


cutoff_date = datetime(2017, 3, 1)

for gender in ('Female', 'Male'):
    results = group_data(constants.fnames, constants.class_names,
                         constants.parsers, constants.compress_fields,
                         filter_key=partial(filter_key, cutoff_date, gender),
                         group_key=lambda row: row.vehicle_make)
    print(f'************ {gender} ************')
    print(list(results))
