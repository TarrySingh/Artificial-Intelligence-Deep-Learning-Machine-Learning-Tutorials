import itertools
from datetime import datetime

import constants
import parse_utils

# for fname, class_name, parser in zip(constants.fnames, constants.class_names, constants.parsers):
#     file_iter = parse_utils.iter_file(fname, class_name, parser)
#     print(fname)
#     for _ in range(3):
#         print(next(file_iter))
#     print()

# gen = parse_utils.iter_combined_plain_tuple(constants.fnames, constants.class_names,
#                                             constants.parsers, constants.compress_fields)
#
# print(list(next(gen)))
# print(list(next(gen)))

# nt = parse_utils.create_combo_named_tuple_class(constants.fnames, constants.compress_fields)
# print(nt._fields)

data_iter = parse_utils.iter_combined(constants.fnames, constants.class_names,
                                      constants.parsers, constants.compress_fields)

for row in itertools.islice(data_iter, 5):
    print(row)

print('-------------------------------')

cutoff_date = datetime(2018, 3, 1)
filtered_iter = parse_utils.filtered_iter_combined(constants.fnames, constants.class_names,
                                                   constants.parsers, constants.compress_fields,
                                                   key=lambda row: row.last_updated >= cutoff_date)
for row in filtered_iter:
    print(row)