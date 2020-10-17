import constants
import parse_utils

# # see a sample of what is in each file
# for fname in constants.fnames:
#     print(fname)
#     with open(fname) as f:
#         print(next(f), end='')
#         print(next(f), end='')
#         print(next(f), end='')
#     print()

# for fname in constants.fnames:
#     print(fname)
#     with open(fname) as f:
#         reader = csv.reader(f, delimiter=',', quotechar='"')
#         print(next(reader))
#         print(next(reader))
#     print()

# # header row (field names)
# for fname in constants.fnames:
#     print(fname)
#     reader = parse_utils.csv_parser(fname, include_header=True)
#     print(next(reader), end='\n')
#
# print('\n\n')
#
# # just the data
# for fname in constants.fnames:
#     print(fname)
#     reader = parse_utils.csv_parser(fname)
#     print(next(reader))
#     print(next(reader), end='\n')

# reader = parse_utils.csv_parser(constants.fname_update_status)
# for _ in range(5):
#     record = next(reader)
#     record = [str(record[0]), parse_utils.parse_date(record[1]), parse_utils.parse_date(record[2])]
#     print(record)

for fname, class_name, parser in zip(constants.fnames, constants.class_names, constants.parsers):
    file_iter = parse_utils.iter_file(fname, class_name, parser)
    print(fname)
    for _ in range(3):
        print(next(file_iter))
    print()