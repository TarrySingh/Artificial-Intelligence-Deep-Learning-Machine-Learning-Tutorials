import csv
from datetime import datetime
from collections import namedtuple


def csv_parser(fname, *, delimiter=',', quotechar='"', include_header=False):
    with open(fname) as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        if not include_header:
            next(f)
        yield from reader


def parse_date(value, *, fmt='%Y-%m-%dT%H:%M:%SZ'):
    return datetime.strptime(value, fmt)


def extract_field_names(fname):
    reader = csv_parser(fname, include_header=True)
    return next(reader)


def create_named_tuple_class(fname, class_name):
    fields = extract_field_names(fname)
    return namedtuple(class_name, fields)


def iter_file(fname, class_name, parser):
    nt_class = create_named_tuple_class(fname, class_name)
    reader = csv_parser(fname)
    for row in reader:
        parsed_data = (parse_fn(value)for value, parse_fn in zip(row, parser))
        yield nt_class(*parsed_data)
