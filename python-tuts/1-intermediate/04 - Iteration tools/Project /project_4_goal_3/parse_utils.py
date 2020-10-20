import csv
from datetime import datetime
from collections import namedtuple
import itertools


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


def create_combo_named_tuple_class(fnames, compress_fields):
    compress_fields = itertools.chain.from_iterable(compress_fields)
    field_names = itertools.chain.from_iterable(extract_field_names(fname) for fname in fnames)
    compressed_field_names = itertools.compress(field_names, compress_fields)
    return namedtuple('Data', compressed_field_names)


def iter_file(fname, class_name, parser):
    nt_class = create_named_tuple_class(fname, class_name)
    reader = csv_parser(fname)
    for row in reader:
        parsed_data = (parse_fn(value)for value, parse_fn in zip(row, parser))
        yield nt_class(*parsed_data)


def iter_combined_plain_tuple(fnames, class_names, parsers, compress_fields):
    compress_fields = tuple(itertools.chain.from_iterable(compress_fields))
    zipped_tuples = zip(*(iter_file(fname, class_name, parser)
                          for fname, class_name, parser in zip(fnames, class_names, parsers)))

    merged_iter = (itertools.chain.from_iterable(zipped_tuple) for zipped_tuple in zipped_tuples)
    for row in merged_iter:
        compressed_row = itertools.compress(row, compress_fields)
        yield tuple(compressed_row)


def iter_combined(fnames, class_names, parsers, compress_fields):
    combo_nt = create_combo_named_tuple_class(fnames, compress_fields)
    compress_fields = tuple(itertools.chain.from_iterable(compress_fields))
    zipped_tuples = zip(*(iter_file(fname, class_name, parser)
                          for fname, class_name, parser in zip(fnames, class_names, parsers)))

    merged_iter = (itertools.chain.from_iterable(zipped_tuple) for zipped_tuple in zipped_tuples)
    for row in merged_iter:
        compressed_row = itertools.compress(row, compress_fields)
        yield combo_nt(*compressed_row)


def filtered_iter_combined(fnames, class_names, parsers, compress_fields, *, key=None):
    iter_combo = iter_combined(fnames, class_names, parsers, compress_fields)
    yield from filter(key, iter_combo)
