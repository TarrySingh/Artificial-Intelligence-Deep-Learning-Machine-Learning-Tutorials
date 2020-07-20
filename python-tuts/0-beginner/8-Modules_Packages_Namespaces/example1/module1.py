# module1.py

print('----- Running {0} -----'.format(__name__))


def pprint_dict(header, d):
    print('\n\n----------------------------------------')
    print('***** {0} *****'.format(header))
    for key, value in d.items():
        print(key, value)
    print('----------------------------------------\n\n')


pprint_dict('module1.globals', globals())

print('----- End of {0} -----'.format(__name__))
