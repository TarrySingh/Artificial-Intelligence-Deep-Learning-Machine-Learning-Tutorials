# run.py

print(f'loading run.py: __name__ = {__name__}')
import module1
import timing


if __name__ == '__main__':
    print('running run.py...')
    # result = timing.timeit('list(range(1_000_000))', repeats=20)
    result = timing.timeit('a=1)')
    print(result)
