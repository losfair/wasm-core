import sys
import os
import json


'''
target_dir
- config.json
- src
  - test_file1
  - test_file2

config.json
{
    'test_name': {
        'file_type': 'file_type:str',
        'source': 'file_path:str',
        'expected': 'output:str',
    },
}
'''


class TestCase:
    file_type_to_test_case = {}

    @classmethod
    def register(cls, file_type, sub_class):
        cls.file_type_to_test_case[file_type] = sub_class

    def __new__(cls, file_type, source, expected):
        sub_class = cls.file_type_to_test_case[file_type]
        return sub_class(file_type, source, expected)

    def __init__(self, file_type, source, expected):
        self.file_type = file_type
        self.source = source
        self.expected = expected
        self._result = None

    def run(self):
        self.preprocess()
        self._result = self.process()
        self.clean()

    def result(self):
        if self._result is None:
            raise RuntimeError('Test Case {} is not finished'.format(self.source))
        return self._result

    def is_passed(self):
        return self.result() == self.expected

    def preprocess(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError


class CTestCase(TestCase):
    def preprocess(self):
        os.system('gcc -o test {!r}'.format(self.source))

    def process(self):
        with os.popen('./test', 'r') as p:
            return ''.join(p)

    def clean(self):
        os.system('rm test')


class PyTestCase(TestCase):
    def preprocess(self):
        pass

    def process(self):
        with os.popen('python {}'.format(self.source), 'r') as p:
            return ''.join(p)

    def clean(self):
        pass


TestCase.register('cc', CTestCase)
TestCase.register('py', PyTestCase)

if __name__ == '__main__':
    target_dir = sys.argv[1]
    print(f"Running tests in directory {target_dir}")
    os.chdir(target_dir)

    with open("config.json", encoding='utf8') as f:
        config = json.loads(f.read())
    os.chdir('src')

    tcs = {k: TestCase(
        file_type=v['file_type'],
        source=v['source'],
        expected=v['expected']
    ) for k, v in config.items()}

    for tcn, tc in tcs.items():
        tc.run()
        if tc.is_passed():
            print(f'Test {repr(tcn)} failed with result:')
            print(tc.result())
        else:
            print(f'Test {repr(tcn)} passed.')

    print('All tests finished.')
