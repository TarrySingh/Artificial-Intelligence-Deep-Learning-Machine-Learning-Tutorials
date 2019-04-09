import setuptools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'requests',
    'wget',
]

setuptools.setup(
    name="word2word",
    version="0.1.6",
    author="Kyubyong Park, Dongwoo Kim, Yo Joong Choe",
    author_email="kbpark.linguist@gmail.com, kimdwkimdw@gmail.com, yjchoe33@gmail.com",
    description="Word Translator for 3,564 Language Pairs",
    install_requires=REQUIRED_PACKAGES,
    license='Apache License 2.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kyubyong/word2word",
    packages=setuptools.find_packages(),
    package_data={'word2word': ['word2word/supporting_languages.txt']},
    python_requires=">=3.6",
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
