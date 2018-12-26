#!/bin/bash

ANTLR_JOB=${1:-ANTLR_Tool}
ST_VERSION=3.1
ANTLR2_VERSION=2.7.7

# find the antlr.jar from the upstream project
JAR=$(ls $WORKSPACE/../../$ANTLR_JOB/lastSuccessful/org.antlr\$antlr/archive/org.antlr/antlr/*/antlr-*-jar-with-dependencies.jar)
echo "antlr.jar=$JAR"

if [ ! -f "$JAR" ]; then
    echo "Could not find antlr.jar"
    exit 1
fi

    
echo "************************************************************************"
echo "Setting up dependencies"
echo

rm -fr $WORKSPACE/tmp
mkdir -p $WORKSPACE/tmp
cd $WORKSPACE

# stringtemplate3
if [ ! -f stringtemplate3-$ST_VERSION.tar.gz ]; then
    wget http://pypi.python.org/packages/source/s/stringtemplate3/stringtemplate3-$ST_VERSION.tar.gz
fi
(cd tmp; tar xzf ../stringtemplate3-$ST_VERSION.tar.gz)
(cd tmp/stringtemplate3-$ST_VERSION; python setup.py install --install-lib=$WORKSPACE)

# antlr2
if [ ! -f antlr-$ANTLR2_VERSION.tar.gz ]; then
    wget http://www.antlr2.org/download/antlr-$ANTLR2_VERSION.tar.gz
fi
(cd tmp; tar xzf ../antlr-$ANTLR2_VERSION.tar.gz)
(cd tmp/antlr-$ANTLR2_VERSION/lib/python; python setup.py install --install-lib=$WORKSPACE)


export CLASSPATH=$JAR

echo "************************************************************************"
echo "Running the testsuite"
echo

cd $WORKSPACE
rm -fr testout/
mkdir -p testout/
python setup.py unittest --xml-output=testout/
python setup.py functest --xml-output=testout/ --antlr-jar="$JAR"


echo "************************************************************************"
echo "Running pylint"
echo

cd $WORKSPACE
pylint --rcfile=pylintrc --output-format=parseable --include-ids=yes antlr3 | tee pylint-report.txt


echo "************************************************************************"
echo "Building dist files"
echo

cd $WORKSPACE
rm -f dist/*
cp -f $JAR dist/
python setup.py sdist --formats=gztar,zip
for PYTHON in /usr/bin/python2.?; do
    $PYTHON setup.py bdist_egg
done
