#!/bin/sh

ANTLR_HOME=`dirname "$0"`/../../..

java -Dfile.encoding=windows-1252 \
    -classpath "$ANTLR_HOME/build/classes:$ANTLR_HOME/build/rtclasses:$ANTLR_HOME/lib/antlr-3.0.jar:$ANTLR_HOME/lib/antlr-2.7.7.jar:$ANTLR_HOME/lib/stringtemplate-3.0.jar" \
    org.antlr.Tool \
    $@
