#!/bin/bash

if [ -e doxygen.sh ]; then
    . doxygen.sh
fi

rm -fr build/doc
mkdir -p build/doc/antlr3

for f in __init__ exceptions constants dfa tokens streams recognizers; do
    sed -e '/begin\[licence\]/,/end\[licence\]/d' antlr3/$f.py \
      >>build/doc/antlr3.py
done

touch build/doc/antlr3/__init__.py

cp -f antlr3/tree.py build/doc/antlr3
cp -f antlr3/treewizard.py build/doc/antlr3

doxygen doxyfile
