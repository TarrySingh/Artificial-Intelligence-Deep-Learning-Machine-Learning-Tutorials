perl -pe 'BEGIN{undef $/;} s#<gallery>.*</gallery>##s' \
  | perl -pe 'BEGIN{undef $/;} s#<timeline>.*</timeline>##s' \
  | sed -e 's/{{abbr|[^|]*|\([^}]\+\)}}/\1/g' \
        -e 's/{{as of|[^|}]*|alt=\([^|}]*\)}}/\1/g' \
        -e 's/{{as of|\([^|}]*\)}}/as of \1/g' \
        -e 's/{{as of|\([^|]*\)|[^}]*}}/as of \1/g' \
				-e 's/{{convert|\([^|]*\)|\([a-zA-Z]*\)[^}]*}}/\1 \2/g' \
  | pandoc --wrap=none -f mediawiki --filter $(dirname $0)/filter_markdown.py -t markdown
