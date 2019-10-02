NUM=${1:-1}

for CURSOR in $(seq 1 $NUM); do
  WORD=`tail -c +$CURSOR arabeyes.wordlist | head -1`
  QUERY='query={"dictionary":"AR-EN-WORD-DICTIONARY","word":"'$WORD'","dfilter":true}'
  echo $QUERY
  URL="http://aratools.com/dict-service?format=json&_=1524853230411"
  CURL="curl -s $URL --data-urlencode '$QUERY'"
  echo $CURL
  RESULT=$($CURL | jq .result[].solution.root | uniq | recode html..)
  echo $WORD '=' $RESULT
done
