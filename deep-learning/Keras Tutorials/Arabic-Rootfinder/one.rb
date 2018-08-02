require 'json'
require 'htmlentities'

wordlist = File.read(ARGV[2] || "arabeyes.wordlist").split("\n")
starti = ARGV[0].to_i
endi = ARGV[1].to_i

puts "WORD,ROOT"
starti.upto(endi).each do |n|
  word = wordlist[n]
  query = 'query=' + ({ dictionary: "AR-EN-WORD-DICTIONARY", word: word, dfilter: true }.to_json)

  url = "http://aratools.com/dict-service?format=json&_=1524853230411"
  curl = "curl -s \"#{url}\" --data-urlencode '#{query}'"
  result = `#{curl}`
  formatted = JSON.parse(result, symbolize_names: true)[:result].map { |r| r[:solution][:root] }.uniq

  if formatted.size == 1
    formatted = HTMLEntities.new.decode(formatted[0])
    puts "#{word},#{formatted}"
  end
end
