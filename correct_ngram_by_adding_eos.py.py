###
# There is a small problem that Transformers will not be happy about later on.
# The 5-gram correctly includes a "Unknown" or <unk>, as well as a begin-of-sentence, <s> token, but no end-of-sentence, </s> token.
# This sadly has to be corrected currently after the build.
# We can simply add the end-of-sentence token by adding the line 0 </s> -0.11831701 below the begin-of-sentence token and increasing the ngram 1 count by 1
###


import sys



with open(sys.argv[1], "r") as read_file, open(sys.argv[1].replace('arpa', 'correct.arpa'), "w") as write_file:
  has_added_eos = False
  for line in read_file:
    if not has_added_eos and "ngram 1=" in line:
      count=line.strip().split("=")[-1]
      write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
    elif not has_added_eos and "<s>" in line:
      write_file.write(line)
      write_file.write(line.replace("<s>", "</s>"))
      has_added_eos = True
    else:
      write_file.write(line)
