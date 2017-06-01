src="/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.en"
target="/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.fr"
codes="/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.codes"
vocab1="/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.vocab.en"
vocab2="/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.vocab.fr"

./subword-nmt/learn_joint_bpe_and_vocab.py --input ${src} ${target} -s 32000 -o ${codes} --write-vocabulary ${vocab1} ${vocab2}
./subword-nmt/apply_bpe.py -c ${codes} --vocabulary ${vocab1} --vocabulary-threshold 50 < ${src} > ${src}.BPE
./subword-nmt/apply_bpe.py -c ${codes} --vocabulary ${vocab2} --vocabulary-threshold 50 < ${target} > ${target}.BPE