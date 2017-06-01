SITE="http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/"

lang1="en"
lang2="he"

OSUB_PREFIX="OpenSubtitles2016.${lang1}-${lang2}"
TMX_FILE="${lang1}-${lang2}.tmx"

wget -nc "${SITE}${TMX_FILE}.gz" -O "${TMX_FILE}.gz";
gunzip "${TMX_FILE}.gz";
cat ${TMX_FILE} | sed -n "s/.*xml:lang=\"${lang1}\"><seg>\(.*\)<\/seg>.*/\1/p" > "${OSUB_PREFIX}.${lang1}";
cat ${TMX_FILE} | sed -n "s/.*xml:lang=\"${lang2}\"><seg>\(.*\)<\/seg>.*/\1/p" > "${OSUB_PREFIX}.${lang2}";
rm ${TMX_FILE};
