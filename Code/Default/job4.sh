#!/bin/bash


MID=$1
echo "${MID}"
python run_test.py Cli ${MID}
python top_k.py Cli ${MID}

echo "${MID}"
python run_test.py Codec ${MID}
python top_k.py Codec ${MID}

echo "${MID}"
python run_test.py Compress ${MID}
python top_k.py Compress ${MID}

echo "${MID}"
python run_test.py Csv ${MID}
python top_k.py Csv ${MID}

echo "${MID}"
python run_test.py Gson ${MID}
python top_k.py Gson ${MID}

echo "${MID}"
python run_test.py JacksonCore ${MID}
python top_k.py JacksonCore ${MID}

echo "${MID}"
python run_test.py JacksonXml ${MID}
python top_k.py JacksonXml ${MID}

echo "${MID}"
python run_test.py Jsoup ${MID}
python top_k.py Jsoup ${MID}

echo "${MID}"
python run_test.py Lang ${MID}
python top_k.py Lang ${MID}

echo "${MID}"
python run_test.py Math ${MID}
python top_k.py Math ${MID}

echo "${MID}"
python run_test.py Mockito ${MID}
python top_k.py Mockito ${MID}

echo "${MID}"
python run_test.py Time ${MID}
python top_k.py Time ${MID}






