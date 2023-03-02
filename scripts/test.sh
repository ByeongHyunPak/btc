echo '= [SCI1K] =' &&
bash ./scripts/test_SCI1K.sh $1 $2 &&

echo '= [SCID] =' &&
bash ./scripts/test_SCID.sh $1 $2 &&

echo '= [SIQAD] =' &&
bash ./scripts/test_SIQAD.sh $1 $2 &&

true