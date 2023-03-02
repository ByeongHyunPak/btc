echo 'SCID-x2' &&
python test.py --config ./configs/test/test-scid-02.yaml --model $1 --gpu $2 &&
echo 'SCID-x3' &&
python test.py --config ./configs/test/test-scid-03.yaml --model $1 --gpu $2 &&
echo 'SCID-x4' &&
python test.py --config ./configs/test/test-scid-04.yaml --model $1 --gpu $2 &&
echo 'SCID-x5' &&
python test.py --config ./configs/test/test-scid-05.yaml --model $1 --gpu $2 &&
echo 'SCID-x6' &&
python test.py --config ./configs/test/test-scid-06.yaml --model $1 --gpu $2 &&
echo 'SCID-x7' &&
python test.py --config ./configs/test/test-scid-07.yaml --model $1 --gpu $2 &&
echo 'SCID-x8' &&
python test.py --config ./configs/test/test-scid-08.yaml --model $1 --gpu $2 &&
echo 'SCID-x9' &&
python test.py --config ./configs/test/test-scid-09.yaml --model $1 --gpu $2 &&
echo 'SCID-x10' &&
python test.py --config ./configs/test/test-scid-10.yaml --model $1 --gpu $2 &&

true