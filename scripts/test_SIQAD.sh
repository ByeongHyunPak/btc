echo 'SIQAD-x2' &&
python test.py --config ./configs/test/test-siqad-02.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x3' &&
python test.py --config ./configs/test/test-siqad-03.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x4' &&
python test.py --config ./configs/test/test-siqad-04.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x5' &&
python test.py --config ./configs/test/test-siqad-05.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x6' &&
python test.py --config ./configs/test/test-siqad-06.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x7' &&
python test.py --config ./configs/test/test-siqad-07.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x8' &&
python test.py --config ./configs/test/test-siqad-08.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x9' &&
python test.py --config ./configs/test/test-siqad-09.yaml --model $1 --gpu $2 &&
echo 'SIQAD-x10' &&
python test.py --config ./configs/test/test-siqad-10.yaml --model $1 --gpu $2 &&

true