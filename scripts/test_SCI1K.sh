echo 'SCI1K-x2' &&
python test.py --config ./configs/test/test-sci1k-02.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x3' &&
python test.py --config ./configs/test/test-sci1k-03.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x4' &&
python test.py --config ./configs/test/test-sci1k-04.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x5' &&
python test.py --config ./configs/test/test-sci1k-05.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x6' &&
python test.py --config ./configs/test/test-sci1k-06.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x7' &&
python test.py --config ./configs/test/test-sci1k-07.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x8' &&
python test.py --config ./configs/test/test-sci1k-08.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x9' &&
python test.py --config ./configs/test/test-sci1k-09.yaml --model $1 --gpu $2 &&
echo 'SCI1K-x10' &&
python test.py --config ./configs/test/test-sci1k-10.yaml --model $1 --gpu $2 &&

true
