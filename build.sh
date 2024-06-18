echo 'Building api-gateway-service'
cd api-gateway-service

./build.sh

cd ..
echo 'Building recognition-service'
cd recognition-service

./build.sh
