echo '*** Build for Virtualization Project ***'

echo 'Cleaning up folders...'
chmod +x clean.sh
./clean.sh

echo 'Starting containters...'
docker compose up -d --build