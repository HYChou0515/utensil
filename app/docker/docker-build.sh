cd ../..
poetry build
cp dist/utensil*.whl app/docker/
cd app
docker build -t app:v1 -f docker/Dockerfile .
