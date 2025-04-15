
#!/bin/bash

# Q1 Refactoring
echo "Q1. How does the if statement that we use for this looks like?: if __name__ == "__main__":"

# Q2 Installing pytest
echo "Q2. What should be the other file?: __init__.py"

# Q3 Writing first unit test
echo "Q3. How many rows should be there in the expected dataframe?: 2"

# Q4 Mocking S3 with Localstack 

echo "Q4. In both cases we should adjust commands for localstack. What option do we need to use for such purposes?: --endpoint-url"

# Q5 Creating test data

docker-compose up -d
sleep 1

BUCKET_NAME="nyc-duration"
AWS_DEFAULT_REGION="us-east-1"

S3_ENDPOINT_URL="http://localhost:4566"

export S3_ENDPOINT_URL=$S3_ENDPOINT_URL

aws --endpoint-url=$S3_ENDPOINT_URL s3 mb s3://$BUCKET_NAME
aws --endpoint-url=$S3_ENDPOINT_URL s3 ls


pipenv run pytest -s integration_test.py

echo "Q5. What's the size of the file?: $(aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in --recursive)"

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
fi

docker-compose down

exit ${ERROR_CODE}
