#!/bin/sh

# Define the number of workers for your app. Between 4 and 12 should be fine.
WORKERS=4

# Define on which port the webserver will be listening. If you have a webserver already listening to port 80
# you should proxy requests to below port to the app
PORT=80

# If using an https connection (you should), define your SSL keys and certificate locations here
KEYFILE=
CERTFILE=

if [ -z "$KEYFILE" ]
then
    gunicorn -k uvicorn.workers.UvicornWorker -b :$PORT -w 4 --access-logfile=/var/lib/text-pair/api_server/access.log --error-logfile=/var/lib/text-pair/api_server/error.log --chdir /var/lib/text-pair/api/ text_pair:app
else
    gunicorn --keyfile=$KEYFILE --certfile=$CERTFILE -k uvicorn.workers.UvicornWorker -b :$PORT -w 4 --access-logfile=/var/lib/text-pair/api_server/access.log --error-logfile=/var/lib/text-pair/api_server/error.log --chdir /var/lib/text-pair/api/ text_pair:app
fi