#!/bin/sh

sudo -H pip3 install -r requirements.txt
sudo rm -rf /var/lib/text-align
sudo mkdir -p /var/lib/text-align
sudo cp -R web /var/lib/text-align/
sudo cp -R config /var/lib/text-align/
sudo mkdir -p /etc/text-align
sudo cp -R web/apache_wsgi.conf /etc/text-align
echo "Make sure you include /etc/text-align/apache_wsgi.conf in your main Apache configuration file in order to enable searching through the web app."

