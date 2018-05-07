#!/bin/sh

sudo pip3 install lib/.[web] --upgrade

echo "\nMoving web application components into place..."
sudo rm -rf /var/lib/text-align
sudo mkdir -p /var/lib/text-align

if [ -d web/web_app/node_modules ]
    then
        sudo rm -rf web/web_app/node_modules
fi
sudo cp -Rf web /var/lib/text-align/
sudo cp -Rf config /var/lib/text-align/

echo "\nMoving global configuration into place..."
sudo mkdir -p /etc/text-align
if [ ! -f /etc/text-align/apache_wsgi.conf ]
    then
        sudo cp -R web/apache_wsgi.conf /etc/text-align
        echo "\nMake sure you include /etc/text-align/apache_wsgi.conf in your main Apache configuration file in order to enable searching through the web app."
else
    echo "/etc/text-align/apache_wsgi.conf already exists, not modifying..."
fi

if [ ! -f /etc/text-align/global_settings.ini ]
    then
        sudo touch /etc/text-align/global_settings.ini
        echo "## DATABASE SETTINGS ##" | sudo tee -a /etc/text-align/global_settings.ini > /dev/null
        echo "[DATABASE]" | sudo tee -a /etc/text-align/global_settings.ini > /dev/null
        echo "database_name =" | sudo tee -a /etc/text-align/global_settings.ini > /dev/null
        echo "database_user =" | sudo tee -a /etc/text-align/global_settings.ini > /dev/null
        echo "database_password =" | sudo tee -a /etc/text-align/global_settings.ini > /dev/null
        echo "Make sure you create a PostgreSQL database with a user with read/write access to that database and configure /etc/text-align/global_settings.ini accordingly."
else
    echo "/etc/text-align/global_settings.ini already exists, not modifying..."
fi