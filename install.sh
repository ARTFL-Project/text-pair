#!/bin/sh

sudo pip3 install lib/.[web] --upgrade
sudo pip3 install https://github.com/ARTFL-Project/text-preprocessing/archive/v0.5.tar.gz

echo "\nMoving web application components into place..."
sudo rm -rf /var/lib/text-pair
sudo mkdir -p /var/lib/text-pair

if [ -d web/web_app/node_modules ]
    then
        sudo rm -rf web/web_app/node_modules
fi
sudo cp -Rf web /var/lib/text-pair/
sudo cp -Rf config /var/lib/text-pair/

echo "\nMoving global configuration into place..."
sudo mkdir -p /etc/text-pair
if [ ! -f /etc/text-pair/apache_wsgi.conf ]
    then
        sudo cp -R web/apache_wsgi.conf /etc/text-pair
        echo "\nMake sure you include /etc/text-pair/apache_wsgi.conf in your main Apache configuration file in order to enable searching through the web app."
else
    echo "/etc/text-pair/apache_wsgi.conf already exists, not modifying..."
fi

if [ ! -f /etc/text-pair/global_settings.ini ]
    then
        sudo touch /etc/text-pair/global_settings.ini
        echo "## DATABASE SETTINGS ##" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "[DATABASE]" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_name =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_user =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_password =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "Make sure you create a PostgreSQL database with a user with read/write access to that database and configure /etc/text-pair/global_settings.ini accordingly."
else
    echo "/etc/text-pair/global_settings.ini already exists, not modifying..."
fi
