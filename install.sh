#!/bin/sh

sudo pip3 install lib/.[web] --upgrade

echo "\nMoving web application components into place..."
sudo mkdir -p /var/lib/text-pair
if [ ! -f /var/lib/text-pair/api_server/web_server.sh ]
    then
        sudo cp -Rf api_server /var/lib/text-pair/api_server/
else
    echo "/var/lib/text-pair/api_server/web_server.sh already exists, not modifying..."
fi

if [ -d web/web_app/node_modules ]
    then
        sudo rm -rf web/web_app/node_modules
fi
sudo cp -Rf api /var/lib/text-pair/
sudo cp -Rf web-app /var/lib/text-pair/
sudo cp -Rf config /var/lib/text-pair/

echo "\nMoving global configuration into place..."
sudo mkdir -p /etc/text-pair
if [ ! -f /etc/text-pair/global_settings.ini ]
    then
        sudo touch /etc/text-pair/global_settings.ini
        echo "[WEB_APP]" | sudo tee -a /etc/topologic/global_settings.ini > /dev/null
        echo "web_app_path =" | sudo tee -a /etc/topologic/global_settings.ini > /dev/null
        echo "[DATABASE]" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_name =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_user =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_password =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "Make sure you create a PostgreSQL database with a user with read/write access to that database and configure /etc/text-pair/global_settings.ini accordingly."
else
    echo "/etc/text-pair/global_settings.ini already exists, not modifying..."
fi

echo "\n## IMPORTANT ##"
echo "In order to start the TextPAIR web app, you need to configure and start up the web_server.sh script."
echo "You can either:"
echo "- Start it manually at /var/lib/text-pair/api_server/web_server.sh"
echo "- Use the systemd init script located at /var/lib/text-pair/api_server/textpair.service: for this you will need to copy it to your OS systemd folder (usually /etc/systemd/system/) and run 'systemctl enable textpair && systemctl start textpair' as root"