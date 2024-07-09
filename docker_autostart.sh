#!/bin/bash
service postgresql stop
rm -f /var/lib/postgresql/14/main/postmaster.pid
service postgresql restart
cd /var/lib/text-pair/api_server
sh /var/lib/text-pair/api_server/web_server.sh &
/bin/bash
