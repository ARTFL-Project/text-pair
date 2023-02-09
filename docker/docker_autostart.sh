#!/bin/bash
if [ ! -d "/data/psql_data" ]; then
su postgres <<'EOF'
service postgresql start
psql -c "create database textpair"
psql -d textpair -c "create extension pg_trgm;"
psql -c "create role textpair with login password 'textpair';"
psql -c "GRANT ALL PRIVILEGES ON database textpair to textpair;"
EOF
perl -pi -e 's/^(local.*)peer$/$1 md5/;' /etc/postgresql/14/main/pg_hba.conf
su postgres <<'EOF'
service postgresql restart
EOF
else
service postgresql stop
su postgres <<'EOF'
rm -f /var/lib/postgresql/14/main/postmaster.pid
service postgresql restart
EOF
fi
cd /var/lib/text-pair/api_server
./web_server.sh &
/bin/bash
