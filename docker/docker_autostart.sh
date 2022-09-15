#!/bin/bash
if [ ! -d "/data/psql_data" ]; then
su postgres <<'EOF'
/usr/lib/postgresql/14/bin/initdb --pgdata=/data/psql_data;
cd /data/psql_data
/usr/lib/postgresql/14/bin/pg_ctl -D /data/psql_data/ -l logfile start
psql -c "create database
textpair ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';"
psql -d textpair -c "create extension pg_trgm;"
psql -c "create role textpair with login password 'textpair';"
psql -c "GRANT ALL PRIVILEGES ON database textpair to textpair;"
EOF
perl -pi -e 's/^(local.*)peer$/md5/;' /data/psql_data/pg_hba.conf
su postgres <<'EOF'
cd /data/psql_data
/usr/lib/postgresql/14/bin/pg_ctl -D /data/psql_data/ restart
EOF
mkdir -p /data/text-pair && ln -s /data/text-pair /var/www/html/text-pair
else
service postgresql stop
chown -R postgres /data/psql_data
su postgres <<'EOF'
cd /data/psql_data
rm -f /data/psql_data/postmaster.pid
/usr/lib/postgresql/14/bin/pg_ctl -D /data/psql_data/ restart
EOF
fi
if [ ! -d /var/www/html/text-pair ]; then
ln -s /data/text-pair /var/www/html/text-pair
fi
cd /var/lib/text-pair/api_server
./web_server.sh &
/bin/bash
