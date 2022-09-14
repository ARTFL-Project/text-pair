FROM artfl/philologic:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y postgresql postgresql-contrib apache2-dev curl git locales libpq-dev

RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash && apt-get install -y nodejs

RUN apt-get clean && rm -rf /var/lib/apt

RUN mkdir textpair && curl -L  https://github.com/ARTFL-Project/text-pair/archive/v2.0-beta.5.tar.gz | tar xz -C textpair --strip-components 1 &&\
    cd textpair && sh install.sh

RUN echo "<Location /text-pair-api>\nProxyPass http://localhost:8000 Keepalive=On\nProxyPassReverse http://localhost:8000\n</Location>\n<Location /text-pair>\nProxyPass http://localhost:8000 Keepalive=On\nProxyPassReverse http://localhost:8000\n</Location>\n" >> /etc/apache2/sites-enabled/000-default.conf

RUN sed -i 's/PORT=80/PORT=8000/g' /var/lib/text-pair/api_server/web_server.sh

RUN echo "[WEB_APP]\nweb_app_path = /var/www/html/text-pair\n[DATABASE]\ndatabase_name = textpair\ndatabase_user = textpair\ndatabase_password = textpair" > /etc/text-pair/global_settings.ini

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN echo "#!/bin/bash\nif [ ! -d \"/data/psql_data\" ]; then\nsu postgres <<'EOF'\n/usr/lib/postgresql/12/bin/initdb --pgdata=/data/psql_data;\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile start\npsql -c \"create database\ntextpair ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';\"\npsql -d textpair -c \"create extension pg_trgm;\"\npsql -c \"create role textpair with login password 'textpair';\"\npsql -c \"GRANT ALL PRIVILEGES ON database textpair to textpair;\"\nEOF\nperl -pi -e 's/^(local.*)peer$/$1md5/;' /data/psql_data/pg_hba.conf\nsu postgres <<'EOF'\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile restart\nEOF\nmkdir -p /data/text-pair && ln -s /data/text-pair /var/www/html/text-pair\nelse\nservice postgresql stop\nchown -R postgres /data/psql_data\nsu postgres <<'EOF'\ncd /data/psql_data\nrm -f /data/psql_data/postmaster.pid\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile restart\nEOF\nfi\nif [ ! -d "/var/www/html/text-pair" ]; then\nln -s /data/text-pair /var/www/html/text-pair\nfi\napachectl start\ncd /var/lib/text-pair/api_server\n./web_server.sh &\n/bin/bash" > /usr/local/bin/init_textpair_db && chmod +x /usr/local/bin/init_textpair_db

CMD ["/usr/local/bin/init_textpair_db"]
