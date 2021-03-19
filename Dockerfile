FROM artfl/philologic:latest

RUN apt update && apt install -y postgresql postgresql-contrib apache2-dev curl git locales

RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash && apt-get install -y nodejs

RUN mkdir textpair && curl -L  https://github.com/ARTFL-Project/text-pair/archive/v2.0-beta.3.tar.gz | tar xz -C textpair --strip-components 1 &&\
    cd textpair && sh install.sh

RUN echo "Include /etc/text-pair/*conf" >> /etc/apache2/apache2.conf && \
    echo "LoadModule wsgi_module \"/usr/local/lib/python3.8/dist-packages/mod_wsgi/server/mod_wsgi-py38.cpython-38-x86_64-linux-gnu.so\"" > /etc/apache2/mods-enabled/wsgi.conf && \
    service apache2 restart && \
    echo "## DATABASE SETTINGS ##\n[DATABASE]\ndatabase_name = textpair\ndatabase_user = textpair\ndatabase_password = textpair" > /etc/text-pair/global_settings.ini

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

RUN echo "#!/bin/bash\nif [ ! -d \"/data/psql_data\" ]; then\nsu postgres <<'EOF'\n/usr/lib/postgresql/12/bin/initdb --pgdata=/data/psql_data;\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile start\npsql -c \"create database\ntextpair ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';\"\npsql -d textpair -c \"create extension pg_trgm;\"\npsql -c \"create role textpair with login password 'textpair';\"\npsql -c \"GRANT ALL PRIVILEGES ON database textpair to textpair;\"\nEOF\nperl -pi -e 's/^(local.*)peer$/$1md5/;' /data/psql_data/pg_hba.conf\nsu postgres <<'EOF'\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile restart\nEOF\nmkdir /data/text-pair && ln -s /data/text-pair /var/www/html/text-pair\nelse\nsu postgres <<'EOF'\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile restart\nEOF\nln -s /data/text-pair /var/www/html/text-pair\nfi\napachectl start\n/bin/bash" > /usr/local/bin/init_textpair_db && chmod +x /usr/local/bin/init_textpair_db

# RUN echo "#!/bin/bash\nif [ ! -d \"/data/psql_data\" ]; then\nsu postgres <<'EOF'\n/usr/lib/postgresql/12/bin/initdb --pgdata=/data/psql_data;\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile start\npsql -c \"create database\ntextpair ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';\"\npsql -d textpair -c \"create extension pg_trgm;\"\npsql -c \"create role textpair with login password 'textpair';\"\npsql -c \"GRANT ALL PRIVILEGES ON database textpair to textpair;\"\nEOF\nperl -pi -e 's/^(local.*)peer$/$1md5/;' /data/psql_data/pg_hba.conf\nsu postgres <<'EOF'\ncd /data/psql_data\n/usr/lib/postgresql/12/bin/pg_ctl -D /data/psql_data/ -l logfile restart\nEOF\nmkdir /data/text-pair && ln -s /data/text-pair /var/www/html/text-pair\nfi\nservice apache2 start\n/bin/bash" > /usr/local/bin/init_textpair_db && chmod +x /usr/local/bin/init_textpair_db

CMD ["/usr/local/bin/init_textpair_db"]