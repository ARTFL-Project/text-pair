FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /textpair
COPY api /textpair/api
COPY api_server /textpair/api_server
COPY config /textpair/config
COPY extras /textpair/extras
COPY lib /textpair/lib
COPY web-app /textpair/web-app
COPY install.sh /textpair/install.sh
COPY docker_autostart.sh /usr/local/bin/init_textpair_db
COPY textpair /textpair/textpair

RUN apt update && apt install -y postgresql postgresql-contrib curl git locales libpq-dev python3-pip sudo ripgrep liblz4-tool && curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash && apt-get install -y nodejs && apt-get clean && rm -rf /var/lib/apt && cd /textpair && ./install.sh && mkdir -p /var/www/html/text-pair && echo "[WEB_APP]\nweb_app_path = /var/www/html/text-pair\napi_server = http://localhost/text-pair-api\n[DATABASE]\ndatabase_name = textpair\ndatabase_user = textpair\ndatabase_password = textpair" > /etc/text-pair/global_settings.ini && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
RUN mkdir /etc/philologic && echo 'database_root = "/var/www/html/philologic"\nurl_root = "http://localhost/philologic/"\nweb_app_dir = "/var/lib/philologic4/web_app/"' > /etc/philologic/philologic4.cfg

USER postgres
RUN service postgresql start && sleep 5 && \
    psql --command "CREATE DATABASE textpair;" && \
    psql --command "CREATE ROLE textpair WITH LOGIN PASSWORD 'textpair';" && \
    psql --command "GRANT ALL PRIVILEGES ON DATABASE textpair TO textpair;" && \
    psql -d textpair --command "CREATE EXTENSION pg_trgm;" && \
    perl -pi -e 's/^(local.*)peer$/$1 md5/;' /etc/postgresql/14/main/pg_hba.conf

USER root
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

CMD ["/usr/local/bin/init_textpair_db"]