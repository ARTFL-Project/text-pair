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

RUN apt update && apt install -y postgresql postgresql-contrib curl git locales libpq-dev python3-pip sudo ripgrep liblz4-tool && curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash && apt-get install -y nodejs && apt-get clean && rm -rf /var/lib/apt && cd /textpair && sh install.sh && mkdir -p /var/www/html/text-pair && echo "[WEB_APP]\nweb_app_path = /var/www/html/text-pair\napi_server = http://localhost/text-pair-api\n[DATABASE]\ndatabase_name = textpair\ndatabase_user = textpair\ndatabase_password = textpair" > /etc/text-pair/global_settings.ini && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

CMD ["/usr/local/bin/init_textpair_db"]
